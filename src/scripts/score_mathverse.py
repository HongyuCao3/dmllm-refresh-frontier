"""Rule-based MathVerse scoring of a frontier campaign, with paired comparisons of every arm against full.

    python src/scripts/score_mathverse.py <campaign name> [--out SUMMARY_JSON] [--dump PER_ROW_JSONL]

No LLM judge (user decision 2026-10-29). The answer segment is the text after the last "Answer" marker, else after
"</think>", else the last 200 characters. Multiple choice: the last standalone option letter (A-E) in that segment, else the
option whose text the segment contains (longest match).
Free-form: every number in the gold answer (fractions evaluated, unit powers such as cm^2 dropped, pi required
when the gold has it) must appear, in order,
as the last numbers of the answer segment, within half a unit of the gold's last decimal place (at least 1e-3
relative). A gold answer without any number is unscorable and counts as wrong for every arm alike.

Reports accuracy per arm, and per arm vs full: paired Δ [95% CI] (bootstrap over problems, 10,000 resamples,
seed 0) and an exact McNemar p-value. Reads all generations*.jsonl shards of the campaign.
"""
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

ACCT = os.popen("id -un").read().strip()
OUT = f"/scratch/{ACCT}/outputs/dmllm_refresh"
NUM = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def answer_segment(text):
    for pat in (r"(?i)answer\s*(?:is)?\s*[:：]?", r"</think>"):
        parts = re.split(pat, text)
        if len(parts) > 1 and parts[-1].strip():
            return parts[-1]
    return text[-200:]


UNIT_POWER = re.compile(r"(?:\\mathrm\{\s*~?\s*)?(cm|mm|km|m|ft|in|units?|metres?|meters?)\s*\}?\s*(?:\^\s*\{?\s*[23]\s*\}?|[²³])")


def numbers(s):
    s = UNIT_POWER.sub(r"\1", s)
    s = re.sub(r"\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    vals = []
    frac = r"\(?(" + NUM.pattern + r")\)?\s*/\s*\(?(" + NUM.pattern + r")\)?"
    for m in re.finditer(frac + "|" + NUM.pattern, s):
        if m.group(1) is not None:
            a, b = float(m.group(1).replace(",", "")), float(m.group(2).replace(",", ""))
            vals.append((a / b if b else math.inf, 6))
        else:
            tok = m.group(0).replace(",", "")
            vals.append((float(tok), len(tok.split(".")[1]) if "." in tok else 0))
    return vals


def norm(s):
    return re.sub(r"\\hline|\\left|\\right|[\s$\\{}]", "", s).lower()


def correct(row, text):
    seg = answer_segment(text)
    if row["question_type"] == "multi-choice":
        letters = re.findall(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", seg)
        if not letters:
            # answered with the option's text: take the longest option text contained in the segment
            opts = re.findall(r"^\s*([A-E])\s*[.:：)]\s*(.+?)\s*$", row["question_for_eval"], re.M)
            hits = sorted((len(norm(t)), k) for k, t in opts if norm(t) and norm(t) in norm(seg))
            letters = [hits[-1][1]] if hits else []
        return bool(letters) and letters[-1] == row["answer"].strip()[:1].upper()
    gold = numbers(row["answer"])
    if not gold:
        return False
    if ("\\pi" in row["answer"] or "π" in row["answer"]) and not re.search(r"\\pi|π|pi", seg):
        return False
    pred = [v for v, _ in numbers(seg)]
    if len(pred) < len(gold):
        return False
    for (g, dec), p in zip(gold, pred[-len(gold):]):
        if abs(p - g) > max(0.5 * 10 ** -dec, 1e-3 * abs(g)) + 1e-9:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign")
    ap.add_argument("--manifest", default=f"/scratch/{ACCT}/data/dmllm_refresh/mathverse_vision_dominant/manifest.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None)
    ap.add_argument("--gen-length", type=int, default=256, help="campaign gen_length; outputs that hit the cap "
                    "have gen_length - 1 tokens (Exp 3: 127 of 128)")
    args = ap.parse_args()
    out = args.out or f"results/phase1/{args.campaign}/accuracy_summary.json"

    meta = {r["sample_id"]: r for r in map(json.loads, open(args.manifest))}
    by = defaultdict(dict)
    for f in sorted(glob.glob(f"{OUT}/{args.campaign}/generations*.jsonl")):
        for r in map(json.loads, open(f)):
            by[r["arm"]][r["sample_id"]] = r
    ids = sorted(set.intersection(*(set(v) for v in by.values())))
    arms = ["full"] + sorted((a for a in by if a != "full"), key=lambda a: int(a.removeprefix("interval")))
    ok = {a: np.array([correct(meta[i], by[a][i]["text"]) for i in ids]) for a in arms}
    full_s = np.array([by["full"][i]["gen_s"] for i in ids])
    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(ids), (10_000, len(ids)))

    rows = {}
    for a in arms:
        gen_s = np.array([by[a][i]["gen_s"] for i in ids])
        rec = dict(accuracy=round(float(ok[a].mean()), 4), n_correct=int(ok[a].sum()),
                   gen_s_mean=round(float(gen_s.mean()), 2), speedup=round(float(full_s.sum() / gen_s.sum()), 2),
                   r_mean=round(float(np.mean([by[a][i]["r"] for i in ids])), 4),
                   out_tokens_mean=round(float(np.mean([len(by[a][i]["tokens"]) for i in ids])), 1),
                   hit_length_cap=int(sum(len(by[a][i]["tokens"]) >= args.gen_length - 1 for i in ids)))
        if a != "full":
            d = ok[a].astype(float) - ok["full"].astype(float)
            ci = np.percentile(d[boot].mean(1), [2.5, 97.5])
            b, c = int(((~ok[a]) & ok["full"]).sum()), int((ok[a] & ~ok["full"]).sum())
            rec.update(delta_vs_full=round(float(d.mean()), 4), delta_ci95=[round(float(v), 4) for v in ci],
                       lost=b, gained=c, mcnemar_p=round(binomtest(c, b + c).pvalue, 4) if b + c else 1.0)
        rows[a] = rec

    by_type = {a: {t: round(float(np.mean([ok[a][j] for j, i in enumerate(ids) if meta[i]["question_type"] == t])), 4)
                   for t in ("multi-choice", "free-form")} for a in arms}
    summary = dict(campaign=args.campaign, n=len(ids), gpu=by["full"][ids[0]]["gpu"],
                   n_prompt_mean=round(float(np.mean([by["full"][i]["n_prompt"] for i in ids])), 1),
                   unscorable_free_form=sum(1 for i in ids if meta[i]["question_type"] == "free-form"
                                            and not numbers(meta[i]["answer"])),
                   arms=rows, accuracy_by_type=by_type)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(summary, open(out, "w"), indent=1)
    if args.dump:
        with open(args.dump, "w") as fh:
            for i in ids:
                for a in arms:
                    fh.write(json.dumps(dict(sample_id=i, arm=a, gold=meta[i]["answer"], correct=bool(ok[a][ids.index(i)]),
                                             segment=answer_segment(by[a][i]["text"])[-200:])) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
