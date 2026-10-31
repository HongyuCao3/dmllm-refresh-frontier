"""Build the plan-v3 training-side data from VLAA-Thinking GRPO-25K: the digit + multiple-choice subsets (the ones
2607.08056 used: CLEVR-Math, ArxivQA, GeoQA170K), deduplicated against MathVerse testmini, split into a dev split
for phase B2 and a training split for phase D.

Prompt: the VLAA answer-format directive (first or last line, e.g. "Answer with the option's letter ...",
"Keep it short and to the point.") is dropped and the question is wrapped like MathVerse's `query_cot` + " /think"
(src/scripts/prepare_mathverse.py), so training and evaluation prompts share one format. `question_type` and
`question_for_eval` follow the MathVerse manifest, so src/scripts/score_mathverse.correct scores both.

Dedup (plan v3, "实验设置"): MathVerse collects its problems from GeoQA, Geometry3K and others, which VLAA's
GeoQA170K also contains. A VLAA item is dropped if (a) its image's 64-bit difference hash is within Hamming
distance `--max-ham` of any MathVerse testmini image (all versions), or (b) its question's word set has Jaccard
>= `--min-jac` with any MathVerse question. Both thresholds are conservative (drop when in doubt); the report
lists the counts and example pairs for a manual check.

CPU-only, interactive node, llada-v env, after the image tarballs are extracted under <out>/images/<source>/:
    python src/scripts/prepare_vlaa.py [--dev 200] [--seed 0]
"""
import argparse
import io
import json
import os
import random
import re
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

ACCT = os.popen("id -un").read().strip()
COT = {"mcq": "Please first conduct reasoning, and then answer the question and provide the correct option letter, "
              "e.g., A, B, C, D, at the end.\nQuestion: ",
       "digit": "Please first conduct reasoning, and then answer the question and provide the final value, "
                "e.g., 1, 2.5, 300, at the end.\nQuestion: "}
OPTION = re.compile(r"^\s*\(?[A-F]\s*[.:：)]")
DIRECTIVE = re.compile(r"(?i)letter|short and to the point|concise|brief|single (word|number)|only the (number|"
                       r"answer)|answer (directly|only)|one word|just the (number|answer)|directly")


def strip_directive(q):
    """Drop the answer-format directive line(s) at either end; returns (question, removed lines)."""
    lines, removed = q.strip().split("\n"), []
    for end in (0, -1):
        if len(lines) > 1 and DIRECTIVE.search(lines[end]) and not OPTION.match(lines[end]):
            removed.append(lines.pop(end))
    return "\n".join(lines).strip(), removed


def dhash(img, size=8):
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten())


def words(s):
    return frozenset(re.findall(r"[a-z]+|\d+", s.lower())) - {"the", "a", "of", "is", "in", "and", "to", "as", "shown",
                                                              "figure", "what", "if", "then", "choices"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"/scratch/{ACCT}/data/dmllm_refresh/vlaa_thinking")
    ap.add_argument("--dev", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-ham", type=int, default=6)
    ap.add_argument("--min-jac", type=float, default=0.6)
    args = ap.parse_args()

    src = hf_hub_download("UCSC-VLAA/VLAA-Thinking", "VLAA-Thinking-GRPO-25K.json", repo_type="dataset")
    rows = [r for r in map(json.loads, (l for l in open(src) if l.strip())) if r["verifier_type"] in COT]

    mv = pq.read_table(hf_hub_download("CaraJ/MathVerse-lmmseval", "testmini.parquet", repo_type="dataset")).to_pylist()
    mv_hash = np.stack([dhash(Image.open(io.BytesIO(r["image"]["bytes"]))) for r in mv if r.get("image")])
    mv_words = list({words(r["question"]) for r in mv if r["question"]})
    print(f"MathVerse testmini: {len(mv)} rows, {len(mv_hash)} images, {len(mv_words)} distinct questions", flush=True)

    kept, dropped, removed_lines = [], Counter(), Counter()
    examples = []
    for r in rows:
        q, rem = strip_directive(r["question"])
        removed_lines.update(l[:60] for l in rem)
        img_path = os.path.join(args.out, "images", r["image"])
        h = dhash(Image.open(img_path))
        ham = np.unpackbits(mv_hash ^ h, axis=1).sum(1).min()
        w = words(q)
        jac = max((len(w & m) / len(w | m) for m in mv_words if w and m), default=0.0)
        reason = "image" if ham <= args.max_ham else "text" if jac >= args.min_jac else None
        if reason:
            dropped[(reason, r["image"].split("/")[0])] += 1
            if len(examples) < 20:
                examples.append(dict(id=r["id"], reason=reason, ham=int(ham), jac=round(jac, 2), question=q[:200]))
            continue
        kept.append(dict(vlaa_id=r["id"], source=r["image"].split("/")[0], verifier_type=r["verifier_type"],
                         question_type="multi-choice" if r["verifier_type"] == "mcq" else "free-form",
                         question=COT[r["verifier_type"]] + q + " /think", question_for_eval=q,
                         answer=r["gt"].strip(), image=img_path))

    random.Random(args.seed).shuffle(kept)
    for i, r in enumerate(kept):
        r["sample_id"] = i
        r["split"] = "dev" if i < args.dev else "train"
    with open(os.path.join(args.out, "manifest.jsonl"), "w") as fh:
        for r in kept:
            fh.write(json.dumps(r) + "\n")
    report = dict(n_subset=len(rows), n_kept=len(kept), n_dev=min(args.dev, len(kept)),
                  kept_by_source=Counter(r["source"] for r in kept),
                  dev_by_source=Counter(r["source"] for r in kept[:args.dev]),
                  dropped={f"{a}/{b}": n for (a, b), n in dropped.items()}, max_ham=args.max_ham,
                  min_jac=args.min_jac, removed_directives=removed_lines.most_common(30), drop_examples=examples)
    json.dump(report, open(os.path.join(args.out, "dedup_report.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k not in ("drop_examples", "removed_directives")}, indent=1))


if __name__ == "__main__":
    main()
