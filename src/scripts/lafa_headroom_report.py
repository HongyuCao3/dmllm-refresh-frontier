"""Report for backup-line experiment E1 (src/scripts/lafa_draft_headroom.py). CPU only, sarlafa env.

Per task and draft variant (diagnostics, not reported quality numbers; the decoded output is AR greedy throughout):
  agree[i]   share of sites whose draft slot i equals y[p+i]
  accept     mean accepted prefix length A (leading agreeing slots, 0-4)
  tpf_est    mean(max(A, 1)) / K: tokens per forward of a self-speculative round with a K-jump draft (a miss still
             commits the AR token with one correction forward). Sites are every `stride`-th position, not the
             positions where real rounds start, so warp1's estimate is checked against the decoder's own
             `accepted_per_forward` (spec_tpf)
  any_agree  mean number of agreeing slots (non-prefix bound: the draft knew the token, the prefix rule lost it)
Oracles: best variant per site among the K=4 variants (schedule chosen per block with hindsight), and per site
best of all variants. Paired differences vs warp1 / warp4 with problem-level bootstrap 95% CIs.

Determination (warp4, K=4 jumps): per slot, the first jump after which the argmax stays at its final value
("predicted"), and the first latent state after which the token nearest to the state stays equal to the final
argmax ("latent"; states exist after jumps 1-3, 4 = only at readout). Order test: over slot pairs i < j with
different determination jumps, the share where the left slot is determined first (0.5 = no order), with a
problem-level bootstrap CI. Breakdown by token class and by AR confidence.

    python src/scripts/lafa_headroom_report.py [--dir /scratch/<acct>/outputs/dmllm_refresh/lafa_e1_headroom]
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ACCT = os.popen("id -un").read().strip()
K_OF = {"warp1": 1, "warp2": 2, "warp4": 4, "l2r4": 4, "rev4": 4, "conf4": 4}
_BOXED = re.compile(r"\\(?:boxed|fbox)\s*\{")


def boxed(text):
    m = list(_BOXED.finditer(text))
    if not m:
        return None
    depth, start = 1, m[-1].end()
    for i in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[start:i].strip()
    return None


def correct(task, text, gold):
    """Scorers as in dKV-ARKV src/dkvar/eval/tasks.py (v1)."""
    if task == "gsm8k":
        gold = gold.split("####")[-1].strip().replace(",", "")
        nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        pred = boxed(text) or (nums[-1] if nums else None)
    else:
        pred = boxed(text)
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        pass
    norm = lambda s: re.sub(r"\s+", "", s.strip().strip("$"))
    if norm(pred) == norm(gold):
        return True
    from math_verify import parse, verify
    return bool(verify(parse(f"${gold}$"), parse(f"${pred}$")))


def boot(per_problem, n=5000, seed=0):
    """Mean and 95% CI of a statistic given as per-problem (numerator, denominator) pairs."""
    a = np.array(per_problem, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    s = a[idx].sum(1)
    est = a[:, 0].sum() / max(a[:, 1].sum(), 1e-9)
    r = s[:, 0] / np.maximum(s[:, 1], 1e-9)
    return [round(est, 4), round(float(np.percentile(r, 2.5)), 4), round(float(np.percentile(r, 97.5)), 4)]


def det_step(seq, final):
    """First index from which seq stays equal to final; len(seq) if never."""
    k = len(seq)
    while k > 0 and seq[k - 1] == final:
        k -= 1
    return k


def token_class(s):
    t = s.strip()
    if not t:
        return "space/newline"
    if re.fullmatch(r"-?\d+(\.\d+)?", t):
        return "number"
    if re.fullmatch(r"[\W_]+", t):
        return "symbol/punct"
    if t.lower() in {"the", "a", "an", "of", "to", "is", "and", "in", "so", "we", "that", "for", "it", "this", "be",
                     "are", "by", "with", "as", "on", "then", "which", "has", "have", "at", "from", "each", "total"}:
        return "function word"
    return "content word"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=f"/scratch/{ACCT}/outputs/dmllm_refresh/lafa_e1_headroom")
    ap.add_argument("--model", default="lafa-hf/lafa-7B")
    ap.add_argument("--out", default="results/lafa_e1/headroom_summary.json")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    rows = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(args.dir, "*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            rows[r["task"]].append(r)
    summary = {}
    for task, rs in rows.items():
        rs.sort(key=lambda r: r["idx"])
        variants = [v for v in K_OF if v in rs[0]["sites"][0]]
        S = dict(n=len(rs), accuracy=round(np.mean([correct(task, r["text"], r["gold"]) for r in rs]), 4),
                 gen_len_mean=round(np.mean([len(r["y"]) for r in rs]), 1),
                 ar_match=boot([(sum(r["ar_match"]), len(r["ar_match"])) for r in rs]),
                 spec_tpf=boot([(r["spec_stats"]["gen_len"], r["spec_stats"]["total_steps"]) for r in rs]),
                 spec_s_mean=round(np.mean([r["spec_s"] for r in rs]), 2), variants={})
        per = {v: defaultdict(list) for v in variants}  # per problem lists
        best4, bestall = [], []
        for r in rs:
            y = r["y"]
            acc = {v: [] for v in variants}
            for s in r["sites"]:
                p = s["p"]
                tgt = y[p:p + 4]
                if len(tgt) < 4:
                    continue
                for v in variants:
                    fin = s[v]["argmax"][-1]
                    ag = [int(fin[i] == tgt[i]) for i in range(4)]
                    A = next((i for i in range(4) if not ag[i]), 4)
                    acc[v].append((ag, A))
            for v in variants:
                ags = np.array([a for a, _ in acc[v]])
                As = np.array([A for _, A in acc[v]])
                n = len(As)
                per[v]["agree"].append([(ags[:, i].sum(), n) for i in range(4)])
                per[v]["accept"].append((As.sum(), n))
                per[v]["tpf"].append((np.maximum(As, 1).sum() / K_OF[v], n))
                per[v]["any"].append((ags.sum(), n))
            v4 = [v for v in variants if K_OF[v] == 4]
            if v4:
                best4.append((np.max([[A for _, A in acc[v]] for v in v4], 0).sum(), len(acc[v4[0]])))
            bestall.append((np.max([[max(A, 1) / K_OF[v] for _, A in acc[v]] for v in variants], 0).sum(),
                            len(acc[variants[0]])))
        for v in variants:
            S["variants"][v] = dict(K=K_OF[v], accept=boot(per[v]["accept"]), tpf_est=boot(per[v]["tpf"]),
                                    any_agree=boot(per[v]["any"]),
                                    agree=[boot([pp[i] for pp in per[v]["agree"]])[0] for i in range(4)])
        # paired differences in accepted prefix length (sites aligned within a problem)
        for ref in ("warp1", "warp4"):
            for v in variants:
                if v == ref or ref not in per:
                    continue
                d = [(a[0] - b[0], a[1]) for a, b in zip(per[v]["accept"], per[ref]["accept"])]
                S["variants"][v][f"accept_minus_{ref}"] = boot(d)
        if best4:
            S["oracle_best_K4_variant_accept"] = boot(best4)
        S["oracle_best_variant_tpf"] = boot(bestall)

        if "warp4" in variants:
            pred_d, lat_d, order_pred, order_lat = defaultdict(list), defaultdict(list), [], []
            by_class, by_conf = defaultdict(list), defaultdict(list)
            for r in rs:
                y, conc_p, conc_l = r["y"], [0, 0], [0, 0]
                for s in r["sites"]:
                    p, w = s["p"], s["warp4"]
                    if p + 4 > len(y):
                        continue
                    fin = w["argmax"][-1]
                    dp = [det_step([w["argmax"][k][i] for k in range(4)], fin[i]) for i in range(4)]
                    dl = [det_step([w["latent"][k][i] for k in range(3)] + [fin[i]], fin[i]) for i in range(4)]
                    for i in range(4):
                        pred_d[i].append(dp[i])
                        lat_d[i].append(dl[i])
                        c = token_class(tok.decode([fin[i]]))
                        by_class[c].append((dp[i], dl[i], int(fin[i] == y[p + i])))
                        q = r["ar_p1"][p + i]
                        by_conf["AR p1<0.5" if q < .5 else "AR p1 0.5-0.9" if q < .9 else "AR p1>=0.9"].append(
                            (dp[i], dl[i], int(fin[i] == y[p + i])))
                    for d, conc in ((dp, conc_p), (dl, conc_l)):
                        for i in range(4):
                            for j in range(i + 1, 4):
                                if d[i] != d[j]:
                                    conc[0] += d[i] < d[j]
                                    conc[1] += 1
                order_pred.append(tuple(conc_p))
                order_lat.append(tuple(conc_l))
            dist = lambda xs, m: [round(float(np.mean(np.array(xs) == k)), 3) for k in range(m + 1)]
            S["determination_warp4"] = dict(
                note="jump index 0-3 = determined after that many jumps (0 = at the first readout); latent 3 = only "
                     "at the final readout",
                predicted_by_slot={i: dict(mean=round(float(np.mean(pred_d[i])), 3), dist=dist(pred_d[i], 3))
                                   for i in range(4)},
                latent_by_slot={i: dict(mean=round(float(np.mean(lat_d[i])), 3), dist=dist(lat_d[i], 3))
                                for i in range(4)},
                left_first_share_predicted=boot(order_pred), left_first_share_latent=boot(order_lat),
                by_token_class={c: dict(n=len(x), predicted_mean=round(float(np.mean([a for a, _, _ in x])), 3),
                                        latent_mean=round(float(np.mean([b for _, b, _ in x])), 3),
                                        draft_agree=round(float(np.mean([g for _, _, g in x])), 3))
                                for c, x in sorted(by_class.items())},
                by_ar_confidence={c: dict(n=len(x), predicted_mean=round(float(np.mean([a for a, _, _ in x])), 3),
                                          latent_mean=round(float(np.mean([b for _, b, _ in x])), 3),
                                          draft_agree=round(float(np.mean([g for _, _, g in x])), 3))
                                  for c, x in sorted(by_conf.items())})
        summary[task] = S
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
