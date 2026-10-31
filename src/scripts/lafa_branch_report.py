"""Report for pilot P1 (src/scripts/lafa_branch_headroom.py). CPU only. Diagnostics, single seed.

Per policy (low / early / rand; 4 branches each, plus the greedy output):
  oracle      any of greedy + branches correct (upper bound of branching, no selector)
  vote        majority answer over greedy + branches (ties broken toward greedy)
  max_lp      the candidate with the highest mean AR log-probability
  changed     share of branches whose answer differs from greedy
  fix / break branches that turn a wrong greedy answer right / a right one wrong
  extra_fwd   forwards of the branches per problem, relative to the greedy decode
All with problem-level bootstrap 95% CIs; policy minus rand and minus greedy are paired.

    python src/scripts/lafa_branch_report.py [--dir /scratch/<acct>/outputs/dmllm_refresh/lafa_p1_branch]
"""
import argparse
import glob
import json
import os
from collections import Counter

import numpy as np

ACCT = os.popen("id -un").read().strip()


def ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    b = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return [round(float(x.mean()), 4), round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=f"/scratch/{ACCT}/outputs/dmllm_refresh/lafa_p1_branch")
    ap.add_argument("--out", default="results/lafa_p1/branch_summary.json")
    args = ap.parse_args()
    rs = sorted((json.loads(l) for f in glob.glob(os.path.join(args.dir, "*.jsonl")) for l in open(f)),
                key=lambda r: r["idx"])
    greedy = np.array([r["greedy"]["correct"] for r in rs], float)
    S = dict(n=len(rs), greedy=ci(greedy), greedy_forwards=round(np.mean([r["greedy"]["forwards"] for r in rs]), 1),
             n_p1_below_05_mean=round(np.mean([r["n_p1_below_05"] for r in rs]), 1), policies={})
    res = {}
    for pol in ("low", "early", "rand", "all"):
        o, v, m, ch, fx, br, ex = [], [], [], [], [], [], []
        for r in rs:
            g = r["greedy"]
            bs = [b for b in r["branches"] if pol == "all" or b["policy"] == pol]
            if pol == "all":
                bs = list({b["p"]: b for b in bs}.values())
            cands = [g] + bs
            o.append(any(c["correct"] for c in cands))
            cnt = Counter(c["answer"] for c in cands if c["answer"] is not None)
            if cnt:
                top = max(cnt.values())
                winners = [a for a, k in cnt.items() if k == top]
                pick = g["answer"] if g["answer"] in winners else winners[0]
                v.append(next(c["correct"] for c in cands if c["answer"] == pick))
            else:
                v.append(g["correct"])
            m.append(max(cands, key=lambda c: c["mean_lp"])["correct"])
            ch.append(np.mean([b["answer"] != g["answer"] for b in bs]) if bs else 0.0)
            fx.append(sum(b["correct"] and not g["correct"] for b in bs))
            br.append(sum(g["correct"] and not b["correct"] for b in bs))
            ex.append(sum(b["forwards"] for b in bs) / max(g["forwards"], 1))
        res[pol] = dict(oracle=np.array(o, float), vote=np.array(v, float), max_lp=np.array(m, float))
        S["policies"][pol] = dict(oracle=ci(o), vote=ci(v), max_lp=ci(m), changed_answer=ci(ch),
                                  fix_per_problem=ci(fx), break_per_problem=ci(br), extra_forwards_x=ci(ex),
                                  oracle_minus_greedy=ci(np.array(o, float) - greedy),
                                  vote_minus_greedy=ci(np.array(v, float) - greedy),
                                  max_lp_minus_greedy=ci(np.array(m, float) - greedy))
    for pol in ("low", "early"):
        for k in ("oracle", "vote", "max_lp"):
            S["policies"][pol][f"{k}_minus_rand"] = ci(res[pol][k] - res["rand"][k])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(S, open(args.out, "w"), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
