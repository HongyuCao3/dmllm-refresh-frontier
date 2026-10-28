"""Gate verdicts (.claude/research.md, thresholds frozen 2026-10-27). Prints one `GATE <name> PASS|FAIL` line;
exit code 0 on PASS, 1 on FAIL, 2 if inputs are incomplete.

    python src/scripts/gate_check.py g1 [--identity ROWS] [--frontier GENERATIONS] [--out SUMMARY_JSON]

G1 (i)  unified loop token-identical to upstream at interval 1 / 4 / 32 (src/scripts/g1_identity.py rows).
G1 (ii) per interval in {1, 2, 4, 8, 16, 32}: speedup over full within ±20 % (relative) of Faster but Different
        Tab. 3, Jaccard vs full within ±0.05 (absolute), both monotone in the interval.
Speedup = mean full time / mean interval time over the same samples (as Tab. 3). Jaccard = over lowercased
alphabetic word sets (paper Sec. 4), against the full output of the same sample, averaged over samples.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

ACCT = os.popen("id -un").read().strip()
OUT = f"/scratch/{ACCT}/outputs/dmllm_refresh"
# Faster but Different (arXiv 2607.29079) Tab. 3: interval -> (seconds, Jaccard); full = 71.90 s.
PAPER_FULL_S = 71.90
PAPER = {32: (6.37, 0.420), 16: (8.12, 0.463), 8: (11.16, 0.482), 4: (17.53, 0.572), 2: (30.11, 0.666),
         1: (55.29, 0.987)}


def words(text):
    return set(re.findall(r"[a-z]+", text.lower()))


def jaccard(a, b):
    a, b = words(a), words(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def g1_identity(path):
    if not os.path.exists(path):
        return False, False, {}
    rows = {(r["sample_id"], r["threshold"], r["k"], r["impl"]): r for r in map(json.loads, open(path))}
    cells = defaultdict(lambda: [0, 0])
    for (sid, thr, k, impl), r in rows.items():
        if impl != "ours":
            continue
        ref = rows.get((sid, thr, k, "upstream"))
        if ref is None:
            continue
        cells[(thr, k)][0] += ref["tokens"] == r["tokens"]
        cells[(thr, k)][1] += 1
    complete = {k for _, k in cells} >= {1, 4, 32}
    ok = complete and all(same == n for same, n in cells.values())
    return ok, complete, {f"threshold={thr} interval={k}": f"{same}/{n}" for (thr, k), (same, n) in sorted(
        cells.items(), key=lambda kv: (str(kv[0][0]), kv[0][1]))}


def g1_frontier(path):
    if not os.path.exists(path):
        return False, False, {}
    by = defaultdict(dict)
    for r in map(json.loads, open(path)):
        by[r["arm"]][r["sample_id"]] = r
    ids = sorted(by["full"])
    full_s = sum(by["full"][i]["gen_s"] for i in ids)
    table, ok, complete = {}, True, True
    for k, (p_s, p_j) in sorted(PAPER.items()):
        arm = by.get(f"interval{k}", {})
        if set(arm) != set(ids):
            complete = False
            continue
        speedup = full_s / sum(arm[i]["gen_s"] for i in ids)
        jac = sum(jaccard(arm[i]["text"], by["full"][i]["text"]) for i in ids) / len(ids)
        p_speedup = PAPER_FULL_S / p_s
        row = dict(speedup=round(speedup, 2), paper_speedup=round(p_speedup, 2),
                   speedup_ok=abs(speedup / p_speedup - 1) <= 0.20,
                   jaccard=round(jac, 3), paper_jaccard=p_j, jaccard_ok=abs(jac - p_j) <= 0.05,
                   exact_match=sum(arm[i]["tokens"] == by["full"][i]["tokens"] for i in ids),
                   gen_s_mean=round(sum(arm[i]["gen_s"] for i in ids) / len(ids), 2),
                   r_mean=round(sum(arm[i]["r"] for i in ids) / len(ids), 4))
        table[k] = row
        ok &= row["speedup_ok"] and row["jaccard_ok"]
    ks = sorted(table)
    mono_speed = all(table[a]["speedup"] <= table[b]["speedup"] for a, b in zip(ks, ks[1:]))
    mono_jac = all(table[a]["jaccard"] >= table[b]["jaccard"] for a, b in zip(ks, ks[1:]))
    ok &= complete and mono_speed and mono_jac
    return ok, complete, dict(n=len(ids), full_gen_s_mean=round(full_s / len(ids), 2),
                              gpu=by["full"][ids[0]]["gpu"], intervals=table,
                              monotone_speedup=mono_speed, monotone_jaccard=mono_jac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=["g1"])
    ap.add_argument("--identity", default=f"{OUT}/phase1_g1_identity/rows.jsonl")
    ap.add_argument("--frontier", default=f"{OUT}/mme_caption_frontier/generations.jsonl")
    ap.add_argument("--out", default="results/phase1/mme_caption_frontier/g1_summary.json")
    args = ap.parse_args()

    ok_i, complete_i, ident = g1_identity(args.identity)
    ok_ii, complete_ii, front = g1_frontier(args.frontier)
    summary = dict(gate="G1", identity=ident, identity_pass=ok_i, frontier=front, frontier_pass=ok_ii)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    if not (complete_i and complete_ii):
        print("GATE G1 INCOMPLETE")
        sys.exit(2)
    print(f"GATE G1 {'PASS' if ok_i and ok_ii else 'FAIL'} (i) {'PASS' if ok_i else 'FAIL'} "
          f"(ii) {'PASS' if ok_ii else 'FAIL'}")
    sys.exit(0 if ok_i and ok_ii else 1)


if __name__ == "__main__":
    main()
