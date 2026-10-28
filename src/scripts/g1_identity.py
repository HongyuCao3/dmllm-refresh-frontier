"""Gate G1(i): the unified loop (src/methods/refresh/loop.py) with FixedInterval(k) must be token-identical
to upstream Fast-dLLM at prefix_refresh_interval = k, for k in {1, 4, 32}, on the same samples.

Settings: MME caption, steps = gen_length = block_length = 128, threshold 0.5 (Faster but Different Sec. 6.2)
and threshold None (fixed 1 token per step). Also runs upstream twice on interval 4 / threshold 0.5 to measure
run-to-run nondeterminism (research.md: measured once in Phase 1).

Writes one jsonl line per (sample, threshold, k, impl); prints the identity table and `G1(i) PASS|FAIL`.
Exit code 0 on PASS, 1 on FAIL. Resumable: finished rows are skipped.
"""
import argparse
import collections
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.methods.refresh import llada_v, loop  # noqa: E402
from src.methods.refresh.policies import FixedInterval  # noqa: E402

ACCT = os.popen("id -un").read().strip()
GEN = dict(steps=128, gen_length=128, block_length=128, stopping_criteria=["<|eot_id|>"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=f"/scratch/{ACCT}/data/dmllm_refresh/mme_caption/manifest.jsonl")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--intervals", default="1,4,32")
    ap.add_argument("--thresholds", default="0.5,none")
    ap.add_argument("--out", default=f"/scratch/{ACCT}/outputs/dmllm_refresh/phase1_g1_identity/rows.jsonl")
    args = ap.parse_args()

    samples = [json.loads(l) for l in open(args.manifest)][: args.n]
    thresholds = [None if s == "none" else float(s) for s in args.thresholds.split(",")]
    intervals = [int(k) for k in args.intervals.split(",")]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    key = lambda r: (r["sample_id"], r["threshold"], r["k"], r["impl"])
    rows = {key(r): r for r in map(json.loads, open(args.out))} if os.path.exists(args.out) else {}

    tokenizer, model, image_processor, load_s = llada_v.load()
    print(f"model load {load_s:.1f}s on {torch.cuda.get_device_name(0)}", flush=True)
    hook = loop.attach(model)
    input_ids = llada_v.prompt_ids(tokenizer, llada_v.CAPTION_PROMPT)

    def upstream(embeds, k, thr):
        return hook._fast_generate_with_embeds(inputs_embeds=embeds, tokenizer=tokenizer, threshold=thr,
                                               prefix_refresh_interval=k, **GEN)

    def ours(embeds, k, thr):
        return loop.generate(hook, embeds, FixedInterval(k), threshold=thr, tokenizer=tokenizer, **GEN)[0]

    jobs = [(s, thr, k, impl) for s in samples for thr in thresholds for k in intervals
            for impl in ("upstream", "ours")]
    jobs += [(s, 0.5, 4, "upstream_rerun") for s in samples if 0.5 in thresholds and 4 in intervals]
    impls = {"upstream": upstream, "ours": ours, "upstream_rerun": upstream}
    with open(args.out, "a") as fh:
        for s, thr, k, impl in jobs:
            if (s["sample_id"], thr, k, impl) in rows:
                continue
            embeds = llada_v.embed(model, image_processor, input_ids, s["image"])
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = impls[impl](embeds, k, thr)
            torch.cuda.synchronize()
            r = dict(sample_id=s["sample_id"], threshold=thr, k=k, impl=impl, gen_s=time.perf_counter() - t0,
                     tokens=out[0].tolist())
            rows[key(r)] = r
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"sample {s['sample_id']} thr={thr} k={k} {impl:15s} {r['gen_s']:.1f}s len={len(r['tokens'])}",
                  flush=True)

    ok = True
    table = collections.defaultdict(lambda: [0, 0])
    for (sid, thr, k, impl), r in rows.items():
        if impl == "upstream":
            continue
        same = rows[(sid, thr, k, "upstream")]["tokens"] == r["tokens"]
        table[(thr, k, impl)][0] += same
        table[(thr, k, impl)][1] += 1
        ok &= same or impl == "upstream_rerun"
    for (thr, k, impl), (same, n) in sorted(table.items(), key=lambda kv: (str(kv[0][0]), kv[0][1], kv[0][2])):
        print(f"threshold={thr} interval={k} {impl:15s} identical to upstream {same}/{n}")
    print(f"G1(i) {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
