"""Gate A (plan v3), technical part: in block decoding the unified loop with the `Confidence()` commit policy
must be token-identical to upstream Fast-dLLM on the same problems.

Settings: MathVerse vision-dominant manifest (first n problems), gen_length 256, block_length 32, KV prefix cache
refreshed only at block starts (prefix_refresh_interval > steps per block), steps in {256, 64} with the fixed
schedule (threshold None) and the threshold rule (0.5). Three implementations per cell: upstream
`_fast_generate_with_embeds`, our loop on its upstream commit path (commit=None) and our loop with
`commit=Confidence()` (src/methods/commit). The other part of gate A (near-monotone accuracy-NFE curves) is read
from the phase-A campaign, not here.

Writes one jsonl line per (problem, steps, threshold, impl); prints the identity table and `GATE A(i) PASS|FAIL`.
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
from src.methods.commit.policies import Confidence  # noqa: E402
from src.methods.refresh import llada_v, loop  # noqa: E402
from src.methods.refresh.policies import FixedInterval  # noqa: E402

ACCT = os.popen("id -un").read().strip()
GEN = dict(gen_length=256, block_length=32, stopping_criteria=["<|eot_id|>"])
BLOCK_START_ONLY = 10 ** 9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    default=f"/scratch/{ACCT}/data/dmllm_refresh/mathverse_vision_dominant/manifest.jsonl")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--steps", default="256,64")
    ap.add_argument("--thresholds", default="none,0.5")
    ap.add_argument("--out", default=f"/scratch/{ACCT}/outputs/dmllm_refresh/v3_phaseA_identity/rows.jsonl")
    args = ap.parse_args()

    samples = [json.loads(l) for l in open(args.manifest)][: args.n]
    thresholds = [None if s == "none" else float(s) for s in args.thresholds.split(",")]
    steps_list = [int(s) for s in args.steps.split(",")]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    key = lambda r: (r["sample_id"], r["steps"], r["threshold"], r["impl"])
    rows = {key(r): r for r in map(json.loads, open(args.out))} if os.path.exists(args.out) else {}

    tokenizer, model, image_processor, load_s = llada_v.load()
    print(f"model load {load_s:.1f}s on {torch.cuda.get_device_name(0)}", flush=True)
    hook = loop.attach(model)

    def upstream(embeds, steps, thr):
        return hook._fast_generate_with_embeds(inputs_embeds=embeds, tokenizer=tokenizer, threshold=thr, steps=steps,
                                               prefix_refresh_interval=BLOCK_START_ONLY, **GEN)

    def ours(embeds, steps, thr, commit=None):
        return loop.generate(hook, embeds, FixedInterval(BLOCK_START_ONLY), steps=steps, threshold=thr,
                             tokenizer=tokenizer, commit=commit, **GEN)[0]

    impls = {"upstream": upstream, "ours_upstream_path": ours,
             "ours_confidence": lambda e, s, t: ours(e, s, t, Confidence())}
    jobs = [(s, st, thr, impl) for s in samples for st in steps_list for thr in thresholds for impl in impls]
    with open(args.out, "a") as fh:
        for s, steps, thr, impl in jobs:
            if (s["sample_id"], steps, thr, impl) in rows:
                continue
            input_ids = llada_v.prompt_ids(tokenizer, s["question"])
            embeds = llada_v.embed(model, image_processor, input_ids, s["image"])
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = impls[impl](embeds, steps, thr)
            torch.cuda.synchronize()
            r = dict(sample_id=s["sample_id"], steps=steps, threshold=thr, impl=impl,
                     gen_s=time.perf_counter() - t0, n_prompt=embeds.shape[1], tokens=out[0].tolist())
            rows[key(r)] = r
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"sample {s['sample_id']} steps={steps} thr={thr} {impl:19s} {r['gen_s']:.1f}s "
                  f"len={len(r['tokens'])}", flush=True)

    ok = True
    table = collections.defaultdict(lambda: [0, 0])
    for (sid, steps, thr, impl), r in rows.items():
        if impl == "upstream":
            continue
        same = rows[(sid, steps, thr, "upstream")]["tokens"] == r["tokens"]
        table[(steps, thr, impl)][0] += same
        table[(steps, thr, impl)][1] += 1
        ok &= same
    for (steps, thr, impl), (same, n) in sorted(table.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2])):
        print(f"steps={steps} threshold={thr} {impl:19s} identical to upstream {same}/{n}")
    print(f"GATE A(i) {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
