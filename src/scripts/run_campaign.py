"""Run one campaign (configs/campaigns/*.json): one task x many refresh-policy arms, one model load.

Arm policies: "full" = upstream generate without cache (the reference); "interval" = unified loop with
FixedInterval(k). All arms of a sample run back to back (samples outer, arms inner), so slow drift of the GPU
hits every arm alike. Timing: one discarded warm-up per arm on the first sample, CUDA-synchronized clocks,
generation time excludes model load and image preprocessing.

Prompt: the manifest row's "question" if present (e.g. MathVerse), else the campaign's task prompt (captions).

Writes /scratch/<acct>/outputs/dmllm_refresh/<campaign>/generations.jsonl, one line per (arm, sample) with
tokens, text, gen_s and the refresh trace summary (forwards, refreshes, r = refreshes / forwards).
With --num-shards S, shard i runs samples with sample_id % S == i and writes generations.shard<i>.jsonl
(one Slurm array task per shard). Resumable: finished (arm, sample) pairs are skipped.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.methods.refresh import llada_v, loop  # noqa: E402
from src.methods.refresh.policies import FixedInterval  # noqa: E402

ACCT = os.popen("id -un").read().strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    cfg = json.load(open(args.campaign))
    task = cfg["task"]
    gen = task["gen"]
    suffix = f".shard{args.shard}" if args.num_shards > 1 else ""
    out = f"/scratch/{ACCT}/outputs/dmllm_refresh/{cfg['name']}/generations{suffix}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {(r["arm"], r["sample_id"]) for r in map(json.loads, open(out))} if os.path.exists(out) else set()
    samples = [s for s in (json.loads(l) for l in open(task["manifest"].format(acct=ACCT)))
               if s["sample_id"] < task["n"] and s["sample_id"] % args.num_shards == args.shard]

    tokenizer, model, image_processor, load_s = llada_v.load()
    gpu = torch.cuda.get_device_name(0)
    print(f"model load {load_s:.1f}s on {gpu}", flush=True)
    n_forward = []
    backbone_forward = model.model.forward

    def counted(*a, **kw):
        n_forward.append(kw["inputs_embeds"].shape[1])
        return backbone_forward(*a, **kw)

    model.model.forward = counted
    hook = loop.attach(model)

    def run(arm, embeds):
        # "full" is upstream without Fast-dLLM, i.e. with the original attention forwards restored.
        if arm["policy"] == "full":
            hook.unregister_hooks()
        else:
            hook.register_hooks()
        n_forward.clear()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if arm["policy"] == "full":
            tokens, trace = model.generate_with_embeds(inputs_embeds=embeds, tokenizer=tokenizer, **gen), None
        else:
            tokens, trace = loop.generate(hook, embeds, FixedInterval(arm["k"]), threshold=arm["threshold"],
                                          tokenizer=tokenizer, **gen)
        torch.cuda.synchronize()
        gen_s = time.perf_counter() - t0
        full_len = embeds.shape[1] + gen["gen_length"]
        refreshes = sum(l == full_len for l in n_forward)
        return dict(gen_s=gen_s, n_prompt=embeds.shape[1], forwards=len(n_forward), refreshes=refreshes,
                    r=refreshes / len(n_forward), tokens=tokens[0].tolist(),
                    text=tokenizer.decode(tokens[0], skip_special_tokens=True),
                    trace=None if trace is None else [[t, b, i, int(f), n] for t, b, i, f, n in trace])

    arms = cfg["arms"]
    warmed = set()
    with open(out, "a") as fh:
        for s in samples:
            embeds = None
            for arm in arms:
                if (arm["name"], s["sample_id"]) in done:
                    continue
                if embeds is None:
                    input_ids = llada_v.prompt_ids(tokenizer, s.get("question", task.get("prompt")))
                    embeds = llada_v.embed(model, image_processor, input_ids, s["image"])
                if arm["name"] not in warmed:
                    run(arm, embeds)
                    warmed.add(arm["name"])
                rec = dict(arm=arm["name"], sample_id=s["sample_id"], mme_id=s.get("mme_id"), gpu=gpu,
                           load_s=load_s, **run(arm, embeds))
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"sample {s['sample_id']} {arm['name']:11s} N={rec['n_prompt']} {rec['gen_s']:.2f}s "
                      f"fwd={rec['forwards']} refresh={rec['refreshes']}", flush=True)


if __name__ == "__main__":
    main()
