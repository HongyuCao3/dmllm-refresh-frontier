"""Phase 2 replacement-error probe: how much does a stale piece of the prompt cache change the next decision?

Along a fixed trajectory (unified loop, FixedInterval(1): a fresh prompt cache at every step), at step t the probe
builds caches in which one cell is replaced by its value from step t - s (age s) and the rest stays fresh, runs
the block forward with each, and compares against the block forward with the fresh cache. The trajectory itself
is never changed, so error is measured without later divergence (plan, "replacement error").

Cells: position group {visual, text (system + question)} x layer band {0-7, 8-15, 16-23, 24-31} at ages {1, 4};
plus the whole prompt, all layers, at ages {1, 2, 4, 8} (D3 time profile). With block = gen_length every answer
position is recomputed at every step, so only the prompt can be stale.

Per (sample, step) one jsonl line: masked answer positions, the positions decoded at the previous step
(decoding order), and per (variant, age): top-1 flips among masked positions, mean KL(fresh || stale) over them,
and whether the committed decision (positions transferred and their tokens, threshold rule) changes.
Diagnostic only: never a reported quality number (.claude/research.md). Resumable per sample.

    python src/scripts/probe_replacement.py [--manifest M] [--n 50] [--out OUT_JSONL]
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava.constants import IMAGE_TOKEN_INDEX  # noqa: E402
from src.methods.refresh import llada_v, loop  # noqa: E402
from src.methods.refresh.policies import FixedInterval  # noqa: E402

ACCT = os.popen("id -un").read().strip()
BANDS = {"L0-7": range(0, 8), "L8-15": range(8, 16), "L16-23": range(16, 24), "L24-31": range(24, 32)}


def variants(n_layers):
    out = [("all", "all", s) for s in (1, 2, 4, 8)]
    out += [(g, b, s) for s in (1, 4) for g in ("visual", "text") for b in BANDS]
    assert all(max(r) < n_layers for r in BANDS.values())
    return out


def mix(fresh, stale, pos, layers):
    """Fresh cache with `pos` (bool over prefix positions) taken from `stale` in `layers`."""
    m = pos.view(1, 1, -1, 1)
    return [tuple(torch.where(m, sk, fk) for sk, fk in zip(stale[l], fresh[l])) if l in layers else fresh[l]
            for l in range(len(fresh))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=f"/scratch/{ACCT}/data/dmllm_refresh/mme_caption/manifest.jsonl")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--prompt", default=llada_v.CAPTION_PROMPT)
    ap.add_argument("--gen-length", type=int, default=128)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default=f"/scratch/{ACCT}/outputs/dmllm_refresh/phase2_probe_mme/rows.jsonl")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = {r["sample_id"] for r in map(json.loads, open(args.out)) if r.get("end")} if os.path.exists(args.out) else set()

    tokenizer, model, image_processor, load_s = llada_v.load()
    print(f"model load {load_s:.1f}s on {torch.cuda.get_device_name(0)}", flush=True)
    hook = loop.attach(model)
    lm = model
    n_layers = len(lm.model.layers)
    var = variants(n_layers)
    max_age = max(s for _, _, s in var)
    samples = [json.loads(l) for l in open(args.manifest)][: args.n]

    for smp in samples:
        if smp["sample_id"] in done:
            continue
        ids = llada_v.prompt_ids(tokenizer, smp.get("question", args.prompt))
        embeds = llada_v.embed(model, image_processor, ids, smp["image"])
        n_prompt = embeds.shape[1]
        p = int((ids[0] == IMAGE_TOKEN_INDEX).nonzero()[0])
        visual = torch.zeros(n_prompt, dtype=torch.bool, device=embeds.device)
        visual[p:p + n_prompt - ids.shape[1] + 1] = True
        groups = {"all": torch.ones_like(visual), "visual": visual, "text": ~visual}
        history, rows, prev_mask = {}, [], []
        t0 = time.perf_counter()

        def block_logits(cache, x_embeds, bs, mask):
            logits = lm.lm_head(lm.model(inputs_embeds=x_embeds[:, bs:], fast_dllm_cache=cache)[0]).float()
            for tok in loop.FORBIDDEN_TOKENS:
                logits[:, :, tok] = torch.where(mask, -float("inf"), logits[:, :, tok])
            return logits

        def decide(logits, mask, x_embeds, bs, be, n_tok):
            x = torch.full(mask.shape, 126336, dtype=torch.long, device=mask.device)
            x0, tr = hook._get_transfer_index(logits, 0.0, "low_confidence", mask, x, n_tok, False,
                                              x_embeds.shape[1] - bs, be - bs, 0, args.threshold)
            return tr, torch.where(tr, x0, -1)

        def probe(t, cache, x_embeds, mask_index, bs, be, n_tok):
            mask = mask_index[:, bs:]
            decoded = (prev_mask[0] & ~mask).nonzero()[:, 1].tolist() if prev_mask else []
            prev_mask[:] = [mask.clone()]
            history[t] = cache
            history.pop(t - max_age - 1, None)
            ref = block_logits(cache, x_embeds, bs, mask)
            ref_tr, ref_tok = decide(ref, mask, x_embeds, bs, be, n_tok)
            m = mask[0]
            ref_top = ref[0, m].argmax(-1)
            ref_logp = torch.log_softmax(ref[0, m], -1)
            row = dict(sample_id=smp["sample_id"], t=t, n_masked=int(m.sum()), decoded_prev=decoded,
                       ref_transfer=ref_tr[0].nonzero()[:, 0].tolist(), cells={})
            for g, b, s in var:
                if t - s not in history:
                    continue
                layers = range(n_layers) if b == "all" else BANDS[b]
                lg = block_logits(mix(cache, history[t - s], groups[g][:bs], set(layers)), x_embeds, bs, mask)
                tr, tok = decide(lg, mask, x_embeds, bs, be, n_tok)
                logp = torch.log_softmax(lg[0, m], -1)
                kl = (ref_logp.exp() * (ref_logp - logp)).nan_to_num(0.0).sum(-1).mean()
                row["cells"][f"{g}|{b}|{s}"] = [int((lg[0, m].argmax(-1) != ref_top).sum()), round(float(kl), 5),
                                                int(not torch.equal(tok, ref_tok))]
            rows.append(row)

        loop.generate(hook, embeds, FixedInterval(1), steps=args.gen_length, gen_length=args.gen_length,
                      block_length=args.gen_length, threshold=args.threshold, tokenizer=tokenizer,
                      stopping_criteria=["<|eot_id|>"], probe=probe)
        with open(args.out, "a") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
            fh.write(json.dumps(dict(sample_id=smp["sample_id"], end=True, n_prompt=n_prompt,
                                     n_visual=int(visual.sum()), steps=len(rows),
                                     probe_s=time.perf_counter() - t0)) + "\n")
        print(f"sample {smp['sample_id']} N={n_prompt} steps={len(rows)} {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
