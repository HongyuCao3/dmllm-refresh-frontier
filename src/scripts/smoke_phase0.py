"""Phase 0 smoke test: does upstream LLaDA-V (+ its Fast-dLLM hook) run here, and what does it cost?

Measures, per (arm, sample): prompt length N (text + anyres visual tokens), prep time (vision tower +
embedding), generation time, number of backbone forwards and how many of them are full-sequence
(= refreshes), peak memory, output text. These replace the guessed runtimes in .claude/cluster.md and
size the Phase 1 and Phase 2 jobs. Not a reported result: 5 samples, no quality claim.

Arms (all deterministic, steps = gen_length = block_length = 128, stop at <|eot_id|>):
    full           upstream generate, no cache: the reference trajectory
    fd_i1          Fast-dLLM hook, refresh every step, no threshold: should reproduce `full` token-for-token
    fd_i32         Fast-dLLM hook, refresh every 32 steps, no threshold: cache effect alone
    fd_i32_thr1    Fast-dLLM hook, interval 32, threshold 1: upstream generate_demo.py default

Timing: one discarded warm-up generation per arm, torch.cuda.synchronize() around every clock read,
generation time excludes model load and image preprocessing (reported separately as prep_s).
Resumable: one jsonl line per (arm, sample), finished pairs are skipped.
"""
import argparse
import copy
import json
import os
import time

import torch
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.hooks.fast_dllm_hook import register_fast_dllm_hook, unregister_fast_dllm_hook
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model

ACCT = os.popen("id -un").read().strip()
ARMS = {
    "full": None,
    "fd_i1": dict(prefix_refresh_interval=1, threshold=None),
    "fd_i32": dict(prefix_refresh_interval=32, threshold=None),
    "fd_i32_thr1": dict(prefix_refresh_interval=32, threshold=1),
}
GEN = dict(steps=128, gen_length=128, block_length=128, stopping_criteria=["<|eot_id|>"])
QUESTION = DEFAULT_IMAGE_TOKEN + "\nPlease describe the image in detail."


class ForwardCounter:
    """Wraps the LLaDA backbone forward; a forward over the whole sequence is a refresh."""

    def __init__(self, backbone):
        self.orig = backbone.forward
        self.lengths = []
        backbone.forward = self

    def __call__(self, *a, **kw):
        self.lengths.append(kw["inputs_embeds"].shape[1])
        return self.orig(*a, **kw)


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=f"/scratch/{ACCT}/data/dmllm_refresh/mme_caption/manifest.jsonl")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default=f"/scratch/{ACCT}/outputs/dmllm_refresh/phase0_smoke/generations.jsonl")
    args = ap.parse_args()

    samples = [json.loads(l) for l in open(args.manifest)][: args.n]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        done = {(r["arm"], r["sample_id"]) for r in map(json.loads, open(args.out))}

    t0 = time.perf_counter()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        "GSAI-ML/LLaDA-V", None, "llava_llada", attn_implementation="sdpa", device_map="cuda:0")
    model.eval()
    load_s = time.perf_counter() - t0
    print(f"model load {load_s:.1f}s", flush=True)
    counter = ForwardCounter(model.model)

    conv = copy.deepcopy(conv_templates["llava_llada"])
    conv.append_message(conv.roles[0], QUESTION)
    conv.append_message(conv.roles[1], None)
    input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX,
                                      return_tensors="pt").unsqueeze(0).cuda()

    @torch.no_grad()
    def run(sample, arm_kw):
        image = Image.open(sample["image"])
        t_a = sync_time()
        image_tensor = [t.to(dtype=torch.float16, device="cuda") for t in
                        process_images([image], image_processor, model.config)]
        _, _, _, _, embeds, _ = model.prepare_inputs_labels_for_multimodal(
            input_ids, None, None, None, None, image_tensor, ["image"], image_sizes=[image.size])
        t_b = sync_time()
        counter.lengths.clear()
        torch.cuda.reset_peak_memory_stats()
        if arm_kw is None:
            out = model.generate_with_embeds(inputs_embeds=embeds, tokenizer=tokenizer, **GEN)
        else:
            out = hook._fast_generate_with_embeds(inputs_embeds=embeds, tokenizer=tokenizer, **GEN, **arm_kw)
        t_c = sync_time()
        n_prompt = embeds.shape[1]
        return dict(n_prompt=n_prompt, prep_s=t_b - t_a, gen_s=t_c - t_b,
                    n_forward=len(counter.lengths),
                    n_refresh=sum(l == n_prompt + GEN["gen_length"] for l in counter.lengths),
                    peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30,
                    tokens=out[0].tolist(),
                    text=tokenizer.decode(out[0], skip_special_tokens=False))

    with open(args.out, "a") as fh:
        for arm in args.arms.split(","):
            arm_kw = ARMS[arm]
            hook = register_fast_dllm_hook(model) if arm_kw is not None else None
            todo = [s for s in samples if (arm, s["sample_id"]) not in done]
            if todo:
                warm = run(samples[0], arm_kw)
                print(f"[{arm}] warm-up {warm['gen_s']:.1f}s", flush=True)
            for s in todo:
                r = run(s, arm_kw)
                rec = dict(arm=arm, sample_id=s["sample_id"], mme_id=s["mme_id"], load_s=load_s,
                           gpu=torch.cuda.get_device_name(0), **r)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"[{arm}] sample {s['sample_id']} N={r['n_prompt']} gen={r['gen_s']:.2f}s "
                      f"fwd={r['n_forward']} refresh={r['n_refresh']} mem={r['peak_mem_gb']:.1f}GB", flush=True)
            if hook is not None:
                unregister_fast_dllm_hook(hook)


if __name__ == "__main__":
    main()
