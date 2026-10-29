"""Export a fixed sample of MathVerse testmini problems (one version split) for the frontier campaigns.

Prompt as upstream LLaDA-V's lmms-eval run (third_party/LLaDA-V/eval/scripts/mathverse.sh): shot_type
"format-prompt", query_type "query_cot" (the dataset's chain-of-thought query, which asks for the option letter
or the final value at the end), then " /think" appended (think_mode "think"). The image token is added by
src/methods/refresh/llada_v.prompt_ids.

CPU-only, run on the interactive node in the llada-v env:
    python src/scripts/prepare_mathverse.py [--split vision_dominant] [--n 300] [--seed 0]
"""
import argparse
import io
import json
import os
import random

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

ACCT = os.popen("id -un").read().strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="vision_dominant")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"/scratch/{ACCT}/data/dmllm_refresh/mathverse_{args.split}"

    path = hf_hub_download("CaraJ/MathVerse-lmmseval", f"testmini_{args.split}.parquet", repo_type="dataset")
    rows = sorted(pq.read_table(path).to_pylist(), key=lambda r: int(r["sample_index"]))
    picked = random.Random(args.seed).sample(rows, args.n)
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    with open(os.path.join(out, "manifest.jsonl"), "w") as fh:
        for i, r in enumerate(picked):
            img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            img_path = os.path.join(out, "images", f"{i:03d}.png")
            img.save(img_path)
            fh.write(json.dumps({
                "sample_id": i, "sample_index": r["sample_index"], "problem_index": r["problem_index"],
                "problem_version": r["problem_version"], "question_type": r["question_type"],
                "subject": r["metadata"]["subject"], "question": r["query_cot"] + " /think",
                "answer": r["answer"], "question_for_eval": r["question_for_eval"],
                "image": img_path, "width": img.width, "height": img.height}) + "\n")
    print(f"{len(rows)} problems in {args.split}; wrote {args.n} (seed {args.seed}) to {out}")


if __name__ == "__main__":
    main()
