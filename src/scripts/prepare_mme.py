"""Export a fixed sample of unique MME images for the detailed-captioning task.

Faster but Different (Tab. 3) measures the fixed-interval frontier on 50 MME images but does not list
which ones, so we draw our own: unique images (MME asks two questions per image), sorted by id, then a
seeded random sample. The manifest order is the run order; the Phase 0 smoke test takes its first 5.

CPU-only, run on the interactive node in the llada-v env:
    python src/scripts/prepare_mme.py
"""
import argparse
import glob
import io
import json
import os
import random

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image

ACCT = os.popen("id -un").read().strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"/scratch/{ACCT}/data/dmllm_refresh/mme_caption")
    args = ap.parse_args()

    snap = snapshot_download("lmms-lab/MME", repo_type="dataset")
    images = {}
    for f in sorted(glob.glob(os.path.join(snap, "data", "*.parquet"))):
        for row in pq.read_table(f, columns=["question_id", "image", "category"]).to_pylist():
            images.setdefault(row["question_id"], row)

    ids = random.Random(args.seed).sample(sorted(images), args.n)
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    with open(os.path.join(args.out, "manifest.jsonl"), "w") as fh:
        for i, qid in enumerate(ids):
            img = Image.open(io.BytesIO(images[qid]["image"]["bytes"])).convert("RGB")
            path = os.path.join(args.out, "images", f"{i:03d}.png")
            img.save(path)
            fh.write(json.dumps({"sample_id": i, "mme_id": qid, "category": images[qid]["category"],
                                 "image": path, "width": img.width, "height": img.height}) + "\n")
    print(f"{len(images)} unique images in {snap}; wrote {args.n} (seed {args.seed}) to {args.out}")


if __name__ == "__main__":
    main()
