#!/bin/bash
# Pull the LLaDA-V checkpoint and its vision tower into the shared HF cache.
#
# CPU-only, network-bound: run on the interactive node after scripts/setup_env.sh.
# Pretrained weights never go into /home (see .claude/cluster.md).
#
# Usage:  bash scripts/download_weights.sh

set -euo pipefail

ACCT=$(id -un)           # NOT $USER: in the OnDemand container $USER is "coder"

module load anaconda3
source activate /scratch/${ACCT}/envs/llada-v

export HF_HOME=/scratch/${ACCT}/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1

python - <<'PY'
import json, os
from huggingface_hub import snapshot_download

path = snapshot_download("GSAI-ML/LLaDA-V")
print("LLaDA-V ->", path)

# The checkpoint names its vision tower in config.json; fetch it too so the first GPU
# job does not download on the compute node.
tower = json.load(open(os.path.join(path, "config.json"))).get("mm_vision_tower")
print("mm_vision_tower =", tower)
if tower and "/" in tower and not os.path.isdir(tower):
    print(tower, "->", snapshot_download(tower))
PY
