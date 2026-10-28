#!/bin/bash
# Build the dedicated conda env for LLaDA-V inference and the refresh-policy code.
#
# A separate env, not the shared research-template: LLaDA-V's llava package targets
# transformers 4.39 / tokenizers 0.15, research-template carries transformers 5.x, and
# research-template is shared with other projects whose jobs a pin bump would break.
# See .claude/cluster.md.
#
# CPU-only: safe to run on the interactive node. ~20-30 min (pip resolve + the torch
# cu121 bundle). Re-running updates the existing env in place.
#
# Usage:  bash scripts/setup_env.sh

set -euo pipefail

ACCT=$(id -un)           # NOT $USER: in the OnDemand container $USER is "coder"
REPO=/home/hcao2/dmllm-refresh-frontier
PREFIX=/scratch/${ACCT}/envs/llada-v

module load anaconda3

if [ -d "$PREFIX" ]; then
    echo "env already exists at $PREFIX -- activating to update"
else
    conda create -p "$PREFIX" python=3.10 -y
fi

source activate "$PREFIX"

export HF_HOME=/scratch/${ACCT}/hf_cache
export PYTHONDONTWRITEBYTECODE=1
export PIP_CACHE_DIR=/scratch/${ACCT}/tmp/pip_cache

pip install --upgrade pip
# Upstream's commented pin is torch 2.1.2 / torchvision 0.16.2.
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.1.2 torchvision==0.16.2
pip install -r "${REPO}/requirements-llada-v.txt"

# third_party/ clones are git-ignored: clone at the pinned commits if missing.
# Commits and roles: third_party/MANIFEST.md.
clone() {  # dir url commit
    local dir="${REPO}/third_party/$1"
    [ -d "$dir" ] || git clone "$2" "$dir"
    git -C "$dir" checkout --quiet "$3"
}
clone LLaDA-V       https://github.com/ML-GSAI/LLaDA-V.git        f8b02ce04b09f4f271fe55a3652059a73bbc7a32
clone Fast-dLLM     https://github.com/NVlabs/Fast-dLLM.git       a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b
clone dKV-Cache     https://github.com/horseee/dKV-Cache.git      49a76fcc43b744ec2d960137f216e419317138b1
clone Elastic-Cache https://github.com/VILA-Lab/Elastic-Cache.git 1960d8fc6231205a1ae4ebba3898d475e339f7e1
clone dLLM-cache    https://github.com/maomaocun/dLLM-cache.git   17235bffc8c5b587a2dc6f7dc76fcd01eab76e3a

# Editable install of LLaDA-V's llava package, so `import llava` resolves without
# PYTHONPATH games. --no-deps: requirements-llada-v.txt is authoritative; the package's
# [train] extras pull deepspeed, wandb and a 2023 pydantic.
pip install -e "${REPO}/third_party/LLaDA-V/train" --no-deps

echo
echo "=== verify ==="
python - <<'PY'
import torch, transformers, tokenizers
print("torch       ", torch.__version__)
print("transformers", transformers.__version__)
print("tokenizers  ", tokenizers.__version__)
print("cuda avail  ", torch.cuda.is_available(), "(False on the interactive node is expected)")
from llava.model.builder import load_pretrained_model
from llava.hooks.fast_dllm_hook import register_fast_dllm_hook
from llava.hooks import register_cache_LLaDA_V
print("llava import OK")
PY
