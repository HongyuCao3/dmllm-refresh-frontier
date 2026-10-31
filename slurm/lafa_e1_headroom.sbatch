#!/bin/bash
# Backup line (plan v3), experiment E1: draft-order / time-schedule headroom of Lafa-7B self-speculative decoding,
# teacher-forced on its own (= AR greedy) output. Script: src/scripts/lafa_draft_headroom.py (self-checks its draft
# loop against the checkpoint's `_denoise_flowmap_block` before the first problem).
#
# Array: task 0-1 = gsm8k shards 0/1, task 2-3 = math-500 shards 0/1; 100 problems per task (50 per shard).
# Env: /scratch/<account>/envs/sarlafa (transformers 5.19, Lafa remote code from the HF cache). GPU: A100.
#
# Runtime estimate (assumptions: 7B bf16 eager, one 4-token draft forward ~20-25 ms with a ~600-token cache;
# stride 2; 6 variants = 1+2+4+4+4+4 = 19 forwards per site; plus one 1-token AR forward per position and one
# self-speculative decode): GSM8K ~250 tokens -> 125 sites x 19 x 0.022 s + ~10 s ~ 60 s per problem;
# MATH-500 ~450 tokens -> ~110 s. Per shard: GSM8K ~50 min, MATH-500 ~95 min; total ~4.7 GPU-h.
#
# Usage:  sbatch --array=0-3 slurm/lafa_e1_headroom.sbatch
#SBATCH --job-name=lafa-e1
#SBATCH --partition=work1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --signal=B:TERM@120
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL,REQUEUE,TIME_LIMIT_90

set -euo pipefail
ACCT=$(id -un)
module load anaconda3
source activate /scratch/${ACCT}/envs/sarlafa
export HF_HOME=/scratch/${ACCT}/hf_cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
cd /home/${ACCT}/dmllm-refresh-frontier
nvidia-smi --query-gpu=name,memory.total --format=csv

TASKS=(gsm8k gsm8k math-500 math-500)
T=${SLURM_ARRAY_TASK_ID:-0}
exec python src/scripts/lafa_draft_headroom.py --task "${TASKS[$T]}" --n 100 --shard $((T % 2)) --num-shards 2 "$@"
