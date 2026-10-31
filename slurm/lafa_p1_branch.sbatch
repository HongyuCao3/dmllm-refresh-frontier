#!/bin/bash
# Backup-line follow-up, pilot P1: branching at AR-uncertain tokens on Lafa-7B, MATH-500 first 200 (upper bound and
# selection signals). Script: src/scripts/lafa_branch_headroom.py.
#
# Array: tasks 0-1 = shards 0/1 of the 200 problems (100 each).
# Env: /scratch/<account>/envs/sarlafa. GPU: A100.
#
# Runtime estimate (E1 measurements: self-speculative MATH-500 decode 3.2 s per problem at ~316 tokens; a branch
# re-decodes the rest, ~1.6-3 s): greedy + up to 12 branch rollouts (3 policies x 4, overlap shared) + 13 clean
# scoring forwards (~0.1 s) ~ 30 s per problem -> 100 problems ~ 50 min per task; total ~1.7 GPU-h.
#
# Usage:  sbatch --array=0-1 slurm/lafa_p1_branch.sbatch
#SBATCH --job-name=lafa-p1
#SBATCH --partition=work1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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

exec python src/scripts/lafa_branch_headroom.py --n 200 --shard ${SLURM_ARRAY_TASK_ID:-0} --num-shards 2 "$@"
