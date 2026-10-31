#!/bin/bash
# Gate A (plan v3), technical part: unified loop with the Confidence() commit policy vs upstream Fast-dLLM in
# block decoding (block 32, gen 256, prefix cache refreshed at block starts), steps 256 / 64, threshold None / 0.5,
# first 5 MathVerse vision-dominant problems. Script: src/scripts/ga_identity.py (exit code 0 = PASS).
#
# Env: /scratch/<account>/envs/llada-v. GPU: any A100 (identity test, timings not compared).
#
# Runtime estimate (no block-decoding measurement yet): Exp 4 interval 32 without blocks took 8.3 s per problem
# at 256 steps; block-start-only refresh at block 32 assumed similar, 256 steps ~10 s, 64 steps ~3 s ->
# 3 implementations x (10 + 3) s x 2 thresholds ~ 80 s per problem; 5 problems ~ 7 min + 1 min load.
#
# Usage:  sbatch slurm/v3_phaseA_identity.sbatch
#SBATCH --job-name=v3a-identity
#SBATCH --partition=work1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --signal=B:TERM@120
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL,REQUEUE,TIME_LIMIT_90

set -euo pipefail
ACCT=$(id -un)
module load anaconda3
source activate /scratch/${ACCT}/envs/llada-v
export HF_HOME=/scratch/${ACCT}/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
cd /home/${ACCT}/dmllm-refresh-frontier
nvidia-smi --query-gpu=name,memory.total --format=csv

exec python src/scripts/ga_identity.py "$@"
