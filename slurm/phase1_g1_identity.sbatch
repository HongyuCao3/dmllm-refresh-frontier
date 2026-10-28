#!/bin/bash
# Gate G1(i): unified loop vs upstream Fast-dLLM, token identity at interval 1 / 4 / 32,
# threshold 0.5 and None, first 5 MME caption images. Script: src/scripts/g1_identity.py
# (exit code 0 = PASS). Also reruns upstream at interval 4 / threshold 0.5 for run-to-run nondeterminism.
#
# Env: /scratch/<account>/envs/llada-v. GPU: any A100 (identity test, timings not compared).
#
# Runtime estimate (from Phase 0 smoke, A100 80GB, N ~ 3k): per sample and implementation,
# interval 1 ~50 s, interval 4 ~15 s (guess: between 1 and 32), interval 32 ~6 s -> ~70 s per threshold;
# 2 thresholds x 2 implementations + 1 rerun ~ 300 s per sample; 5 samples ~ 25 min + 30 s load.
#
# Usage:  sbatch slurm/phase1_g1_identity.sbatch
#SBATCH --job-name=g1-identity
#SBATCH --partition=work1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
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

exec python src/scripts/g1_identity.py "$@"
