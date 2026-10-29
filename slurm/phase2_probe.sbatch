#!/bin/bash
# Phase 2 replacement-error probe (D3 time profile, D1 position groups, layer bands) on the 50 MME caption
# images. Script: src/scripts/probe_replacement.py. Resumable per sample.
#
# Env: /scratch/<account>/envs/llada-v. GPU: A100 80GB (keeps 9 prompt caches of ~1.6 GB each plus mixed copies;
# timings are not compared, but the 80GB card is required for memory).
#
# Runtime estimate (GUESS, from the caption campaign: refresh forward ~0.43 s, block forward ~0.056 s): per step
# 1 refresh + 1 reference + up to 20 probe block forwards ~ 1.6 s; ~112 steps -> ~3 min per image;
# 50 images ~ 2.5 h + load. MEASURED in smoke job 18600727 (2 images, A100 80GB): 83 s (N = 1042) and 115 s
# (N = 3715) per image -> ~1.4 h for 50; --time 3:00.
#
# Usage:  sbatch --time=00:30:00 slurm/phase2_probe.sbatch --n 2 --out /scratch/<account>/outputs/dmllm_refresh/phase2_probe_smoke/rows.jsonl
#         sbatch slurm/phase2_probe.sbatch
#SBATCH --job-name=p2-probe
#SBATCH --partition=work1
#SBATCH --gpus=a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
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

exec python src/scripts/probe_replacement.py "$@"
