#!/bin/bash
# Run one campaign (configs/campaigns/*.json: one task x many refresh policies, one model load).
# Script: src/scripts/run_campaign.py. Resumable: requeue / resubmit continues where it stopped.
#
# Env: /scratch/<account>/envs/llada-v.
# GPU: A100 80GB, pinned with --constraint so every wall-clock row of a speed-quality table comes from the
# same GPU type (.claude/cluster.md). Timing: one discarded warm-up per arm, CUDA-synchronized clocks,
# generation time excludes model load and image preprocessing.
#
# Runtime estimate for mme_caption_frontier (50 images x 7 arms, gen_length = steps = 128, threshold 0.5):
# per image full 58 s, interval 1 ~45 s, 2 ~25 s, 4 ~14 s, 8 ~9 s, 16 ~7 s, 32 ~6 s ~ 165 s (full and
# interval 1/32 measured in Phase 0 smoke; 2-16 and the threshold-0.5 effect are guesses) -> ~2.3 h.
# --time 4:00 leaves margin; under the 4 GPU-hour line.
#
# Usage:  sbatch slurm/campaign.sbatch configs/campaigns/mme_caption_frontier.json
#SBATCH --job-name=campaign
#SBATCH --partition=work1
#SBATCH --gpus=a100:1
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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

exec python src/scripts/run_campaign.py "$@"
