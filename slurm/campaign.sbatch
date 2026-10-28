#!/bin/bash
# Run one campaign (configs/campaigns/*.json: one task x many refresh policies, one model load).
# Script: src/scripts/run_campaign.py. Resumable: requeue / resubmit continues where it stopped.
#
# Env: /scratch/<account>/envs/llada-v.
# GPU: A100 80GB, pinned with --constraint so every wall-clock row of a speed-quality table comes from the
# same GPU type (.claude/cluster.md). Timing: one discarded warm-up per arm, CUDA-synchronized clocks,
# generation time excludes model load and image preprocessing.
#
# Runtime, mme_caption_frontier (50 images x 7 arms, gen_length = steps = 128, threshold 0.5), MEASURED in
# job 18513394: per image full 65.6 s, interval 1 48.3, 2 26.5, 4 15.7, 8 10.4, 16 7.7, 32 6.3 s = 181 s
# -> 2.5 h generation, ~2.9 h wall. --time 4:00; under the 4 GPU-hour line. Other campaigns: state their own
# estimate in the submission message (gen_length and sample count change it).
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
