#!/bin/bash
# Phase 0 smoke: upstream LLaDA-V full recompute vs Fast-dLLM hook (interval 1, 32, 32 + threshold 1)
# on the first 5 images of the MME caption manifest. Measures model load, prompt length N, per-sample
# generation time, forwards/refreshes per sample, peak memory. Script: src/scripts/smoke_phase0.py.
#
# Env: /scratch/<account>/envs/llada-v. GPU: any A100 (gpu name logged per row; smoke numbers only size
# later jobs, they are not compared across GPU types).
# Timing: one discarded warm-up per arm, CUDA-synchronized clocks, generation time excludes model load
# and image preprocessing.
#
# Runtime estimate (guesses marked): model load 1-3 min (guess: 16 GB from /scratch). Per sample, from
# Faster but Different Tab. 3 (their GPU, guess that A100 is similar): full 72 s, interval 1 ~72 s
# (no threshold: 128 full forwards, like full), interval 32 without threshold ~15-25 s (guess),
# interval 32 + threshold 6 s. 4 arms x (5 samples + 1 warm-up) ~ 6 x 170 s ~ 17 min. --time 1:30 leaves
# margin for N being 3-5k instead of ~1k.
#
# Usage:  sbatch slurm/phase0_smoke.sbatch
#SBATCH --job-name=p0-smoke
#SBATCH --partition=work1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
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

exec python src/scripts/smoke_phase0.py "$@"
