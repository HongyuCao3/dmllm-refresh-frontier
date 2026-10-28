# Cluster Rules (Palmetto 2 / Slurm) — read before anything that submits a job

Inherited from `~/ConceptAttentionPlus/.claude/cluster.md`, which was verified against real jobs on this
account. This file records what applies here; items marked *inherited* were not re-verified in this repo.

## Interactive node vs compute node

- The interactive node has **no GPU**. CPU-only work (env build, weight download, pytest, manifest building,
  aggregation, plotting) runs there; anything that loads LLaDA-V goes through `sbatch`.
- `python` / `conda` are not on PATH until `module load anaconda3`.
- **Derive the account with `id -un`, not `$USER`.** In the OnDemand container `$USER` is `coder`, and `sbatch`
  exports the submitting environment verbatim, so a `$USER`-derived path points at a nonexistent `/scratch/coder`.
- `sinfo` is not on PATH and `scontrol show partition` returns empty in the container (*inherited*).

## Two environments, deliberately (exception to `global.md`)

`global.md` names `research-template` as the env. **The experiment code runs in a second env**, because of a
hard version conflict, not a preference:

| | shared `research-template` | LLaDA-V (`third_party/LLaDA-V/train`) |
|---|---|---|
| transformers | 5.x | 4.39.x (upstream pins `tokenizers~=0.15.2`) |
| torch | 2.13 | 2.1.2 (upstream's commented pin) |
| numpy | 2.x | `<2` |

- LLaDA-V env: `/scratch/<account>/envs/llada-v`, built by `scripts/setup_env.sh` from
  `requirements-llada-v.txt`. Everything under `src/methods/`, `src/inference/`, `src/scripts/` that touches the
  model runs here.
- Template-side env (`src/main.py`, `tests/`, Hydra/Lightning/MLflow scaffold): the shared
  `/scratch/<account>/envs/research-template`.
- **Never `pip install` into `research-template` for this project.** It is shared with other projects. Add a
  dependency to `requirements-llada-v.txt` first, then re-run `scripts/setup_env.sh`.
- Attention backend is `sdpa` (as in upstream's `generate_demo.py`). flash-attn is not needed; if it ever is,
  follow the wheel-cache rule in `~/.claude/CLAUDE.md`, never the GitHub prebuilt wheel.
- Every sbatch script states which env it activates in its header comment.

## GPU sizing (project-specific, to be replaced by measured numbers)

LLaDA-V is LLaDA-8B plus a SigLIP vision tower and projector: ~17 GB of weights in bf16/fp16. An A100 40GB
holds the weights; the online replacement probe (`.claude/research.md`) keeps a second KV copy per probed cell,
so diagnostics request the 80GB card.

| job | request |
|---|---|
| generation / frontier runs | `--gpus a100:1` |
| replacement probe, oracle, anything holding two KV copies | `--gpus a100:1 --constraint gpu_a100_80gb` |

**Measured (Phase 0 smoke, job 18490708, A100 80GB PCIe, MME caption, gen_length = steps = 128):** model
load 27 s; prompt length N = 1.0–3.7k (anyres visual tokens; typical ~3k); full recompute 20–79 s/sample
(mean 58 s); Fast-dLLM interval 1 49 s, interval 32 6.6 s; peak memory 31 GB (full), 27 GB (Fast-dLLM).
A 40GB card holds one KV copy; two copies at N ≈ 3.7k should still fit 80GB. Source:
`results/phase0/smoke/smoke_summary.json`.

**Wall-clock comparisons run on one GPU type.** Every row of a speed–quality table comes from the same
`--constraint`, and the sbatch header names it. A40 / L40S / A100 numbers are never mixed in one table.

## Paths

- Conda envs: `/scratch/<account>/envs/{llada-v,research-template}`.
- HF cache: `export HF_HOME=/scratch/<account>/hf_cache`. Pretrained weights never go into `/home`.
  `scripts/download_weights.sh` pulls `GSAI-ML/LLaDA-V` and its vision tower.
- Benchmark data: `/scratch/<account>/data/dmllm_refresh/`.
- Results: `results/<phase>/<run-name>/`. Per-sample generations (`*.jsonl`) and per-step probe dumps live
  under `/scratch/<account>/outputs/dmllm_refresh/`; the small `*_summary.json` is committed next to the claim
  it supports.
- Job logs: `slurm/logs/<name>-%j.out`, gitignored except for the directory.
- **A batch job cannot read `/tmp`.** It is node-local. Anything a job executes lives in the repo or under
  `/scratch/<account>/tmp/`.

## Job submission

- Default partition `work1`, no `--account`; owner partitions may preempt `work1` (*inherited*).
- Every sbatch script sets `--signal=B:TERM@120` and `--requeue`, and **the final command is `exec python ...`**,
  so SIGTERM reaches python. Generation is resumable by construction: write one jsonl line per sample and skip
  samples already written.
- One job loads the model once and runs a campaign (`configs/campaigns/*.json`: one task x many refresh policies),
  never one job per policy. Measure the load time in the smoke test and quote it.
- Sweeps over many campaigns use a Slurm job array over a one-spec-per-line file, never a loop of `sbatch`.
- Notifications: `--mail-type END,FAIL,REQUEUE,TIME_LIMIT_90`. No other notification mechanism.
- Timing runs: warm up (one discarded sample), synchronize CUDA before reading the clock, and report the
  generation time only, without model load or image preprocessing. State this in the header.

## Before any long-running job

`CLAUDE.md` requires a runtime estimate up front. State GPU type, number of samples, generation length,
denoising steps, number of policies in the campaign and, for probes, the number of extra forwards per step.
**Name which parts of the estimate are guesses.** Put it in the sbatch header next to the `--time` it
justifies, and replace it with the measured number after the first run.

**Above 4 GPU-hours estimated, explain in plain language why the experiment is necessary and how it serves the
main line, before submitting.**
