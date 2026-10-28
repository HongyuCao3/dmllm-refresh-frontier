# CLAUDE.md — Project rule router

## What this project is

Cache refresh scheduling for the multimodal diffusion LLM **LLaDA-V**: reach the output quality of
fixed-interval KV refresh (Fast-dLLM) with a lower full-refresh ratio r, i.e. lower wall-clock at matched
quality, against fixed interval and Elastic-Cache. The plan is `plans/research_plan_v2.md` (error map →
oracle upper bound → static then online refresh policies, with gates G1–G4).

The project is **inference-only**: a released checkpoint, no training. `run_train`, `run_optuna`,
`mode=bench` and the `ModelCheckpoint` machinery of the template scaffold do not produce this project's
numbers. Every result comes from generation runs under `src/methods/` / `src/inference/` / `src/scripts/`,
driven by `slurm/` jobs.

## Rule priority

1) .claude/global.md
2) .claude/research.md — gates, controls, lever maps, result interpretation
3) .claude/cluster.md — anything submitted with sbatch, any /scratch path, any conda env
4) .claude/architecture.md (template scaffold only; see below)
5) module-level rules: .claude/data.md, .claude/methods.md, .claude/inference.md, .claude/train.md,
   .claude/configs.md

If rules conflict, follow .claude/global.md, and flag the conflict to the user.

## Where the project lives

| Path | What |
|---|---|
| `plans/research_plan_v2.md` | The research plan. Authoritative for scope, baselines and phases |
| `docs/PROGRESS.md` | Run log: status, user decisions, findings, next steps. Update after every job or decision |
| `third_party/` | Upstream clones at pinned commits, git-ignored; `third_party/MANIFEST.md` lists them. **Read-only — never edit a file under it.** |
| `requirements-llada-v.txt` | Pins of the `llada-v` env — **not** upstream's `train/requirements.txt` |
| `scripts/setup_env.sh`, `scripts/download_weights.sh` | CPU-only setup, run on the interactive node |
| `slurm/` | Every GPU job; read `.claude/cluster.md` first |
| `configs/campaigns/` | One task x many refresh policies per job (one model load) |
| `results/<phase>/<run>/` | Committed `*_summary.json`; bulky per-sample outputs live under `/scratch` |

The template modules (`architecture.md`, `methods.md`, `inference.md`, `train.md`, `configs.md`, `data.md`)
apply to the template scaffold: `src/main.py`, `src/train.py`, `src/methods/{linear,mlp}`, `src/callbacks/`,
and the Hydra groups under `configs/` other than `campaigns/`. They do not apply to project code that does not
run through that stack.

When editing code:
- If you add or modify the DataModule layer (`src/data/`, targeted by `configs/dataset/*.yaml`) → read .claude/data.md.
  `src/data/` is currently absent from the tree: the root `.gitignore` pattern `data/` also matches `src/data/`.
- If you modify files under src/methods → read .claude/methods.md
- If you modify files under src/inference → read .claude/inference.md
- If you modify training loop / trainer usage → read .claude/train.md
- If you change configs/** → read .claude/configs.md

When editing docs:
- read .claude/docs.md

For long-running experiments:
- Before generating or changing experiment code, provide a best-effort runtime estimate.
- State the assumptions used for the estimate (dataset size, epochs, batch size, device, number of steps).
- If exact runtime is unknown, give an order-of-magnitude estimate and identify the missing information.

For experiment notifications:
- Do not assume Claude can send email directly.
- If requested, generate or update a wrapper script that sends email on normal exit or failure.
- Prefer a shell or Python wrapper in scripts/ or tools/ with email credentials supplied via environment variables.

Do not infer architecture from single example files.
If rules conflict, follow .claude/global.md.

Before modifying files, suggest running:
git pull.
