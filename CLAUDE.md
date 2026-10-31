# CLAUDE.md — Project rule router

## What this project is

Commit (unmasking) policies for the multimodal diffusion LLM **LLaDA-V**: at every denoising step, decide
which answer positions to commit. The prior is the model's own confidence; the question is how far a
posterior signal (answer reward, reference trajectories, gold answers) can correct it, whether the prior is
worse on image-dependent tokens, and whether commits and KV-cache refresh can be decided jointly. The plan is
`plans/research_plan_v3.md` (read the closest work → block baselines → order upper bound → multimodal diagnosis →
learned policy → commit + cache, with gates A–E). Results already established in prior work (listed in the plan)
are cited, not re-tested; the closest competitor is arXiv 2607.08056 (RL order policy on MMaDA). A backup main
line (plan v3, "备用主线", user 2026-10-31) moves the question to Lafa, a text-only block-causal flow LM whose commit is a
left prefix of each 4-token block; it is switched to only by the user.

The earlier main line, cache refresh scheduling at a lower refresh ratio r (`plans/research_plan_v2.md`,
gates G1–G4), is a **concluded sub-direction** (user decision 2026-10-30): with block = gen_length, the
7.3 s per-item floor caps any refresh saving at about 12%, below G3's 40%. Its code, campaigns and results stay
and are reused; its findings feed phases B and E of v3.

**The base checkpoint stays frozen.** Phases A–C are inference only. From phase D on, the project trains only
a small commit policy on frozen LLaDA-V features; that training follows `.claude/global.md` (held-out
reporting split, ≥ 3 training seeds). `run_optuna`, `mode=bench` and the `ModelCheckpoint` machinery of the
template scaffold still do not produce this project's numbers. Decoding results come from generation runs under
`src/methods/` / `src/inference/` / `src/scripts/`, driven by `slurm/` jobs.

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
| `plans/research_plan_v3.md` | The research plan (main line: commit policies). Authoritative for scope, baselines and phases |
| `plans/research_plan_v2.md` | Concluded sub-direction: cache refresh scheduling (results reused by v3) |
| `docs/PROGRESS.md` | Run log: status, user decisions, findings, next steps. Update after every job or decision |
| `third_party/` | Upstream clones at pinned commits, git-ignored; `third_party/MANIFEST.md` lists them. **Read-only — never edit a file under it.** |
| `requirements-llada-v.txt` | Pins of the `llada-v` env — **not** upstream's `train/requirements.txt` |
| `scripts/setup_env.sh`, `scripts/download_weights.sh` | CPU-only setup, run on the interactive node |
| `slurm/` | Every GPU job; read `.claude/cluster.md` first |
| `configs/campaigns/` | One task x many decoding policies (refresh or commit) per job (one model load) |
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
