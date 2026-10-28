# Third-party clones

Clones are gitignored and read-only: do not edit, vendor, or `pip install` them, with one exception:
`scripts/setup_env.sh` installs `LLaDA-V/train` (the `llava` package) editable with `--no-deps` into the
`llada-v` env. Re-create everything with `bash scripts/setup_env.sh`, which clones and checks out the commits
below. Cloned 2026-10-27.

| Dir | Origin | Commit | Commit date | License | Role |
|---|---|---|---|---|---|
| `LLaDA-V` | https://github.com/ML-GSAI/LLaDA-V.git | `f8b02ce04b09` | 2026-03-23 | Apache-2.0 (`train/LICENSE`) | Model and loader (`llava.model.builder`), generation loop, built-in Fast-dLLM hook (`llava/hooks/fast_dllm_hook.py`) and dLLM-Cache hook (`llava/cache`, `llava/hooks/cache_hook_LLaDA_V.py`), lmms-eval fork (`eval/`) |
| `Fast-dLLM` | https://github.com/NVlabs/Fast-dLLM.git | `a9b81e4caa24` | 2026-05-30 | Apache-2.0 | Fixed-interval frontier reference; `fast_dvlm/` is the VLM variant. Same commit as `~/dKV-ARKV/third_party` |
| `dKV-Cache` | https://github.com/horseee/dKV-Cache.git | `49a76fcc43b7` | 2025-05-22 | none stated: read only, do not vendor | dKV-Cache-Decode baseline (LLaDA / Dream text models only; port to LLaDA-V needed). Same commit as `~/dKV-ARKV/third_party` |
| `Elastic-Cache` | https://github.com/VILA-Lab/Elastic-Cache.git | `1960d8fc6231` | 2026-07-13 | Apache-2.0 | Strongest adaptive baseline. **Released code covers `llada/` and `dream/` only; no LLaDA-V entry point**, although the paper evaluates LLaDA-V. Port to LLaDA-V needed |
| `dLLM-cache` | https://github.com/maomaocun/dLLM-cache.git | `17235bffc8c5` | 2026-05-01 | Apache-2.0 | Dual-interval baseline; has `demo_LLaDA_V.py`. LLaDA-V's own `llava/cache` is a copy of it; this clone is the reference for checking that copy |

## Ports this project has to write

- dKV-Cache-Decode and Elastic-Cache on LLaDA-V. Both become policies in the project's own generation loop
  (`src/methods/`), checked against the upstream text-model implementations on LLaDA-8B-Instruct
  (already in the HF cache) before being trusted on LLaDA-V.
