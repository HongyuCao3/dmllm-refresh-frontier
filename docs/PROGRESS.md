# PROGRESS — dmllm-refresh-frontier run log

Authoritative run log, decisions and lever maps. Plan: `plans/research_plan_v2.md`. Rules: `.claude/research.md`,
`.claude/cluster.md`. Newest entries at the bottom of each section.

## Status (2026-10-27)

| Plan item | Status |
|---|---|
| Router + rules | done (commit `9ced305`): `CLAUDE.md` project statement and "Where the project lives"; `.claude/cluster.md`; `.claude/research.md` |
| Third-party code | done: 5 clones at pinned commits, `third_party/MANIFEST.md` |
| Env `llada-v` | scripts written (`scripts/setup_env.sh`, `requirements-llada-v.txt`), **not built**. The run was blocked by the auto-mode permission check; the user runs it. Pins inferred from upstream (`tokenizers~=0.15.2` → transformers 4.39.3, torch 2.1.2 cu121), untested |
| Weights | `scripts/download_weights.sh` written, **not run**. `GSAI-ML/LLaDA-V` is not yet in `/scratch/hcao2/hf_cache` |
| Phase 0 smoke | not started |
| Phases 1–5 | not started |

## Decisions (user)

| Date | Decision |
|---|---|
| 2026-10-27 | Repetition: deterministic decoding, one run per arm, paired bootstrap `Δ [95% CI] (n)` over samples + paired tests (McNemar / Wilcoxon, Holm over anchors). Appendix: 50 samples × 3 sampling seeds (temperature 0.5), the sign of Δ must agree in all three. Recorded as a project exception in `.claude/global.md` |
| 2026-10-27 | Gates frozen: G1 token-identical to upstream at interval 1 / 4 / 32, speedup ±20% relative and Jaccard ±0.05 of the published table, both monotone; G2 top 20% cells ≥ 50% of error; G3 oracle ≥ 40% lower wall-clock on ≥ 2 anchors; G4 ≥ 25% on ≥ 2 anchors, Holm-significant, ≥ 50% of oracle gain |
| 2026-10-27 | Matched quality (proposed with the gates, accepted): not significantly below the anchor and within 1 accuracy point (or 0.01 Jaccard) |

## Findings so far (setup)

- Upstream LLaDA-V (`train/llava`) already ships the Fast-dLLM hook and a dLLM-Cache copy, so the
  fixed-interval frontier and dLLM-Cache baselines run from upstream code.
- **Elastic-Cache's released code has no LLaDA-V entry point** (only `llada/`, `dream/`), although the paper
  evaluates LLaDA-V. dKV-Cache is also text-only. Both need ports to LLaDA-V, validated first on
  LLaDA-8B-Instruct against upstream. Phase 1 may take +1–2 weeks.
- Inference uses `sdpa` (as upstream's demo); flash-attn is not needed.
- Open measurement: the sequence length N with anyres visual tokens may be 3–5k, not the ~1000 in the plan's
  storage estimate. Measure in the Phase 0 smoke test.

## Next steps

1. User: `bash scripts/setup_env.sh && bash scripts/download_weights.sh` (CPU, ~30 min + download). Fix pins if
   the import check fails.
2. Phase 0 smoke (`slurm/`): upstream full recompute + interval 32 on 5 MME images, A100. Measure model load,
   N, per-step time, seconds per sample; replace the runtime estimates in `.claude/cluster.md` and the plan.
3. Unified generation loop skeleton with a `RefreshPolicy` interface; token-identity test against upstream
   Fast-dLLM (G1 part i).
4. `src/scripts/gate_check.py` once the Phase 1 output format is fixed.
5. Sync the online plan artifact (claude.ai/artifact/Aw3jJnaReYrrrsWJ2R8tEC) with the frozen gates and the
   repetition protocol, if wanted.
