# PROGRESS — dmllm-refresh-frontier run log

Authoritative run log, decisions and lever maps. Plan: `plans/research_plan_v2.md`. Rules: `.claude/research.md`,
`.claude/cluster.md`. Newest entries at the bottom of each section.

## Status (2026-10-28)

| Plan item | Status |
|---|---|
| Router + rules | done (commit `9ced305`): `CLAUDE.md` project statement and "Where the project lives"; `.claude/cluster.md`; `.claude/research.md` |
| Third-party code | done: 5 clones at pinned commits, `third_party/MANIFEST.md` |
| Env `llada-v` | **built 2026-10-28** at `/scratch/hcao2/envs/llada-v`: torch 2.1.2+cu121, transformers 4.39.3, tokenizers 0.15.2; `llava` + Fast-dLLM hook + dLLM-Cache hook import OK on CPU (log `/scratch/hcao2/tmp/logs/setup_llada_v_20261028_1034.log`). GPU load not yet tested |
| Weights | **downloaded 2026-10-28**: `GSAI-ML/LLaDA-V` (16 GB, snapshot `a10ba17`) and vision tower `google/siglip2-so400m-patch14-384` (4.3 GB, snapshot `e8e4872`) in `/scratch/hcao2/hf_cache`. llava loads the tower with its own `SigLipVisionModel`, not transformers' siglip2 |
| Phase 0 smoke | **done 2026-10-28** (job 18490708, 11 min): upstream path runs; see "Phase 0 smoke" below |
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

1. ~~Env + weights~~ done 2026-10-28.
2. ~~Phase 0 smoke~~ done 2026-10-28 (see above). Open: paper settings (threshold, Jaccard definition) for G1(ii).
3. Unified generation loop skeleton with a `RefreshPolicy` interface; token-identity test against upstream
   Fast-dLLM (G1 part i).
4. `src/scripts/gate_check.py` once the Phase 1 output format is fixed.
5. Sync the online plan artifact (claude.ai/artifact/Aw3jJnaReYrrrsWJ2R8tEC) with the frozen gates and the
   repetition protocol, if wanted.

## Phase 0 smoke (2026-10-28)

Data: `src/scripts/prepare_mme.py` → `/scratch/hcao2/data/dmllm_refresh/mme_caption/` (50 unique MME images,
seed-0 sample of 1187; Faster but Different does not list its 50, so G1 compares ratios on our own 50). Job:
`slurm/phase0_smoke.sbatch` → `src/scripts/smoke_phase0.py`, first 5 images, arms `full`, `fd_i1`, `fd_i32`,
`fd_i32_thr1` (upstream code, steps = gen_length = block = 128).

Lever map (written before the run):

| Outcome | Meaning | Next action |
|---|---|---|
| All arms run, `fd_i1` tokens == `full` | Upstream path works; interval 1 is a true no-cache reference | Phase 1 unified loop, G1(i) test against these outputs |
| `fd_i1` != `full` | Hook itself changes numerics (bf16/autocast path or prompt handling) | Locate the divergence before building the loop; G1(i) reference becomes `fd_i1` only if the user agrees |
| Load or OOM failure | Env pins or 40GB card insufficient | Fix pins / request 80GB, rerun |
| N ≥ 3k | Storage estimate in the plan is ≥3x low | Rescale Phase 2 storage plan (online only, fewer layers) |
| full gen_s far from 72 s | Different GPU or settings than the paper | Only ratios enter G1; note the absolute gap |

Result (job 18490708, A100 80GB PCIe, 5 images; `results/phase0/smoke/smoke_summary.json`; smoke only, not a
reported result):

| Arm | gen s/sample | speedup vs full | forwards (refreshes) | token-identical to full | word Jaccard vs full |
|---|---|---|---|---|---|
| full | 58.4 | 1.00 | 128 (128) | — | 1.000 |
| fd_i1 | 49.3 | 1.18 | 128 (128) | 5/5 | 1.000 |
| fd_i32 | 6.6 | 8.89 | 128 (4) | 0/5 | 0.385 |
| fd_i32_thr1 | 5.3 | 11.07 | 100 (3.4) | 0/5 | 0.336 |

- Lever-map row hit: **all arms run and `fd_i1` == `full` token-for-token on 5/5** → Phase 1 unified loop next.
  The 1.18x at interval 1 comes from the hook's attention path (sdpa, no mask) and autocast, not from fewer forwards.
- **N = 1.0–3.7k (typical ~3k)**, 3x the plan's ~1000: the raw-KV storage estimate is ≥ 3x low (≈ 200 GB per
  sample), which confirms online-only error measurement for Phase 2.
- Model load 27 s; peak memory 31 GB (full), 27 GB (Fast-dLLM).
- Our ratios are close to Faster but Different Tab. 3 at interval 32 only **with threshold 1** (11.07x vs 11.3x;
  without threshold 8.89x). Interval 1 is 1.18x vs their 1.3x. Open: which threshold and which Jaccard
  definition (word set? token set?) the paper used. This decides the G1(ii) comparison settings and must be
  settled before Phase 1 runs; it is not decided here.
