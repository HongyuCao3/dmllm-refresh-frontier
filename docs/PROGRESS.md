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
| Phase 1 | G1 PASS with recorded deviation (2026-10-28): unified loop token-identical to upstream; MME frontier reproduced. Baselines (Elastic-Cache, dKV-Cache ports) and MathVerse frontier not started |
| Phases 2–5 | not started |

## Decisions (user)

| Date | Decision |
|---|---|
| 2026-10-27 | Repetition: deterministic decoding, one run per arm, paired bootstrap `Δ [95% CI] (n)` over samples + paired tests (McNemar / Wilcoxon, Holm over anchors). Appendix: 50 samples × 3 sampling seeds (temperature 0.5), the sign of Δ must agree in all three. Recorded as a project exception in `.claude/global.md` |
| 2026-10-27 | Gates frozen: G1 token-identical to upstream at interval 1 / 4 / 32, speedup ±20% relative and Jaccard ±0.05 of the published table, both monotone; G2 top 20% cells ≥ 50% of error; G3 oracle ≥ 40% lower wall-clock on ≥ 2 anchors; G4 ≥ 25% on ≥ 2 anchors, Holm-significant, ≥ 50% of oracle gain |
| 2026-10-27 | Matched quality (proposed with the gates, accepted): not significantly below the anchor and within 1 accuracy point (or 0.01 Jaccard) |
| 2026-10-28 | **G1 accepted as PASS with a recorded deviation**: interval-2 Jaccard 0.615 vs paper 0.666 misses the ±0.05 tolerance by 0.001; the paper's value lies inside our 95% bootstrap CI at every interval, and the paper's 50-image subset is unpublished. Thresholds unchanged. The 50-image frontier (`results/phase1/mme_caption_frontier/g1_summary.json`) is the reference curve |

## Findings so far (setup)

- Upstream LLaDA-V (`train/llava`) already ships the Fast-dLLM hook and a dLLM-Cache copy, so the
  fixed-interval frontier and dLLM-Cache baselines run from upstream code.
- **Elastic-Cache's released code has no LLaDA-V entry point** (only `llada/`, `dream/`), although the paper
  evaluates LLaDA-V. dKV-Cache is also text-only. Both need ports to LLaDA-V, validated first on
  LLaDA-8B-Instruct against upstream. Phase 1 may take +1–2 weeks.
- Inference uses `sdpa` (as upstream's demo); flash-attn is not needed.
- ~~Open measurement: N with anyres visual tokens~~ measured in Phase 0: N = 1.0–3.7k (typical ~3k), 3x the
  plan's ~1000 (see "Phase 0 smoke").

## Findings so far (Phase 0–1, 2026-10-28)

- **The upstream fixed-interval frontier reproduces on our hardware and images** (G1 PASS with a recorded
  deviation, see Decisions). Speedup ladder at threshold 0.5: 1.36x / 2.47x / 4.19x / 6.31x / 8.47x / 10.34x for
  interval 1 / 2 / 4 / 8 / 16 / 32. This is the reference curve every policy is compared against.
- **The target zone is interval 2–4: Jaccard 0.62–0.55 at 2.5–4.2x.** Interval 1 → 2 drops Jaccard by 0.37
  (0.982 → 0.615) but intervals 4 → 32 by only 0.12. Most of the content drift happens as soon as any step reuses
  the cache; this is the region where a better-placed refresh has room to win.
- **Interval-2 Jaccard is bimodal per image** (6/50 exact match, range 0.24–1.0, sd 0.24). Paired tests over
  images, not means, must carry quality comparisons near interval 2.
- **The unified loop adds no cost**: token-identical to upstream and equal time (`src/methods/refresh/loop.py`).
  Run-to-run nondeterminism on this path: 0/5 token mismatches.
- **Threshold 0.5 barely parallelizes caption decoding**: 105–114 forwards of 128 at every interval. Measured r
  equals 1/k (r = 0.502 at interval 2, 0.035 at 32), so r is set by the interval, not by early termination.
- **Cost profile (A100 80GB PCIe)**: full recompute 65.6 s/image; interval 1 48.3, 2 26.5, 4 15.7, 8 10.4,
  16 7.7, 32 6.3 s. Model load 27–53 s; peak memory 31 GB. work1 jobs can be preempted (job 18513394 was, at
  270/350 rows); `--requeue` plus per-row resume recovered it without duplicates.

## Next steps

Done 2026-10-28: env + weights; Phase 0 smoke; paper settings (threshold 0.5, word-set Jaccard); unified loop
with `RefreshPolicy` (G1(i)); `src/scripts/gate_check.py g1`; MME frontier (G1(ii)).

Remaining Phase 1 work (plan: frontier + baselines on captioning and MathVerse):

1. **MathVerse frontier** (proposed next, no new code): MathVerse data prep, a thinking-mode campaign
   (gen_length 256–512), full + intervals 1–32 via `slurm/campaign.sbatch`. Needs: prompt and answer parsing
   for thinking mode, accuracy scorer. Runtime to estimate once gen_length and sample count are fixed.
2. **Elastic-Cache port to LLaDA-V** (strongest baseline; released code has `llada/`, `dream/` only). Validate
   first on LLaDA-8B-Instruct against upstream Elastic-Cache, then port. Plan budget +1–2 weeks.
3. **dKV-Cache-Decode port** (text-only upstream), same validation path.
4. dLLM-Cache baseline from upstream LLaDA-V's own copy (`register_cache_LLaDA_V`); add as a campaign arm.
5. Same-budget random refresh policy (`policies.py`), needed by every later comparison.
6. Optional: sync the online plan artifact (claude.ai/artifact/Aw3jJnaReYrrrsWJ2R8tEC) with the frozen gates,
   the repetition protocol and the measured frontier.

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
  settled before Phase 1 runs; it is not decided here. **Settled 2026-10-28** from the paper: threshold 0.5,
  lowercased alphabetic word-set Jaccard (see "Phase 1").

## Phase 1 (started 2026-10-28)

Faster but Different settings (read from arXiv 2607.29079, Sec. 4 and 6.2): prompt "Please describe the image
in detail.", gen_length = block = steps = 128, **threshold 0.5** while the interval varies, deterministic
decoding, RTX 6000 Ada, times exclude load and preprocessing. Their 50 images are a fixed-seed subset of 300 MME
images that is not published. **Jaccard = over lowercased alphabetic word sets**, against the unaccelerated
output (`use_fast_dllm=False`) of the same image. Our Phase 1 frontier uses these settings.

Unified loop: `src/methods/refresh/loop.py` (copy of upstream's loop; only the refresh decision goes through a
`RefreshPolicy`, `src/methods/refresh/policies.py`). G1(i) test: `slurm/phase1_g1_identity.sbatch` →
`src/scripts/g1_identity.py`, 5 images, intervals 1/4/32, thresholds 0.5 and None.

Lever map (written before the run):

| Outcome | Meaning | Next action |
|---|---|---|
| Identical at all 6 (k, threshold) cells | Loop reproduces upstream; policies only change refresh timing | G1(ii) frontier campaign on 50 images |
| Mismatch, and upstream rerun also mismatches itself | Run-to-run nondeterminism, not a loop bug | Report the rate; G1(i) cannot be token-exact as frozen; ask the user how to read the gate |
| Mismatch, upstream rerun self-identical | Bug in the copy | Find first divergent step, fix, rerun |

**G1(i) PASS** (job 18510589, A100 40GB, 27 min; `results/phase1/g1_identity/g1_identity_summary.json`): the
unified loop is token-identical to upstream on 5/5 images in all 6 cells (interval 1/4/32 x threshold 0.5/None),
with equal time (e.g. interval 4, threshold 0.5: 15.6 s both). The upstream rerun is 5/5 identical, so
run-to-run nondeterminism on this path is 0/5. Lever-map row hit: first row → G1(ii) frontier campaign.

G1(ii) frontier: `configs/campaigns/mme_caption_frontier.json` via `slurm/campaign.sbatch`
(`src/scripts/run_campaign.py`), 50 images x {full, interval 1/2/4/8/16/32 at threshold 0.5}, A100 80GB pinned.
Verdict: `python src/scripts/gate_check.py g1` → `results/phase1/mme_caption_frontier/g1_summary.json`.

Lever map (written before the run):

| Outcome | Meaning | Next action |
|---|---|---|
| G1 PASS | Our frontier reproduces the published one on our hardware and images | Freeze the frontier as the reference curve; start Phase 1 baselines (Elastic-Cache / dKV-Cache ports) and MathVerse frontier |
| Speedups out of ±20 %, Jaccard in range | Hardware cost profile differs (A100 vs RTX 6000 Ada); output behaviour matches | Report both; ask the user whether G1(ii) speed should be read on the same GPU type as the paper before proceeding |
| Jaccard out of ±0.05 at some intervals | Image subset or decoding differs from the paper's | Check the per-image spread (5 vs 50 images, our seed-0 subset); if the gap is within subset variance, present to user; otherwise inspect settings |
| Not monotone | Noise at neighbouring intervals (e.g. 8 vs 16, paper 0.482 vs 0.463) | Present the paired CI; the gate is frozen, the decision how to proceed is the user's |

**G1 verdict: FAIL — (i) PASS, (ii) FAIL on one cell** (job 18513394, A100 80GB PCIe for all 350 rows, 50 images;
preempted once at 270 rows and resumed via requeue; `results/phase1/mme_caption_frontier/g1_summary.json`).

| Interval | speedup (paper) | Jaccard (paper) | Jaccard 95% CI over images | exact match | r |
|---|---|---|---|---|---|
| 1 | 1.36 (1.30) | 0.982 (0.987) | [0.961, 0.995] | 35/50 | 1.000 |
| 2 | 2.47 (2.39) | **0.615 (0.666)** | [0.550, 0.679] | 6/50 | 0.502 |
| 4 | 4.19 (4.10) | 0.553 (0.572) | [0.500, 0.606] | 0/50 | 0.253 |
| 8 | 6.31 (6.44) | 0.501 (0.482) | [0.457, 0.547] | 0/50 | 0.130 |
| 16 | 8.47 (8.85) | 0.459 (0.463) | [0.424, 0.495] | 0/50 | 0.067 |
| 32 | 10.34 (11.29) | 0.432 (0.420) | [0.400, 0.467] | 0/50 | 0.035 |

Full recompute 65.6 s/image (paper 71.9 s on RTX 6000 Ada). All speedups within ±20 %, both monotone, 5/6
Jaccards within ±0.05. The miss is interval 2: |0.615 − 0.666| = 0.051 vs tolerance 0.05. Bootstrap (10,000
resamples, seed 0) puts the paper's value inside our 95% CI at every interval; at interval 2 the per-image
Jaccard is bimodal (6 images at 1.0, range 0.24–1.0, sd 0.24), so the CI half-width (±0.065) exceeds the frozen
tolerance. Lever-map row hit: "Jaccard out of ±0.05 … within subset variance → present to user". The gate is
frozen; how to read this is the user's decision: **accepted as PASS with a recorded deviation (2026-10-28,
Decisions table)**.
