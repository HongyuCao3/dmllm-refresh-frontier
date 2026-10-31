# PROGRESS — dmllm-refresh-frontier run log

Authoritative run log, decisions and lever maps. Plan: `plans/research_plan_v3.md` (main line: commit policies);
`plans/research_plan_v2.md` (cache refresh) is a concluded sub-direction, logged below as Phases 0–2. Rules: `.claude/research.md`,
`.claude/cluster.md`. Newest entries at the bottom of each section.

## Status (2026-10-30)

| Plan item | Status |
|---|---|
| Router + rules | done (commit `9ced305`): `CLAUDE.md` project statement and "Where the project lives"; `.claude/cluster.md`; `.claude/research.md` |
| Third-party code | done: 5 clones at pinned commits, `third_party/MANIFEST.md` |
| Env `llada-v` | **built 2026-10-28** at `/scratch/hcao2/envs/llada-v`: torch 2.1.2+cu121, transformers 4.39.3, tokenizers 0.15.2; `llava` + Fast-dLLM hook + dLLM-Cache hook import OK on CPU (log `/scratch/hcao2/tmp/logs/setup_llada_v_20261028_1034.log`); GPU load verified in the Phase 0 smoke. `pyarrow<18` added for benchmark prep |
| Weights | **downloaded 2026-10-28**: `GSAI-ML/LLaDA-V` (16 GB, snapshot `a10ba17`) and vision tower `google/siglip2-so400m-patch14-384` (4.3 GB, snapshot `e8e4872`) in `/scratch/hcao2/hf_cache`. llava loads the tower with its own `SigLipVisionModel`, not transformers' siglip2 |
| Phase 0 smoke | **done 2026-10-28** (job 18490708, 11 min): upstream path runs; see "Phase 0 smoke" below |
| Phase 1 | G1 PASS with recorded deviation (2026-10-28): unified loop token-identical to upstream; MME frontier reproduced. Bad-case analysis of the frontier done (Obs 1, 2026-10-29). MathVerse frontier (Exp 4) **done 2026-10-29** (job 18603291, n = 300): no accuracy loss at any k up to 32; no lever-map row fits, decision with the user. Elastic-Cache / dKV-Cache ports deferred until G2 / G3 (user, 2026-10-29) |
| Docs | Quarto site filled 2026-10-29: Exp 1–3, Obs 1, claims C 1–C 5, task definition, method (loop), dashboard. `cd docs && quarto render` |
| Phase 2 | Exp 5 done 2026-10-29 (job 18601279): **G2 PASS on captions** (60.8%); MathVerse not probed. Page `docs/experiments/exp5_replacement_error_map.qmd` |
| Phases 3–5 (v2) | **not run**: sub-direction concluded 2026-10-30 (see Decisions). Elastic-Cache / dKV-Cache ports dropped unless phase E needs them |
| Plan v3 | **draft 2026-10-30, revised after a literature check the same day; revised again 2026-10-31** (gate B main condition, self-select check, 788-problem proposal, phase-C calibration target, D-order / D-count, training-set dedup, phase-D compute 300–500 GPU-h, backup line on Lafa): `plans/research_plan_v3.md`; gates A–E and L1 proposed in `.claude/research.md`, to be frozen by the user before phase A runs |
| Plan v3 phase 0 | **done 2026-10-31** (`docs/literature_review.qmd`): both 2607.08056 and SAS are D-order, neither reports NFE or uses LLaDA-V / MathVerse / caches (gaps hold). SAS: base frozen, code public, sequential one-token policy that degrades under parallel top-k, reward needs gold reasoning text. 2607.08056: no code, freezing unstated, single-run gains. Changes: left-to-right baseline added to phase A; SAS-style baseline retrained under our k with a reward choice for the user; 2607.08056-style baseline rebuilt from DCoLT's UPM with the base frozen |
| Backup line L0 | **done 2026-10-31** (CPU): on GSM8K a training-free min-p ≥ 0.9 step gate already saves 32–66% at ≥ 99.4% token agreement, and accuracy is saturated (0.96): no headroom; only a harder task (MATH-500, GPU) is left open. Details under Next steps |

## Decisions (user)

| Date | Decision |
|---|---|
| 2026-10-27 | Repetition: deterministic decoding, one run per arm, paired bootstrap `Δ [95% CI] (n)` over samples + paired tests (McNemar / Wilcoxon, Holm over anchors). Appendix: 50 samples × 3 sampling seeds (temperature 0.5), the sign of Δ must agree in all three. Recorded as a project exception in `.claude/global.md` |
| 2026-10-27 | Gates frozen: G1 token-identical to upstream at interval 1 / 4 / 32, speedup ±20% relative and Jaccard ±0.05 of the published table, both monotone; G2 top 20% cells ≥ 50% of error; G3 oracle ≥ 40% lower wall-clock on ≥ 2 anchors; G4 ≥ 25% on ≥ 2 anchors, Holm-significant, ≥ 50% of oracle gain |
| 2026-10-27 | Matched quality (proposed with the gates, accepted): not significantly below the anchor and within 1 accuracy point (or 0.01 Jaccard) |
| 2026-10-28 | **G1 accepted as PASS with a recorded deviation**: interval-2 Jaccard 0.615 vs paper 0.666 misses the ±0.05 tolerance by 0.001; the paper's value lies inside our 95% bootstrap CI at every interval, and the paper's 50-image subset is unpublished. Thresholds unchanged. The 50-image frontier (`results/phase1/mme_caption_frontier/g1_summary.json`) is the reference curve |

| 2026-10-29 | **Order of work**: record and analyse all Phase 0–1 results in the docs first; then (1) MathVerse frontier, (2) Phase 2 error map, D3 and D5 first. Elastic-Cache / dKV-Cache ports wait until G2 / G3 |
| 2026-10-30 | **Main line changed to commit (unmasking) policies, plan v3.** Cache refresh becomes a concluded sub-direction: with block = gen_length the 7.3 s per-item floor caps any refresh saving at ~12% (< G3's 40%), and Exp 4 shows staleness flips answers symmetrically (lost ≈ gained), which points at commit decisions, not cache age. Base model stays frozen; a small commit policy may be trained from phase D on (`CLAUDE.md`, `.claude/global.md`, `.claude/research.md` updated) |
| 2026-10-31 | **Backup main line: commit policies on Lafa** (plan v3, "备用主线"). Lafa commits a left prefix (1–4 tokens) of each 4-token block after K Euler steps, so the decision is prefix length and step count, not order. Trigger and gate L1 thresholds still to be confirmed by the user |
| 2026-10-29 | **No analysis-paper fallback for now**: if G2 or G3 fails, the response is bad-case analysis aimed at an improvement and a positive result, not a pivot to the "error is uniform" paper. Thresholds unchanged |

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

## Findings: frontier bad cases (Obs 1, 2026-10-29)

CPU analysis of the 350 Exp 3 generations: `src/scripts/analyze_frontier.py` →
`results/phase1/mme_caption_frontier/analysis_summary.json`, figures in `docs/experiments/figures/obs1_*`.
Exploratory (questions posed after the run). Page: `docs/experiments/obs1_frontier_bad_cases.qmd`.

- **What goes stale**: with block = gen_length = 128 every reuse step recomputes all answer positions; only the
  prompt's (visual + question) keys/values are reused. Staleness = prompt states that have not yet attended to the
  answer tokens decoded since the last refresh.
- **Near k = 2, Jaccard vs full tracks trajectory, not quality**: Jaccard(k=2, k=4) = 0.63 ≈ Jaccard(k=2, full)
  = 0.61. Visual check of the 4 worst k = 2 images (Jaccard 0.24–0.33, one reader, qualitative): the k = 2
  caption is as good or better in 3/4 (e.g. full says "two glasses of beer", image shows beer + water).
- **Small perturbations flip trajectories**: k = 1 differs from full only by threshold-0.5 parallel decoding
  (112 vs 128 forwards); 15/50 not identical, one at Jaccard 0.507.
- **Divergence is decided early**: median first divergent answer token 22 at k = 2 and 4, 8.5 at k ≥ 8; 40% of
  images split within 16 tokens at k = 2; Spearman(Jaccard, split position) = 0.75.
- **Fragility is per image**: per-image Jaccard rank-correlates 0.50–0.75 across neighbouring cached k; not
  explained by prompt length (Spearman −0.05). Low: artwork, celebrity; high: landmark.
- All 350 outputs hit the 127-token cap: every caption is cut off.
- Levers this suggests (each against fixed interval and same-budget random refresh): dense refresh at early
  steps; refresh triggered by the number of tokens decoded since the last refresh; per-sample budget.
- Consequence for the main line: the k = 2–4 target zone was defined by Jaccard; MathVerse accuracy must set the
  anchors (claim C 5 in `docs/claims.qmd`).

## Next steps

### Plan v3 (2026-10-30)

**Pilot for phase B from Exp 4 data (2026-10-31, CPU, exploratory, not a reported result).** Treating the 6 Exp 4
cache arms as 6 perturbed decodings with the order unchanged: any-of-6 correct 35.7% vs full 30.3% (+5.3 points;
multiple-choice 49.1% vs 42.8%, free-form 17.3% vs 13.4%); all-of-6 correct 25.0%. Perturbation alone already
clears the proposed 3-point headroom, so gate B's discriminating condition is order@K vs token sampling. Computed
from `score_mathverse.py --dump` on `/scratch/.../mathverse_vd_frontier/`. MathVerse testmini vision-dominant has
788 problems (parquet checked); the 300 are a seeded subset.

**Backup line, phase L0 (2026-10-31, CPU, exploratory, not a reported result).** Re-read of dKV-ARKV L-1 traces
(`~/dKV-ARKV/results/l1_lafa/18003407/`, Lafa-8B, GSM8K train first 100, CFG 8 + self-cond, single seed; per-row
argmax / min-p, no per-token confidences stored):

| K | acc | stop at first row with min p ≥ 0.9: rows used / NFE saved | token agreement with final | final row min p < 0.9 |
|---|---|---|---|---|
| 2 | 0.93 | 0.68 / 32% | 1.0000 | 26% of blocks |
| 4 | 0.96 | 0.49 / 51% | 0.9991 | 6.8% |
| 8 | 0.96 | 0.34 / 66% | 0.9936 | 2.1% |

- **The prior already takes the step-stopping slack.** A fixed confidence gate (no training) saves 32–66% of the
  rows at ≥ 99.4% token agreement, so the proposed gate L1 ("the prior threshold does not reach the oracle point")
  would fail on GSM8K: no room left for a posterior on step count.
- **Accuracy is saturated.** K 4 is 0.96 vs AR 0.97 (4 wrong of 100), so prefix length cannot buy accuracy here.
  Per-problem share of low-confidence blocks separates wrong from right only weakly (AUROC 0.54–0.77, n_wrong 4–7,
  no inference).
- **Literature:** step / token early stopping and adaptive commit are well covered on masked DLMs (Prophet
  2508.19982, Jot 2602.11133, LESS 2606.16908, C⁴ / LATCH 2607.28166, x-Prediction Flow 2606.29066, Ada-DLM ACL
  2026). Nothing found on block-causal continuous flow LMs like Lafa, but the training-free gate above is the
  direct transfer and already works.
- **Reading:** on GSM8K the backup line has no headroom on either axis. The only open case is a harder task where
  flow accuracy is not saturated (MATH-500): whether prefix length or step count then changes accuracy needs GPU runs
  (L1). The efficiency side still has to be compared with self-speculation (×8.5 faster than serial K 4 flow).

Open decisions for the user before phase A runs: (1) freeze gates A–E (proposals in `.claude/research.md`);
(2) the policy training set (candidates: Geo170K, MathV360K, GeoQA train; any MathVerse version is excluded,
it shares the evaluation problems); (3) K and the perturbation for the phase-B order upper bound.

Phase A (block baselines), no new model code beyond a commit-policy interface:

1. Add `src/methods/commit/` with a `CommitPolicy` interface on the unified loop (`src/methods/refresh/loop.py`):
   top-k confidence (LLaDA), confidence threshold (Fast-dLLM), random order at the same NFE.
2. Block decoding (block 32, gen_length 256) on the 300 MathVerse vision-dominant problems; G-A token-identity
   test against upstream LLaDA-V block decoding.
3. Measure s/problem with blocks; update the runtime table in plan v3 (estimate before measuring: 10–30 s per
   problem, 1–2.5 GPU-h per arm).
4. Write the phase-A lever map into the docs before the run (`.claude/research.md`, *Lever maps*).

Literature check (2026-10-30, details in plan v3 "已有工作与空白"): the closest prior work is 2607.08056 (GRPO
order module on MMaDA-8B, multimodal understanding incl. MathVista; no MathVerse, no LLaDA-V, no NFE, no
image-token analysis, no cache). SAS (2606.23567) trains an order policy over a frozen LLaDA-8B with GRPO. Both
become phase-D baselines. A new phase 0 (read both in full: frozen base?, code?, inputs and reward, NFE) comes
before phase A; the proposed gate B threshold drops from 5 to 3 points.

### Plan v2 (concluded sub-direction; kept for the record)

Done 2026-10-28: env + weights; Phase 0 smoke; paper settings (threshold 0.5, word-set Jaccard); unified loop
with `RefreshPolicy` (G1(i)); `src/scripts/gate_check.py g1`; MME frontier (G1(ii)).

Order set by the user on 2026-10-29: (1) MathVerse frontier, (2) Phase 2 error map with D3 / D5 first;
ports (items 2–3 below) wait until G2 / G3.

State on 2026-10-29 (superseded: both jobs finished, Exp 5 recorded; Exp 4 result below, open decision with the user): when the probe ends, run `python src/scripts/analyze_probe.py` (a
background waiter does this), write the Exp 5 page and read the result against the Exp 5 lever map. When the
MathVerse array ends, score it and read it against the 300-problem lever map. If the interim pattern holds (no
accuracy loss up to k = 32), no lever-map row fits; that case goes to the user (see "Interim, n = 126" under Exp 4),
gates and lever maps stay unchanged.

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

## Exp 4 — MathVerse frontier (started 2026-10-29)

Settings (user, 2026-10-29): MathVerse testmini **vision-dominant**, 300 problems (seed 0 of 788;
`src/scripts/prepare_mathverse.py` → `/scratch/hcao2/data/dmllm_refresh/mathverse_vision_dominant/`, 173 multiple
choice / 127 free-form), upstream prompt (format-prompt, `query_cot`, " /think"), **gen_length = steps = block =
256**, threshold 0.5, arms full + k ∈ {2, 4, 8, 16, 32}. **Rule-based scorer** (no GPT-4o): option letter for
multiple choice, numeric / normalized match for free-form, spot-checked by hand. k = 1 is left out: anchors and
policies all use threshold 0.5, so it is not needed for G3 / G4, and it would add ~35% cost.

Pilot first: `configs/campaigns/mathverse_vd_pilot.json` (first 8 problems), then
`sbatch --array=0-5 slurm/campaign.sbatch configs/campaigns/mathverse_vd_frontier.json`. Estimate (guess, scaled
from captions): ~260 s / problem → ~22 GPU-h for 300. Why it is needed: accuracy is the plan's primary quality
metric, and Obs 1 showed caption Jaccard near k = 2 tracks trajectory, not quality, so the anchors for G3 / G4
must come from this run.

Lever map, pilot (written before the run):

| Outcome | Meaning | Next action |
|---|---|---|
| Most full outputs end (eot) before 256 tokens with an extractable answer | Length fits think mode | Submit the 300-problem array with the measured time |
| Most full outputs hit 256 without an answer | Reasoning truncated; accuracy would measure truncation | Ask the user: 512, or no-think mode |
| Time per problem > 2× the estimate | Prompt longer than assumed | Re-estimate; ask before submitting > 40 GPU-h |

Lever map, 300-problem frontier (written before the run):

| Outcome | Meaning | Next action |
|---|---|---|
| Accuracy at k = 2–8 not significantly below full and within 1 point (McNemar) | Caching costs no accuracy up to k = 8 (C 5 confirmed): the room is at k ≥ 16 | Anchors move to where accuracy drops; Phase 2 / 3 target that region |
| Accuracy drops monotonically from k = 2 with significant gaps | k = 2 / 4 / 8 are real quality anchors | Keep the planned anchors; Phase 2 error map on MathVerse |
| Accuracy flat at every k, full included, and low (< 15%) | Task too hard at this length; no signal | Ask the user: other split, MathVista, or longer generation |
| Non-monotone, CIs overlapping | Accuracy too noisy at n = 300 | Extend to all 788 problems before choosing anchors |

## Exp 5 — Phase 2 replacement-error map on captions (started 2026-10-29)

Probe `src/scripts/probe_replacement.py` (job `slurm/phase2_probe.sbatch`), 50 MME images, trajectory = unified
loop at k = 1, threshold 0.5. At each step t, one cell of the prompt cache is replaced by its value from step
t − s and the block forward is compared with the fresh-cache forward. With block = gen_length only the prompt can be
stale, so D5 ("does local staleness spread to another group") reduces here to the visual-vs-text comparison of
D1; it becomes its own test once a task uses block < gen_length.

Cells: {visual, text} × layer bands {0–7, 8–15, 16–23, 24–31} × step, at ages s ∈ {1, 4}; plus whole prompt ×
all layers at s ∈ {1, 2, 4, 8} for the D3 time profile. Per cell: top-1 flips among masked answer positions,
mean KL, and whether the committed decision (transferred positions + tokens) changes. The probe also records
decoding order (positions decoded per step). Estimate (guess): ~3 min / image, ~2.5 h for 50; smoke on 2 images
first.

**G2 as computed here** (frozen threshold, cell definition fixed before the run): cells = position group × layer
band × step (2 × 4 × steps), error = top-1 flips summed over images, at age 4 (k = 4 anchor); G2 passes on
captions if the top 20% of cells hold ≥ 50% of flips. Age 1 reported alongside.

Lever map (written before the run):

| Outcome | Meaning | Next action |
|---|---|---|
| Whole-prompt flips concentrated in early steps (first 20% of steps ≥ 50% of flips, D3) | Obs 1's early-split lever is real at the cache level | Static early-dense schedule (S1) vs fixed interval and same-budget random at matched r, on captions and MathVerse |
| Visual and text groups differ ≥ 2× in flips per position (D1) | Groups need different refresh periods | Group-wise refresh (refresh text every step, visual every k, or the reverse) |
| One layer band holds ≥ 50% of flips | Staleness matters in a few layers | Partial-layer refresh (close to Elastic-Cache's deep-layer idea: check overlap) |
| G2 met | Error concentrated: an oracle has room | Phase 3 oracle on the concentrated cells |
| Flips uniform over steps, groups and bands (G2 not met) | No static cell is cheap to skip | Bad-case analysis (user, 2026-10-29): look at which decisions flip and at token-count triggers, not an analysis paper |
| Decision flips rare (< 5% of steps at age 4) while Jaccard drops 0.43 at k = 4 | Divergence comes from rare single flips that cascade | Target the few flip steps: online trigger on confidence margin of the committed tokens |

**Pilot result** (job 18600102, A100 80GB PCIe, 8 problems; smoke only, not a reported result):
- N = 1.4k–3.7k (mean 2.4k). Full 101 s / problem; k = 2 / 4 / 8 / 16 / 32: 23.5 / 16.4 / 13.7 / 10.5 / 8.8 s
  (speedup 4.3× / 6.2× / 7.4× / 9.7× / 11.5×). 174 s / problem in all → ~14.5 GPU-h for 300.
- Every output fills all 256 positions, but none is truncated: the model pads with repeated lines and always ends
  with "The answer is …". No `</think>` tag appears. Lever-map row hit: "extractable answer" → array submitted.
- Scorer checked by hand on all 48 rows; added a fix for multiple-choice answers given as option text
  (e.g. "This is a function" → A). Pilot accuracy full 4/8, k = 2 4/8, k ≥ 4 2/8 (n = 8, no inference).
- Degenerate repetition at k = 32 (e.g. "z = 40 \n 40 \n 40 …") still ends in the same final answer.

300-problem array: job 18603291 (6 shards, `sbatch --array=0-5 slurm/campaign.sbatch
configs/campaigns/mathverse_vd_frontier.json`). Score: `python src/scripts/score_mathverse.py mathverse_vd_frontier`.

**Interim, n = 126 of 300** (2026-10-29, ~1.1 h into the array; partial shards, not a reported result, CIs too
wide to read a lever-map row):

| Arm | Accuracy | Speedup | Δ vs full [95% CI] | lost / gained | McNemar p |
|---|---|---|---|---|---|
| full | 30.2% | 1.0× | — | — | — |
| k = 2 | 29.4% | 4.1× | −0.8 [−2.4, 0.0] | 1 / 0 | 1.0 |
| k = 4 | 27.8% | 6.2× | −2.4 [−5.6, 0.0] | 3 / 0 | 0.25 |
| k = 8 | 30.2% | 8.5× | 0.0 [−4.0, +4.0] | 3 / 3 | 1.0 |
| k = 16 | 31.0% | 10.4× | +0.8 [−3.2, +4.8] | 3 / 4 | 1.0 |
| k = 32 | 31.0% | 11.6× | +0.8 [−4.0, +5.6] | 4 / 5 | 1.0 |

By type: multiple choice 38–42%, free-form 10–14%, in every arm. Reading if it holds at n = 300: fixed interval
loses no accuracy even at k = 32, so accuracy leaves no room for a better schedule on this task and G3 cannot pass
here. Neither "flat and low (< 15%)" nor "k = 2–8 flat, room at k ≥ 16" fits: full is at 30%, and k ≥ 16 does
not drop either. Options to put to the user then: a setting more sensitive to staleness (longer generation,
semi-autoregressive blocks < gen_length, another task), or a search for the k where accuracy starts to drop
(k > 32).

**Probe smoke** (job 18600727, 2 images): 83 s (N = 1042) and 115 s (N = 3715) per image; full probe job 18601279
(50 images, ~1.4 h). Smoke-level hints only (n = 2): flips grow toward late steps, the deep bands 16–31 hold ~80% of
flips, text positions flip 11–15× more per position than visual ones.

**Result** (job 18601279, 50 images, 1 h 54 min, 136 s / image; `results/phase2/probe_mme/probe_summary.json`,
page `docs/experiments/exp5_replacement_error_map.qmd`):
- **G2 PASS on captions**: top 20% of 952 (group × band × step) cells hold 60.8% of flips at age 4. Reference
  shares: structure only (group × band, even over steps) 50.5%; time only 42.1%. The concentration is mostly static.
- D3: first 20% of steps hold 36.0% of flips (masked-position share there: 35.9%) → early-dense row **not hit**.
  Flip rate rises late (age 4: 0.052 → 0.114); decision changes are U-shaped (19.6% first fifth, 17.8% last bin).
- D1: text positions (~61 of 3112, 2%) cause 30.3% of flips, **21.7×** visual per position → group-wise row **hit**.
- Bands: deepest band 45.9% (< 50%, row narrowly **not hit**); layers 16–31 together 78.1% (Elastic-Cache overlap).
- Decision changes at 13.5% of steps at age 4 → "rare flips" row **not hit**.
- Rows hit: "G2 met" (Phase 3 oracle on concentrated cells) and "groups differ" (group-wise refresh). Proposed
  first candidate: text prompt every step + visual every k (text refresh estimated ~2% of a prompt refresh).
  Open: C 2 needs G2 on MathVerse too, and the interim Exp 4 shows no accuracy loss to recover there.

## Exp 4 — result (2026-10-29)

Job 18603291 (6 shards, A100 80GB PCIe, 2 h 17 min – 2 h 35 min each, all COMPLETED). Scored with
`src/scripts/score_mathverse.py mathverse_vd_frontier` → `results/phase1/mathverse_vd_frontier/accuracy_summary.json`.
n = 300 (173 multiple choice / 127 free-form), paired over problems.

| Arm | Accuracy | gen s | Speedup | r | Δ vs full [95% CI] | lost / gained | McNemar p |
|---|---|---|---|---|---|---|---|
| full | 30.3% (91) | 100.0 | 1.0× | 1.000 | — | — | — |
| k = 2 | 29.7% (89) | 24.5 | 4.1× | 0.502 | −0.7 [−2.3, +1.0] | 4 / 2 | 0.69 |
| k = 4 | 28.0% (84) | 16.2 | 6.2× | 0.253 | −2.3 [−5.0, 0.0] | 11 / 4 | 0.12 |
| k = 8 | 30.0% (90) | 11.9 | 8.4× | 0.129 | −0.3 [−3.3, +2.3] | 10 / 9 | 1.0 |
| k = 16 | 30.7% (92) | 9.6 | 10.4× | 0.067 | +0.3 [−3.0, +3.7] | 12 / 13 | 1.0 |
| k = 32 | 30.7% (92) | 8.3 | 12.0× | 0.036 | +0.3 [−3.0, +3.7] | 13 / 14 | 1.0 |

- **Fixed interval loses no significant accuracy up to k = 32.** The k = 4 dip (−2.3, p = 0.12) does not
  recur at k ≥ 8: noise, not a trend. Multiple choice 40–44%, free-form 11–14% in every arm.
- **Staleness changes answers, not accuracy**: discordant problems grow from 6 (k = 2) to 27 (k = 32), but lost ≈
  gained at every k ≥ 8.
- Every output fills 256 positions (repetition padding, then "The answer is …"), as in the pilot.
- **Wall-clock is linear in r with a large floor** (fit over the 5 cached arms, max residual 0.22 s):
  gen s ≈ 7.3 + 34.5 r. The floor is the reuse steps, which recompute all 256 answer positions (block = gen_length).
  Refresh share of time: 71% at k = 2, 54% at 4, 37% at 8, 15% at 32. Captions: 4.8 + 43.2 r, 24% at k = 32.
- **Consequence for G3 on this setting**: the cheapest fixed interval at full accuracy is k = 32 (8.3 s). Even
  r → 0 at that accuracy saves at most 12%, below the 40% of G3. Anchors k = 2 / 4 / 8 are not distinct quality
  levels here.
- Lever-map row hit: **none** ("k = 2–8 flat, room at k ≥ 16" fails because k ≥ 16 does not drop either; "flat
  and low" fails because full is 30%, not < 15%). Gates and lever maps unchanged; next setting is the user's
  decision (options: k > 32 sweep; semi-AR blocks < gen_length; another task or longer generation).

