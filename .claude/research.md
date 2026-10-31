# research.md — how experiments serve the main line

Read before planning an experiment, interpreting a result, or recommending a next step. Adapted from
`~/dKV-ARKV/.claude/research.md`. The plan is `plans/research_plan_v3.md`; it is authoritative for scope. `plans/research_plan_v2.md` (cache refresh) is a
concluded sub-direction; its gates G1–G4 below are kept as a record.

## The main line

On LLaDA-V, learn **which answer positions to commit at each denoising step** so that accuracy at a matched
NFE (forward passes per item) beats the model's own confidence (the prior: top-k confidence as in LLaDA,
confidence threshold as in Fast-dLLM). The posterior signals are answer reward (RL), successful decoding
orders and trajectory-survival labels. A literature check (2026-10-30, plan v3 "已有工作与空白") found that
"order matters", "confidence is not optimal", "best-of-K orders help", "an RL order policy over a frozen denoiser
works" (SAS, 2606.23567) and even "an RL order policy on a multimodal MDM" (2607.08056, MMaDA) are established;
they are cited, not re-tested. The contribution is: an understanding-only dMLLM (LLaDA-V) on vision-dominant math;
whether learned gains concentrate on image-dependent tokens beyond attention-based visual signals; accuracy at
matched NFE; and a learned joint commit + cache policy. Main ground:
MathVerse vision-dominant (300 problems, the Exp 4 set) with block decoding (block 32, gen_length 256).

The base model is a frozen released checkpoint. Generation length, block size and the NFE budget are fixed per
comparison; **only the commit policy (and, in phase E, the refresh policy) changes** between arms. Policy
training data never overlaps the evaluation problems; other MathVerse versions share its problems and are
excluded.

## Gates of plan v3 (proposed, not frozen)

Thresholds below are proposals. They are frozen by the user before the first run of the phase they gate, and
not changed afterwards. Until frozen, no gate verdict is reported.

| Gate | After | Proposed pass condition |
|---|---|---|
| A | Phase A block baselines | Unified loop token-identical to upstream LLaDA-V in block decoding; baseline accuracy–NFE curves (near-)monotone |
| B | Phase B order upper bound | **Main condition:** at matched NFE and K, order@K significantly above K token-sampling runs with the order fixed (paired bootstrap CI excludes 0). Secondary: Acc(order@K) − Acc(confidence) ≥ 3 points (a free Exp 4 pilot already gives +5.3 from 6 cache perturbations without any order change, so this alone does not discriminate). Multiple-choice and free-form reported separately; a pass on multiple-choice only is read as guessing. self-select@K is reported but does not gate |
| C | Phase C multimodal diagnosis | Calibration error of image-dependent tokens significantly above that of other tokens (paired bootstrap CI excludes 0), and image dependence still predicts errors after controlling for attention-based visual signals (VISAGE, VRCD) |
| D | Phase D learned policy | ≥ 1 policy beats the confidence baseline by ≥ 3 points at matched NFE, Holm-significant, reaches ≥ 50% of the phase-B headroom, and is not below SAS-style and 2607.08056-style policies re-implemented on LLaDA-V, nor below best-of-K + self-select on the accuracy–NFE curve. D-order (fixed commit count) and D-count (policy also sets the count; matched mean NFE) are reported separately. Assumes the 788-problem evaluation set; at n = 300 re-derive the threshold from a power calculation before freezing |
| E | Phase E commit + cache | ≥ 15% lower wall-clock than the strongest training-free joint method (EntropyCache, Polestar) at matched accuracy |
| L1 | Backup line (Lafa) phase L1 | Oracle prefix length / step stopping within ±1 point of Lafa's default flow setting (K 4, CFG 8, self-cond) at ≥ 30% fewer NFE, and the prior confidence-threshold baseline does not reach this point |

Backup line (plan v3, "备用主线"): commit policies on Lafa (text-only block-causal flow LM, block 4, prefix commit,
no order choice). Switched to only by the user, on a failed gate B main condition or if phase 0 finds the LLaDA-V gap
covered by prior work; a failed gate C does not trigger it. Lafa's self-speculative mode is the reference for any
efficiency claim.

## Gates of plan v2 (concluded sub-direction, record only)

Each phase ended in a gate decided by `src/scripts/gate_check.py`: its exit code and printed `GATE ...` line are
the verdict. Thresholds below were frozen by the user on 2026-10-27, before any data existed. Outcome: G1 PASS
(recorded deviation), G2 PASS on captions, G3 not run: with block = gen_length the per-item floor caps the
saving at about 12% (Exp 4), so the sub-direction was concluded by the user on 2026-10-30.

| Gate | After | Pass condition (frozen) |
|---|---|---|
| G1 | Phase 1 frontier | (i) The unified generation loop is token-identical to upstream LLaDA-V + Fast-dLLM on fixed interval 1, 4 and 32 (bf16, same samples). (ii) On the 50 MME images, for every interval in {1, 2, 4, 8, 16, 32}: our speedup over full recompute is within ±20% (relative) of Faster but Different Tab. 3, our Jaccard vs full output is within ±0.05 (absolute) of it, and both are monotone in the interval |
| G2 | Phase 2 error map | Top 20% of (position group x layer x step) cells hold ≥ 50% of total replacement error (top-1 flip rate), on ≥ 1 main task |
| G3 | Phase 3 oracle | Oracle wall-clock at matched quality ≤ 0.6 x fixed interval (≥ 40% lower), on ≥ 2 of the anchors interval 2 / 4 / 8 |
| G4 | Phase 4 policies | ≥ 1 policy cuts matched-quality wall-clock by ≥ 25% on ≥ 2 of the anchors interval 2 / 4 / 8, each significant after Holm correction over the 3 anchors, and reaches ≥ 50% of the oracle's wall-clock gain |

G1 compares **ratios** (speedup, Jaccard), not seconds: the published seconds come from a different GPU.
"Matched quality" means the policy's quality is not significantly below the anchor's (paired test below) and
its point estimate is within 1 accuracy point (or 0.01 Jaccard) of it.

## When a gate fails

- A failed gate selects a lever, not a stop. **User decision 2026-10-29: the analysis-paper fallback ("error is
  uniform, fixed interval is near-optimal") is off the table for now.** G2 or G3 failing leads to bad-case
  analysis aimed at an improvement lever and a positive result. Stopping or pivoting stays the user's decision;
  present ranked levers.

## Lever maps

Before proposing any experiment, diagnostic or re-run, write its lever map (outcome → meaning → next main-line
action) into the docs before the results come in. If every outcome leads to the same action, do not run it.

## Repetition and variance (user decision, 2026-10-27)

Under LLaDA-V's default decoding (greedy with low-confidence remasking) repeated runs of one config are
near-identical, so seed std is not a meaningful spread. The uncertainty that matters is "does the conclusion
hold on other samples", so:

- **Main results: deterministic decoding, one run per arm, paired over samples.** Every arm runs on the same
  samples. Report the per-sample paired difference against the reference arm as `Δ [95% CI] (n)`: paired
  bootstrap over samples, 10,000 resamples, percentile interval, resampling seed fixed and logged.
- Paired tests: McNemar (exact) on per-sample correctness for accuracy; Wilcoxon signed-rank on per-sample
  wall-clock, Jaccard and BERTScore. α = 0.05; Holm correction across the 3 quality anchors where a gate
  counts anchors.
- **Robustness check (appendix, not the headline):** for each claimed win, 50 samples x 3 sampling seeds with
  stochastic decoding (temperature 0.5, fixed before the run). The check passes if the sign of Δ (policy vs
  fixed interval at the same anchor) is the same in all 3 seeds. It tests sensitivity to small trajectory
  changes, which the paired bootstrap cannot see.
- Run-to-run bf16 nondeterminism is measured once in Phase 1 (same config twice, token mismatch rate) and
  reported; it is not a substitute for either of the above.

## Controls that a positive result never relaxes

- Every arm runs on the same samples (no selection), the same prompts, and the same generation length, steps and
  threshold. Forced-reasoning variants are reported separately from official settings.
- Speed and quality come from the same runs, on one GPU type, timed as in `.claude/cluster.md`.
- r is reported as measured, and any probe or signal cost is converted into an equivalent r increment.
- Every proposed commit policy is compared against the **confidence baselines, block-wise left-to-right (SAS
  Table 11: it ties or beats SAS at short lengths), a random commit order, a
  training-free visual re-ranking method (VRCD or VISAGE), and SAS-style and 2607.08056-style learned policies,
  all at the same NFE**; a phase-E policy is compared against EntropyCache and Polestar; every upper bound from best-of-K orders is compared against best-of-K token sampling with the
  order fixed, so that "more tries" is not read as "better order".
- (v2, refresh policies) Every adaptive refresh policy is compared against **same-budget random refresh**, and
  every win is re-tested stacked with Elastic-Cache's deep-layer refresh to show the gains add.
- Replacement error (KL, top-1 flip rate) is a diagnostic and steers decisions; it is never a reported quality
  number. Quality is task accuracy (primary) and agreement with full recompute (Jaccard, BERTScore).
