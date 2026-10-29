# research.md — how experiments serve the main line

Read before planning an experiment, interpreting a result, or recommending a next step. Adapted from
`~/dKV-ARKV/.claude/research.md`. The plan is `plans/research_plan_v2.md`; it is authoritative for scope.

## The main line

On LLaDA-V, reach the output quality of fixed-interval refresh with a **lower full-refresh ratio r**, measured
as **lower end-to-end wall-clock at matched quality**. The opponents are the fixed-interval frontier
(Fast-dLLM, intervals 1–32) and Elastic-Cache. Long generation (detailed captioning, MathVerse/MathVista in
thinking mode) is the main ground; short-answer VQA is boundary evidence only.

The model is a frozen released checkpoint. Generation length, denoising steps and the parallel-decoding
threshold are fixed per task; **only the refresh policy changes** between arms.

## Gates

Each phase ends in a gate decided by `src/scripts/gate_check.py`: its exit code and printed `GATE ...` line are
the verdict. Thresholds below were frozen by the user on 2026-10-27, before any data existed, and are not
changed afterwards.

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
- Every proposed adaptive policy is compared against **same-budget random refresh**, and every win is re-tested
  stacked with Elastic-Cache's deep-layer refresh to show the gains add.
- Replacement error (KL, top-1 flip rate) is a diagnostic and steers decisions; it is never a reported quality
  number. Quality is task accuracy (primary) and agreement with full recompute (Jaccard, BERTScore).
