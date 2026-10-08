# Analysis plan for the additional experiments

Written on 2026-10-08, **before** any of the analyses below was computed and before
the new runs were launched. The main study (DDPG / TD3 × critic LayerNorm × 10 seeds,
plus TD3 with β = 0.75) was already finished and its results known.

All four analyses will be reported (in the report, or in one sentence there with the
full results in the repository), labelled as exploratory. What gets a figure in the
report is chosen for its relevance to Q1/Q2, not for its p-values.

Per-run metrics are those of the main study: final return (mean of the last 3
evaluations), bias (mean of Q1 − G over the evaluations from 150k to 300k steps),
success rate (share of the 30 final evaluation episodes with return ≥ 200).
Same hyper-parameters (Table 1 of the report) and same seeds (0–9).

## A. Timing: does early bias predict later improvement? (existing 60 runs)
- Early bias: mean bias over the evaluations at 50k–150k steps.
- Later improvement: final return − mean evaluation return at 140k–160k steps.
- Test: Spearman correlation computed on ranks within each of the 6 conditions and
  pooled over the 60 runs; p-value from 10,000 permutations within conditions.
  Also reported per algorithm variant (n = 20, LayerNorm settings pooled the same way).

## B. Bias and performance within conditions (existing 60 runs)
- Bias and final return are z-scored within each of the 6 conditions.
- Test: pooled Pearson correlation (n = 60); p-value from 10,000 permutations within conditions.

## C. Dose-response on the TD3 target weight β (60 new runs)
- TD3 with β ∈ {0.5, 0.25, 0} × critic LayerNorm {off, on} × seeds 0–9, analysed
  together with the existing β ∈ {0.75, 1}.
- Tests, per LayerNorm setting: Spearman correlation between β and each per-run metric
  (n = 50); each β against β = 1 (difference of means, 95% bootstrap CI, Welch's t-test);
  Holm correction over this family (3 β × 3 metrics × 2 LayerNorm settings = 18 tests).

## D. Ablation: DDPG plus one TD3 change (60 new runs)
- DDPG + twin critics with min target (`--twin 1`), DDPG + target policy smoothing
  (`--smooth 1`), DDPG + delayed actor and target updates (`--policy-delay 2`),
  × critic LayerNorm {off, on} × seeds 0–9, compared with the existing DDPG and TD3 runs.
- Tests: each variant against DDPG, per LayerNorm setting (difference of means, 95%
  bootstrap CI, Welch's t-test); Holm correction over this family
  (3 variants × 3 metrics × 2 LayerNorm settings = 18 tests).

## Execution note
New runs use the same code (`td3.py`, whose defaults reproduce the main runs bit for bit)
and the same library versions, on other PPTI machines (ppti-gpu-4, ppti-gpu-2,
ppti-14-407-06). Runs on different CPUs are not bit-identical but are statistically
equivalent repeats.
