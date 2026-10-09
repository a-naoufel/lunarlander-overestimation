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

## E and F. Depth and update-to-data ratio (80 new runs) — added on 2026-10-09, before these runs were launched
- TD3 × critic LayerNorm {off, on} × seeds 0–4, **100k environment steps** (300k would make UTD 5 too long
  for the deadline), all other hyper-parameters as in the main study.
- E (depth): 1, 3, 4 or 5 hidden layers of 256 units (UTD 1). F (UTD): 2, 3, 4 or 5 gradient steps per
  environment step (2 hidden layers).
- The default point (2 layers, UTD 1) is taken from the main TD3 runs, seeds 0–4, using only their
  evaluations up to 100k steps: up to that step they are identical to 100k-step runs (same seed, same code;
  the replay buffer is not yet full).
- Per-run metrics, scaled to 100k steps: final return = mean of the evaluations at 80k, 90k and 100k;
  bias = mean over 50k–100k; success = share of those 30 final episodes with return ≥ 200.
- Tests, per experiment and per LayerNorm setting: Spearman correlation between the factor value and each
  per-run metric (n = 25); each value against the default (difference of means, 95% bootstrap CI, Welch's
  t-test); Holm correction over each experiment (4 values × 3 metrics × 2 LayerNorm settings = 24 tests).

## G. Longer training (60 new runs) — added on 2026-10-09 at 19:50, before these runs were launched
- Question: does DDPG reach a return of 200 when trained longer? Asked after seeing that no DDPG run ends at 200 or
  above after 300k steps (our code) or 150k steps (SB3 code, `mp-sb3`).
- Our code: DDPG and TD3 × critic LayerNorm {off, on} × seeds 0–9, **600k environment steps** (twice the main
  budget; longer runs would not finish before the deadline), all other hyper-parameters as in the main study. These
  runs use other CPUs than the main runs, so even their first 300k steps are new, statistically equivalent repeats.
- SB3 code: its core grid (DDPG / TD3 × critic LayerNorm × seeds 0–4) with **300k steps** instead of 150k, all other
  settings unchanged.
- Per-run metrics at the end of training, as defined in each code base: final return (mean of the last 3 evaluations
  in our code, of the last 5 in the SB3 code), success (share of those episodes with return ≥ 200), bias.
- Reported, per code base: (1) the number of DDPG runs whose final return is ≥ 200; (2) per condition, final return at
  the end vs. at half of the training (300k / 150k steps) within the same runs, paired t-test; (3) TD3 − DDPG final
  return per LayerNorm setting (difference of means, 95% CI, Welch's t-test). Holm correction over the 6 tests of
  (2) and (3) in each code base.
