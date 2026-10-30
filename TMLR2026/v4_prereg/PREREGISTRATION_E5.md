# E5 pre-declaration: decoupled horizon cohort

Written 2026-10-30, before any E5 outcome existed. It responds to the v4 review finding that E3 ran under the
coupled default optimizer, where reward scale alone moves the crossing (E1: A − S = −0.074 at 300k). File hashes
are appended to `HASHES.sha256` immediately before launch. Platform, environment and estimator are as in
`PREREGISTRATION_v4.md`.

## Design
- **Learner:** the E1 `separate_no_clip` / `VecNormalize` cell: disjoint actor and critic Adam optimizers, no
  gradient clipping, `VecNormalize(norm_obs=False, norm_reward=True, gamma=0.99, clip_reward=10, epsilon=1e-8)`.
- **Arms:** B (curvature only), A (losses ×7) and S (both branches ×7), all with reference z₀ = −2.6281398333,
  as in E1.
- **Grid:** the nine E3 cells {.2, .4, .6, .65, .7, .75, .8, .85, 1}.
- **Seeds:** 42–51.
- **Budget:** 3.6M requested agent transitions, evaluated every 150k (24 checkpoints).
- **Size:** 3 × 9 × 10 = **270 trajectories**.
- **Runner:** `e5_decoupled_horizon.py`, which reuses `optimizer_factorial.py` construction unchanged.
- **Pre-launch gate:** a 300k run of E5's construction path for (A, w = .6, seed 42) must reproduce the
  completed E1 run (separate_no_clip, VecNormalize, A, .6, 42) bit-for-bit (evaluation summary and all ten
  episodes). Launch proceeds only if this holds.

## Analysis (fixed now)
- **Estimator.** Firth w50 per arm and checkpoint; 2,000 paired seed-block draws, bootstrap seed 20261015.
  Out-of-grid and non-converged draws are retained and counted.
- **Primary contrasts at 300k, 900k, 1.8M, 3.6M:**
  - A − S (asymmetry);
  - A − B (loss multiplier);
  - S − B (symmetric scaling).
- **Secondary:**
  - absolute w50(T) per arm;
  - stability (3.6M − 1.8M interval within ±.05);
  - recovery and relapse by 3.6M, conditional on the 300k label;
  - all 24 checkpoints, reported descriptively.

## Predictions and interpretation (declared now)
1. **At 300k, A − S > 0 and S − B includes 0.** This replicates the E1 decoupled cell on a different grid.
2. **Persistence.** If A − S excludes 0 at every primary budget, the asymmetric effect persists under the
   decoupled learner over the tested budgets. If it excludes 0 at 300k but not at 3.6M, the effect is
   finite-budget and fades. Either outcome is reported; no infinite-horizon claim is made.
3. **Symmetric scaling.** If S − B includes 0 at every primary budget, the scale artifact stays absent with
   budget. If S − B excludes 0 at any budget, that budget is reported as a residual scale effect under
   decoupling.
4. **Comparison with E3** is descriptive only, because E3 uses reference 0 and the λ7 / curvature arms. The
   paper's horizon claim will rest on E5 for asymmetry and on E3 for default-learner budget dependence.

## Stopping rule
Run all 270 declared trajectories. Rerun failures only with the identical index and configuration. Add or drop
no arm, cell, seed or analysis after inspecting outcomes. Log deviations in `DEVIATIONS.md`.
