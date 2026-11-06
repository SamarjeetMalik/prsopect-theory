# E9 and E10 pre-declaration: clip-threshold dose-response; critic-target replication with more seeds

Written 2026-11-06, before any E9 or E10 run. **Status: approved by the authors (the request covered "all" remaining
experiments) and frozen before the gate:** hashes of this file, `e9_e10_runner.py` and `e9_e10_design.json` are
appended to `HASHES.sha256` before any run.

Platform, environment, label, estimator and bootstrap are E1's (`PREREGISTRATION_v4.md`) unless stated.

Frozen 2026-11-06T11:56:42Z.

**Smoke test** (2,560 transitions, Slurm job 5527; not part of E9/E10): it exercised the E9 τ = 30000 path (index 630),
the E10 PopArt path (index 960) and the gate override path (τ = 0.5). All three completed with the declared settings.
No outcome was analysed.

## E9: clip-threshold dose-response (640 trajectories, indices 0–639)
**Question.** Does the symmetric shift fall as the shared clip threshold τ rises?

Section 4 of the paper predicts that under a shared clip the actor's applied gradient is κ‖g_a‖ with κ = min(1, τ/‖g‖),
and the critic's gradient dominates ‖g‖: about 1,300 (B) and 12,000 (S) per minibatch in E1. So raising τ past B's
critic norm, and then past S's, should raise the S/B applied-gradient ratio from 0.12 towards 1. The crossing shift
should follow.

**Design.**
- E1's default learner: one Adam, one global clip, raw rewards, 300k.
- `max_grad_norm` = τ ∈ {300, 3000, 10000, 30000}.
- Arms B and S on the E1 grid, seeds 42–51.
- The end points are E1 runs, used as references, not re-run:
  - τ = 0.5: E1 `global_clip/raw` B, S;
  - τ = ∞: E1 `separate_no_clip/raw` B, S.

    Splitting one Adam into two changes nothing by itself, because Adam is per-parameter.

**Analysis (fixed now).**
- Firth w₅₀ per (τ, arm), with E1's 2,000 seed-block draws (seed 20261015), so every contrast is paired with E1.
- Reported for each τ:
  - S − B;
  - the measured S/B ratio of applied actor gradients;
  - the critic's share of the squared joint norm (from `updates.jsonl.gz`).

**Predictions.**
1. **Mechanism check.** The measured S/B applied-gradient ratio rises monotonically in τ across {0.5, 300, 3000,
   10000, 30000} (point estimates), and is above 0.5 at τ = 30000.
2. **Dose-response.** S − B at τ = 30000 is below S − B at τ = 0.5. The paired difference is negative, with a 95%
   interval excluding zero.
3. **End point.** S − B at τ = 30000 does not differ detectably from the no-clip value: the paired difference
   includes zero.
4. **Shape, descriptive.** S − B against log(measured S/B ratio) across the six τ levels is reported. No
   functional form is tested.

**Interpretation.**
- (2) and (3) together support the clip-threshold pathway as a dose-response.
- If (2) fails while (1) holds, the clip factor moves but the crossing does not, and the shared-clip account of the
  E6 intervention is weakened. We will say so.

## E10: critic-target replication with more seeds (640 trajectories, indices 640–1279)
**Question.** Does the PopArt reduction of the residual symmetric shift hold at the coverage-calibrated level in a
larger independent cohort? In E8 it held at the declared 95% level, but not at the calibrated 99% level in ten
fresh seeds.

**Design.** E7's learner (separate optimizers without clipping, raw rewards), plain or PopArt critic. Arms B and S,
E1 grid, **fresh seeds 62–81** (20 seeds).

**Analysis (fixed now).**
- **Primary:** seeds 62–81 alone. Firth w₅₀ with 2,000 multinomial draws over the 20 seed blocks (seed 20261108).
  - S − B under each critic;
  - the change in S − B from plain to PopArt;
  - 95% and 99% percentile intervals.
- **Secondary:** pooled seeds 42–81 (E1/E7 for seeds 42–51, E8 part B for 52–61, E10 for 62–81), with 2,000
  multinomial draws over 40 seed blocks (seed 20261109).

**Predictions.**
5. **Primary.** PopArt S − B has a 95% interval including zero, and the change is negative with a **99%** interval
   excluding zero.
   - **If the 99% interval includes zero but the 95% interval excludes it:** report as supported at the declared
     level only, as in E8.
   - **If the PopArt S − B 95% interval excludes zero (positive):** target normalization does not remove the
     residual.
6. **Replication.** Plain S − B is positive with a 95% interval excluding zero. This replicates E1 and E8.

## Pre-launch gate (must pass before the main array starts)
1. The runner with `--tau 0.5` (B, w = 0.6, seed 42) reproduces E1 run 40 bit-for-bit: evaluation summary and all
   ten episodes.
2. The runner with `--critic popart` (B, w = 0.6, seed 42) reproduces E7 run 360 bit-for-bit.
3. Main-array indices 630 (E9, τ = 30000) and 640 (E10) complete.

If any check fails, the main array does not start, and the failure is logged in `DEVIATIONS.md`.

## Stopping rule
- Run all 1,280 declared trajectories.
- Rerun failures only with the identical index.
- Add or drop no arm, cell, seed or analysis after inspecting outcomes.
