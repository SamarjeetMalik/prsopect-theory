# E8 pre-declaration: online moment-matched symmetric control, and an independent-seed replication of E7 P1

Written 2026-11-05, after E7's outcomes were known and before any E8 run. **Approved by the authors and frozen on
2026-11-05T22:48:51Z**, together with the E8 sources and `e8_targets.json`. Their hashes were appended to `HASHES.sha256` before the
gate or any main-array run.

`e8_targets.json` was built on 2026-11-06 by `analysis/e8_build_targets.py` from the per-rollout logs of E1 and E7 A
(cluster job 4930), with SHA-256 prefix `514270f3031287f6`. Building it read no outcome label.

**Smoke test (2,560 transitions, Slurm job 4944; not part of E8).**
- It exercised the static (index 40), online popart (index 120) and part-B (index 160) paths; all three completed.
- Online tracking: the first adapted rollout's mean and SD were −27.7 and 63.9, against targets of −28.9 and 64.3.
- Largest PopArt output drift: 3×10⁻⁵.
- No outcome was analysed or kept.

Platform, environment, learner, estimator and bootstrap are E7's (`PREREGISTRATION_E7.md`).

## Why
E7 left two results that this cohort can address.

1. **The moment control under the raw-target critic.**
   - E7's control M is a static affine map of B's reward. Its constants are frozen to give A's E1 training-reward
     mean and SD under **E1's** policies.
   - M's own policies differ: at mid-grid they mostly collapse, unlike the B policies whose moments set M's
    constants, and B's reward is lower under a collapsed policy. So M's achieved mean was more negative than A's
     (−13.5 vs −6.6 with the plain critic), which failed E7's 25% matching check (P5).
   - With the plain critic, A − M was +0.050 [0.000, +0.100]. Whether that near-null result reflects the reward's
     moments, or M's more negative mean, is not identified.
2. **The critic-target attribution.**
   - E7 P1 was inconclusive: under PopArt, S − B was +0.075 [−0.032, +0.164], and the change was −0.085
     [−0.183, +0.006]. The power note anticipated this outcome.
   - These E7 results are now **known**. Adding seeds to the same cohort and re-testing would be optional stopping.
   - Part B is therefore a fresh cohort with its own primary analysis. Its pooled estimate is secondary.

## Part A: online moment-matched control M_on (160 trajectories)
**Arm.** M_on applies r ↦ a_r·r_B + b_r to B's curvature-only reward. This is an increasing affine map, so it
treats gains and losses identically.
- Before each rollout r ≥ 1, the constants are set from the moving averages μ̂_B and σ̂_B of B's per-rollout reward
  mean and SD under M_on's current policy:
  - a_r = clip(σ*_r / max(σ̂_B, 10⁻³), a_w/4, 4a_w);
  - b_r = μ*_r − a_r·μ̂_B;
  - the moving averages use weight 0.5;
  - a_w is E7's static scale for the cell.
- Rollout 0 uses E7's static constants.

**Targets.**
- μ*_r and σ*_r are the seed-averaged mean, and the root-mean-square over seeds of the within-rollout SD, of A's
  transformed training reward at rollout r.
- They come from the same critic mode: plain from E1 `separate_no_clip/raw` A, popart from E7 `popart` A.
- They are built by `analysis/e8_build_targets.py` from the per-rollout logs and frozen in `e8_targets.json` before
  any E8 run. The construction reads only A's training-reward logs, never an outcome label.

**What differs between A and M_on, and what does not.**
- An affine map cannot change the shape of B's reward distribution.
- M_on is constructed to share A's first two training-reward moments along training, under its own policy.
- What A − M_on still varies:
  - the distribution's shape: A is sign-asymmetric, with a heavier, longer loss tail;
  - the per-transition mapping from outcome to reward, which is the asymmetric feature the contrast is meant to probe.
- Quantiles, the fraction negative, and critic-target (rollout-buffer return) moments are reported descriptively;
  they are not matched.

**Confound introduced.**
- M_on's reward map changes between rollouts, whereas A's is fixed.
- Under the plain critic, the changing shift b_r moves the value targets over time.
- Both are reported. A − M_on is read as a contrast against a moment-matched, nonstationary symmetric control, not as a
  pure manipulation of sign asymmetry.

**Grid and seeds.** E1 grid {0, .2, .4, .5, .6, .7, .85, 1}; seeds 42–51; both critic modes. Indices 0–159.

## Part B: independent-seed replication of the critic-target test (320 trajectories)
B and S under both critic modes with fresh seeds 52–61, on the same grid and learner. Indices 160–479.

## Pre-launch gate (must pass before the main array starts)
1. E8 index 40 run with `--static` (plain critic, w = .6, seed 42, E7's constants) reproduces E7 run 280 bit-for-bit:
   evaluation summary and all ten episodes. This shows that the E8 code path equals E7's when the map is static.
2. An online part-A run (index 120: popart, w = .6, seed 42) completes at 300k and logs 469 rollout records.
3. A part-B run (index 160) completes at 300k.

If any check fails, the main array does not start; the failure and any fix go into `DEVIATIONS.md`.

## Analysis (fixed now)
- **Estimator.** Firth w₅₀; seed-block bootstrap with 2,000 draws, seed 20261015, so part-A contrasts against the E1
  and E7 arms (seeds 42–51) stay paired. Isotonic crossings are a sensitivity check.
- **Intervals.**
  - Primary: 95% percentile intervals, as for E1–E7.
  - Declared sensitivity: 99% percentile intervals. A coverage simulation of this design (`stat_checks.py`, run before
    E8) found the nominal-95% seed-block intervals cover at 0.88–0.92, and the 99% intervals at ≥0.94.
- **Part A contrasts, within each critic mode:**
  - A − M_on, M_on − B, A − S (from E7);
  - M_on − M (E7's static control).
- **Part A interaction:** Δ(A − M_on), popart minus plain.
- **Part B, primary (seeds 52–61 alone):**
  - under popart: S − B, and the change in S − B from plain to popart;
  - under plain: S − B, as a check that the residual symmetric shift replicates.
- **Part B, secondary:** the same quantities pooled over seeds 42–61, using 2,000 multinomial seed-block draws over
  20 seeds (seed 20261107).
- **Matching diagnostics:**
  - M_on's achieved training-reward mean and SD versus A's, pooled over the eight cells exactly as in E7 P5;
  - per-cell values;
  - per-rollout tracking error;
  - quantiles p10/p50/p90 and the fraction negative;
  - critic-target moments.

## Predictions and interpretation (declared now)
1. **Matching.** Pooled over the eight cells, M_on's achieved mean and SD are each within 10% of A's (relative to
   |A|), under both critic modes. Per-cell values within 25% are descriptive.
   - **If not met:** parts 2–3 are reported with that qualification, and M_on is not called moment-matched.
2. **A − M_on under popart** is positive, with an interval excluding zero.
3. **A − M_on under plain.** No directional prediction.
   - **Positive with an interval excluding zero:** the plain-critic asymmetric contrast is not explained by the
     training reward's first two moments, and the E7 plain reading (possibly a moment effect) is revised.
   - **Interval includes zero:** a moment effect under the raw-target critic remains possible, and the abstract's
     qualification stays.
4. **M_on − M** is reported in each mode; it shows how much E7's static control was moved by its mean mismatch. No
   directional prediction.
5. **Part B, primary.** Under popart, S − B has an interval including zero, and the plain-to-popart change in S − B is
   negative with an interval excluding zero.
   - **Both hold:** the critic-target-scale attribution is supported in an independent cohort.
   - **Popart S − B positive, excluding zero:** output normalization of the targets is not the channel.
   - **Otherwise:** inconclusive, reported as such.
   - The pooled 20-seed result is reported alongside, labelled secondary. It cannot overturn the primary verdict.

## Size and cost
480 trajectories (160 + 320) at about 8.4 minutes each: about 67 core-hours, or about 8.5 hours at eight concurrent tasks.

## Stopping rule
- Run all 480 declared trajectories.
- Rerun failures only with the identical index and configuration.
- Add or drop no arm, cell, seed or analysis after inspecting outcomes.
- Log every deviation in `DEVIATIONS.md`.
