# X1 and X2 pre-declaration: fresh-seed replication of the clipping intervention, with optimizer-state logging

Written 2026-11-18, before any X1 or X2 run. **Status:** approved by the authors on 2026-11-18 ("Run X1 + X2") and
frozen before the gate. The SHA-256 of this file, `x1_x2_runner.py`, `x1_x2_design.json` and
`analysis/phase3_x1x2.py` is appended to `HASHES.sha256` before any main-array run.

**Why.** A calibration study added in revision (`analysis/calibration_v5.py`) simulated 27 scenarios with 10,000
replicates each. In it, the pre-declared seed-block percentile intervals under-cover at ten seeds: 0.86–0.92 at
nominal 95%. A seed-level jackknife-t interval was near nominal in every scenario. With jackknife-t
(`analysis/jackknife_v5.py`):
- E6's per-group change in S − B still excludes zero at 95%, but not at 99%;
- E11's per-group change no longer excludes zero.

X1 and X2 test both changes again on fresh seeds, at a seed count chosen from the simulated power.

## Common design
- Supply chain, raw transformed rewards, arms B and S (z₀ = −2.6281398333), the E1 grid
  {0, .2, .4, .5, .6, .7, .85, 1}, 300k requested transitions.
- **Fresh seeds 82–101** (20 seeds); none has been used before.
- The collapse label, its thresholds and the evaluation protocol are E1's.
- **Primary inference:** seed-level jackknife-t on the pooled Firth crossing, with t(n − 1) quantiles, at 95%, on the
  fresh seeds alone.
- **Secondary:**
  - jackknife-t at 99%;
  - the seed-block percentile bootstrap (2,000 multinomial draws, seed 20261119);
  - pooling with the earlier seeds 42–51 (E6 for X1, E11 for X2): 30 seeds, draws seed 20261120.

## X1: SB3 PPO (960 runs, indices 0–959)
**Modes:** E6's three modes, run with E6's classes:
- one Adam step with a global clip of 0.5;
- actor and critic each clipped to 0.5;
- disjoint Adam optimizers without clipping.

**Logging.** In addition to E6's per-step applied-gradient and parameter-update norms, every optimizer step records
three norms for the actor's parameters:
- Adam's bias-corrected first moment, m̂;
- √v̂, from Adam's bias-corrected second moment v̂;
- the step direction m̂ / (√v̂ + ε).

The per-minibatch shared-clip factor κ = min(1, 0.5/‖g‖) comes from the logged total norm. Logging only reads tensors
and optimizer state.

**Predictions** (P1–P3 use the primary analysis):
1. With the global clip, S − B > 0: the confound replicates.
2. The change in S − B, per-group minus global clip, is negative. **This is the primary test.**
3. Without clipping, S − B > 0: the residual not explained by clipping replicates.
4. **Mechanism, on the optimizer state.** Take the seed-paired S/B ratio of the actor's Adam step direction (the
   geometric mean over each seed's cells, with a jackknife-t interval over seeds). Under the global clip it is below 1,
   with its 95% interval excluding 1. Under the per-group clip its interval lies entirely above the global-clip
   interval.

## X2: CleanRL-style PPO (640 runs, indices 960–1599)
**Design:** E11's port, run unchanged through `e11_cleanrl.train`, with two conditions:
- raw rewards, one global clip;
- raw rewards, actor and critic clipped separately.

**Predictions:**
5. Under the raw condition, S − B > 0.
6. The change in S − B, per-group minus raw, is negative. **This is the primary test.**

## Interpretation (declared now)
**P2 met.** The per-group intervention is confirmed on fresh seeds at the calibrated level; the paper states the SB3
clipping result at that level.

**P2 not met, P1 met.** The clipping pathway's share of the shift is not confirmed. The paper then says the per-group
reduction is supported only in the original cohort.

**P6.** If P6 is not met, the second implementation reproduces the shift (P5) but not, at this power, the per-group
reduction. Simulated power for a 0.1 change is about 0.65 on fresh seeds alone and about 0.8 pooled.

**P4.** P4 is descriptive support for the Adam pathway. It cannot establish causation, which comes from the per-group
intervention.

## Pre-launch gate (all checks must pass before either main array starts)
1. **X1 bit-identity.** With seed 42, B, w = 0.6, the runner reproduces E6 runs 40, 280 and 520 bit-for-bit, one per
   mode, on the evaluation file. This shows the added logging does not change learning.
2. **X2 bit-identity.** The runner reproduces E11 runs 40 (raw) and 520 (per-group) bit-for-bit.
3. **Completion.** Main-array indices 0 and 960 complete.

If any check fails, no array starts, and the failure is logged in `DEVIATIONS.md`.

## Stopping rule
- Run all 1,600 declared trajectories.
- Rerun failures only with the identical index.
- Add or drop no arm, cell, seed or analysis after outcomes are seen.

## Smoke
Slurm job 9149: 2,560 transitions, every mode, plus the logging-identity check at that budget. The smoke runs are not
part of X1 or X2.

**Result:** passed. All five modes complete. In each X1 mode, evaluation and per-step update norms are bit-identical to
the E6 runner at that budget. The Adam fields are present. No outcome label was analysed.

Frozen 2026-11-18T02:36:31Z.
