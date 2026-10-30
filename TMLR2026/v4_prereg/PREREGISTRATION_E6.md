# E6 pre-declaration: clipping isolation with parameter-update logging

Written 2026-10-30, before any E6 outcome existed. It answers two points of the external reviews:
- the E1 separate-optimizer arm removes clipping as well;
- gradient norms are not the parameter updates Adam applies.

Hashes are appended to `HASHES.sha256` before launch. Platform, environment and estimator are as in
`PREREGISTRATION_v4.md`.

## Design
- **Arms:** B, A, S (as in E1; z₀ = −2.6281398333).
- **Grid:** the eight E1 cells {0, .2, .4, .5, .6, .7, .85, 1}.
- **Seeds:** 42–51.
- **Budget:** 300k. **Rewards:** raw.
- **Optimizer modes:**
  - `global_clip`: one Adam step, a single global clip of 0.5 (the SB3 default).
  - `groupwise_clip`: one Adam step, actor and critic gradients clipped to 0.5 separately. This is the new cell;
    the class `GroupwiseClipPPO` already exists in the frozen source.
  - `separate_no_clip`: disjoint actor/critic Adam, no clipping.
- **Size:** 3 × 3 × 8 × 10 = **720 trajectories**.
- **Runner:** `e6_clip_isolation.py`. It adds, per optimizer step and parameter group, the applied (post-clip)
  gradient norm and the norm of the actual parameter change. No existing source file is modified.
- **Why no shared-Adam / no-clip arm:** Adam is per-parameter, so one Adam over actor and critic and two separate
  Adams over the same parameters, with the same settings and step counts, perform identical updates. Removing
  clipping is the only difference between `global_clip` and `separate_no_clip`.

## Pre-launch gate (must pass before the main array starts)
1. E6 `global_clip`, A, w = .6, seed 42 (index 120) reproduces E1 run 120 (global_clip, raw, A, .6, 42)
   bit-for-bit: evaluation summary and all ten episodes.
2. E6 `separate_no_clip`, A, .6, 42 (index 600) reproduces E1 run 760 bit-for-bit.
3. A short `groupwise_clip` smoke (index 360) completes and writes `param_updates.jsonl.gz`.

If any check fails, the main array does not start; the failure and any fix are logged in `DEVIATIONS.md`.

## Analysis (fixed now)
- **Estimator.** Firth w50 per (mode, arm); 2,000 paired seed-block draws, seed 20261015; isotonic sensitivity.
- **Contrasts within each mode:** S − B, A − B and A − S.
- **Interactions relative to `global_clip`:** Δ(S − B), Δ(A − S).
- **Update diagnostics, per (mode, arm):** mean actor and critic parameter-update norms; the actor update ratio
  S/B and A/B; the actor update-to-applied-gradient ratio over training.

## Predictions and interpretation (declared now)
1. **Shared-clip throttling.** Under `global_clip`, the actor's parameter-update norm is smaller for S than for B
   (ratio < 1, seed-paired interval excluding 1). Under `groupwise_clip` and `separate_no_clip` the ratio is not
   detectably different from 1.
2. **Clipping isolation.** If S − B under `groupwise_clip` is not detectably different from zero, and Δ(S − B)
   relative to `global_clip` is negative with an interval excluding zero, the symmetric-scaling artifact is
   attributed to the shared clip: the critic taking the actor's clip budget. If S − B stays positive under
   `groupwise_clip`, a channel other than the shared clip remains (for example critic target magnitude), as the
   E1 separate-optimizer residual (+0.160) suggests.
3. **Asymmetry.** A − S under `groupwise_clip` is reported against the E1 rule: positive with an interval
   excluding zero supports the asymmetric effect without the shared-clip channel.
4. **Reproduction.** Labels in the `global_clip` and `separate_no_clip` modes are expected to equal E1's
   (same seeds and platform; logging is observational). Any mismatch is reported.

## Stopping rule
Run all 720 declared trajectories. Rerun failures only with the identical index and configuration. Add or drop
no arm, cell, seed or analysis after inspecting outcomes. Log deviations in `DEVIATIONS.md`.

## Launch timing
E6 starts after E5 (Slurm job 3227) finishes, so E5 keeps the cluster's eight slots. The gate runs first; the
main array depends on the gate succeeding.
