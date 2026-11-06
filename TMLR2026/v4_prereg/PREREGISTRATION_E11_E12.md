# E11 and E12 pre-declaration: a second PPO implementation; local observations with agent identity

Written 2026-11-06, before any E11 or E12 run. **Status: approved by the authors (the request covered "all" remaining
experiments) and frozen before the gate:** hashes of this file, `e11_cleanrl.py`, `e11_design.json`,
`e12_localobs.py` and `e12_design.json` are appended to `HASHES.sha256` before any run.

Environment, reward arms (B, A, S with z₀ = −2.6281398333), mixture grid, seeds 42–51, 300k budget, the collapse
label, the estimator and the bootstrap (2,000 seed-block draws, seed 20261015) are E1's.

Frozen 2026-11-06T12:01:25Z.

## E11: a second PPO implementation (720 trajectories, indices 0–719)
**Question.** Is the shared-clip confound specific to Stable-Baselines3?

**Implementation.** `e11_cleanrl.py` is a single-file port of CleanRL's `ppo_continuous_action.py`. It keeps CleanRL's
defaults:
- networks and initialization;
- clipped value loss;
- learning-rate annealing;
- observation normalization;
- reward normalization;
- one global clip of 0.5.

Rollout and minibatch sizes are matched to E1's budget arithmetic (128 steps × 5 lanes; 5 minibatches of 128).
Evaluation is E1's `horizon_supply.evaluate`, through a `predict` that freezes the observation normalization.

**Conditions.**
- `cleanrl_raw`: CleanRL defaults without reward normalization.
- `cleanrl_default`: CleanRL defaults, with reward normalization.
- `cleanrl_group`: as `cleanrl_raw`, but with actor and critic clipped separately.

Each condition runs arms B, A and S.

**Predictions.**
1. **`cleanrl_raw`:** S − B > 0, with a 95% interval excluding zero. The confound reproduces outside SB3.
2. **`cleanrl_group` vs `cleanrl_raw`:** the change in S − B is negative, with an interval excluding zero.
3. **`cleanrl_default`:** S − B has an interval including zero.
4. **`cleanrl_default`:** A − S > 0, with an interval excluding zero.

**Diagnostics.** S/B applied-actor-gradient ratio and critic share per condition.

**Interpretation.**
- If prediction 1 fails, the confound is not shown outside SB3, and the paper says so.
- If 1 holds but 2 fails, clipping is not the pathway in this implementation.

## E12: local observations with agent identity (480 trajectories, indices 0–479)
**Question.** Do the findings hold when the shared policy is not a homogeneous controller?

**Observation.** Each node observes only:
- its own retail demand;
- its own inventory;
- its own incoming order pipeline (zero-padded);

plus a one-hot node identifier. `e12_localobs.py` applies this view in training and in evaluation. Evaluation is
otherwise `horizon_supply.evaluate`, unchanged.

**Learners.**
- `default`: E1's default learner (one Adam, global clip, raw rewards).
- `decoupled`: separate optimizers without clipping, plus `VecNormalize` rewards.

Each learner runs arms B, A and S.

**Predictions.**
5. **`default`:** S − B > 0, with an interval excluding zero.
6. **`decoupled`:** S − B has an interval including zero.
7. **`decoupled`:** A − S > 0, with an interval excluding zero.

**Descriptive only.** Crossings compared with E1's global-observation crossings, for the same learner and arm. No
test.

## Statistics
- Primary: 95% seed-block percentile intervals.
- Declared sensitivity: 99% intervals.
- All contrasts are paired over seeds 42–51.

## Pre-launch gate (must pass before either main array starts)
1. E11 index 40 run twice at 300k gives bit-identical evaluations (determinism).
2. E11 indices 440 (`cleanrl_default`) and 520 (`cleanrl_group`) complete at 300k.
3. E12 indices 40 (`default`) and 360 (`decoupled`) complete at 300k.

If any check fails, no main array starts, and the failure is logged in `DEVIATIONS.md`.

## Smoke test
Slurm job 5531, 2,560 transitions; not part of E11 or E12.
- All six smoke runs completed.
- E11 is deterministic.
- At this budget the critic's share of the squared joint gradient norm was:
  - 0.98 under `cleanrl_raw`, where the actor's applied gradient was 0.055 against 0.76 pre-clip;
  - 0.08 under `cleanrl_default`.
- Under `cleanrl_group`, the actor's applied gradient was 0.50.
- No outcome label was analysed.

## Stopping rule
- Run all 1,200 declared trajectories.
- Rerun failures only with the identical index.
- Add or drop no arm, cell, seed or analysis after inspecting outcomes.
