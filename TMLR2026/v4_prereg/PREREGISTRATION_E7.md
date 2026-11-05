# E7 pre-declaration: critic-target-scale isolation and a moment-matched reward control

Written 2026-11-05, before any E7 outcome existed, in response to the 5 November external review. Approved by
the authors and frozen the same day: its hash and the E7 source hashes are in `HASHES.sha256`, appended before the
gate or any main-array run.

Platform, environment, estimator and bootstrap are as in `PREREGISTRATION_v4.md`.

## Questions
E6 attributed part of the symmetric-scaling artifact to the shared gradient clip. The rest survives without
clipping: under separate optimizers and raw rewards, E1 gave S − B = +0.160 [+0.078, +0.254]. It coincides with a
poorly fitted critic (explained variance 0.68 for B, 0.30 for S) and disappears under reward normalization.

Reward normalization changes more than the critic's targets: it also rescales the rewards the actor's advantages
are built from. E7 asks two questions.
1. **Critic-target scale.** Does normalizing only the critic's targets remove the residual symmetric effect, with
   the actor's rewards left raw?
2. **Reward moments.** Does the asymmetric contrast survive against a symmetric control with the same training-reward
   mean and SD as A? In E1, A's transformed rewards have a negative mean while S's are positive, so A − S changes
   the mean as well as the gain-to-loss ratio.

## Design
- **Learner.** Separate actor and critic Adam optimizers, no clipping (`SeparateNoClipPPO`), so the shared-clip
  pathway is absent throughout. Raw transformed rewards. 300k transitions. Everything else as in E1.
- **Critic modes.**
  - `plain`: the critic regresses raw returns. This is identical to E1 `separate_no_clip/raw`.
  - `popart`: the critic's output layer is reparametrized as V(x) = σ·head(x) + μ (van Hasselt et al., 2016).
    - μ and σ are running moments of the rollout returns, updated before each PPO update with β = 0.1; the first
      rollout initializes them directly, and σ is floored at 0.01.
    - The head is rescaled so that its outputs are preserved exactly. The runner asserts this on every update,
      allowing a drift of at most 10⁻³ relative.
    - Rewards, advantages, the actor loss and the actor optimizer are untouched.
    - Because Adam's step size does not scale with the loss, the reparametrization, not a loss rescaling, is what
      changes the critic's ability to track large targets.
- **Arms.**
  - B, A and S as in E1 (z₀ = −2.6281398333).
  - **M, new:** r ↦ a_w·r_B + b_w, an increasing affine map of B's curvature-only reward. It is symmetric in gains
    and losses.
    - a_w and b_w are frozen per mixture weight in `e7_design.json`.
    - They are chosen so that M has A's seed-pooled training-reward mean and SD as measured in E1
      `separate_no_clip/raw`: a_w = sd_A/sd_B, b_w = mean_A − a_w·mean_B.
    - The constants range over a_w = 3.39–4.33 and b_w = −38.0 to −12.8.
- **Grid and seeds.** The eight E1 cells {0, .2, .4, .5, .6, .7, .85, 1}; seeds 42–51.
- **Runner.** `e7_critic_scale.py`, with constants in `e7_design.json`. No existing source file is modified.
- **Size.** The full design is 2 × 4 × 8 × 10 = 640 cells (indices 0–639).
  - The 240 `plain` × {B, A, S} cells (indices 0–239) are, by construction, E1 runs. They are **reused from E1**,
    not rerun, subject to gate 1.
  - **The main array is the 400 new trajectories, indices 240–639:** `plain`/M plus all four `popart` arms.
  - Expected cost is about 57 core-hours, roughly 7 h at eight concurrent tasks.

## Pre-launch gate (must pass before the main array starts)
1. E7 index 120 (`plain`, A, w = .6, seed 42) reproduces E1 run 760 bit-for-bit: evaluation summary and all ten
   episodes. This licenses reusing E1's 240 `plain` × {B, A, S} runs.
2. A `popart` S run (index 520) at 300k completes, writes `diagnostics/popart.json`, and never triggers the
   output-preservation assertion.
3. An M run (index 280) at 300k completes, and its recorded reward moments are written to `COMPLETE.json`.

If any check fails, the main array does not start; the failure and any fix are logged in `DEVIATIONS.md`.

**Smoke tests already run (2,560 transitions, Slurm jobs 4354 and 4358; not part of E7).**
- PopArt S: completed; σ ≈ 158, μ ≈ −87; largest output drift on rescale 2×10⁻⁵.
- M: the first version crashed when saving its checkpoint. Its reward wrapper, a `VecEnvWrapper` subclass defined in
  `__main__`, cannot be cloudpickled. It was replaced by a closure around the vector env's `step_wait`. The rerun
  completed and recorded 2,560 transformed rewards, which confirms the affine map is applied.
- No outcome of either run has been analysed or kept.

## Analysis (fixed now)
- **Estimator.** Firth w₅₀ per (critic mode, arm); 2,000 paired seed-block draws, seed 20261015 (the same draws
  as E1, so the reused cells stay paired); isotonic sensitivity.
- **Contrasts within each critic mode:** S − B, A − S, A − M, M − B, A − B.
- **Interactions, popart minus plain:** Δ(S − B), Δ(A − S), Δ(A − M).
- **Diagnostics per (mode, arm):**
  - critic explained variance at the last update;
  - per-minibatch critic and actor gradient norms;
  - for `popart`, the final σ;
  - for M, the achieved training-reward mean and SD against A's, from the same E7 cells.

## Predictions and interpretation (declared now)
1. **Critic-target pathway (primary).** Under `popart`, S − B has an interval that includes zero, and Δ(S − B) is
   negative with an interval excluding zero.
   - **If both hold:** the residual symmetric effect is attributed to critic target scale.
   - **If S − B stays positive with an interval excluding zero under `popart`:** target scale, as addressed by
     output normalization, is not the channel. We will say so and keep "consistent with" out of the text.
   - **If S − B under `popart` includes zero but Δ(S − B) does not exclude zero:** the result is inconclusive and
     reported as such.
2. **Critic fit.** Under `popart`, S's explained variance is within 0.1 of B's. This is descriptive; a failure does
   not override prediction 1.
3. **Asymmetry without the critic pathway.** Under `popart`, A − S is positive with an interval excluding zero.
4. **Moment control.** A − M is positive with an interval excluding zero under both critic modes.
   - **If it holds:** the asymmetric effect is not explained by the training reward's mean and SD.
   - **If A − M includes zero under either mode:** the A − S result is reported as possibly a moment effect, and
     the abstract is changed accordingly.
5. **Matching check.** M's achieved training-reward mean and SD are within 25% of A's, pooled over the eight cells.
   Matching is approximate because the moments depend on the learned policy. If it fails, prediction 4 is reported
   with that qualification.
6. **Reproduction.** Gate 1 passing implies the reused `plain` B/A/S labels equal E1's. Any mismatch is reported and
   E1's runs are not reused.

**Power note.** The plain-mode residual (+0.160) is about twice its interval half-width. With ten seeds, a partial
reduction under `popart` may leave Δ(S − B) undetectable. The inconclusive branch of prediction 1 exists for this
case.

## Stopping rule
- Run all 400 declared trajectories.
- Rerun failures only with the identical index and configuration.
- Add or drop no arm, cell, seed or analysis after inspecting outcomes.
- Log every deviation in `DEVIATIONS.md`.
