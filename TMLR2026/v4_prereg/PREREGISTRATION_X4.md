# X4 pre-declaration: does the critic carry the symmetric-scaling shift, and does its fit?

**Status: approved by the authors on 2026-11-18 (design) and 2026-11-19 (part M, 40 seeds) and frozen on 2026-11-18T19:08:43Z, before any X4 run.** Hashes of this file,
`experiments/x4_runner.py`, `experiments/x4_design.json` and the shared sources are in `HASHES.sha256`. The arrays start
only after the gate below passes (`afterok`).

## Why
X1–X3 rule out the shared gradient clip as necessary for the shift S − B. The account left is the critic. Our evidence
for it is indirect:
- the critic's explained variance is lower under S in every mode;
- normalizing rewards leaves no detectable shift;
- PopArt reduces the shift on pooled cohorts only.

**What follows from the learner's definition.** In SB3 PPO with no gradient clipping, a symmetric reward scale s can
change training only through the critic's regression targets, apart from Adam's ε. That is because of:
- `ent_coef = 0`;
- an MSE value loss with no `clip_range_vf`;
- advantages normalized per minibatch;
- no trainable parameter shared by actor and critic;
- Adam's per-parameter invariance to a constant gradient scale.

Two questions follow:
- **Channel.** Does removing the critic's target scale remove the shift? (part M)
- **H-fit.** Is it the critic's *fit quality* that carries it? (parts D-plain and D-pop)

The alternative to H-fit, **H-other**, is that some property of the scaled critic other than its fit as measured carries
the shift.

## Design: fresh seeds 122–161 (40), arms B and S, the E1 grid, 300k, raw rewards, separate Adam, no gradient clipping
| Part | Critic | Runs | Purpose |
|---|---|---|---|
| D-plain | E7's plain critic (= E1 `separate_no_clip`/raw) | 640 | replication; advantage quality under the shift |
| D-pop | E7's PopArt critic | 640 | **H-fit test:** fit equalized, does the shift persist? |
| M | plain critic with its output fixed-scaled: V = c·head(x), c = 1 (B) or 7 (S) | 640 | **channel test:** S trains as B does apart from Adam's ε |
| **Total** | | **1,920** | |

**Logging (all parts).** Logging reads tensors only; its random draws come from its own generators, and the gate checks
bit-identity. At every rollout a separate **reference critic** is used for measurement only:
- **Architecture:** 64-64 tanh.
- **Targets:** discounted Monte-Carlo returns, standardized per rollout. The supply chain terminates every 30 steps, so
  returns are exact for every transition whose episode ends inside the rollout.
- **Out of sample:** at rollout k, the reference was trained on rollouts 0..k−1 only.

All metrics are scale-free:
- **critic fit:** the explained variance of the learner's values against Monte-Carlo returns (`ev_train_mc`);
- **value error:** RMSE against those returns, divided by their SD;
- **GAE error:** RMS difference between the standardized learner advantages and the standardized reference advantages.
  The reference advantages use the same GAE with the learner's rewards, γ and λ, with values replaced by the
  reference's;
- **rank agreement:** Spearman correlation of the two;
- **sign disagreement:** the fraction of transitions where they have opposite signs after centering.

Per run, the metrics are averaged over rollouts 1 to the end. Each quarter of the run is also kept, for exploratory use
only.

## Predictions
Inference: jackknife-t over 40 seeds (39 df). The seed-block percentile bootstrap (2,000 draws, seed 20261122) is
secondary. Metric contrasts are seed-paired: per seed, the mean over the eight cells of log(S/B) for ratios, or of S − B
for differences.
1. **Replication (D-plain):** S − B > 0, with the 95% interval excluding zero. If this fails, it is reported first, and
   X4 cannot test the critic account.
2. **Advantage degradation (D-plain):** value error, GAE error and sign disagreement each have S/B > 1, and rank agreement
   has S − B < 0, each by its 95% interval. P2 is met when all four are.
3. **PopArt equalizes advantage quality (D-pop):** each of the four S/B ratios lies within 1 ± 0.10 by its 90% interval.
4. **Manipulation check (M):** the critic-fit gap S − B (`ev_train_mc`) lies within ±0.05 by its 90% interval. If this
   fails, M is uninformative.
5. **Channel test (M), conditional on 4:** the paired change (S−B under M) − (S−B under D-plain) is below zero by its 95%
   interval, **and** S − B under M lies within ±0.10 by its 90% interval.
6. **H-fit test (D-pop):** the fit gap S − B under D-pop lies within ±0.05 by its 90% interval. Given that, the shift
   under D-pop is read by the table below.

## Declared interpretation
**Channel (M), given P4:**

| Result | Reading |
|---|---|
| P5 met | Fixing the critic's output scale removes the shift. The critic's target scale carries it at this budget. |
| S − B under M above zero (95%) | A channel other than the critic's target scale exists. The elimination argument is incomplete in practice (for example, Adam's ε). |
| Otherwise | Inconclusive. |

**H-fit (D-pop), given P6's fit equalization:**

| Result | Reading |
|---|---|
| S − B under D-pop above zero (95%), and the change D-pop − D-plain not shown below zero | **H-fit is falsified as stated:** equal fit, same shift. |
| S − B under D-pop above zero (95%), and the change shown below zero | The fit carries at most part of the shift; a residual persists with fit equalized. |
| S − B under D-pop within ±0.10 (90%) | Consistent with H-fit. PopArt changes more than the fit, so this is not proof. |
| Otherwise | Inconclusive. |

If PopArt does not equalize the fit, there is no H-fit test. Non-significance is never read as equivalence.
Associations between per-seed degradation and crossings are exploratory.

## Gate (before any array)
1. D-plain with logging (seed 42, B, w = 0.6) reproduces E1 run 680 bit-for-bit: the evaluation file and every minibatch
   record.
2. M arm B (c = 1), on the same seed, reproduces that D-plain run bit-for-bit, including the X4 log.
3. D-pop (seed 42, B, w = 0.6) reproduces E7 run 360 bit-for-bit.
4. Indices 0, 640 and 1280 complete with 469 X4 records each.

## Compute (measured in the smoke)
- **Per run:** 8.5–9.0 min. That is the same as E7, so logging adds nothing measurable.
- **Total:** 1,920 runs, or about **275 core-hours**.
- **Wall-clock:** about 6 h at 48 concurrent tasks (the raised limit lapses around 19 Nov), about 1.4 days at 8. The
  user's own jobs take priority.

## Stopping rule
- Run all declared trajectories once. Rerun failures only with the identical index.
- Add or drop no arm, part, cell, seed or analysis after outcomes are seen.
- Seeds 122–161 are used once.

## History before freezing (smoke and calibration, no X4 seed at 300k)
- **The draft's part F** gave the critic extra fitting on each rollout's own value targets. It was dropped after
  calibration (seed 43, w = 0.6, 300k).
  - The drafted stopping rule (stop when EV improves by less than 0.005 per epoch) stopped after one epoch.
  - Fixed extra epochs lowered B's fit against Monte-Carlo returns (last quarter: 0.78 standard, 0.45 with +10, 0.11
    with +50) and left S's unchanged (0.14–0.15).
  - F would have equalized fit by degrading B. M replaces it.
- **The manipulation check uses `ev_train_mc`, not SB3's explained variance.** SB3's version compares values with
  λ-returns built from those same values, so a noisy critic inflates it.
- **Seeds were raised from 20 to 40** after a power estimate: at 20 seeds the paired change between parts would have a
  standard error of about 0.057.
- **"Falsified" now requires a shift shown above zero, not a non-significant change.**
- **Part-M smoke (job 12062, seed 42, 2,560 transitions):** M arm B is bit-identical to D-plain (updates, X4 log,
  evaluation summary and episodes). M arm S tracks arm B for the first rollouts (critic fit equal to three decimals) and
  then diverges, as Adam's ε allows, so M's prediction is distributional, not bit-identity.
- **Smoke-only seeds:** the 32k timing smoke used seed 122. None of those runs reached the 300k evaluation, and no
  outcome from them is used.
