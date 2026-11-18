# X3 pre-declaration: which clipped group carries the symmetric-scaling shift?

Written 2026-11-18, after the X1/X2 results and before any X3 run. **Status:** approved by the authors on 2026-11-18
("Reframe and add X3") and frozen before the gate. The SHA-256 of this file, `x3_runner.py`, `x3_design.json` and
`analysis/phase3_x3.py` is appended to `HASHES.sha256` before the main array.

**Why.** On fresh seeds X1 showed that clipping actor and critic separately restores the actor's applied gradient and
Adam step, but leaves the symmetric shift S − B in place: +0.275, against +0.248 under the shared clip. In that mode
the critic is still clipped. Without any clipping, S − B was +0.076 on fresh seeds and +0.103 pooled. The hypothesis
formed from these data is that clipping the *critic* carries the shift and clipping the *actor* does not. X3 tests
this on new seeds.

## Design (1,280 runs, indices 0–1279)
- **Learner:** the supply chain, raw rewards, one Adam step. Two factors are crossed:
  - the actor's gradient is clipped to 0.5, or not;
  - the critic's gradient is clipped to 0.5, or not.
  Each group is clipped separately. This gives four modes: `both` (= E6/X1 groupwise), `actor_only`, `critic_only`,
  `neither`.
- **Arms and seeds:** arms B and S, the E1 grid, **fresh seeds 102–121**, 300k. The label and evaluation are E1's.
- **Logging:** X1's per-step records, plus the bias-corrected Adam quantities for the critic. Logging only reads
  tensors and optimizer state.

## Outcome and effects
The outcome is S − B of the pooled Firth crossing, for each mode. The factorial effects are:
- **critic effect** = ½[(SB_critic_only − SB_neither) + (SB_both − SB_actor_only)];
- **actor effect** = ½[(SB_actor_only − SB_neither) + (SB_both − SB_critic_only)];
- **interaction** = (SB_both − SB_actor_only) − (SB_critic_only − SB_neither).

**Inference:**
- Primary: seed-level jackknife-t, with t on 19 df.
- Secondary: the seed-block percentile bootstrap (2,000 draws, seed 20261121).

## Predictions
1. **Critic effect > 0.** The 95% jackknife-t interval excludes zero. **This is the primary test.**
2. **Actor effect is negligible.** Its 90% jackknife-t interval lies within ±0.10. This is an equivalence test
   (TOST at α = 0.05); a non-significant actor effect alone would not count.
3. **S − B in `both` > 0.** The 95% interval excludes zero; this replicates X1 on new seeds.

The interaction is reported with no prediction. The optimizer-state ratios (applied gradients, updates, Adam step
directions, explained variance) are reported descriptively.

## Interpretation (declared now)
| P1 | P2 | Reading |
|---|---|---|
| met | met | The shift runs through clipping of the critic, not of the actor. The paper's mechanism becomes "a clipped, unnormalized critic"; the actor-starvation account is reported as measured but not causal for the crossing. |
| met | not met | Both groups may contribute. Report both effects with intervals. |
| not met, critic effect estimate near 0 | — | Neither group's clipping is shown to carry the shift. The no-clip shift on fresh seeds, and reward/target normalization, are then the only located factors. |

## Pre-launch gate (all checks must pass before the main array)
1. **Bit-identity.** With seed 82, B, w = 0.6, mode `both` reproduces X1 run 400 (groupwise_clip) bit-for-bit, on the
   evaluation file and the per-step update norms.
2. **Completion.** Indices 0, 320, 640 and 960, one per mode, complete.

If any check fails, no array starts, and the failure is logged in `DEVIATIONS.md`.

## Stopping rule
- Run all 1,280 declared trajectories.
- Rerun failures only with the identical index.
- Add or drop no arm, cell, seed or analysis after outcomes are seen.

## Smoke
Slurm job 10758: 2,560 transitions, not part of X3.
- All four modes complete and clip as declared. Post-clip norms at the last minibatch:
  - both: actor 0.5, critic 0.5;
  - actor_only: actor 0.5, critic 123;
  - critic_only: actor 1.04, critic 0.5;
  - neither: actor 0.99, critic 63.
- `both` with seed 82 is bit-identical to X1 index 400, on the evaluation and on the per-step update norms.
- The critic's Adam fields are logged.
- No outcome label was analysed.

Frozen 2026-11-18T08:20:40Z.
