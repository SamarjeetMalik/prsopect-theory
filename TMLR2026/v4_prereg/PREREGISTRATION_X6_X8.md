# X6–X8 pre-declaration: local critics, the shared clip under decentralized execution, and fit at fixed target scale

**Status: approved by the authors and frozen on 2026-11-20T12:24:37Z, before any X6–X8 run.** Written on 2026-11-20, after X5. It was
frozen after the authors approved it, with hashes of this file, `experiments/x6_x8_runner.py`, `experiments/x6_x8_design.json`,
`analysis/phase3_x6_x8.py` and the shared sources appended to `HASHES.sha256`. Each experiment's array starts only after
its gate passes (`afterok`). X8's critic learning-rate multiplier is set by a calibration on seeds that are not X8
seeds, before the freeze (below).

## Why
An outside review of the X5 revision named four gaps. This document addresses three of them; the fourth, a second
environment, gets its own pilot and pre-declaration.

1. **X5's critic is centralized.** X5 shows that the shift remaining without clipping survives per-node actors on local
   views, but its critic sees the global state. With a parameter-shared actor and critic, both on local views and under
   the shared clip (E12, ten seeds), the shift was not detected, which is inconclusive. **X6** gives every node its own
   local critic.
2. **X5 runs without clipping.** It does not show that the larger shift under SB3's default shared clip (+0.357 in E1)
   transfers to decentralized execution, or how much of that larger shift runs through the critic's targets. **X7**
   trains X5's architecture with one Adam step and the shared clip. It removes the target scale with critics that also
   send gradients of B's size, and compares shared with per-group clipping.
3. **Fit is a candidate mediator, not an isolated one.** In X4 and X5 the interventions that remove the target scale
   also equalize the critic's fit. **X8** degrades the critic's fit for B while leaving its targets at B's scale.

## Design (all three): the E1 grid, 300k transitions, raw rewards, arms on fresh seeds, 40 seeds each

| Exp. | Seeds | Execution | Learner | Parts | Arms | Runs |
|---|---|---|---|---|---|---|
| X6 | 202–241 | per-node actors (X5's) and **per-node local critics** | separate Adam, no clip | plain; fixed-scale head M | B, S | 1,280 |
| X7 | 242–281 | X5's: per-node actors, centralized critic | **one Adam, shared global clip 0.5** (E1's default) | plain; normhead; normpopart; groupclip | B, S | 2,560 |
| X8 | 282–321 | shared controller (X4's) | separate Adam, no clip | plain | B, S, **Bdeg** | 960 |
| **Total** | | | | | | **4,800** |

**X6, local critics.** Each node's critic is a separate 64-64 tanh trunk on the node's local view and one-hot
identifier, with its own output row. Nothing in any critic reads the global state. Part M wraps the output as
V = c·head(x), c = 1 (B) or 7 (S), as X4's part M.

**X7, the shared clip under decentralized execution.** X4's fixed-scale head and E7's PopArt critic both compute the
value loss on raw returns. Under a shared clip their critic gradients therefore still grow with the reward and still
dominate the clip. X7 uses normalized-loss versions:
- `normhead`: V = c·head(x), with the gradient reaching V scaled by 1/c². This is exactly the loss on targets
  divided by c, so the critic regresses B-sized targets and sends B-sized gradients. At c = 1 it is the plain critic.
- `normpopart`: E7's PopArt statistics, with the gradient reaching V scaled by 1/σ² (PopArt's normalized loss).
- `groupclip`: the plain critic, with actor and critic each clipped to 0.5 under the same Adam step (E6/X1's per-group
  clip).

**X8, fit at fixed target scale.** `Bdeg` is arm B with the critic's Adam learning rate multiplied by k < 1. Its
critic fits worse, while its targets, rewards and actor are B's.
- **Calibration of k (before the freeze):** on seeds 42–45 (not X8 seeds), run B, S and Bdeg for each
  k in {1/2, 1/4, 1/8, 1/16, 1/32} on the eight cells.
- **Rule:** the frozen k is the candidate whose mean `ev_train_mc` gap (Bdeg − B) is closest to the S − B gap on the
  same seeds.
- No crossing is computed from the calibration runs.

**Logging.** X4's reference-critic logging, unchanged, in all three experiments. It measures `ev_train_mc`, value
error, GAE error, rank agreement and sign disagreement against an out-of-sample reference critic. The per-minibatch
pre- and post-clip gradient norms are logged as in E1.

## Inference
- **Primary:** jackknife-t over 40 seeds (39 df).
- **Secondary:** the seed-block percentile bootstrap (2,000 draws; seeds 20261124, 20261125, 20261126 for X6, X7, X8).
- **Pairing:** contrasts and changes are seed-paired.
- **Equivalence:** by the 90% jackknife-t interval within the stated margin.
- **Non-significance** is never read as equivalence.
- **Power:** the minimum detectable S − B at 80% power is about 0.08 (X4's standard error).

## Predictions and declared readings

### X6 (local critics)
1. **P1:** S − B under the plain local critic > 0 (95%).
2. **P2:** the plain local critic's advantages are worse for S: value error, GAE error and sign disagreement S/B > 1,
   and rank agreement S − B < 0, each by its 95% interval. Reported whether or not P1 is met.
3. **P3:** the fixed-scale head equalizes the fit: the S − B gap in `ev_train_mc` under M lies within ±0.05 (90%).
4. **P4 (tested only if P1 is met):** the change (S−B under M) − (S−B plain) < 0 (95%), **and** S − B under M lies
   within ±0.10 (90%).

| Result | Reading |
|---|---|
| P1 met | The shift is present with local critics; the claim extends to fully decentralized actor–critic training on the supply chain. |
| P1 met, P3 and P4 met | The critic's target scale carries it there, at this budget, without clipping. |
| P1 not met, S − B within ±0.10 (90%) | No shift at this margin with local critics: the confound, as measured, needs a critic that sees the global state. |
| Otherwise | Not detected and not shown absent: inconclusive. |

### X7 (shared clip, decentralized execution)
1. **P1:** S − B under the plain critic with the shared clip > 0 (95%).
2. **P2 (manipulation check):** the normalized-loss critics send B-sized gradients. For `normhead` and for
   `normpopart`, the seed-paired S/B ratio of the median pre-clip critic gradient norm lies within [0.8, 1.25] (90%).
   Under the plain critic the same ratio exceeds 1 (95%).
3. **P3:** `normhead` removes the shift: the change (S−B normhead) − (S−B plain) < 0 (95%), **and** S − B under
   normhead lies within ±0.10 (90%).
4. **P4:** `normpopart` removes the shift: the same two criteria for normpopart.
5. **Q5 (estimate, no directional prediction):** the change (S−B groupclip) − (S−B plain). X1 and X3 found no
   detectable per-group reduction on the shared controller.

| Result | Reading |
|---|---|
| P1 met | The shift under SB3's default shared clip transfers to decentralized execution. |
| P1, P2, P3 met, Q5 within ±0.10 (90%) | With the shared clip, making the critic's targets and gradients B-sized removes the shift, and per-group clipping does not detectably reduce it. The clipped shift then runs through the critic's target scale, not through the clip's starving of the actor. |
| P1, P2, P3 met, Q5 below zero (95%) | Both routes contribute; X7 does not apportion them. |
| P1, P2 met, S − B under normhead above zero (95%) | A channel other than the critic's target and gradient scale exists under the shared clip. |
| P2 not met | The manipulation failed; P3 and P4 are descriptive only. |
| Otherwise | Inconclusive for that part. |

P4 (normpopart) is read the same way as P3. If P1 is not met, P3–Q5 are descriptive only.

### X8 (fit at fixed target scale)
1. **P1:** S − B > 0 (95%): replication on fresh seeds.
2. **P2 (manipulation check):** the frozen k matches the fit. The difference between the fit gaps,
   (Bdeg − B) − (S − B) in `ev_train_mc`, lies within ±0.05 (90%).
3. **P3:** Bdeg − B > 0 (95%).

| Result | Reading |
|---|---|
| P2 met, P3 met, (S−B) − (Bdeg−B) within ±0.10 (90%) | A critic degraded to S's fit at B's target scale reproduces the shift. A worse fit is sufficient, which is consistent with fit mediating the target-scale effect. |
| P2 met, Bdeg − B within ±0.10 (90%) | Degrading the fit this way does not reproduce the shift: the target-scale effect is not explained by the fit as measured. |
| P2 met, otherwise | Partial or inconclusive; the estimates are reported. |
| P2 not met | The manipulation failed; descriptive only. |

The advantage-quality measures for Bdeg against B are reported beside S against B. A slower critic is one way to
degrade the fit; a different degradation could act differently, and the reading is limited to this one.

## Gates (before each array; seed 42 or a non-X seed, w = 0.6)
- **All:** one index run twice gives bit-identical results (evaluation, every minibatch record, X4 log). With and
  without logging, the learner gives bit-identical evaluation and minibatch records.
- **X6:**
  - M at c = 1 reproduces plain bit-for-bit;
  - the independence check passes: disjoint actor and critic parameters per node; zero gradient into a node's actor
    or critic from other nodes' samples; no critic output depends on global-state entries.
- **X7:**
  - normhead at c = 1 reproduces plain bit-for-bit;
  - the actor independence check passes;
  - the normpopart and groupclip indices complete.
- **X8:**
  - plain arm B on seed 122, w = 0.6 reproduces X4's D-plain run of that seed and cell (X4 index 160) bit-for-bit;
  - Bdeg at k = 1 reproduces plain bit-for-bit.
- **Completion:** each gate's first index of every part completes with 469 X4 records.

## Compute
Measured in the smoke (8 processes on 16 hyperthreads, so each had a physical core):

| Exp. | Minutes per 300k run | Runs | Core-hours |
|---|---|---|---|
| X6 | 10.7–12.4 (11.6 used) | 1,280 | ~247 |
| X7 | 10.1–11.8 (11.0 used) | 2,560 | ~469 |
| X8 | 9.2 | 960 | ~147 |
| X8 calibration (seeds 42–45) | 9.2 | 224 | ~34 |
| **Total** | | **5,024** | **~897** |

**Wall-clock:** about 14 hours at 64 concurrent tasks, which the current limit of 128 CPUs allows, or about 4.7 days at
8. The user's own jobs take priority.

## Stopping rule
- Run all declared trajectories once, and rerun failures only with the identical index.
- Add or drop no arm, part, cell, seed or analysis after outcomes are seen.
- Seeds 202–321 are used once.

## History before freezing
- **Smoke (jobs 16002 and 16004; seed 42, w = 0.6, 2,560 transitions; not an X6–X8 seed).**
  - **Passed:** determinism, logging neutrality, the c = 1 identity (X6's M and X7's normhead against plain) and the
    k = 1 identity (X8), each bit-for-bit.
  - **Independence checks:** passed. Each X6 node has 6,595 actor and 6,465 critic parameters.
  - **Job 16002:** X7's independence check crashed after training, because only the per-group learner had the
    read-only `parameter_groups` helper. It was added to the other X7 learners, where it does not affect training,
    and all X7 smoke runs were repeated in job 16004.
- **Manipulation in the smoke (X7, 2,560 transitions; descriptive):**
  - Median pre-clip critic gradient: 54.9 (B, plain), 405 (S, plain), 53.8 (S, normhead), 0.97 (S, normpopart).
  - The actor's applied gradient under the shared clip is 0.006 for B and 0.001 for S with the plain critic, 0.006 for
    S under normhead and 0.27 for S under normpopart.
  - So normhead gives S B's clip regime, in which the critic still dominates the clip, while normpopart lifts the
    clip's starvation for both arms. P3 and P4 therefore test different things: P3 tests target scale with B's clip
    regime held fixed; P4 tests it with the critic no longer dominating the clip.
- **X8 calibration (job 16005; seeds 42–45, 224 runs, no crossings computed).**
  - Mean `ev_train_mc` gap to B: S −0.265.
  - Bdeg: k = 1/2 −0.133, 1/4 −0.236, 1/8 −0.285, 1/16 −0.318, 1/32 −0.341.
  - By the declared rule **k = 1/8** (closest, |difference| 0.019), now fixed in `x6_x8_design.json`.
- **Analysis dry run:** `phase3_x6_x8.py` was run on X5's per-run table, relabelled as X6, X7 and X8 data. Variants
  exercised every reading branch of X8 (fit match met and failed). Neither is X6–X8 data.
