# X5 pre-declaration: does the symmetric-scaling shift, and its critic-target channel, survive decentralized execution?

**Status: design chosen by the authors on 2026-11-19 (full design, 1,920 runs); frozen on 2026-11-19T11:57:19Z, before any X5
run.** Hashes of this file, `experiments/x5_runner.py`, `experiments/x5_design.json`, `analysis/phase3_x5.py` and
the shared sources are in `HASHES.sha256`. The arrays start only after the gate below passes (`afterok`).

## Why
Every symmetric shift found so far on the supply chain comes from a fully observed, parameter-shared controller. Under
local observation with agent identity (E12, ten seeds) the shift was not detectable (+0.069). X4 located the
shared-controller shift in the scale of the critic's regression targets. X5 asks whether either result holds when the
agents act on their own observations with their own parameters.

## Design: fresh seeds 162–201 (40), arms B and S, the E1 grid, 300k, raw rewards, separate Adam, no clipping
**Execution.**
- **Actors.** Each of the five trained nodes has its own actor network (64-64 tanh), its own action head and its own
  log-std. No actor parameter is shared between nodes. A node's actor sees only its local view (E12's: own retail
  demand, own inventory, own incoming order pipeline, zero-padded) and a one-hot node identifier.
- **Critic.** One centralized critic (64-64 tanh) sees the full global state and the node identifier. The identifier
  is needed because each node's mixed reward differs.
- **Training.** Training is centralized and execution decentralized. At evaluation each node's policy input is its
  own decentralized observation (E12's evaluation, with this view).

| Part | Critic head | Runs | Purpose |
|---|---|---|---|
| plain | E7's plain head | 640 | does the shift appear under decentralized execution? |
| popart | E7's PopArt head | 640 | normalizing the critic's targets, with its fit equalized |
| M | plain head with output fixed-scaled, V = c·head(x), c = 1 (B) or 7 (S) | 640 | channel test, as in X4 |
| **Total** | | **1,920** | |

**Logging.** Logging is X4's reference-critic logging, unchanged:
- **Reference critic:** 64-64 tanh, fitted out of sample to standardized Monte-Carlo returns.
- **Metrics:** `ev_train_mc`, value error, GAE error, rank agreement and sign disagreement, averaged over rollouts
  1 to the end.

## Predictions
**Inference.**
- **Primary:** jackknife-t over 40 seeds (39 df).
- **Secondary:** the seed-block percentile bootstrap (2,000 draws, seed 20261123).
- **Seed pairing:** metric contrasts are seed-paired as in X4.

**Predictions.**
1. **Symmetric shift under decentralized execution (plain):** S − B > 0, with the 95% interval excluding zero.
2. **Advantage degradation (plain):** value error, GAE error and sign disagreement each have S/B > 1, and rank
   agreement has S − B < 0, each by its 95% interval. This is reported whether or not P1 is met.
3. **PopArt equalizes the critic's fit:** the fit gap S − B (`ev_train_mc`) lies within ±0.05 by its 90% interval.
4. **The fixed-scale head equalizes the critic's fit:** the same, under M.
5. **Channel (M):** the change (S−B under M) − (S−B under plain) is below zero by its 95% interval, **and** S − B under
   M lies within ±0.10 by its 90% interval.
6. **PopArt:** the change (S−B under PopArt) − (S−B under plain) is below zero by its 95% interval, **and** S − B under
   PopArt lies within ±0.10 by its 90% interval.

## Declared interpretation
**Scope (P1):**

| Result | Reading |
|---|---|
| P1 met | A symmetric shift is present under decentralized execution. The paper's claim extends to this setting. |
| P1 not met, and S − B (plain) within ±0.10 by its 90% interval | No symmetric shift under decentralized execution at this margin. The shift is established for the shared controller only. |
| Otherwise | Not detected and not shown absent: inconclusive at this power. The claim stays restricted to the shared controller. |

**Channel and PopArt, only if P1 is met** (the readings follow X4's tables):

| Result | Reading |
|---|---|
| P4 and P5 met | Fixing the critic's output scale removes the shift. The critic's target scale carries it under decentralized execution at this budget. |
| P4 met, S − B under M above zero (95%) | A channel other than the critic's target scale exists in this setting. |
| P3 and P6 met | With the fit equalized, the shift is equivalent to zero and reduced. This is consistent with the critic-target account, not proof of it. |
| P3 met, S − B under PopArt above zero (95%), change not shown below zero | The fit account is falsified as stated in this setting. |
| Otherwise | Inconclusive for that part. |

If P1 is not met, P5 and P6 do not test a channel, and their numbers are reported as descriptive only. Non-significance
is never read as equivalence. The minimum detectable S − B is about 0.08 at 80% power (from X4's standard error at 40
seeds).

## Gate (before any array)
1. Index 0 run twice gives bit-identical results: evaluation, every minibatch record and the X5 log.
2. Part M arm B (c = 1) reproduces part plain arm B bit-for-bit (seed 42, w = 0.6, not an X5 seed), including the X5
   log.
3. Logging is RNG-neutral: plain arm B with and without logging gives bit-identical evaluation and minibatch records
   (seed 42, w = 0.6).
4. The independence check passes on indices 0, 640 and 1280 (plain, PopArt, M):
   - each node's actor parameters are disjoint from every other node's;
   - a node's actor receives exactly zero gradient from other nodes' samples;
   - the critic's output does not depend on the local-only entries of the observation.
5. Indices 0, 640 and 1280 complete with 469 X5 records.

## Compute (measured in the smoke)
- **Per run:** about 9.2 minutes. Profiling 12,800 transitions (job 14016) put X5's cost at 1.04 times X4's, and X4's
  arrays averaged 8.76 minutes per run. The environment step dominates both.
- **Why the smoke timing is not used:** it gave 19.6 minutes per run, but it ran 12 processes on 12 hyperthreads, that
  is 6 physical cores. That doubles per-run time, as measured in Phase 0.
- **Total:** 1,920 runs, about 295 core-hours.
- **Wall-clock:** about 6 hours at 48 concurrent tasks, about 37 hours at 8. The user's own jobs take priority.

## Stopping rule
- Run all declared trajectories once. Rerun failures only with the identical index.
- Add or drop no arm, part, cell, seed or analysis after outcomes are seen.
- Seeds 162–201 are used once.

## History before freezing
- **Smoke (jobs 14013 and 14014):** seed 42 only, which is not an X5 seed. Every check passed in both jobs. Each ran:
  - 2,560-transition checks of independence, logging neutrality, M (c = 1) against plain, determinism and PopArt;
  - 32,000-transition timing runs for all six part × arm cells.
- **Actor routing changed between the two smoke jobs.** Each sample is now routed through its own node's network
  only, instead of masking all five networks' outputs. The two are the same function. The change did not alter run
  time, and profiling (job 14016) showed that the environment dominates.
- No smoke run reaches an X5 seed, and no smoke outcome is used.
- **Analysis dry run:** `phase3_x5.py` was run on X4's per-run table, relabelled as X5 (seeds +40), and on a copy with
  S's labels set equal to B's. This checked both branches of the scope reading. Neither is X5 data.
