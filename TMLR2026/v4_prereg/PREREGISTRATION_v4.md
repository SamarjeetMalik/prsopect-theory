# v4 pre-declaration: experiments E1–E4

Written 2026-10-23, before any E1–E4 outcome existed. File hashes are in `HASHES.sha256`, computed immediately
before the first launch. Platform: Ashoka (`debayanlab`, Slurm QoS standard), env `cc4` (versions in
`PHASE0_REPORT.md`). Every trajectory in every experiment runs on this one platform.

**Common to all experiments:**
- Firth logistic midpoint w50 = −b0/b1;
- 2,000 paired training-seed-block bootstrap draws, bootstrap RNG seed 20261015;
- fits and draws outside the grid are retained and counted, never clipped;
- the collapse label is unchanged from v3;
- **stopping rule:** run every declared trajectory; rerun failures only with the identical index and config;
  no arm, cell, seed or analysis is added or dropped after outcomes are inspected;
- deviations are appended to `DEVIATIONS.md` with a timestamp.

## E1: optimizer × reward-normalisation factorial (frozen v2 design, v2.1 source)
**Design.** `AAMAS2027/optimizer_factorial_v2/design.json`, unchanged:
- optimizer: `global_clip` vs `separate_no_clip`;
- reward: `raw` vs `VecNormalize`;
- arms B, A, S, R (R = ×3.451);
- grid {0, .2, .4, .5, .6, .7, .85, 1};
- seeds 42–51; 300k; reference −2.6281398333.

**Runner.** `optimizer_factorial.py` (v2.1, deviation D1).

**Size: all 1,280 trajectories.** The 240 archived global_clip/raw cells are **not reused**, because
cross-platform runs are not bit-reproducible (Phase-0 gate). The 240 reruns are also compared label-by-label with
the archive and reported as cross-platform agreement.

**Primary comparisons and interpretation.** Exactly as in `optimizer_factorial_v2/PROTOCOL.md`, including its
equivalence margin [−.05, .05] and its four-way interpretation rule.

## E2: Adam-ε addendum (new)
**Design.**
- global_clip, raw, Adam ε = 1e-8 (SB3 default 1e-5);
- arms B/A/S/R × the E1 grid × seeds 42–51 = **320** trajectories;
- runner `eps_addendum.py`.

**Comparisons.**
- Within E2: A − S, S − B, R − B.
- Against E1 global_clip/raw, paired by seed: the change in S − B and A − B caused by ε alone.

**Interpretation (declared now).**
- S − B and A − B shrink under E2 versus E1 global_clip/raw, with the paired difference-of-contrasts CI excluding
  0 → the clip effect acts through Adam's ε floor.
- They do not shrink, while E1 separate_no_clip removes them → clipping acts through another route (e.g. the
  global norm's minibatch-varying scale factor).
- Otherwise → inconclusive.

## E3: single-platform refined horizon cohort
**Design.** `AAMAS2027/orchestration_v4_e3/design.json`:
- conditions null / curvature / λ7;
- grid {.2, .4, .6, .65, .7, .75, .8, .85, 1};
- seeds 42–51 = **270** trajectories to 3.6M, evaluated every 150k;
- runner `orchestration_v4_e3/run.py`, on the hash-verified `instrumented_source_v2`.

**Comparisons.**
- λ7 − curvature at 300k, 900k, 1.8M, 3.6M;
- absolute w50(T) per condition;
- recovery and relapse conditional on the 300k label;
- practical stability: the 3.6M − 1.8M change CI wholly within ±.05 (the v3 ±.02 rule is also reported);
- cross-platform label agreement on the 5 archived cells × 3 conditions × 10 seeds, descriptive.

## E4: LBF under frozen source (design declared now; code frozen after a timing smoke)
**Design.**
- MAPPO on LBF (5×5, 2 players, 2 food; legacy `exp_mappo_lbf.py` environment, wrapper and collapse rule
  θ = 0.626 native team return);
- training reward includes the −0.02(1−w) LOAD cost, as in legacy;
- arms: linear (α = β = 1, λ = 1), curvature (.88, λ = 1), A (.88, λ = 4), S (.88, both branches ×4);
- optimizer: global_clip vs separate_no_clip (ported to the MAPPO learner);
- grid {0, .1, .2, .3, .5, .65, .75, .85, .9, .95}; seeds 42–51; 2M timesteps.

That is 4 × 2 × 10 × 10 = **800** trajectories.

**Sizing rule (cost-based, not outcome-based, declared now).** If the timing smoke shows a mean cost above 1.0
run-hour per trajectory, drop seeds 47–51 (giving 400 trajectories) before launch. Nothing else changes.

**Predictions (declared now).**
1. Under global_clip, curvature − linear > 0 (|x| < 1, so x^.88 enlarges magnitudes). Legacy found +0.177.
2. A − linear > 0.
3. Under separate_no_clip, the curvature − linear and S − curvature shifts shrink relative to global_clip.

**Code freeze.** The ported runner's hash is appended to `HASHES.sha256` before E4 launches. The port changes only
the optimizer step, mirroring `SeparateNoClipPPO`.

## Launch order
E1 + E2 first (1,600 trajectories, ~1 day at 8 concurrent), then E3 (270 × ~1.7 h), then E4.
