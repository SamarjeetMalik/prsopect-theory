# v4 Phase 3 result (2026-10-28)

Analysis: `analysis/phase3.py`. Full output: `results/phase3_report.txt` and `results/*.csv`. Every estimate is a
Firth w50 with 2,000 paired seed-block bootstrap draws (seed 20261015), as pre-declared. 1,870/1,870 trajectories
completed and validated. All 2,000 bootstrap draws were usable for every fit, and no interval had zero width.

## E1/E2 contrasts (w50 differences, 95% CI; * = excludes 0)

| Cell | A − S (asymmetry) | A − B (loss ×7) | S − B (symmetric ×7) | R − B (symmetric ×3.45) |
|---|---|---|---|---|
| global_clip / raw (v3 setup) | −0.074 [−0.150, +0.000] | +0.283* | +0.357* | +0.256* |
| global_clip / VecNormalize | **+0.237* [+0.169, +0.305]** | +0.287* | +0.050 [−0.035, +0.144] | +0.060 [−0.068, +0.179] |
| separate_no_clip / raw | **+0.174* [+0.083, +0.255]** | +0.334* | +0.160* | +0.027 [−0.024, +0.069] |
| separate_no_clip / VecNormalize | **+0.237* [+0.136, +0.346]** | +0.188* | −0.049 [−0.117, +0.017] | −0.064 [−0.176, +0.024] |
| E2: Adam ε 1e-8 / raw | +0.048 [−0.013, +0.111] | +0.307* | +0.258* | +0.165* |

Paired change relative to global_clip/raw:

| Cell | Δ(S − B) | Δ(A − S) | Δ(A − B) |
|---|---|---|---|
| VecNormalize | −0.308* | +0.311* | +0.003 (ns) |
| separate_no_clip | −0.197* | +0.248* | +0.051 (ns) |
| both interventions | −0.407* | +0.311* | −0.096 (ns) |
| ε 1e-8 | −0.099 (ns, [−0.243, +0.033]) | +0.122* | +0.023 (ns) |

## Applying the pre-declared rules (PROTOCOL.md v2; PREREGISTRATION_v4 §E2)
1. **"A − S interval excluding zero after both interventions → evidence for a sign-asymmetric effect under this
   implementation."** **MET.** With both interventions, A − S = +0.237 [+0.136, +0.346]. It is also met under
   each intervention alone.
2. **"Collapse of A − B and S − B after uncoupling or normalisation → reward-scale/optimizer mechanism."**
   **PARTLY.**
   - S − B collapses to ≈0 under normalisation and under both interventions. It shrinks but survives under
     uncoupling alone (+0.160).
   - R − B collapses under every intervention.
   - **A − B does not collapse anywhere** (+0.19 to +0.33).
3. **"Persistence of A − B and S − B with A − S near zero → scale effect outside the clipping pathway."**
   **NOT MET.**
4. **E2 (ε).** The ε route was pre-declared as "S − B shrinks with the difference-of-contrasts CI excluding 0".
   **NOT MET** (−0.099 [−0.243, +0.033]). ε is not the main channel, so my Adam-ε hypothesis is not supported.

## Verdict
In the paper's original configuration (global clip, raw rewards), the symmetric-scaling effect is an optimizer
artifact:
- Every minibatch is clipped.
- The critic carries about 100% of the gradient energy.
- The post-clip actor gradient is 0.0015 (B), 0.0009 (A) and 0.0002 (S).
- Under these conditions symmetric ×7 moves the crossing as far as the loss multiplier, which is why v3 had to
  retract the asymmetry claim.

Once that channel is removed:
- the symmetric multipliers no longer move the crossing (S − B, R − B ≈ 0);
- **the loss-only multiplier still does** (A − B ≈ +0.19 to +0.29);
- A − S becomes positive and excludes zero.

**So the loss-multiplier shift is not explained by reward scale. It reflects the sign-asymmetric reward
transform, under this implementation.** The scale artifact acted mainly through critic-target magnitude: it is
removed by VecNormalize, and only partly by separate optimizers. It did not act through Adam's ε.

### Further findings that must be reported
- **The baseline crossing itself is optimizer-dependent.** B = 0.538 in the standard setup, against 0.31–0.45
  under the interventions. Absolute w50 is conditional on the optimizer configuration, not only on the reward
  design.
- **Cross-platform labels.** 221/240 (92.1%) agree with the authors' archived direct control; 164/240 are
  bit-identical. 3,289/3,600 (91.4%) agree with the archived horizon cohort. The A-arm crossing reproduces
  exactly (0.821 [0.775, 0.864]).

## E3 (single platform, 9-cell grid)
- **λ7 − curvature persists but shrinks:**

  | Budget | λ7 − curvature |
  |---|---|
  | 300k | +0.291* |
  | 900k | +0.287* |
  | 1.8M | +0.187* |
  | 3.6M | +0.121* [+0.064, +0.177] |

- **Absolute w50:**
  - null: 0.663 → 0.512 → 0.535 → 0.578;
  - curvature: 0.587 → 0.473 → 0.523 → 0.559;
  - λ7: 0.877 → 0.760 → 0.710 → 0.681.

  Null and curvature are **non-monotone**: they fall to 900k, then rise again, through relapse.
- **Stability** (3.6M − 1.8M within ±.05): **met by no condition** (null [−0.008, +0.095], curvature
  [−0.029, +0.105], λ7 [−0.071, +0.001]). The refined grid removed the v3 zero-width intervals; the boundary is
  genuinely still moving.
- **Recovery and relapse by 3.6M:**

  | Condition | Recovered (of collapsed at 300k) | Relapsed (of non-collapsed at 300k) |
  |---|---|---|
  | null | 14/39 | 8/51 |
  | curvature | 4/28 | 4/62 |
  | λ7 | 34/76 | 0/14 |

  Collapse is not absorbing, and non-collapse is not either.
- **Curvature − null** is −0.076* at 300k and not significant thereafter.

## Implementation notes (analysis code only; no design change)
- The w grid was rounded to 4 dp before pivoting (float keys).
- `null` was read as a string, not NaN.

Both bugs raised errors before any fit was computed.

---

# E4 result: MAPPO on LBF (2026-10-30)

Analysis: `analysis/phase3_e4.py`; output in `results/phase3_e4_report.txt` and `results/E4_*.csv`.
800/800 trajectories completed and validated. Same estimator, same bootstrap seed.

## Crossings (w50 [95% CI])

| Arm | global_clip | separate_no_clip |
|---|---|---|
| linear | −0.289 [−1.333, −0.001], out of grid | 0.003 [−0.179, 0.068] |
| curvature | −0.370 [−2.288, 0.009], out of grid | 0.077 [−0.087, 0.196] |
| A (λ = 4) | **0.525 [0.188, 0.719]** | **0.557 [0.250, 0.724]** |
| S (both branches ×4) | 0.107 [−0.065, 0.225] | 0.077 [−0.126, 0.237] |

The linear and curvature arms rarely collapse at all (overall collapse rate 5–16%, almost entirely at w ≤ 0.1),
so their crossings sit at or below the grid edge and are effectively unidentified.

## Contrasts
- **A − S (asymmetry): +0.418 [+0.218, +0.553] (global_clip); +0.479 [+0.289, +0.613] (separate_no_clip).**
  The difference between optimizers is +0.061 [−0.137, +0.249], not significant.
- **A − linear: +0.814 [+0.422, +1.725] (global, inflated by the out-of-grid linear fit); +0.554
  [+0.310, +0.718] (separate).**
- S − curvature: +0.476 [−0.005, +2.529] (global, unidentified); +0.001 [−0.167, +0.185] (separate).
- Curvature − linear: −0.080 [−1.621, +0.673] (global, unidentified); +0.073 [−0.029, +0.163] (separate).

## Pre-declared predictions
- **P1 (global_clip curvature − linear > 0): NOT MET.** The legacy +0.177 does not reproduce: under this rerun
  neither arm has an interior crossing.
- **P2 (A − linear > 0): MET** under both optimizers.
- **P3 (separate_no_clip shrinks curvature − linear and S − curvature): NOT MET.** Both interactions are
  unidentified (CIs span ±1 or more).

## Why P1/P3 failed, and what E4 does show
P1 and P3 rested on the scale/clipping pathway found in the supply chain. The LBF diagnostics show that pathway
is absent here:
- the critic gradient norm is ≈0.07, against an actor norm of ≈1.1;
- the actor carries 98% of the gradient energy;
- clipping, where it fires, trims the actor itself.

With no critic domination there is no scale throttle for curvature or symmetric scaling to act through. That is
consistent with the E1 mechanism, but it was not what P1/P3 predicted.

What E4 does show is the same qualitative result as E1 under decoupling, now in a second environment and under
both optimizer modes: **the loss-only multiplier moves the crossing (+0.42 to +0.48 relative to symmetric
scaling), while symmetric scaling does not.**

Mechanistically, λ multiplies only LBF's −0.02(1−w) LOAD cost. S scales food rewards and cost together, leaving
their ratio unchanged. The training-reward loss fraction under A falls to 3%, against 14% under linear: agents stop
attempting loads. So in LBF the asymmetric effect is an incentive change on the costly cooperative action. It is a
genuine reward-design effect, but not evidence of a loss-aversion "psychology".

## E4 addendum: relabel at the pre-declared θ = 0.626 (deviation D4, 2026-10-30)
The stored E4 labels used a placeholder θ = 0.5. Recomputed at θ = 0.626 from stored team returns
(`results/phase3_e4_report.txt`, `results/E4_theta_sensitivity.csv`):

| Contrast | Global clip | Separate, no clip |
|---|---|---|
| A − S | +0.447 [+0.312, +0.559]* | +0.490 [+0.315, +0.619]* |
| A − curvature (identified) | +0.543 [+0.404, +0.638]* | +0.617 [+0.394, +0.767]* |

- Curvature now has an interior crossing (0.23). The linear crossing under the global clip remains below the grid.
- **Predictions:**
  - P1 is positive but NOT IDENTIFIED.
  - P2 is MET; it is identified under separate optimizers.
  - P3a is positive but NOT IDENTIFIED.
  - P3b is NOT MET.
- **A − S holds** for θ ∈ {0.5, 0.626, 0.75} and under the isotonic estimator.
- **Loss occupancy:** under A, 1–2% of training rewards are below zero at w ≤ .75 (agents stop LOAD attempts).

## Stage I diagnostics (existing data)
- **Isotonic vs Firth:** they agree within 0.055 (E1/E2) and 0.047 (E3); E4 differs by up to 0.161.
- **Critic explained variance:**
  - separate_no_clip/raw: B 0.685, S 0.300;
  - VecNormalize: B 0.668, S 0.664.
- **Loss-branch occupancy under fixed z₀:** about 0.58 at w = 0, falling to about 0.07 at w = 1, and policy-dependent.

---

# E5 and E6 results (2026-11-04)
Analysis: `analysis/phase3_e5e6.py`; output in `results/phase3_e5e6_report.txt` and `results/E5_*.csv`, `results/E6_*.csv`.
- **E5:** 270/270 trajectories × 24 checkpoints.
- **E6:** 720/720 runs.
- **E6 gate (job 3262):** passed. It is bit-identical to E1 runs 120 and 760.

## E5: decoupled horizon (separate_no_clip + VecNormalize, B/A/S, 9 cells, to 3.6M)

| Budget | A − S | A − B | S − B |
|---|---|---|---|
| 300k | +0.234 [+0.165, +0.322]* | +0.178* | −0.056 [−0.147, +0.043] |
| 900k | +0.197 [+0.086, +0.328]* | +0.141* | −0.056 [−0.146, +0.016] |
| 1.8M | +0.119 [+0.034, +0.244]* | +0.028 | −0.091 [−0.262, +0.054] |
| 3.6M | +0.154 [+0.053, +0.256]* | +0.105 | −0.049 [−0.240, +0.099] |

- **Predictions:** P1 MET; P2 (persistence of A − S at every budget) MET; P3 (S − B undetectable at every budget) MET.
- **Crossings rise after 900k:**
  - B: 0.425 → 0.385 → 0.639 → 0.732;
  - A: 0.603 → 0.526 → 0.667 → 0.837;
  - S: 0.369 → 0.329 → 0.548 → 0.683.
- **Relapse among runs non-collapsed at 300k:** 36/72 (B), 36/60 (A), 32/76 (S).
- **Stability:** met by no arm.

## E6: clipping isolation (raw rewards, B/A/S, E1 grid, 300k)

| Mode | S − B | A − S |
|---|---|---|
| global_clip | +0.357* | −0.074 |
| groupwise_clip | **+0.150 [+0.095, +0.226]*** | **+0.168 [+0.077, +0.240]*** |
| separate_no_clip | +0.160* | +0.174* |

- **Δ(S − B), groupwise vs global:** −0.207 [−0.336, −0.085]*. Groupwise and no-clip crossings are nearly identical.
- **Actor parameter-update ratio S/B (seed-paired):**
  - global clip 0.391 [0.378, 0.405];
  - groupwise clip 0.949 [0.936, 0.962];
  - no clip 0.956 [0.944, 0.967].

  The applied actor-gradient ratio under the global clip is 0.121, so Adam partly compensates: the update shrinks
  less than the gradient.
- **Critic explained variance at S:** 0.28 (global), 0.25 (groupwise), 0.30 (no clip); about 0.61–0.68 for B.
- **Predictions:**
  - **P1 NOT MET as worded.** The global-clip ratio is < 1 (met), but the groupwise and no-clip ratios (0.95)
    have intervals excluding 1.
  - **P2: a channel other than the shared clip remains.** The shared clip accounts for Δ = −0.207 of the +0.357
    artifact. The residual (+0.15) is unaffected by clipping and is removed by reward normalization (E1).
  - **P3 MET.**
  - **P4:** 480/480 labels and profits are bit-identical to E1.

## Verdict
The symmetric-scaling artifact under SB3 defaults has two parts:
1. **A shared-clip part.** The critic takes the actor's clip budget. The actor's actual updates fall to 0.39×, and
   per-group clipping removes this part.
2. **A critic-target-scale part.** It is unaffected by clipping, comes with poor critic fit, and is removed by
   reward normalization.

Under the learner that removes both (separate optimizers + VecNormalize), symmetric scaling stays undetectable
through 3.6M, while the gain-to-loss-ratio effect A − S persists at every budget.

**Analysis-code fix (no design change):** `c6.mode` resolved to the DataFrame method; it now uses `c6["mode"]`.
It raised an error before any E6 verdict was printed.

# E7 results (2026-11-05)
Analysis: `analysis/phase3_e7.py`; output in `results/phase3_e7_report.txt` and `results/E7_*.csv`.
- **Runs:** 400/400 new runs (main array 4369). The 240 plain-critic B/A/S cells are E1 runs.
- **Gate (job 4368):** passed. E7 index 120 is bit-identical to E1 run 760; PopArt and M runs completed.

| Critic | S − B | A − S | A − M | M − B |
|---|---|---|---|---|
| plain | +0.160 [+0.078, +0.254]* | +0.174* | +0.050 [+0.000, +0.100] | +0.285* |
| PopArt | +0.075 [−0.032, +0.164] | +0.207* | +0.268 [+0.211, +0.317]* | +0.014 |

- **Change in S − B, PopArt minus plain:** −0.085 [−0.183, +0.006].
- **Predictions:**
  - **P1 INCONCLUSIVE.** PopArt S − B includes 0, and so does the change.
  - **P2 MET.** PopArt explained variance is S 0.69 vs B 0.68.
  - **P3 MET.**
  - **P4 NOT MET.** The plain-critic A − M interval touches 0.
  - **P5 NOT MET.** M's achieved mean is −13.5 vs A's −6.6 (plain).
  - **P6 MET.**

# E8 results (2026-11-06)
Analysis: `analysis/phase3_e8.py`; output in `results/phase3_e8_report.txt` and `results/E8*.csv`.
- **Smoke (job 4944):** passed. Frozen in commit `f64245b`.
- **Gate (job 4947):** passed. Index 40 `--static` is bit-identical to E7 run 280.
- **Runs:** 480/480 (main array 4948).

| Critic | A − M_on | M_on − B | M_on − M |
|---|---|---|---|
| plain | +0.050 [+0.020, +0.082]* | +0.284* | −0.001 |
| PopArt | +0.246 [+0.197, +0.300]* | +0.036 | +0.022 |

| Part B | plain S − B | PopArt S − B | Change |
|---|---|---|---|
| seeds 52–61 (primary) | +0.162* | +0.006 [−0.070, +0.085] | −0.156 [−0.303, −0.016]* |
| seeds 42–61 (secondary) | +0.159* | +0.044 | −0.115 [−0.209, −0.032]* |

- **Predictions:**
  - **P1 MET.** Pooled mean and SD are within 2%/1% (plain) and 5%/1% (PopArt) of A's. 15/160 runs touched the lower
    scale clamp.
  - **P2 MET.**
  - **P3: positive, excluding 0.** The plain-critic asymmetric contrast is not explained by the first two moments.
    The E7 reading is revised; the residual is small.
  - **P4:** M_on − M is about 0 under both critics.
  - **P5 MET.** It is not met at the calibrated 99% level in the primary cohort: [−0.344, +0.025]. The pooled
    20-seed change excludes 0 at 99%.

## Verdict, updated
1. **Critic-target-scale part (verdict item 2).** It is attributed to critic target scale at the declared level, in an
   independent cohort.
2. **The gain-to-loss contrast.** It survives a symmetric control matched in training-reward mean and SD under both
   critics. With a raw-target critic, the part not explained by those moments is small.

# E9 and E10 results (runs completed 2026-11-07; analysed 2026-11-17)
Analysis: `analysis/phase3_e9_e12.py`; output in `results/phase3_e9_e12_report.txt`, `results/E9*.csv` and `results/E10_contrasts.csv`.
- **Smoke (job 5527):** passed. Frozen in commit `f93cfd2`.
- **Gate (job 5528):** passed. τ = 0.5 reproduces E1 run 40 and `--critic popart` reproduces E7 run 360 bit-for-bit.
- **Runs:** 640/640 (E9, array 5529) and 640/640 (E10, array 5530); none failed. Extraction job 9142.

| τ | applied S/B | S − B |
|---|---|---|
| 0.5 (E1) | 0.121 | +0.357 [+0.249, +0.464]* |
| 300 | 0.137 | +0.303* |
| 3000 | 0.428 | +0.252* |
| 10000 | 0.716 | +0.176* |
| 30000 | 0.943 | +0.153 [+0.071, +0.244]* |
| no clip (E1) | 1.087 | +0.160 [+0.078, +0.254]* |

- **E9 predictions:**
  - **P1 MET.** The applied S/B ratio rises monotonically and exceeds 0.5 at τ = 30000.
  - **P2 MET at the declared 95% level.** S − B at τ = 30000 minus τ = 0.5 is −0.204 [−0.369, −0.032]; 99% [−0.417, +0.018].
  - **P3 MET.** τ = 30000 minus no clip is −0.007 [−0.075, +0.048].
  - **P4 (descriptive).** S − B falls with the applied ratio; about +0.16 remains without any clipping.
- **E10 (seeds 62–81, primary):** plain S − B +0.111 [+0.033, +0.194]*; PopArt S − B +0.010 [−0.039, +0.056]; change
  −0.101 [−0.199, −0.002]* (99% [−0.229, +0.033]). Pooled 42–81 (secondary): change −0.108 [−0.174, −0.043]*, 99%
  [−0.194, −0.024].
  - **P5 supported at the declared 95% level only**, as in E8.
  - **P6 MET.**

# E11 and E12 results (runs completed 2026-11-08; analysed 2026-11-17)
- **Smoke (job 5531):** passed. Frozen in commit `dcfc32f`.
- **Gate (job 5532):** passed. E11 index 40 twice bit-identical; E11 440/520 and E12 40/360 complete.
- **Runs:** 720/720 (E11, array 5533) and 480/480 (E12, array 5534); none failed.

| E11 condition | critic share | applied S/B | S − B | A − S |
|---|---|---|---|---|
| cleanrl_raw | 0.987 | 0.33 | +0.288 [+0.171, +0.393]* | +0.298* |
| cleanrl_group | 0.978 | 1.00 | +0.187 [+0.119, +0.250]* | +0.225* |
| cleanrl_default | 0.014 | 1.00 | +0.000 [−0.057, +0.057] | +0.297 [+0.222, +0.382]* |

- **E11 predictions:** P1 MET; P2 MET (−0.101 [−0.207, −0.000]; 99% includes 0); P3 MET; P4 MET.

| E12 learner | S − B | A − S | w50 B / A / S (E1 full observation) |
|---|---|---|---|
| default | +0.069 [−0.021, +0.154] | +0.091 [+0.040, +0.146]* | 0.582 / 0.743 / 0.651 (0.538 / 0.821 / 0.895) |
| decoupled | −0.008 [−0.166, +0.264] | +0.562 [+0.483, +0.654]* | 0.075 / 0.629 / 0.067 (0.414 / 0.602 / 0.365) |

- **E12 predictions:**
  - **P5 NOT MET.** Under local observation the default learner's S − B is not detectable.
  - **P6 MET, weakly informative.** B and S collapse almost only at w = 0, so their crossings sit at the grid's lower
    edge and the interval is wide.
  - **P7 MET.**

## Verdict, updated
1. **Generality of the confound.** The shared-clip confound reproduces in a second, CleanRL-style implementation, and
   reward normalization removes it there too.
2. **Clipping pathway.** S − B follows the clip threshold, and about +0.16 remains without clipping (E9; E11's
   per-group clip leaves +0.19).
3. **Critic targets.** A second fresh cohort repeats E8: supported at the declared level only.
4. **Information structure.** With local observations and agent identity, A − S holds under both learners; the
   default learner's symmetric shift is no longer detectable.

# X1 and X2 results (runs completed 2026-11-18; analysed the same day)
Analysis: `analysis/phase3_x1x2.py`, frozen in commit `7e7a9c6`; output in `results/phase3_x1x2_report.txt`,
`results/X1X2_contrasts.csv` and `results/X1_optimizer_state.csv`.
- **Smoke (job 9149):** passed.
- **Gate (job 9150):** passed. X1 reproduces E6 runs 40/280/520 and X2 reproduces E11 runs 40/520 bit-for-bit.
- **Runs:** 1,600/1,600 (arrays 9151, 9152); none failed. Extraction job 10757.

Fresh seeds 82–101 (primary), jackknife-t 95%:

| Quantity | Fresh 82–101 | Pooled with seeds 42–51 |
|---|---|---|
| X1 S − B, global clip | +0.248 [+0.191, +0.305]* | +0.285* |
| X1 S − B, per-group clip | +0.275 [+0.214, +0.336]* | +0.236* |
| X1 S − B, no clip | +0.076 [−0.013, +0.166] | +0.103 [+0.035, +0.171]* |
| X1 change, per-group − global | **+0.027 [−0.071, +0.125]** | −0.049 [−0.138, +0.040] |
| X2 S − B, raw | +0.370 [+0.260, +0.479]* | +0.338* |
| X2 change, group − raw | −0.112 [−0.244, +0.021] | −0.105 [−0.194, −0.016]* |

Optimizer state, X1 seed-paired S/B (geometric mean):

| Quantity | Global clip | Per-group clip | No clip |
|---|---|---|---|
| applied actor gradient | 0.131 | 1.000 | 1.053 |
| Adam m̂ | 0.161 | 0.995 | 1.034 |
| √v̂ | 0.160 | 1.000 | 1.044 |
| Adam direction / actor update | 0.400 | 0.940 | 0.957 |
| κ (mean), S/B | 0.150 | — | — |

- **Predictions:**
  - **P1 MET.**
  - **P2 NOT MET.** On fresh seeds, per-group clipping restores the actor's update but does not reduce S − B; the
    point estimate is +0.027.
  - **P3 NOT MET** on fresh seeds. It is positive when pooled.
  - **P4 MET.**
  - **P5 MET.**
  - **P6 NOT MET** on fresh seeds. It is negative when pooled at 95%, but not at 99%.
- **Declared consequence (P2 not met, P1 met):** the clipping pathway's share of the shift is not confirmed. The
  per-group reduction is supported only in the original cohort (E6).

## Verdict, updated
1. **The symmetric-scaling shift is robust.** It appears on fresh seeds and in both implementations.
2. **The shared clip attenuates the actor's applied gradient and Adam step for S.** This is directly measured,
   including Adam's moments, which shrink with κ. In X1, Adam's normalization undoes only part of the attenuation,
   and the step direction ratio is 0.40.
3. **Restoring the actor's step (per-group clip) does not remove the behavioural shift on fresh seeds.** Actor
   attenuation is therefore not shown to cause the crossing shift.
4. **What removes the shift:** reward normalization (both implementations). PopArt removes the no-clip residual when
   pooled over 40 seeds.

## X3 (pre-registered 2026-11-18, frozen 08:20:40Z before the gate; analysed with the frozen `phase3_x3.py`)
1,280 of 1,280 runs complete (Slurm arrays 10760 and 10761). Actor clipped (0.5) or not × critic clipped (0.5) or
not, one Adam step, raw rewards, arms B and S, the E1 grid, fresh seeds 102–121. Output: `results/phase3_x3_report.txt`,
`results/X3_contrasts.csv`, `results/X3_optimizer_state.csv`.

| Quantity | Estimate | Jackknife-t 95% | Jackknife-t 90% |
|---|---|---|---|
| S − B, both clipped | +0.213 | [+0.125, +0.301] | [+0.140, +0.285] |
| S − B, actor only | +0.172 | [+0.104, +0.239] | [+0.116, +0.227] |
| S − B, critic only | +0.176 | [+0.106, +0.246] | [+0.119, +0.234] |
| S − B, neither | +0.175 | [+0.091, +0.259] | [+0.106, +0.244] |
| critic effect | +0.021 | [−0.051, +0.093] | [−0.038, +0.081] |
| actor effect | +0.017 | [−0.050, +0.084] | [−0.039, +0.072] |
| interaction | +0.040 | [−0.097, +0.177] | [−0.073, +0.153] |

- **Predictions:**
  - **P1 NOT MET** (primary): the critic-clipping effect's 95% interval includes zero.
  - **P2 MET:** the actor-clipping effect's 90% interval lies within ±0.10.
  - **P3 MET:** S − B with both groups clipped excludes zero.
- **Descriptive:** in every mode the critic's explained variance is lower for S by 0.39–0.40; the actor's update S/B
  is 0.94–0.96 and the critic's 0.68–0.72; without its clip the critic's applied gradient S/B is about 9.
- **Declared consequence (P1 not met, critic effect near 0):** neither group's clipping is shown to carry the shift.
  The no-clip shift on fresh seeds, and reward/target normalization, are the only located factors.
- X1's no-clip estimate on seeds 82–101 (+0.076, interval including 0) motivated X3's hypothesis; on seeds 102–121
  the no-clip shift is +0.175 and excludes zero.

## Verdict after X3
The symmetric-scaling shift is a property of PPO with an unnormalized critic. It is present with or without
gradient clipping. The shared clip measurably starves the actor but is not shown to move the crossing (E6's reduction
did not replicate in X1, and neither group's clip matters in X3). In the CleanRL-style implementation a small per-group
reduction appears only in the pooled secondary analysis (−0.105 [−0.194, −0.016]). Normalizing rewards removes the
shift in both implementations; normalizing only the critic's targets reduces it on the pooled 40-seed evidence.

**Exploratory addendum (not pre-registered; `analysis/explore_x13_pooled.py`, written after X3's results).** Pooling
X1 and X3 (seeds 82–121) for the configurations they share: S − B without clipping +0.126 [+0.065, +0.188], with
per-group clipping +0.244 [+0.193, +0.295], difference +0.118 [+0.049, +0.187]. The cohorts disagree (per-group minus
no clip ≈ 0.20 on X1's seeds, 0.04 on X3's). Reading: clipping is not necessary for the shift and may enlarge it; its
contribution is unresolved.

## X4 (pre-registered 2026-11-18/19, frozen 19:08:43Z before the gate; analysed with the frozen `phase3_x4.py`)
1,920 of 1,920 runs complete (Slurm arrays 12064 and 12065; gate 12063 passed all four checks). Separate actor and
critic Adam, no clipping, raw rewards, arms B and S, the E1 grid, fresh seeds 122–161 (forty). Parts: D-plain (E7's
plain critic), D-pop (E7's PopArt critic), M (plain critic with output fixed-scaled, V = c·h(x), c = 1 for B, 7 for S;
called the "fixed-scale head" in the paper to avoid a clash with E7's control M). Output:
`results/phase3_x4_report.txt`, `results/X4_contrasts.csv`, `results/X4_advantage_quality.csv`,
`results/X4_predictions.csv`.

| Quantity | Estimate | Jackknife-t 95% | Jackknife-t 90% |
|---|---|---|---|
| S − B, D-plain | +0.143 | [+0.090, +0.195] | [+0.099, +0.186] |
| S − B, D-pop | +0.010 | [−0.040, +0.060] | [−0.032, +0.051] |
| S − B, M | +0.012 | [−0.033, +0.057] | [−0.026, +0.049] |
| change M − D-plain | −0.131 | [−0.175, −0.087] | [−0.168, −0.094] |
| change D-pop − D-plain | −0.133 | [−0.212, −0.055] | [−0.198, −0.068] |

- **Predictions (all met):**
  - **P1:** S − B under D-plain excludes zero.
  - **P2:** under D-plain S's advantages are worse on all four measures. Value error S/B 1.45, GAE error 1.17, sign
    disagreement 1.20, rank agreement −0.076.
  - **P3:** under D-pop all four S/B ratios lie within 1 ± 0.10 (each within 0.006 of 1).
  - **P4:** under M the fit gap (EV against Monte-Carlo returns) is −0.004, with 90% interval [−0.009, −0.000],
    against −0.261 under D-plain.
  - **P5:** the change M − D-plain is below zero, and S − B under M lies within ±0.10 (90%).
  - **P6:** under D-pop the fit gap is +0.001, with 90% interval [−0.004, +0.005].
- **Declared readings:**
  - **Channel (M):** fixing the critic's output scale removes the shift; the critic's target scale carries it at this
    budget.
  - **H-fit (D-pop):** consistent with H-fit; the shift is equivalent to zero within ±0.10 with the fit equalized. This
    is not proof, because PopArt changes more than the fit.
- **Descriptive:** M arm B equals D-plain arm B run by run (label, profit and X4 fit identical in all 320 pairs), as the gate requires.
- **Not used:** the single-seed calibration on seed 43, in which S still collapsed under PopArt, was declared
  descriptive only. On forty seeds PopArt's shift is +0.010.
- **Exploratory only (not pre-declared):** under D-plain, S's in-sample EV after the update, against SB3's own returns,
  exceeds B's (+0.295), while its EV against Monte-Carlo returns is far lower. That measure is self-referential, which
  is why it was rejected as the manipulation check.

## Verdict after X4
The symmetric-scaling shift in the no-clip SB3 learner runs through the scale of the critic's regression targets.
Removing the scale, with a fixed-scale head or with PopArt, removes the shift on forty fresh seeds (equivalent to zero
within ±0.10). With the plain critic, S fits the returns worse and builds worse advantages. X4 does not isolate the
critic's fit as the mediator: both interventions that equalize the fit also remove the target scale.

## Revision v6 (19 Nov 2026): jackknife-t as the reported interval throughout (not pre-registered)
`analysis/jackknife_all.py` re-executes the frozen E1–E12 analysis scripts with their bootstrap weight matrices
replaced by leave-one-seed-out weights. It writes `results_jt/`, with `ci_*` holding the jackknife-t interval and
`pct_*` the pre-declared percentile interval. `results/` is unchanged.

- **Agreement check.** The jackknife-t intervals agree exactly with `jackknife_v5.py` wherever both compute one;
  `render_v4.py` asserts this.
- **Verdict differences.** Sixteen reported zero-exclusion verdicts differ between the two procedures. In every case
  the percentile interval excludes zero and the jackknife-t interval does not (paper Table `tab:pct_sensitivity`).
  The new ones are:
  - the change in S − B from removing the clip (E1/E6, −0.197);
  - the Adam-ε change in A − S (E2);
  - two non-identified E4 curvature contrasts;
  - eight non-primary E5 checkpoints. E5's A − S now excludes zero at 18/24 checkpoints, against 23/24 before.
- **Already reported as calibration-sensitive:** the E8, E9, E10 and E11 changes.
- **Other effects:**
  - E7's plain-critic A − M now clearly includes zero (it touched zero before).
  - E8's plain-critic A − M_on excludes zero at 95% but not at 99%.
- **Pre-declared verdicts are unchanged.** They are judged with the intervals they were declared with; the paper
  says where the two differ.
