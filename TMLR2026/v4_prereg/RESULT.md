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
