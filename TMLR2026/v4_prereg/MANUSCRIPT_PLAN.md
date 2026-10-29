# v4 manuscript plan (after E1–E4)

## Framing
The paper's claim changes from "the evidence does not isolate asymmetry" (v3) to this:

> Under the default PPO configuration, critic-dominated global clipping makes symmetric reward scaling move the
> crossing as far as a loss multiplier. Removing that coupling eliminates the symmetric effect, while the
> loss-only multiplier still moves the crossing, in two environments.

Working title: **"Optimizer Coupling Masks Loss-Asymmetric Shifts in Coordination-Collapse Crossings of
Reward-Mixed Cooperative MARL"**. Keep "finite-budget" in the abstract.

**Contributions.**
1. The measurement protocol, unchanged.
2. A pre-registered optimizer × normalisation factorial identifying the scale artifact and its route: critic
   target magnitude (removed by VecNormalize), not Adam's ε.
3. The asymmetric effect survives decoupling:
   - supply chain: A − S +0.17 to +0.24;
   - LBF: A − S +0.42 to +0.48, under both optimizers.
4. A single-platform refined-horizon cohort:
   - the λ7 − curvature shift is positive at every budget but shrinks (+0.29 → +0.12);
   - the crossings of null and curvature are non-monotone in T;
   - no condition is stable by 3.6M;
   - collapse is not absorbing.
5. Label-level cross-platform reproducibility, 92%.

**Mandatory caveats:**
- implementation-specific (SB3 2.8.0, one optimizer family);
- the baseline crossing itself is optimizer-dependent (B: 0.54 → 0.31–0.45);
- in LBF the asymmetric effect works through the effort-cost incentive;
- E4 P1 and P3 failed (crossings at the grid edge; no critic domination in LBF);
- the ε hypothesis failed;
- deviation D1 and the two analysis-code fixes.

## Section changes

| Section | Change |
|---|---|
| Abstract, §1, §8 | Rewrite around the framing above, with no hand-typed numbers. |
| §3 Method | Add the x_i = r_i + w Σ_{j≠i} r_j decomposition. State the SB3 configuration (one Adam step, ε = 1e-5, global clip 0.5, separate actor/critic MLPs) and define the four optimizer × reward cells. Specify S̄ once; give the isotonic convention, z₀ calibration, s_in/s_out per control, and budget arithmetic (5 lanes × 128 = 640; 300k → 300,160). |
| §5 Horizon | Replace with E3: 9-cell grid, 270 trajectories, absolute w50(T), λ7 − curvature at the four budgets, stability ±.05, recovery/relapse, cross-platform agreement. Drop the v3 degenerate-interval discussion, since the refined grid removed it. |
| §6 Transforms | Keep the 300k decomposition (Table 2 legacy sweep). Replace the three-arm control with E1/E2: a 5-cell × 4-arm crossing table, contrast table, interaction table, and the diagnostics figure (post-clip actor norm, actor energy share). |
| New §7 LBF | E4 crossings and contrasts. State that the critic-domination pathway is absent, and that P1/P3 failed and why. |
| Limitations | One section: budget, label, two environments, one optimizer family, action representation, optimizer dependence of absolute w50, platform non-bit-reproducibility. |
| Appendices | Per-cell k/n tables for E1–E4; the pre-registration and deviations summary; compute (≈1,870 + 800 runs); the cross-platform tables. |
| All Tier 2–5 items | From `FIX_PLAN_v4_2026-10-23.md` (dangling references, unimodality n/power, copy-edits, BibTeX bracing, the 1,005.85 sum). |

## Mechanics
- Add `render_v4.py` and `export_v4_macros.py`, reading `v4_prereg/results/*.csv` and writing
  `TMLR2026/generated/v4_*.tex`.
- Hook them into `rebuild_tmlr.py`. No hand-typed numbers.
- **Artifact size.** Raw E1–E4 runs are ≈45 GB (checkpoints and per-minibatch diagnostics), too large for a
  supplement. Ship instead:
  - the extracted per-run CSVs (≈2 MB);
  - the prereg, hashes and source;
  - the extraction scripts.

  Host the raw runs on an anonymised archive (e.g. an anonymous Zenodo or OSF deposit) only if reviewers ask.
- **Anonymity.** Scrub `PHASE0_REPORT.md`, `HASHES.sha256` comments and the logs of the Ashoka username and paths
  before packaging. grep for "samarjeet", "ashoka", "debayanlab", "PuffBear" and "Agriya".
- **Verification.** Do a clean-room rebuild from the new artifact and independently re-rate the result.
