# Deviations from the frozen optimizer-factorial v2 design

## D1 (2026-10-23, before any outcome was observed): missing `condition` key
`optimizer_factorial.create()` built `model.protocol` without the `condition` key that
`horizon_supply.evaluate()` reads at the 300k checkpoint. All 8 equality-gate trajectories (Ashoka job 185,
indices 80, 90, …, 150) trained to 300,160 transitions and then raised `KeyError: 'condition'` before writing an
evaluation or a checkpoint, so no label was produced.

**Fix (v2.1).** Add `condition="lambda7" if arm == "A" else "curvature"`, the mapping used by the completed direct
control in `scale_control.create()`. The evaluation reads only native profit and base-stock, so the condition
affects nothing but the evaluation environment's construction, exactly as in the archived control.

**Effect.** The source hash of `optimizer_factorial.py` changes. The design (arms, grid, seeds, reference, budget,
analysis) is unchanged.

## D4 (found 2026-10-30, after E4 analysis): E4 labels used the placeholder threshold
`env/lbf_wrapper.is_collapsed_lbf()` loads θ from `docs/exp_lbf/lbf_collapse_threshold.json`. That file is absent from
the frozen source, so it silently fell back to its placeholder `_DEFAULT_THRESHOLD = 0.5`. All 800 E4 `collapsed`
labels therefore use θ = 0.5, not the pre-declared θ = 0.626.

**Fix.** Every run stores its team return, so labels are recomputed at θ = 0.626 from `E4_runs.csv` (no rerun). The
θ = 0.5 analysis is kept as a sensitivity result. At θ = 0.626, A − S is +0.447 [+0.312, +0.559] (global clip) and
+0.490 [+0.315, +0.619] (separate); at θ = 0.5 it is +0.418 and +0.479; at θ = 0.75 it is +0.370 and +0.447.
