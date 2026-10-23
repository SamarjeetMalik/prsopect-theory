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
