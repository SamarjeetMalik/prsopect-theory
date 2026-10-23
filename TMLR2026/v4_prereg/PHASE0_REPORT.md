# Phase 0 report (2026-10-23)

Branch `tmlr-v4` from `origin/codex/tmlr-revision-2026-10-15` @ 46ef14a. Local clone:
`~/Downloads/Projects/coordcollapse-v4`. Remote working dir: `/scratch/samarjeet_malik/coordcollapse_v4`.

## Environment (Ashoka, Slurm)
- Env `~/.local/share/mamba/envs/cc4`, pinned to the archived runs' `manifest.json`:
  - Python 3.11.13
  - numpy 1.26.4
  - torch 2.6.0 (CPU wheel)
  - stable-baselines3 2.8.0
  - supersuit 3.11.0
  - pettingzoo 1.26.1
  - gymnasium 1.2.3
  - pandas 1.5.3
  - scipy 1.16.3
  - cloudpickle 3.1.2
- Analysis env (local): `AAMAS2027/analysis/requirements.lock` on Python 3.12.
- **Compute cap.** QoS `standard`, MaxTRESPerUser cpu=16, gres/gpu=2, mem=72G; partition max 48 h.
  - Slurm allocates whole cores (2 hyperthreads), so at most **8 single-thread trajectories run concurrently**.
  - Packing two processes per core gave no throughput gain: about 990 s per run packed, against about 490 s per
    run unpacked, per core.
- **Run cost.** One 300k supply trajectory takes ~490 s wall on Ashoka (~0.136 h). The archived rate was 0.256
  CPU-h.

## Tests
PASS:
- `test_instrumentation.py` (the instrumented PPO leaves learning bit-identical);
- `test_boundary.py`, `test_checkpoint_recovery.py`, `test_evaluation_adapter.py`, `test_pilot_grid_gate.py`,
  `test_reward_diagnostics.py`;
- `test_scale_confirmation.py`. It runs only in the artifact layout; it passed locally there.

## Rebuild of v3
`python AAMAS2027/analysis/rebuild_tmlr.py` on a fresh unpack of the artifact reproduces v3 **exactly**:
- every generated `.tex` table is byte-identical;
- the PDF text diff against `~/Downloads/main (3).pdf` is 0 lines;
- 0 undefined references.

## Bug found in the frozen factorial (deviation D1)
`optimizer_factorial.create()` omitted `protocol['condition']`. Every run crashed at the 300k evaluation, after
training. Fixed in v2.1; see `DEVIATIONS.md`. The frozen design was never smoke-tested before this.

## Equality gate (arm A, seed 42, 8 cells, global_clip/raw, factorial v2.1 on Ashoka vs the archive)
- Labels are identical in 8/8 cells.
- Bit-exact in the 6 collapsed cells (w ≤ .7, lazy floor −128.1). They differ at w = .85 (mean_S 0 vs .673;
  both collapsed) and w = 1 (profit 159.7 vs 461.0; both non-collapsed).
- Attribution check at w = 1:
  - the original executed source on Ashoka is **bit-identical** to factorial v2.1 on Ashoka, so the refactor
    preserves behaviour;
  - a repeat of v2.1 on Ashoka is bit-identical, so Ashoka is deterministic;
  - both differ from the archive. **The divergence is cross-platform** (authors' PBS hardware/torch build vs
    Ashoka), not code.

## Consequences
1. **E1.** Do not reuse the 240 archived global_clip/raw B/A/S cells. Rerun all 1,280 on Ashoka, so every
   factorial contrast is single-platform. Bonus: those 240 reruns give a full cross-platform label-agreement
   statistic against the archive, to report as reproducibility evidence.
2. **E3.** The 150 archived horizon trajectories are from the other platform. Mixing them with new Ashoka cells
   in one fit would confound platform with w. Either rerun the 150 on Ashoka (~245 run-hours, ~1.3 days at 8
   concurrent), or keep the refined cells as a separate same-platform cohort. **Needs the user's decision.**
3. The paper's v4 text should state this. Learned policies are not bit-reproducible across hardware/BLAS builds;
   collapse labels were (8/8 here, full 240 after E1). Reproducibility claims are made at the label level.
