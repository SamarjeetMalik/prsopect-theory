"""E5 (v4 pre-declared): decoupled horizon cohort.

The E1 separate_no_clip / VecNormalize cell (disjoint actor and critic Adam optimizers, no gradient clipping, online
reward normalization) for arms B, A, S, trained to 3.6M requested agent transitions on the nine E3 mixture cells and
evaluated every 150k. Construction, reference point, arm scales, normalization and evaluation are those of
optimizer_factorial.py; only the arm subset, grid, budget and checkpoint list differ.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import torch
from stable_baselines3.common.vec_env import VecNormalize

import optimizer_factorial as of
import scale_control as sc
from instrumented_ppo import RolloutDiagnostics

ARMS = ["B", "A", "S"]
GRID = [0.2, 0.4, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 1.0]
SEEDS = list(range(42, 52))
BUDGET = 3_600_000
CHECKPOINTS = list(range(150_000, BUDGET + 1, 150_000))


def decode(index):
    assert 0 <= index < len(ARMS) * len(GRID) * len(SEEDS)
    seed = SEEDS[index % len(SEEDS)]
    q = index // len(SEEDS)
    w = GRID[q % len(GRID)]
    arm = ARMS[q // len(GRID)]
    return arm, w, seed


def create(out, index, budget=BUDGET, checkpoints=CHECKPOINTS):
    arm, w, seed = decode(index)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    vec = sc.vector(w, arm, seed, of.arm_scale(arm), of.REFERENCE)
    vec = VecNormalize(vec, norm_obs=False, norm_reward=True, gamma=0.99, clip_reward=10.0, epsilon=1e-8)
    model = of.FactorialSeparatePPO("MlpPolicy", vec, n_steps=128, batch_size=256, seed=seed, verbose=0, device="cpu")
    model.diagnostics_dir = str(out / "diagnostics")
    model.output = str(out)
    model.started = time.monotonic()
    model.completed = []
    model.protocol = dict(
        design="e5-decoupled-horizon-v4",
        index=index,
        optimizer_mode="separate_no_clip",
        reward_mode="vecnormalize",
        arm=arm,
        arm_scale=of.arm_scale(arm),
        w=w,
        seed=seed,
        condition="lambda7" if arm == "A" else "curvature",
        reference=of.REFERENCE,
        checkpoints=list(checkpoints),
        total_budget=budget,
        phase="confirmation",
    )
    model.verify_upstream()
    (out / "protocol.json").write_text(json.dumps(model.protocol, indent=2) + "\n")
    here = Path(__file__)
    sources = [here, here.with_name("optimizer_factorial.py"), here.with_name("instrumented_ppo.py"), here.with_name("scale_control.py")]
    (out / "manifest.json").write_text(json.dumps({"sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}, indent=2) + "\n")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--index", required=True, type=int)
    ap.add_argument("--smoke", action="store_true", help="short run with two checkpoints, for pipeline checks only")
    ap.add_argument("--check300k", action="store_true", help="300k run with one checkpoint, for the pre-launch equality gate against E1")
    a = ap.parse_args()
    budget, cks = (2560, [1280, 2560]) if a.smoke else ((300_000, [300_000]) if a.check300k else (BUDGET, CHECKPOINTS))
    model = create(a.out, a.index, budget, cks)
    model.learn(total_timesteps=budget, callback=RolloutDiagnostics())
    Path(a.out, "COMPLETE.json").write_text(json.dumps({"requested": budget, "actual_transitions": model.num_timesteps}, indent=2) + "\n")
    print("COMPLETE", a.index, model.num_timesteps, flush=True)


if __name__ == "__main__":
    main()
