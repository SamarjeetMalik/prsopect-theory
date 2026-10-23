"""Frozen factorial test of reward scale, normalization, and optimizer coupling."""

from pathlib import Path
import argparse
import hashlib
import json
import time

import torch
from stable_baselines3.common.vec_env import VecNormalize

import horizon_supply as h
import scale_control as sc
from instrumented_ppo import InstrumentedPPO, SeparateNoClipPPO, RolloutDiagnostics


GRID = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]
ARMS = ["B", "A", "S", "R"]
OPTIMIZERS = ["global_clip", "separate_no_clip"]
REWARD_MODES = ["raw", "vecnormalize"]
SEEDS = list(range(42, 52))
REFERENCE = -2.6281398333333335
RMS_MATCH = 30.359 / 8.796


class FactorialGlobalPPO(sc.CheckpointPPO, InstrumentedPPO):
    pass


class FactorialSeparatePPO(sc.CheckpointPPO, SeparateNoClipPPO):
    pass


def decode(index):
    assert 0 <= index < len(OPTIMIZERS) * len(REWARD_MODES) * len(ARMS) * len(GRID) * len(SEEDS)
    seed = SEEDS[index % len(SEEDS)]
    q = index // len(SEEDS)
    w = GRID[q % len(GRID)]
    q //= len(GRID)
    arm = ARMS[q % len(ARMS)]
    q //= len(ARMS)
    reward_mode = REWARD_MODES[q % len(REWARD_MODES)]
    optimizer_mode = OPTIMIZERS[q // len(REWARD_MODES)]
    return optimizer_mode, reward_mode, arm, w, seed


def arm_scale(arm):
    return {"B": 1.0, "A": 1.0, "S": 7.0, "R": RMS_MATCH}[arm]


def create(out, index, budget=300_000):
    optimizer_mode, reward_mode, arm, w, seed = decode(index)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    vec = sc.vector(w, arm, seed, arm_scale(arm), REFERENCE)
    if reward_mode == "vecnormalize":
        vec = VecNormalize(vec, norm_obs=False, norm_reward=True, gamma=0.99, clip_reward=10.0, epsilon=1e-8)
    cls = FactorialGlobalPPO if optimizer_mode == "global_clip" else FactorialSeparatePPO
    model = cls("MlpPolicy", vec, n_steps=128, batch_size=256, seed=seed, verbose=0, device="cpu")
    model.diagnostics_dir = str(out / "diagnostics")
    model.output = str(out)
    model.started = time.monotonic()
    model.completed = []
    model.protocol = dict(
        design="optimizer-factorial-v2",
        index=index,
        optimizer_mode=optimizer_mode,
        reward_mode=reward_mode,
        arm=arm,
        arm_scale=arm_scale(arm),
        w=w,
        seed=seed,
        # Read by horizon_supply.evaluate; same mapping as scale_control.create (v2.1 fix).
        condition="lambda7" if arm == "A" else "curvature",
        reference=REFERENCE,
        checkpoints=[budget],
        total_budget=budget,
        phase="confirmation",
    )
    model.verify_upstream()
    (out / "protocol.json").write_text(json.dumps(model.protocol, indent=2) + "\n")
    sources = [Path(__file__), Path(__file__).with_name("instrumented_ppo.py"), Path(__file__).with_name("scale_control.py")]
    (out / "manifest.json").write_text(json.dumps({"sources": {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}, indent=2) + "\n")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--budget", type=int, default=300_000)
    args = parser.parse_args()
    model = create(args.out, args.index, args.budget)
    model.learn(total_timesteps=args.budget, callback=RolloutDiagnostics())
    Path(args.out, "COMPLETE.json").write_text(json.dumps({"requested": args.budget, "actual_transitions": model.num_timesteps}, indent=2) + "\n")
    print("COMPLETE", args.index, model.num_timesteps, flush=True)


if __name__ == "__main__":
    main()
