"""E2 (v4 pre-declared addendum): global clipping, raw reward, Adam eps = 1e-8 instead of SB3's 1e-5.

Isolates Adam's epsilon from gradient clipping: same arms, grid, seeds, reference and budget as the
optimizer factorial v2.1 (global_clip / raw cells); only the optimizer epsilon differs.
"""

from pathlib import Path
import argparse
import hashlib
import json
import time

import torch

import optimizer_factorial as of
import scale_control as sc
from instrumented_ppo import RolloutDiagnostics

EPS = 1e-8


def decode(index):
    n = len(of.ARMS) * len(of.GRID) * len(of.SEEDS)
    assert 0 <= index < n
    seed = of.SEEDS[index % len(of.SEEDS)]
    q = index // len(of.SEEDS)
    w = of.GRID[q % len(of.GRID)]
    arm = of.ARMS[q // len(of.GRID)]
    return arm, w, seed


def create(out, index, budget=300_000):
    arm, w, seed = decode(index)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    vec = sc.vector(w, arm, seed, of.arm_scale(arm), of.REFERENCE)
    model = of.FactorialGlobalPPO("MlpPolicy", vec, n_steps=128, batch_size=256, seed=seed, verbose=0, device="cpu",
                                  policy_kwargs=dict(optimizer_kwargs=dict(eps=EPS)))
    eps_used = model.policy.optimizer.defaults["eps"]
    assert eps_used == EPS, eps_used
    model.diagnostics_dir = str(out / "diagnostics")
    model.output = str(out)
    model.started = time.monotonic()
    model.completed = []
    model.protocol = dict(
        design="eps-addendum-v4",
        index=index,
        optimizer_mode="global_clip",
        reward_mode="raw",
        adam_eps=eps_used,
        arm=arm,
        arm_scale=of.arm_scale(arm),
        w=w,
        seed=seed,
        condition="lambda7" if arm == "A" else "curvature",
        reference=of.REFERENCE,
        checkpoints=[budget],
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
