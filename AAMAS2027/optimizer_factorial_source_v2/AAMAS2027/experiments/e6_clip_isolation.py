"""E6 (v4 pre-declared): clipping isolation with parameter-update logging.

Optimizer modes on the supply chain, raw transformed rewards, arms B/A/S, the E1 grid and seeds, 300k:
  global_clip       one Adam step, one global clip of 0.5 over actor and critic (SB3 default; E1 rerun with logging)
  groupwise_clip    one Adam step, actor and critic gradients clipped to 0.5 separately (new)
  separate_no_clip  disjoint actor/critic Adam optimizers, no clipping (E1 rerun with logging)
Every optimizer step additionally records, per parameter group, the applied (post-clip) gradient norm and the
norm of the actual parameter change made by Adam. Logging only reads tensors, so learning is unchanged; the
pre-launch gate checks this bit-for-bit against E1. Existing source files are not modified (E5 hash-checks them).
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import time

import numpy as np
import torch

import optimizer_factorial as of
import scale_control as sc
from instrumented_ppo import InstrumentedPPO, GroupwiseClipPPO, SeparateNoClipPPO, RolloutDiagnostics

ARMS = ["B", "A", "S"]
OPTIMIZERS = ["global_clip", "groupwise_clip", "separate_no_clip"]
GRID = of.GRID
SEEDS = of.SEEDS


def group_of(name):
    if "value_net" in name:
        return "critic"
    if "policy_net" in name or "action_net" in name or "log_std" in name:
        return "actor"
    return "shared"


class UpdateLogging:
    """Records per-group applied-gradient and parameter-update norms around each optimizer step."""

    def step_optimizers(self):
        named = [(group_of(n), p) for n, p in self.policy.named_parameters() if p.requires_grad]
        before = [p.detach().clone() for _, p in named]
        grad = {"actor": 0.0, "critic": 0.0, "shared": 0.0}
        for g, p in named:
            if p.grad is not None:
                grad[g] += float(torch.sum(p.grad.detach() ** 2))
        super().step_optimizers()
        upd = {"actor": 0.0, "critic": 0.0, "shared": 0.0}
        for (g, p), b in zip(named, before):
            upd[g] += float(torch.sum((p.detach() - b) ** 2))
        self._update_log.append({**{f"applied_{g}_grad_norm": float(np.sqrt(v)) for g, v in grad.items()},
                                 **{f"{g}_update_norm": float(np.sqrt(v)) for g, v in upd.items()}})

    def train(self):
        self._update_log = []
        super().train()
        path = Path(self.diagnostics_dir) / "param_updates.jsonl.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "at") as f:
            f.write(json.dumps(dict(actual_transitions=self.num_timesteps, n_updates=self._n_updates, steps=self._update_log)) + "\n")


class E6Global(sc.CheckpointPPO, UpdateLogging, InstrumentedPPO):
    pass


class E6Groupwise(sc.CheckpointPPO, UpdateLogging, GroupwiseClipPPO):
    pass


class E6Separate(sc.CheckpointPPO, UpdateLogging, SeparateNoClipPPO):
    pass


CLASSES = {"global_clip": E6Global, "groupwise_clip": E6Groupwise, "separate_no_clip": E6Separate}


def decode(index):
    assert 0 <= index < len(OPTIMIZERS) * len(ARMS) * len(GRID) * len(SEEDS)
    seed = SEEDS[index % len(SEEDS)]
    q = index // len(SEEDS)
    w = GRID[q % len(GRID)]
    q //= len(GRID)
    arm = ARMS[q % len(ARMS)]
    opt = OPTIMIZERS[q // len(ARMS)]
    return opt, arm, w, seed


def create(out, index, budget=300_000):
    opt, arm, w, seed = decode(index)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    vec = sc.vector(w, arm, seed, of.arm_scale(arm), of.REFERENCE)
    model = CLASSES[opt]("MlpPolicy", vec, n_steps=128, batch_size=256, seed=seed, verbose=0, device="cpu")
    if opt == "groupwise_clip":
        model.parameter_groups()  # aborts if any trainable parameter is shared between actor and critic
    model.diagnostics_dir = str(out / "diagnostics")
    model.output = str(out)
    model.started = time.monotonic()
    model.completed = []
    model.protocol = dict(design="e6-clip-isolation-v4", index=index, optimizer_mode=opt, reward_mode="raw", arm=arm,
                          arm_scale=of.arm_scale(arm), w=w, seed=seed, condition="lambda7" if arm == "A" else "curvature",
                          reference=of.REFERENCE, checkpoints=[budget], total_budget=budget, phase="confirmation")
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
    ap.add_argument("--budget", type=int, default=300_000)
    a = ap.parse_args()
    model = create(a.out, a.index, a.budget)
    model.learn(total_timesteps=a.budget, callback=RolloutDiagnostics())
    Path(a.out, "COMPLETE.json").write_text(json.dumps({"requested": a.budget, "actual_transitions": model.num_timesteps}, indent=2) + "\n")
    print("COMPLETE", a.index, model.num_timesteps, flush=True)


if __name__ == "__main__":
    main()
