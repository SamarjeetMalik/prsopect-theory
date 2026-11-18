"""X3 (v5, pre-declared): which clipped group carries the symmetric-scaling shift?

A 2 x 2 factorial under one Adam step on the supply chain with raw rewards: the actor's gradient is clipped to 0.5 or
not, and the critic's gradient is clipped to 0.5 or not, each group separately:
  both          actor and critic clipped separately (= E6/X1 groupwise_clip)
  actor_only    actor clipped, critic unclipped
  critic_only   critic clipped, actor unclipped
  neither       no clipping, one Adam step
Arms B and S, the E1 grid, fresh seeds 102-121, 300k. Logging is X1's (E6's per-step norms, the actor's Adam moments)
plus the same three Adam quantities for the critic. Logging only reads tensors and optimizer state. Nothing here
modifies an existing source file.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import torch

import optimizer_factorial as of
import scale_control as sc
import e6_clip_isolation as e6
import x1_x2_runner as x12
from instrumented_ppo import GroupwiseClipPPO, RolloutDiagnostics

DESIGN = json.loads(Path(__file__).with_name("x3_design.json").read_text())
SEEDS, MODES, ARMS, GRID = DESIGN["seeds"], DESIGN["modes"], DESIGN["arms"], of.GRID
N = len(MODES) * len(ARMS) * len(GRID) * len(SEEDS)            # 1280
CLIP = {"both": (True, True), "actor_only": (True, False), "critic_only": (False, True), "neither": (False, False)}


class FactorialClipPPO(GroupwiseClipPPO):
    """One Adam step; the actor and critic groups are each clipped to max_grad_norm or left unclipped."""
    clip_actor, clip_critic = True, True

    def clip_gradients(self):
        groups = self.parameter_groups(); out = dict(clip_mode=f"actor={self.clip_actor},critic={self.clip_critic}")
        for g, on in [("actor", self.clip_actor), ("critic", self.clip_critic)]:
            if on:
                pre = float(torch.nn.utils.clip_grad_norm_(groups[g], self.max_grad_norm))
            else:
                pre = float(torch.sqrt(sum(torch.sum(p.grad.detach() ** 2) for p in groups[g] if p.grad is not None)))
            out[f"{g}_clip_pre_norm"] = pre; out[f"{g}_group_clipped"] = bool(on and pre > self.max_grad_norm)
        return out


class CriticAdamLogging:
    """Adds the critic's Adam state to the per-step record (after X1's actor fields)."""

    def step_optimizers(self):
        super().step_optimizers()
        critic = [p for n, p in self.policy.named_parameters() if p.requires_grad and e6.group_of(n) == "critic"]
        opt = self.policy.optimizer; b1, b2 = opt.param_groups[0]["betas"]; eps = opt.param_groups[0]["eps"]; m2 = v2 = d2 = 0.0
        for p in critic:
            st = opt.state.get(p)
            if st:
                t = float(st["step"]); mh = st["exp_avg"].detach() / (1 - b1 ** t); vh = st["exp_avg_sq"].detach() / (1 - b2 ** t)
                m2 += float(torch.sum(mh ** 2)); v2 += float(torch.sum(vh)); d2 += float(torch.sum((mh / (vh.sqrt() + eps)) ** 2))
        self._update_log[-1].update(critic_adam_m_norm=float(np.sqrt(m2)), critic_adam_sqrt_v_norm=float(np.sqrt(v2)),
                                    critic_adam_direction_norm=float(np.sqrt(d2)))


def make_class(mode):
    ca, cc = CLIP[mode]
    base = type(f"X3Clip_{mode}", (FactorialClipPPO,), dict(clip_actor=ca, clip_critic=cc))
    return type(f"X3_{mode}", (sc.CheckpointPPO, CriticAdamLogging, x12.AdamLogging, e6.UpdateLogging, base), {})


def decode(index):
    assert 0 <= index < N
    seed = SEEDS[index % len(SEEDS)]; q = index // len(SEEDS); w = GRID[q % len(GRID)]; q //= len(GRID)
    return dict(mode=MODES[q // len(ARMS)], arm=ARMS[q % len(ARMS)], w=w, seed=seed)


def run(spec, out, budget, index):
    out = Path(out); out.mkdir(parents=True, exist_ok=False); t0 = time.monotonic(); torch.set_num_threads(1)
    arm, w, seed = spec["arm"], spec["w"], spec["seed"]
    vec = sc.vector(w, arm, seed, of.arm_scale(arm), of.REFERENCE)
    model = make_class(spec["mode"])("MlpPolicy", vec, n_steps=128, batch_size=256, seed=seed, verbose=0, device="cpu")
    model.parameter_groups()
    model.diagnostics_dir = str(out / "diagnostics"); model.output = str(out); model.started = time.monotonic(); model.completed = []
    model.protocol = dict(design="x3-v5", index=index, experiment="X3", mode=spec["mode"], clip_actor=CLIP[spec["mode"]][0],
                          clip_critic=CLIP[spec["mode"]][1], optimizer_mode=f"x3_{spec['mode']}", reward_mode="raw", arm=arm,
                          arm_scale=of.arm_scale(arm), w=w, seed=seed, condition="lambda7" if arm == "A" else "curvature",
                          reference=of.REFERENCE, checkpoints=[budget], total_budget=budget, phase="confirmation")
    model.verify_upstream()
    (out / "protocol.json").write_text(json.dumps(model.protocol, indent=2) + "\n")
    here = Path(__file__)
    srcs = [here, here.with_name("x3_design.json"), here.with_name("x1_x2_runner.py"), here.with_name("e6_clip_isolation.py"),
            here.with_name("optimizer_factorial.py"), here.with_name("instrumented_ppo.py"), here.with_name("scale_control.py"),
            here.with_name("horizon_supply.py")]
    (out / "manifest.json").write_text(json.dumps({"sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in srcs},
                                                   "torch": torch.__version__, "numpy": np.__version__}, indent=2) + "\n")
    model.learn(total_timesteps=budget, callback=RolloutDiagnostics())
    (out / "COMPLETE.json").write_text(json.dumps({"requested": budget, "actual_transitions": model.num_timesteps,
                                                   "elapsed_seconds": time.monotonic() - t0}, indent=2) + "\n")
    print("COMPLETE", index, model.num_timesteps, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--index", type=int)
    ap.add_argument("--budget", type=int, default=300_000)
    ap.add_argument("--mode"); ap.add_argument("--arm"); ap.add_argument("--w", type=float); ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    spec = decode(a.index) if a.index is not None else dict(mode=a.mode, arm=a.arm, w=a.w, seed=a.seed)
    run(spec, a.out, a.budget, a.index if a.index is not None else -1)


if __name__ == "__main__":
    main()
