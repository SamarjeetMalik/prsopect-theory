"""X1 and X2 (v5, pre-declared): fresh-seed replication of the clipping intervention, with optimizer-state logging.

X1 (indices 0-959): E6's design (SB3 PPO, raw rewards, one Adam step with a global clip of 0.5 / actor and critic
clipped to 0.5 separately / disjoint Adam optimizers without clipping), arms B and S, the E1 grid, fresh seeds
82-101, 300k. On top of E6's per-step logging (applied gradient and parameter-update norms per group) every optimizer
step also records, for the actor's parameters, the norms of Adam's bias-corrected first moment, of the square root of
its bias-corrected second moment, and of the resulting step direction m_hat / (sqrt(v_hat) + eps). The per-minibatch
shared-clip factor kappa = min(1, tau / ||g||) follows from the total norm that InstrumentedPPO already logs. All
logging only reads tensors and optimizer state; the gate checks bit-identity against E6.

X2 (indices 960-1599): E11's CleanRL-style PPO, raw rewards, one global clip vs actor and critic clipped separately,
arms B and S, the E1 grid, seeds 82-101, 300k. It calls e11_cleanrl.train unchanged.

Nothing here modifies an existing source file.
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
import e11_cleanrl as e11
from instrumented_ppo import InstrumentedPPO, GroupwiseClipPPO, SeparateNoClipPPO, RolloutDiagnostics

DESIGN = json.loads(Path(__file__).with_name("x1_x2_design.json").read_text())
SEEDS, GRID = DESIGN["seeds"], of.GRID
X1_MODES, X2_CONDS, ARMS = DESIGN["x1_modes"], DESIGN["x2_conditions"], DESIGN["arms"]
N1 = len(X1_MODES) * len(ARMS) * len(GRID) * len(SEEDS)        # 960
N2 = len(X2_CONDS) * len(ARMS) * len(GRID) * len(SEEDS)        # 640


class AdamLogging:
    """Adds the actor's Adam state, read after each step, to E6's per-step record."""

    def _actor_optimizer_state(self):
        actor = [p for n, p in self.policy.named_parameters() if p.requires_grad and e6.group_of(n) == "actor"]
        opts = [self.actor_optimizer] if hasattr(self, "actor_optimizer") else [self.policy.optimizer]
        out = []
        for opt in opts:
            b1, b2 = opt.param_groups[0]["betas"]; eps = opt.param_groups[0]["eps"]
            for p in actor:
                st = opt.state.get(p)
                if st:
                    out.append((st["exp_avg"].detach(), st["exp_avg_sq"].detach(), float(st["step"]), b1, b2, eps))
        return out

    def step_optimizers(self):
        super().step_optimizers()
        m2 = v2 = d2 = 0.0
        for m, v, t, b1, b2, eps in self._actor_optimizer_state():
            mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
            m2 += float(torch.sum(mh ** 2)); v2 += float(torch.sum(vh)); d2 += float(torch.sum((mh / (vh.sqrt() + eps)) ** 2))
        self._update_log[-1].update(actor_adam_m_norm=float(np.sqrt(m2)), actor_adam_sqrt_v_norm=float(np.sqrt(v2)),
                                    actor_adam_direction_norm=float(np.sqrt(d2)))


class X1Global(sc.CheckpointPPO, AdamLogging, e6.UpdateLogging, InstrumentedPPO):
    pass


class X1Groupwise(sc.CheckpointPPO, AdamLogging, e6.UpdateLogging, GroupwiseClipPPO):
    pass


class X1Separate(sc.CheckpointPPO, AdamLogging, e6.UpdateLogging, SeparateNoClipPPO):
    pass


CLASSES = {"global_clip": X1Global, "groupwise_clip": X1Groupwise, "separate_no_clip": X1Separate}


def decode(index):
    """0..959: X1 mode x arm x w x seed; 960..1599: X2 condition x arm x w x seed (seed fastest)."""
    assert 0 <= index < N1 + N2
    i = index if index < N1 else index - N1
    seed = SEEDS[i % len(SEEDS)]; q = i // len(SEEDS); w = GRID[q % len(GRID)]; q //= len(GRID); arm = ARMS[q % len(ARMS)]
    if index < N1:
        return dict(exp="X1", mode=X1_MODES[q // len(ARMS)], arm=arm, w=w, seed=seed)
    return dict(exp="X2", mode=X2_CONDS[q // len(ARMS)], arm=arm, w=w, seed=seed)


def manifest(out, extra):
    here = Path(__file__)
    srcs = [here, here.with_name("x1_x2_design.json"), here.with_name("e6_clip_isolation.py"), here.with_name("e11_cleanrl.py"),
            here.with_name("optimizer_factorial.py"), here.with_name("instrumented_ppo.py"), here.with_name("scale_control.py"),
            here.with_name("horizon_supply.py")]
    (out / "manifest.json").write_text(json.dumps({"sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in srcs},
                                                   "torch": torch.__version__, "numpy": np.__version__, **extra}, indent=2) + "\n")


def run(spec, out, budget, index):
    out = Path(out); out.mkdir(parents=True, exist_ok=False); t0 = time.monotonic()
    arm, w, seed = spec["arm"], spec["w"], spec["seed"]
    proto = dict(design="x1-x2-v5", index=index, experiment=spec["exp"], mode=spec["mode"], arm=arm, arm_scale=of.arm_scale(arm), w=w,
                 seed=seed, condition="lambda7" if arm == "A" else "curvature", reference=of.REFERENCE, checkpoints=[budget],
                 total_budget=budget, phase="confirmation")
    if spec["exp"] == "X1":
        torch.set_num_threads(1)
        vec = sc.vector(w, arm, seed, of.arm_scale(arm), of.REFERENCE)
        model = CLASSES[spec["mode"]]("MlpPolicy", vec, n_steps=128, batch_size=256, seed=seed, verbose=0, device="cpu")
        if spec["mode"] == "groupwise_clip":
            model.parameter_groups()
        model.diagnostics_dir = str(out / "diagnostics"); model.output = str(out); model.started = time.monotonic(); model.completed = []
        model.protocol = dict(proto, optimizer_mode=spec["mode"], reward_mode="raw")
        model.verify_upstream()
        (out / "protocol.json").write_text(json.dumps(model.protocol, indent=2) + "\n"); manifest(out, {})
        model.learn(total_timesteps=budget, callback=RolloutDiagnostics())
        comp = {"requested": budget, "actual_transitions": model.num_timesteps}
    else:
        (out / "protocol.json").write_text(json.dumps(dict(proto, condition_name=spec["mode"]), indent=2) + "\n"); manifest(out, {})
        steps, summ = e11.train(spec["mode"], arm, w, seed, budget, out)
        comp = {"requested": budget, "actual_transitions": steps, "diagnostics": summ}
    comp["elapsed_seconds"] = time.monotonic() - t0
    (out / "COMPLETE.json").write_text(json.dumps(comp, indent=2) + "\n")
    print("COMPLETE", index, comp["actual_transitions"], flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--index", type=int)
    ap.add_argument("--budget", type=int, default=300_000)
    ap.add_argument("--exp"); ap.add_argument("--mode"); ap.add_argument("--arm"); ap.add_argument("--w", type=float); ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    spec = decode(a.index) if a.index is not None else dict(exp=a.exp, mode=a.mode, arm=a.arm, w=a.w, seed=a.seed)
    run(spec, a.out, a.budget, a.index if a.index is not None else -1)


if __name__ == "__main__":
    main()
