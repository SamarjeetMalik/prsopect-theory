"""E9 and E10 (v4, pre-declared; PREREGISTRATION_E9_E10.md).

E9, clip-threshold dose-response. The default learner of E1 (one Adam, one global gradient-norm clip, raw rewards)
with the clip threshold tau raised from 0.5 to {300, 3000, 10000, 30000}. Arms B and S on the E1 grid with seeds
42-51. Raising tau lets first B's and then S's critic-dominated joint gradient escape the clip, so the S/B ratio of
applied actor gradients should rise from 0.12 towards 1. The pre-declared prediction is that S - B falls from its
tau = 0.5 value (+0.357, E1) towards its no-clip value (+0.160, E1 separate_no_clip/raw).

E10, critic-target replication with more seeds. E7's design (separate optimizers without clipping, raw rewards,
plain or PopArt critic), arms B and S, with fresh seeds 62-81.

Nothing here modifies an existing source file. Classes come from optimizer_factorial (E1) and e7_critic_scale (E7).
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
import e7_critic_scale as e7

DESIGN = json.loads(Path(__file__).with_name("e9_e10_design.json").read_text())
TAUS, E10_SEEDS, GRID = DESIGN["e9_taus"], DESIGN["e10_seeds"], of.GRID
N9 = len(TAUS) * 2 * len(GRID) * len(of.SEEDS)               # 640
N10 = 2 * 2 * len(GRID) * len(E10_SEEDS)                      # 640


def decode(index):
    """0..639: E9 tau x arm (B, S) x w x seed (42-51). 640..1279: E10 critic x arm (B, S) x w x seed (62-81)."""
    if index < N9:
        seed = of.SEEDS[index % 10]; q = index // 10; w = GRID[q % len(GRID)]; q //= len(GRID)
        arm = ["B", "S"][q % 2]; tau = TAUS[q // 2]
        return dict(exp="E9", tau=tau, critic=None, arm=arm, w=w, seed=seed)
    i = index - N9
    assert i < N10
    seed = E10_SEEDS[i % len(E10_SEEDS)]; q = i // len(E10_SEEDS); w = GRID[q % len(GRID)]; q //= len(GRID)
    arm = ["B", "S"][q % 2]; critic = e7.CRITIC[q // 2]
    return dict(exp="E10", tau=None, critic=critic, arm=arm, w=w, seed=seed)


def create(out, spec, budget=300_000, index=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    arm, w, seed = spec["arm"], spec["w"], spec["seed"]
    vec = sc.vector(w, arm, seed, of.arm_scale(arm), of.REFERENCE)
    if spec["exp"] == "E9":
        model = of.FactorialGlobalPPO("MlpPolicy", vec, n_steps=128, batch_size=256, seed=seed, verbose=0, device="cpu",
                                      max_grad_norm=float(spec["tau"]))
        opt_mode, reward_mode = "global_clip", "raw"
    else:
        model = e7.CLASSES[spec["critic"]]("MlpPolicy", vec, n_steps=128, batch_size=256, seed=seed, verbose=0, device="cpu")
        opt_mode, reward_mode = "separate_no_clip", "raw"
    model.diagnostics_dir = str(out / "diagnostics"); model.output = str(out); model.started = time.monotonic(); model.completed = []
    model.protocol = dict(design="e9-e10-v4", index=index, experiment=spec["exp"], optimizer_mode=opt_mode, reward_mode=reward_mode,
                          max_grad_norm=spec["tau"], critic_mode=spec["critic"], arm=arm, arm_scale=of.arm_scale(arm), w=w, seed=seed,
                          condition="curvature", reference=of.REFERENCE, checkpoints=[budget], total_budget=budget, phase="confirmation")
    model.verify_upstream()
    (out / "protocol.json").write_text(json.dumps(model.protocol, indent=2) + "\n")
    here = Path(__file__)
    srcs = [here, here.with_name("e9_e10_design.json"), here.with_name("optimizer_factorial.py"), here.with_name("e7_critic_scale.py"),
            here.with_name("e7_design.json"), here.with_name("instrumented_ppo.py"), here.with_name("scale_control.py")]
    (out / "manifest.json").write_text(json.dumps({"sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in srcs}}, indent=2) + "\n")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--index", type=int)
    ap.add_argument("--budget", type=int, default=300_000)
    # Gate-only overrides: run one explicitly specified configuration instead of a design index.
    ap.add_argument("--exp", choices=["E9", "E10"]); ap.add_argument("--tau", type=float); ap.add_argument("--critic")
    ap.add_argument("--arm"); ap.add_argument("--w", type=float); ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    spec = decode(a.index) if a.index is not None else dict(exp=a.exp, tau=a.tau, critic=a.critic, arm=a.arm, w=a.w, seed=a.seed)
    model = create(a.out, spec, a.budget, a.index)
    model.learn(total_timesteps=a.budget, callback=RolloutDiagnostics())
    extra = {}
    if spec["exp"] == "E10" and isinstance(model, e7.E7PopArt):
        model.save_popart(a.out); extra["popart_final"] = model._popart_log[-1] if model._popart_log else None
    Path(a.out, "COMPLETE.json").write_text(json.dumps({"requested": a.budget, "actual_transitions": model.num_timesteps, **extra}, indent=2) + "\n")
    print("COMPLETE", a.index, model.num_timesteps, flush=True)


if __name__ == "__main__":
    main()
