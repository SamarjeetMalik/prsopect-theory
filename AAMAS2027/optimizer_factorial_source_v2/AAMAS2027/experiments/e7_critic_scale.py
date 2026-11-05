"""E7 (v4, pre-declared): critic-target-scale isolation and a mean-and-scale-matched reward control.

Supply chain, separate actor/critic Adam without clipping (so the shared-clip pathway is absent), raw transformed
rewards, the E1 grid and seeds, 300k. Two critic modes:
  plain   the critic regresses raw returns (identical to E1 separate_no_clip/raw)
  popart  the critic's output layer is reparametrised as V = sigma * head(x) + mu, with mu and sigma running
          moments of the rollout returns and PopArt's output-preserving rescale of the head whenever they change
          (van Hasselt et al., 2016). The actor's rewards, advantages and loss are unchanged.
Arms: B, A, S as in E1, and M: an affine map of B's reward, r -> scale_w * r + shift_w, with per-w constants frozen
in e7_design.json so that M has A's training-reward mean and SD as measured in E1 (separate_no_clip/raw). M is
symmetric in gains and losses; A - M holds the first two moments of the reward roughly fixed.
Existing source files are not modified (E5/E6 hash-check them).
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
from instrumented_ppo import SeparateNoClipPPO, RolloutDiagnostics

DESIGN = json.loads(Path(__file__).with_name("e7_design.json").read_text())
ARMS = DESIGN["arms"]                      # ["B", "A", "S", "M"]
CRITIC = DESIGN["critic_modes"]            # ["plain", "popart"]
GRID = of.GRID
SEEDS = of.SEEDS
BETA, SIGMA_MIN = DESIGN["popart_beta"], DESIGN["popart_sigma_min"]
AFFINE = {}


def add_affine_reward(vec, scale, shift):
    """r -> scale * r + shift on every agent's reward, by wrapping the vector env's own step_wait. A closure, not a
    VecEnvWrapper subclass: the training env is cloudpickled at checkpoints, and an ABC subclass defined in
    __main__ cannot be. Returns a dict that accumulates the resulting reward moments."""
    stats = dict(n=0, s1=0.0, s2=0.0); inner = vec.step_wait; scale, shift = float(scale), float(shift)

    def step_wait():
        obs, rew, done, info = inner()
        rew = scale * rew + shift
        stats["n"] += rew.size; stats["s1"] += float(rew.sum()); stats["s2"] += float((rew ** 2).sum())
        return obs, rew, done, info

    vec.step_wait = step_wait
    return stats


def moments(stats):
    n = max(stats["n"], 1); m = stats["s1"] / n
    return dict(n=stats["n"], mean=m, rms=float(np.sqrt(stats["s2"] / n)), sd=float(np.sqrt(max(stats["s2"] / n - m * m, 0.0))))


class PopArtHead(torch.nn.Module):
    """V(x) = sigma * linear(x) + mu. The wrapped nn.Linear keeps its parameter objects, so optimizers built on
    them before wrapping still own them; its name keeps the 'value_net' prefix used for parameter grouping."""

    def __init__(self, linear):
        super().__init__(); self.linear = linear
        self.register_buffer("mu", torch.zeros(1)); self.register_buffer("sigma", torch.ones(1))

    def forward(self, x):
        return self.sigma * self.linear(x) + self.mu

    @torch.no_grad()
    def update(self, returns, first):
        r = returns.flatten().double()
        m, s2 = float(r.mean()), float((r ** 2).mean())
        if first:
            mu, nu = m, s2
        else:
            mu = (1 - BETA) * float(self.mu) + BETA * m
            nu = (1 - BETA) * (float(self.sigma) ** 2 + float(self.mu) ** 2) + BETA * s2
        sigma = max(np.sqrt(max(nu - mu * mu, 0.0)), SIGMA_MIN)
        old_mu, old_sigma = float(self.mu), float(self.sigma)
        self.linear.weight.mul_(old_sigma / sigma)
        self.linear.bias.mul_(old_sigma).add_(old_mu - mu).div_(sigma)
        self.mu.fill_(mu); self.sigma.fill_(sigma)
        return old_mu, old_sigma, mu, sigma


class E7Plain(sc.CheckpointPPO, SeparateNoClipPPO):
    pass


class E7PopArt(sc.CheckpointPPO, SeparateNoClipPPO):
    def _setup_model(self):
        super()._setup_model()
        self.policy.value_net = PopArtHead(self.policy.value_net)
        self._popart_first = True; self._popart_log = []

    def train(self):
        # Rescale the critic head to the current rollout's returns before the update, preserving its outputs.
        buf = self.rollout_buffer
        obs = torch.as_tensor(buf.observations.reshape(-1, buf.observations.shape[-1])[:64], device=self.device)
        with torch.no_grad():
            before = self.policy.predict_values(obs).flatten()
        old_mu, old_sigma, mu, sigma = self.policy.value_net.update(torch.as_tensor(buf.returns), self._popart_first)
        with torch.no_grad():
            after = self.policy.predict_values(obs).flatten()
        drift = float((after - before).abs().max())
        assert drift <= 1e-3 * max(1.0, float(before.abs().max())), f"PopArt rescale changed outputs by {drift}"
        self._popart_first = False
        self._popart_log.append(dict(t=self.num_timesteps, mu=mu, sigma=sigma, max_output_drift=drift))
        super().train()

    def save_popart(self, out):
        (Path(out) / "diagnostics").mkdir(parents=True, exist_ok=True)
        (Path(out) / "diagnostics/popart.json").write_text(json.dumps(self._popart_log) + "\n")


CLASSES = {"plain": E7Plain, "popart": E7PopArt}


def decode(index):
    assert 0 <= index < len(CRITIC) * len(ARMS) * len(GRID) * len(SEEDS)
    seed = SEEDS[index % len(SEEDS)]
    q = index // len(SEEDS)
    w = GRID[q % len(GRID)]
    q //= len(GRID)
    arm = ARMS[q % len(ARMS)]
    critic = CRITIC[q // len(ARMS)]
    return critic, arm, w, seed


def create(out, index, budget=300_000):
    critic, arm, w, seed = decode(index)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    base_arm = "B" if arm == "M" else arm
    vec = sc.vector(w, base_arm, seed, of.arm_scale(base_arm), of.REFERENCE)
    affine = None
    if arm == "M":
        c = DESIGN["M_affine"][f"{w:g}"]
        affine = add_affine_reward(vec, c["scale"], c["shift"])
    model = CLASSES[critic]("MlpPolicy", vec, n_steps=128, batch_size=256, seed=seed, verbose=0, device="cpu")
    model.diagnostics_dir = str(out / "diagnostics")
    model.output = str(out)
    model.started = time.monotonic()
    model.completed = []
    AFFINE["wrapper"] = affine
    model.protocol = dict(design="e7-critic-scale-v4", index=index, optimizer_mode="separate_no_clip", reward_mode="raw",
                          critic_mode=critic, arm=arm, arm_scale=of.arm_scale(base_arm), w=w, seed=seed,
                          condition="lambda7" if arm == "A" else "curvature", reference=of.REFERENCE,
                          M_affine=DESIGN["M_affine"].get(f"{w:g}") if arm == "M" else None,
                          checkpoints=[budget], total_budget=budget, phase="confirmation")
    model.verify_upstream()
    (out / "protocol.json").write_text(json.dumps(model.protocol, indent=2) + "\n")
    here = Path(__file__)
    sources = [here, here.with_name("e7_design.json"), here.with_name("optimizer_factorial.py"), here.with_name("instrumented_ppo.py"),
               here.with_name("scale_control.py")]
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
    extra = {}
    if isinstance(model, E7PopArt):
        model.save_popart(a.out); extra["popart_final"] = model._popart_log[-1] if model._popart_log else None
    if AFFINE.get("wrapper") is not None:
        extra["M_reward_moments"] = moments(AFFINE["wrapper"])
    Path(a.out, "COMPLETE.json").write_text(json.dumps({"requested": a.budget, "actual_transitions": model.num_timesteps, **extra}, indent=2) + "\n")
    print("COMPLETE", a.index, model.num_timesteps, flush=True)


if __name__ == "__main__":
    main()
