"""E12 (v4, pre-declared): the supply chain with local observations and agent identity.

In E1-E10 every trainable node's policy lane receives the same full or-gym observation, with no agent identifier,
so the parameter-shared policy is a homogeneous controller. Here each node observes only its own part of the state,
plus a one-hot identifier of the node:
  - the customer demand on its retail links (if any);
  - its own on-hand inventory;
  - the pipeline of orders it has placed with its suppliers (one entry per period of lead time per incoming link).
The local vectors are zero-padded to a common length, as SuperSuit pads, and the identifier is appended. Rewards,
transforms, arms, learners, budget and the collapse label are E1's. Evaluation is horizon_supply.evaluate with one
change, applied identically in training: each agent's policy input is its local observation and identifier.

Learners: default (one Adam, global clip 0.5, raw rewards) and decoupled (separate optimizers without clipping,
VecNormalize rewards), with E1's classes. Nothing here modifies an existing source file.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import gymnasium as gym
import supersuit as ss
import torch
from pettingzoo.utils import BaseParallelWrapper
from stable_baselines3.common.vec_env import VecNormalize

import horizon_supply as h
import optimizer_factorial as of
import scale_control as sc
from instrumented_ppo import InstrumentedPPO, SeparateNoClipPPO, RolloutDiagnostics

DESIGN = json.loads(Path(__file__).with_name("e12_design.json").read_text())
LEARNERS, ARMS, GRID, SEEDS = DESIGN["learners"], DESIGN["arms"], of.GRID, of.SEEDS


def base_of(env):
    e = env
    while not hasattr(e, "unwrapped_env"):
        e = e.env
    return e


def local_index(base, agents):
    """Indices of each trainable node's local entries in or-gym's state vector [demand, inventory, pipeline]."""
    raw = base.unwrapped_env
    nr, nm = len(raw.retail_links), len(raw.main_nodes)
    idx = {}
    for a in agents:
        node = base.agent_name_to_id[a]; ix = [j for j, l in enumerate(raw.retail_links) if l[0] == node]
        ix += [nr + k for k, n in enumerate(raw.main_nodes) if n == node]
        pos = nr + nm
        for link, L in raw.lead_times.items():
            if L == 0:
                continue
            if link[1] == node:
                ix += list(range(pos, pos + L))
            pos += L
        idx[a] = np.array(ix, int)
    assert pos == raw.obs_dim, "state layout differs from [demand, inventory, pipeline]"
    return idx


class LocalObs(BaseParallelWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.agents_order = list(env.possible_agents); self.base = base_of(env); self.idx = local_index(self.base, self.agents_order)
        self.width = max(len(v) for v in self.idx.values()); self.n = len(self.agents_order)
        self._spaces = {a: gym.spaces.Box(-np.inf, np.inf, (self.width + self.n,), np.float64) for a in self.agents_order}
        assert len({tuple(v) for v in self.idx.values()}) == self.n, "local views must differ across nodes"

    def observation_space(self, agent):
        return self._spaces[agent]

    def local(self, a, full):
        v = np.zeros(self.width + self.n); x = np.asarray(full, np.float64)[self.idx[a]]
        v[:len(x)] = x; v[self.width + self.agents_order.index(a)] = 1.0
        return v

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return {a: self.local(a, o) for a, o in obs.items()}, info

    def step(self, actions):
        obs, r, te, tr, info = self.env.step(actions)
        return {a: self.local(a, o) for a, o in obs.items()}, r, te, tr, info


def vector(w, arm, seed):
    env = LocalObs(sc.build(w, arm, seed, of.arm_scale(arm), of.REFERENCE))
    vec = ss.pettingzoo_env_to_vec_env_v1(ss.pad_action_space_v0(env))
    vec = ss.concat_vec_envs_v1(vec, 1, num_cpus=0, base_class="stable_baselines3"); vec.seed = lambda s: None
    return vec


class LocalPolicy:
    """Wraps a trained model for horizon_supply.evaluate, which passes each agent's full observation: the wrapper
    maps it to that agent's local view. The agent order of evaluate's loop is the environment's, so a counter over
    the agents present reproduces the identity."""
    def __init__(self, model, w, arm, seed):
        self.model = model; self.view = LocalObs(sc.build(w, arm, seed, of.arm_scale(arm), of.REFERENCE)); self.queue = []

    def bind(self, agents):
        self.queue = list(agents)

    def predict(self, obs, deterministic=True):
        a = self.queue.pop(0) if self.queue else None
        assert a is not None, "agent order not bound"
        return self.model.predict(self.view.local(a, obs), deterministic=deterministic)


def evaluate(model, protocol, episodes=10):
    """horizon_supply.evaluate with each agent's policy input replaced by its local view (and nothing else changed)."""
    pol = LocalPolicy(model, protocol["w"], protocol["arm"], protocol["seed"])
    orig_build = h.build

    def build(w, condition, seed):
        env = orig_build(w, condition, seed)
        step, reset = env.step, env.reset

        def reset2(seed=None, options=None):
            o, i = reset(seed=seed, options=options); pol.bind([a for a in env.agents]); return o, i

        def step2(actions):
            out = step(actions); pol.bind([a for a in env.agents]); return out
        env.reset, env.step = reset2, step2
        return env
    h.build = build
    try:
        return h.evaluate(pol, protocol, episodes)
    finally:
        h.build = orig_build


class LocalGlobal(InstrumentedPPO):
    """E1's default learner without E1's checkpoint evaluation (which would pass global observations)."""


class LocalSeparate(SeparateNoClipPPO):
    """E1's decoupled learner without E1's checkpoint evaluation."""


def decode(index):
    assert 0 <= index < len(LEARNERS) * len(ARMS) * len(GRID) * len(SEEDS)
    seed = SEEDS[index % 10]; q = index // 10; w = GRID[q % len(GRID)]; q //= len(GRID)
    arm = ARMS[q % len(ARMS)]; learner = LEARNERS[q // len(ARMS)]
    return learner, arm, w, seed


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--index", required=True, type=int)
    ap.add_argument("--budget", type=int, default=300_000); a = ap.parse_args()
    learner, arm, w, seed = decode(a.index); out = Path(a.out); out.mkdir(parents=True, exist_ok=False); t0 = time.monotonic()
    torch.set_num_threads(1)
    vec = vector(w, arm, seed)
    if learner == "decoupled":
        vec = VecNormalize(vec, norm_obs=False, norm_reward=True, gamma=0.99, clip_reward=10.0, epsilon=1e-8)
    cls = LocalGlobal if learner == "default" else LocalSeparate
    model = cls("MlpPolicy", vec, n_steps=128, batch_size=256, seed=seed, verbose=0, device="cpu")
    model.diagnostics_dir = str(out / "diagnostics"); model.output = str(out); model.started = time.monotonic(); model.completed = []
    protocol = dict(design="e12-localobs-v4", index=a.index, learner=learner, optimizer_mode="global_clip" if learner == "default" else "separate_no_clip",
                    reward_mode="raw" if learner == "default" else "vecnormalize", arm=arm, arm_scale=of.arm_scale(arm), w=w, seed=seed,
                    condition="lambda7" if arm == "A" else "curvature", reference=of.REFERENCE, checkpoints=[a.budget], total_budget=a.budget,
                    observation="local+id", phase="confirmation")
    model.protocol = protocol; model.verify_upstream()
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    here = Path(__file__); srcs = [here, here.with_name("e12_design.json"), here.with_name("optimizer_factorial.py"), here.with_name("instrumented_ppo.py"),
                                   here.with_name("scale_control.py"), here.with_name("horizon_supply.py")]
    (out / "manifest.json").write_text(json.dumps({"sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in srcs}}, indent=2) + "\n")
    model.learn(total_timesteps=a.budget, callback=RolloutDiagnostics())
    res = evaluate(model, protocol, 10); res.update(requested_checkpoint=a.budget, actual_agent_transitions=model.num_timesteps)
    (out / f"evaluation_{a.budget}.json").write_text(json.dumps(res, indent=2))
    Path(a.out, "COMPLETE.json").write_text(json.dumps({"requested": a.budget, "actual_transitions": model.num_timesteps, "elapsed_seconds": time.monotonic() - t0}, indent=2) + "\n")
    print("COMPLETE", a.index, model.num_timesteps, flush=True)


if __name__ == "__main__":
    main()
