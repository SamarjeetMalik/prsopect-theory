"""X5 (pre-declared): does the symmetric-scaling shift, and its critic-target-scale channel, appear under decentralized
execution?

Supply chain, E12's local views. Each of the five trained nodes has its OWN actor network (no parameter sharing): its
input is the node's local view (own retail demand, own inventory, own incoming order pipeline, zero-padded) plus a
one-hot node identifier, and it has its own action head and log-std. One CENTRALIZED critic sees the full global state
and the node identifier (MAPPO-style; the identifier is needed because each node's mixed reward differs). Separate
actor and critic Adam without clipping, raw rewards, arms B and S, the E1 grid, fresh seeds 162-201, 300k. Three critic
heads: E7's plain head, E7's PopArt head, and X4's fixed-scale head (V = c * head, c = 1 for B, 7 for S).
X4's reference-critic logging is reused unchanged. Evaluation is horizon_supply.evaluate with each node's policy input
replaced by its decentralized observation, as in E12. Existing source files are not modified.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import gymnasium as gym
import numpy as np
import supersuit as ss
import torch
from torch import nn
from pettingzoo.utils import BaseParallelWrapper
from stable_baselines3.common.policies import ActorCriticPolicy

import horizon_supply as h
import optimizer_factorial as of
import scale_control as sc
import e7_critic_scale as e7
import e12_localobs as e12
from x4_runner import X4Logging, FixedScaleHead
from instrumented_ppo import RolloutDiagnostics

DESIGN = json.loads(Path(__file__).with_name("x5_design.json").read_text())
SEEDS, ARMS, PARTS, GRID = DESIGN["seeds"], DESIGN["arms"], DESIGN["parts"], of.GRID
N = len(PARTS) * len(ARMS) * len(GRID) * len(SEEDS)            # 1,920
HID = DESIGN["actor"]["hidden"]


# ------------------------------------------------------------------ observations
class DecObs(BaseParallelWrapper):
    """Each node's observation is [local view and one-hot id (E12) | full global state]. The actor reads only the
    first part, the critic only the global state and the id."""

    def __init__(self, env):
        super().__init__(env)
        self.local_view = e12.LocalObs(env)                       # reuses E12's index map and padding
        self.agents_order = self.local_view.agents_order; self.n = self.local_view.n
        self.L = self.local_view.width + self.n; self.G = int(np.prod(env.observation_space(self.agents_order[0]).shape))
        self._spaces = {a: gym.spaces.Box(-np.inf, np.inf, (self.L + self.G,), np.float64) for a in self.agents_order}

    def observation_space(self, agent):
        return self._spaces[agent]

    def dec(self, a, full):
        return np.concatenate([self.local_view.local(a, full), np.asarray(full, np.float64)])

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return {a: self.dec(a, o) for a, o in obs.items()}, info

    def step(self, actions):
        obs, r, te, tr, info = self.env.step(actions)
        return {a: self.dec(a, o) for a, o in obs.items()}, r, te, tr, info


def vector(w, arm, seed):
    env = DecObs(sc.build(w, arm, seed, of.arm_scale(arm), of.REFERENCE))
    vec = ss.pettingzoo_env_to_vec_env_v1(ss.pad_action_space_v0(env))
    vec = ss.concat_vec_envs_v1(vec, 1, num_cpus=0, base_class="stable_baselines3"); vec.seed = lambda s: None
    return vec, env.n, env.L, env.G


# ------------------------------------------------------------------ policy
def mlp(i, hidden):
    layers, d = [], i
    for k in hidden:
        layers += [nn.Linear(d, k), nn.Tanh()]; d = k
    return nn.Sequential(*layers)


class DecExtractor(nn.Module):
    """policy_net: one MLP and action-mean head per node; value_net: one MLP on [global state, id].
    latent_pi = [action mean of the acting node, its one-hot id]; latent_vf = critic features."""

    def __init__(self, n, L, G, act_dim):
        super().__init__()
        self.n, self.L, self.G, self.A = n, L, G, act_dim
        self.policy_net = nn.ModuleList([nn.Sequential(mlp(L, HID), nn.Linear(HID[-1], act_dim)) for _ in range(n)])
        self.value_net = mlp(G + n, HID)
        self.latent_dim_pi, self.latent_dim_vf = act_dim + n, HID[-1]

    def ids(self, x):
        return x[:, self.L - self.n:self.L]

    def forward_actor(self, x):
        """Each sample goes through its own node's network only."""
        idv = self.ids(x); loc = x[:, :self.L]; k = idv.argmax(1)
        mean = loc.new_zeros(len(x), self.A)
        for j, net in enumerate(self.policy_net):
            sel = k == j
            if bool(sel.any()):
                mean = mean.index_put((sel.nonzero(as_tuple=True)[0],), net(loc[sel]))
        return torch.cat([mean, idv], 1)

    def forward_critic(self, x):
        return self.value_net(torch.cat([x[:, self.L:], self.ids(x)], 1))

    def forward(self, x):
        return self.forward_actor(x), self.forward_critic(x)


class DecPolicy(ActorCriticPolicy):
    def __init__(self, *args, n=5, L=34, G=68, **kwargs):
        self.n_agents, self.L, self.G = n, L, G
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = DecExtractor(self.n_agents, self.L, self.G, int(np.prod(self.action_space.shape)))

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        A = int(np.prod(self.action_space.shape))
        self.action_net = nn.Identity()                                   # the per-node heads live in policy_net
        self.log_std = nn.Parameter(torch.full((self.n_agents, A), float(self.log_std_init)))   # per-node log-std
        for net in self.mlp_extractor.policy_net:                        # SB3's action-head initialisation, per node
            nn.init.orthogonal_(net[-1].weight, gain=0.01); nn.init.zeros_(net[-1].bias)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi):
        A = int(np.prod(self.action_space.shape)); mean, idv = latent_pi[:, :A], latent_pi[:, A:]
        return self.action_dist.proba_distribution(mean, idv @ self.log_std)


# ------------------------------------------------------------------ learners
class X5Plain(X4Logging, e7.E7Plain):
    pass


class X5Pop(X4Logging, e7.E7PopArt):
    pass


class X5Matched(X4Logging, e7.E7Plain):
    def wrap_critic(self, c):
        self.policy.value_net = FixedScaleHead(self.policy.value_net, c)


class NoLog:
    """Gate 3 only: the same learner with X4's logging switched off."""

    def train(self):
        e7.E7Plain.train(self)


class X5PlainNoLog(NoLog, X5Plain):
    pass


def independence_check(model):
    """Gate 4: disjoint per-node actor parameters, zero cross-node actor gradients, a critic without local-only input."""
    pol = model.policy; ex = pol.mlp_extractor; n = ex.n
    sets = [{id(p) for p in net.parameters()} for net in ex.policy_net]
    assert all(not (sets[i] & sets[j]) for i in range(n) for j in range(i + 1, n)), "actor parameters are shared"
    groups = model.parameter_groups()
    assert {id(p) for p in groups["actor"]} == set().union(*sets) | {id(pol.log_std)}, "actor group is not the per-node nets and log-std"
    obs = torch.as_tensor(np.asarray(model.rollout_buffer.observations).reshape(-1, ex.L + ex.G)[:256], dtype=torch.float32)
    for k in range(n):
        sel = obs[obs[:, ex.L - n + k] > 0.5]
        if not len(sel):
            continue
        pol.zero_grad(); dist = pol.get_distribution(sel); dist.distribution.mean.sum().backward()
        for j, net in enumerate(ex.policy_net):
            g = sum(float(p.grad.abs().sum()) for p in net.parameters() if p.grad is not None)
            assert (g > 0) == (j == k), f"node {k}'s samples move node {j}'s actor (|grad| {g})"
        assert all(float(sel[:, ex.L - n + j].sum()) == 0 for j in range(n) if j != k), "a sample carries two identifiers"
    pol.zero_grad()
    x = obs[:8].clone().requires_grad_(True); pol.predict_values(x).sum().backward()
    assert float(x.grad[:, :ex.L - n].abs().sum()) == 0.0, "the critic reads local-only entries"
    pol.zero_grad()
    return dict(n_nodes=n, actor_params_per_node=sum(p.numel() for p in ex.policy_net[0].parameters()),
                critic_params=sum(p.numel() for p in groups["critic"]))


# ------------------------------------------------------------------ evaluation (E12's, with the decentralized view)
class DecPolicyEval(e12.LocalPolicy):
    def __init__(self, model, w, arm, seed):
        self.model = model; self.view = DecObs(sc.build(w, arm, seed, of.arm_scale(arm), of.REFERENCE)); self.queue = []

    def predict(self, obs, deterministic=True):
        a = self.queue.pop(0) if self.queue else None
        assert a is not None, "agent order not bound"
        return self.model.predict(self.view.dec(a, obs), deterministic=deterministic)


def evaluate(model, protocol, episodes=10):
    pol = DecPolicyEval(model, protocol["w"], protocol["arm"], protocol["seed"]); orig_build = h.build

    def build(w, condition, seed):
        env = orig_build(w, condition, seed); step, reset = env.step, env.reset

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


# ------------------------------------------------------------------ run
def decode(index):
    assert 0 <= index < N
    seed = SEEDS[index % len(SEEDS)]; q = index // len(SEEDS); w = GRID[q % len(GRID)]; q //= len(GRID)
    return dict(part=PARTS[q // len(ARMS)], arm=ARMS[q % len(ARMS)], w=w, seed=seed)


def run(spec, out, budget, index, log=True, check=False):
    out = Path(out); out.mkdir(parents=True, exist_ok=False); t0 = time.monotonic(); torch.set_num_threads(1)
    part, arm, w, seed = spec["part"], spec["arm"], spec["w"], spec["seed"]
    vec, n, L, G = vector(w, arm, seed)
    cls = {"plain": X5Plain if log else X5PlainNoLog, "popart": X5Pop, "M": X5Matched}[part]
    model = cls(DecPolicy, vec, n_steps=128, batch_size=256, seed=seed, verbose=0, device="cpu", policy_kwargs=dict(n=n, L=L, G=G))
    if part == "M":
        model.wrap_critic(of.arm_scale(arm))
    model._x4_init(seed)
    model.diagnostics_dir = str(out / "diagnostics"); model.output = str(out); model.started = time.monotonic(); model.completed = []
    model.protocol = dict(design=DESIGN["design"], index=index, experiment="X5", part=part, execution="decentralized",
                          critic_mode={"plain": "plain", "popart": "popart", "M": "scale_matched"}[part],
                          critic_output_scale=of.arm_scale(arm) if part == "M" else None, optimizer_mode="separate_no_clip", reward_mode="raw",
                          arm=arm, arm_scale=of.arm_scale(arm), w=w, seed=seed, condition="curvature", reference=of.REFERENCE,
                          observation="decentralized: local+id actor, global+id critic", logging=bool(log),
                          checkpoints=[], total_budget=budget, phase="confirmation")
    model.verify_upstream()
    (out / "protocol.json").write_text(json.dumps(model.protocol, indent=2) + "\n")
    here = Path(__file__)
    srcs = [here, here.with_name("x5_design.json"), here.with_name("x4_runner.py"), here.with_name("x4_design.json"),
            here.with_name("e12_localobs.py"), here.with_name("e7_critic_scale.py"), here.with_name("e7_design.json"),
            here.with_name("optimizer_factorial.py"), here.with_name("instrumented_ppo.py"), here.with_name("scale_control.py"),
            here.with_name("horizon_supply.py")]
    (out / "manifest.json").write_text(json.dumps({"sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in srcs},
                                                   "torch": torch.__version__, "numpy": np.__version__}, indent=2) + "\n")
    model.learn(total_timesteps=budget, callback=RolloutDiagnostics())
    extra = {}
    if check:
        extra["independence_check"] = independence_check(model)
    if isinstance(model, e7.E7PopArt):
        model.save_popart(out); extra["popart_final"] = model._popart_log[-1] if model._popart_log else None
    res = evaluate(model, model.protocol, 10); res.update(requested_checkpoint=budget, actual_agent_transitions=model.num_timesteps)
    (out / f"evaluation_{budget}.json").write_text(json.dumps(res, indent=2))
    (out / "COMPLETE.json").write_text(json.dumps({"requested": budget, "actual_transitions": model.num_timesteps,
                                                   "elapsed_seconds": time.monotonic() - t0, **extra}, indent=2) + "\n")
    print("COMPLETE", index, model.num_timesteps, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--index", type=int)
    ap.add_argument("--budget", type=int, default=300_000)
    ap.add_argument("--part"); ap.add_argument("--arm"); ap.add_argument("--w", type=float); ap.add_argument("--seed", type=int)
    ap.add_argument("--no-log", action="store_true", help="gate 3 only")
    ap.add_argument("--check", action="store_true", help="run the independence check after training (gate 4)")
    a = ap.parse_args()
    spec = decode(a.index) if a.index is not None else dict(part=a.part, arm=a.arm, w=a.w, seed=a.seed)
    run(spec, a.out, a.budget, a.index if a.index is not None else -1, log=not a.no_log, check=a.check)


if __name__ == "__main__":
    main()
