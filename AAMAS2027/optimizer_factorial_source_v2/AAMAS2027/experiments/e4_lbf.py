"""E4 (v4 pre-declared): centralized-critic MAPPO on Level-Based Foraging, optimizer x transform design.

Environment, reward wrapper, MAPPO policy, PPO hyper-parameters, seeding, evaluation and collapse rule are the
legacy ones (experiments/exp_mappo_lbf.py, exp_ppo_lbf.py, env/lbf_wrapper.py). The only additions are:
- arm S, implemented as in the supply-chain control: CPT output divided by 1/4 (both branches x4);
- optimizer modes global_clip (SB3 update, instrumented) and separate_no_clip (disjoint actor/critic Adam,
  no clipping), with parameter groups that include MAPPO's separate feature extractors.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]

import argparse
import hashlib
import json
import time
import warnings

import numpy as np
import supersuit as ss
import torch
from stable_baselines3.common.callbacks import BaseCallback

from env.lbf_wrapper import make_lbf_env_with_w, is_collapsed_lbf, DEFAULT_ENV_ID, MAX_CYCLES
from env.global_state_wrapper import GlobalStateWrapper
from agents.cpt_wrapper import CPTRewardWrapper
from agents.mappo_policy import CentralizedCriticPolicy
from instrumented_ppo import InstrumentedPPO, SeparateNoClipPPO
from reward_diagnostics import moments

GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.65, 0.75, 0.85, 0.9, 0.95]
ARMS = {"linear": (1.0, 1.0, 1.0, 1.0), "curvature": (0.88, 0.88, 1.0, 1.0),
        "A": (0.88, 0.88, 4.0, 1.0), "S": (0.88, 0.88, 1.0, 4.0)}   # alpha, beta, lambda, symmetric scale
OPTIMIZERS = ["global_clip", "separate_no_clip"]
SEEDS = list(range(42, 52))
N_ENV_COPIES = 8
PPO_KWARGS = dict(n_steps=256, batch_size=512, ent_coef=0.01)
BUDGET = 2_000_000


def group_of(name):
    if "value_net" in name or "vf_features_extractor" in name:
        return "critic"
    if any(k in name for k in ("policy_net", "action_net", "log_std", "pi_features_extractor", "features_extractor")):
        return "actor"
    return "shared"


class _LBFGroups:
    def _sums(self, prefix=""):
        sums = {"actor": 0.0, "critic": 0.0, "shared": 0.0}
        for name, p in self.policy.named_parameters():
            if p.grad is not None:
                sums[group_of(name)] += float(torch.sum(p.grad.detach() ** 2))
        total = sum(sums.values())
        return sums, total

    def before_clip(self):
        sums, total = self._sums(); norm = float(np.sqrt(total))
        return {**{k + "_grad_norm": float(np.sqrt(v)) for k, v in sums.items()}, "total_grad_norm": norm,
                "actor_grad_energy_share": float(sums["actor"] / total if total else 0.0), "gradient_clipped": bool(norm > self.max_grad_norm)}

    def after_clip(self):
        sums, total = self._sums()
        return {**{"post_" + k + "_grad_norm": float(np.sqrt(v)) for k, v in sums.items()}, "post_total_grad_norm": float(np.sqrt(total)),
                "post_actor_grad_energy_share": float(sums["actor"] / total if total else 0.0)}

    def parameter_groups(self):
        groups = {"actor": [], "critic": [], "shared": []}
        for name, p in self.policy.named_parameters():
            if p.requires_grad: groups[group_of(name)].append(p)
        if groups["shared"] or not groups["actor"] or not groups["critic"]:
            raise RuntimeError({k: len(v) for k, v in groups.items()})
        return groups


class LBFGlobalPPO(_LBFGroups, InstrumentedPPO):
    pass


class LBFSeparatePPO(_LBFGroups, SeparateNoClipPPO):
    pass


class LBFRewardDiagnostics(BaseCallback):
    """Per-rollout moments of the raw blended and transformed training rewards (pooled over lanes)."""
    def _on_rollout_start(self): self.samples = []
    def _on_step(self):
        for info in self.locals["infos"]:
            if "blended_reward" in info: self.samples.append((info["blended_reward"], info["cpt_reward"]))
        return True
    def _on_rollout_end(self):
        if not self.samples: return
        a = np.array(self.samples, float)
        rec = dict(actual_transitions=self.num_timesteps, blend=moments(a[:, 0]), transformed=moments(a[:, 1]))
        import gzip
        p = Path(self.model.diagnostics_dir) / "rollouts.jsonl.gz"; p.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(p, "at") as f: f.write(json.dumps(rec) + "\n")


def decode(index):
    n = len(OPTIMIZERS) * len(ARMS) * len(GRID) * len(SEEDS); assert 0 <= index < n
    seed = SEEDS[index % len(SEEDS)]; q = index // len(SEEDS)
    w = GRID[q % len(GRID)]; q //= len(GRID)
    arm = list(ARMS)[q % len(ARMS)]; opt = OPTIMIZERS[q // len(ARMS)]
    return opt, arm, w, seed


def build_env(w, arm):
    alpha, beta, lam, c = ARMS[arm]
    local_w = make_lbf_env_with_w(w, env_id=DEFAULT_ENV_ID, max_cycles=MAX_CYCLES)
    params = {a: {"lambda": lam, "alpha": alpha, "beta": beta} for a in local_w.possible_agents}
    cpt = CPTRewardWrapper(local_w, params, reward_scale=1.0, global_reward_weight=0.0)
    if c != 1.0: cpt.output_scale = 1.0 / c
    return GlobalStateWrapper(cpt)


def evaluate_lbf(model, env, num_episodes=10, base_seed=42):
    totals = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=base_seed + ep); R = 0.0
        while env.agents:
            actions = {a: int(model.predict(obs[a], deterministic=True)[0]) for a in env.agents if a in obs}
            obs, rewards, term, trunc, info = env.step(actions)
            for a in rewards:
                R += float(info[a]["true_reward"]) if a in info and "true_reward" in info[a] else float(rewards[a])
        totals.append(R)
    return float(np.mean(totals)), totals


def create(out, index, budget=BUDGET):
    opt, arm, w, seed = decode(index)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); np.random.seed(seed); warnings.filterwarnings("ignore")
    env = build_env(w, arm); local_dim = env.local_dim
    vec = ss.concat_vec_envs_v1(ss.pettingzoo_env_to_vec_env_v1(ss.pad_observations_v0(ss.pad_action_space_v0(env))),
                                N_ENV_COPIES, num_cpus=0, base_class="stable_baselines3")
    inner = getattr(vec, "venv", vec); vec.seed = lambda s, _v=inner: _v.reset(seed=int(s))
    cls = LBFGlobalPPO if opt == "global_clip" else LBFSeparatePPO
    model = cls(CentralizedCriticPolicy, vec, verbose=0, seed=seed, device="cpu",
                policy_kwargs=dict(local_dim=local_dim, net_arch=dict(pi=[64, 64], vf=[128, 128])), **PPO_KWARGS)
    model.parameter_groups()  # abort early if grouping is incomplete
    model.verify_upstream()
    model.diagnostics_dir = str(out / "diagnostics"); model.inner = inner
    model.protocol = dict(design="e4-lbf-v4", index=index, optimizer_mode=opt, arm=arm, alpha=ARMS[arm][0], beta=ARMS[arm][1],
                          lambda_loss=ARMS[arm][2], symmetric_scale=ARMS[arm][3], w=w, seed=seed, budget=budget,
                          env=DEFAULT_ENV_ID, max_cycles=MAX_CYCLES, n_env_copies=N_ENV_COPIES, ppo_kwargs=PPO_KWARGS)
    (out / "protocol.json").write_text(json.dumps(model.protocol, indent=2) + "\n")
    here = Path(__file__); srcs = [here, here.with_name("instrumented_ppo.py"), ROOT / "env/lbf_wrapper.py", ROOT / "env/global_state_wrapper.py",
                                   ROOT / "agents/mappo_policy.py", ROOT / "agents/cpt_wrapper.py"]
    import importlib.metadata as md
    pk = {}
    for p in ["numpy", "torch", "stable-baselines3", "supersuit", "pettingzoo", "gymnasium", "lbforaging"]:
        try: pk[p] = md.version(p)
        except md.PackageNotFoundError: pk[p] = None
    (out / "manifest.json").write_text(json.dumps(dict(packages=pk, sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in srcs}), indent=2) + "\n")
    return model


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--budget", type=int, default=BUDGET); a = ap.parse_args()
    t0 = time.monotonic(); model = create(a.out, a.index, a.budget); p = model.protocol
    model.inner.reset(seed=int(p["seed"]))
    model.learn(total_timesteps=a.budget, callback=LBFRewardDiagnostics())
    mean_ret, eps = evaluate_lbf(model, build_env(p["w"], p["arm"]), num_episodes=10, base_seed=p["seed"] * 10)
    res = dict(team_return=mean_ret, episodes=eps, collapsed=bool(is_collapsed_lbf(mean_ret)), actual_transitions=model.num_timesteps,
               elapsed_seconds=time.monotonic() - t0)
    Path(a.out, "evaluation.json").write_text(json.dumps(res, indent=2) + "\n")
    Path(a.out, "COMPLETE.json").write_text(json.dumps(dict(requested=a.budget, actual_transitions=model.num_timesteps)) + "\n")
    print("COMPLETE", a.index, model.num_timesteps, f"{res['elapsed_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
