"""E11 (v4, pre-declared): a second PPO implementation. A single-file port of CleanRL's ppo_continuous_action.py
(Huang et al., 2022), written for the supply-chain vector environment of E1 and evaluated by E1's protocol.

CleanRL defaults kept: separate actor and critic MLPs (64-64, tanh) with orthogonal initialization (sqrt 2; actor head
0.01, critic head 1.0); state-independent log-std parameter; one Adam (lr 3e-4, eps 1e-5) over all parameters with
linear learning-rate annealing; GAE (gamma 0.99, lambda 0.95); 10 update epochs; per-minibatch advantage
normalization; policy clip 0.2; clipped value loss (coefficient 0.5); zero entropy coefficient; one global gradient-
norm clip of 0.5 over all parameters; observation normalization (running mean/variance, clip 10); reward
normalization by the running SD of discounted returns (clip 10); actions clipped to the action-space bounds.
Rollouts of 128 steps on the environment's five agent lanes (640 transitions) with 5 minibatches of 128, so the
300k budget executes 300,160 transitions, as in E1.

Conditions (E11 design):
  cleanrl_raw      CleanRL defaults but raw rewards (no reward normalization), global clip
  cleanrl_default  CleanRL defaults (reward normalization), global clip
  cleanrl_group    CleanRL defaults but raw rewards, actor and critic clipped separately to 0.5
Nothing here modifies an existing source file.
"""
from pathlib import Path
import argparse
import hashlib
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import horizon_supply as h
import optimizer_factorial as of
import scale_control as sc

DESIGN = json.loads(Path(__file__).with_name("e11_design.json").read_text())
CONDITIONS, ARMS, GRID, SEEDS = DESIGN["conditions"], DESIGN["arms"], of.GRID, of.SEEDS
N_STEPS, N_MB, EPOCHS, LR, GAMMA, LAM, CLIP, VF, ENT, MAXN = 128, 5, 10, 3e-4, 0.99, 0.95, 0.2, 0.5, 0.0, 0.5


class RunningMeanStd:
    """gymnasium's RunningMeanStd (parallel variance update)."""
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, np.float64); self.var = np.ones(shape, np.float64); self.count = epsilon

    def update(self, x):
        bm, bv, bc = np.mean(x, axis=0), np.var(x, axis=0), x.shape[0]
        delta = bm - self.mean; tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std); torch.nn.init.constant_(layer.bias, bias_const); return layer


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.critic = nn.Sequential(layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(), layer_init(nn.Linear(64, 64)), nn.Tanh(), layer_init(nn.Linear(64, 1), std=1.0))
        self.actor_mean = nn.Sequential(layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(), layer_init(nn.Linear(64, 64)), nn.Tanh(), layer_init(nn.Linear(64, act_dim), std=0.01))
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x); std = torch.exp(self.actor_logstd.expand_as(mean)); dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), self.critic(x)


class Policy:
    """predict() interface used by horizon_supply.evaluate: frozen observation normalization, deterministic mean
    action clipped to the action bounds."""
    def __init__(self, agent, obs_rms, low, high):
        self.agent, self.obs_rms, self.low, self.high = agent, obs_rms, low, high

    def norm(self, o):
        return np.clip((o - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -10, 10)

    def predict(self, obs, deterministic=True):
        assert deterministic
        x = torch.as_tensor(self.norm(np.asarray(obs, np.float64)[None]), dtype=torch.float32)
        with torch.no_grad():
            a = self.agent.actor_mean(x).numpy()[0]
        return np.clip(a, self.low, self.high), None


def decode(index):
    assert 0 <= index < len(CONDITIONS) * len(ARMS) * len(GRID) * len(SEEDS)
    seed = SEEDS[index % 10]; q = index // 10; w = GRID[q % len(GRID)]; q //= len(GRID)
    arm = ARMS[q % len(ARMS)]; cond = CONDITIONS[q // len(ARMS)]
    return cond, arm, w, seed


def grad_norm(params):
    return float(torch.sqrt(sum(torch.sum(p.grad.detach() ** 2) for p in params if p.grad is not None)))


def train(cond, arm, w, seed, budget, out):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(1)
    vec = sc.vector(w, arm, seed, of.arm_scale(arm), of.REFERENCE)
    n_env = vec.num_envs; obs_dim = vec.observation_space.shape[0]; act_dim = vec.action_space.shape[0]
    low, high = vec.action_space.low.astype(np.float32), vec.action_space.high.astype(np.float32)
    agent = Agent(obs_dim, act_dim); opt = torch.optim.Adam(agent.parameters(), lr=LR, eps=1e-5)
    actor_params = list(agent.actor_mean.parameters()) + [agent.actor_logstd]; critic_params = list(agent.critic.parameters())
    obs_rms = RunningMeanStd((obs_dim,)); ret_rms = RunningMeanStd(()); ret_acc = np.zeros(n_env)
    norm_rew = cond == "cleanrl_default"
    batch = N_STEPS * n_env; mb = batch // N_MB; iters = -(-budget // batch)
    O = torch.zeros((N_STEPS, n_env, obs_dim)); A = torch.zeros((N_STEPS, n_env, act_dim)); LP = torch.zeros((N_STEPS, n_env))
    R = torch.zeros((N_STEPS, n_env)); D = torch.zeros((N_STEPS, n_env)); V = torch.zeros((N_STEPS, n_env))

    def nobs(o):
        obs_rms.update(o); return np.clip((o - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10)

    nxt = torch.as_tensor(nobs(np.asarray(vec.reset(), np.float64)), dtype=torch.float32); nd = torch.zeros(n_env)
    diag = dict(actor=[], critic=[], total=[], post_actor=[], clipped=[]); raw_rew = []; steps = 0; ev = float("nan")
    for it in range(1, iters + 1):
        for g in opt.param_groups:
            g["lr"] = (1.0 - (it - 1.0) / iters) * LR
        for t in range(N_STEPS):
            O[t] = nxt; D[t] = nd
            with torch.no_grad():
                a, lp, _, v = agent.get_action_and_value(nxt)
            V[t] = v.flatten(); A[t] = a; LP[t] = lp
            o, r, done, _ = vec.step(np.clip(a.numpy(), low, high))
            r = np.asarray(r, np.float64); raw_rew.append(r.copy()); steps += n_env
            if norm_rew:
                ret_acc = ret_acc * GAMMA + r; ret_rms.update(ret_acc); r = np.clip(r / np.sqrt(ret_rms.var + 1e-8), -10, 10)
                ret_acc[np.asarray(done, bool)] = 0.0
            R[t] = torch.as_tensor(r, dtype=torch.float32)
            nxt = torch.as_tensor(nobs(np.asarray(o, np.float64)), dtype=torch.float32); nd = torch.as_tensor(np.asarray(done, np.float32))
        with torch.no_grad():
            nv = agent.get_value(nxt).reshape(1, -1); adv = torch.zeros_like(R); last = 0
            for t in reversed(range(N_STEPS)):
                nnt = 1.0 - (nd if t == N_STEPS - 1 else D[t + 1]); nvt = nv if t == N_STEPS - 1 else V[t + 1]
                delta = R[t] + GAMMA * nvt * nnt - V[t]; adv[t] = last = delta + GAMMA * LAM * nnt * last
            ret = adv + V
        bo, bl, ba, bad, br, bv = O.reshape(-1, obs_dim), LP.reshape(-1), A.reshape(-1, act_dim), adv.reshape(-1), ret.reshape(-1), V.reshape(-1)
        idx = np.arange(batch)
        for _ in range(EPOCHS):
            np.random.shuffle(idx)
            for s in range(0, batch, mb):
                m = idx[s:s + mb]
                _, nlp, ent, nv_ = agent.get_action_and_value(bo[m], ba[m]); ratio = (nlp - bl[m]).exp()
                madv = (bad[m] - bad[m].mean()) / (bad[m].std() + 1e-8)
                pg = torch.max(-madv * ratio, -madv * torch.clamp(ratio, 1 - CLIP, 1 + CLIP)).mean()
                nv_ = nv_.view(-1); vu = (nv_ - br[m]) ** 2; vc = bv[m] + torch.clamp(nv_ - bv[m], -CLIP, CLIP)
                vloss = 0.5 * torch.max(vu, (vc - br[m]) ** 2).mean()
                loss = pg - ENT * ent.mean() + VF * vloss
                opt.zero_grad(); loss.backward()
                ga, gc = grad_norm(actor_params), grad_norm(critic_params)
                if cond == "cleanrl_group":
                    nn.utils.clip_grad_norm_(actor_params, MAXN); nn.utils.clip_grad_norm_(critic_params, MAXN); clipped = ga > MAXN or gc > MAXN
                else:
                    tot = nn.utils.clip_grad_norm_(agent.parameters(), MAXN); clipped = float(tot) > MAXN
                diag["actor"].append(ga); diag["critic"].append(gc); diag["total"].append(float(np.hypot(ga, gc)))
                diag["post_actor"].append(grad_norm(actor_params)); diag["clipped"].append(bool(clipped))
                opt.step()
        yp, yt = bv.numpy(), br.numpy(); ev = float(1 - np.var(yt - yp) / np.var(yt)) if np.var(yt) > 0 else float("nan")
    pol = Policy(agent, obs_rms, low, high)
    proto = dict(design="e11-cleanrl-v4", condition_name=cond, arm=arm, arm_scale=of.arm_scale(arm), w=w, seed=seed,
                 condition="curvature" if arm != "A" else "lambda7", reference=of.REFERENCE, total_budget=budget)
    res = h.evaluate(pol, proto, 10)
    res.update(requested_checkpoint=budget, actual_agent_transitions=steps)
    (out / f"evaluation_{budget}.json").write_text(json.dumps(res, indent=2))
    rr = np.concatenate(raw_rew); share = np.array(diag["critic"]) ** 2 / np.maximum(np.array(diag["total"]) ** 2, 1e-300)
    summ = dict(mb_actor_grad_norm=float(np.mean(diag["actor"])), mb_critic_grad_norm=float(np.mean(diag["critic"])),
                mb_post_actor_grad_norm=float(np.mean(diag["post_actor"])), mb_critic_share=float(share.mean()),
                mb_clip_freq=float(np.mean(diag["clipped"])), explained_variance_last=ev,
                reward_transformed_mean=float(rr.mean()), reward_transformed_rms=float(np.sqrt(np.mean(rr ** 2))))
    return steps, summ


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--index", required=True, type=int)
    ap.add_argument("--budget", type=int, default=300_000); a = ap.parse_args()
    cond, arm, w, seed = decode(a.index); out = Path(a.out); out.mkdir(parents=True, exist_ok=False); t0 = time.monotonic()
    proto = dict(design="e11-cleanrl-v4", index=a.index, condition_name=cond, arm=arm, arm_scale=of.arm_scale(arm), w=w, seed=seed,
                 reference=of.REFERENCE, total_budget=a.budget, checkpoints=[a.budget])
    (out / "protocol.json").write_text(json.dumps(proto, indent=2) + "\n")
    here = Path(__file__); srcs = [here, here.with_name("e11_design.json"), here.with_name("optimizer_factorial.py"), here.with_name("scale_control.py"),
                                   here.with_name("horizon_supply.py")]
    (out / "manifest.json").write_text(json.dumps({"sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in srcs},
                                                   "torch": torch.__version__, "numpy": np.__version__}, indent=2) + "\n")
    steps, summ = train(cond, arm, w, seed, a.budget, out)
    Path(a.out, "COMPLETE.json").write_text(json.dumps({"requested": a.budget, "actual_transitions": steps, "elapsed_seconds": time.monotonic() - t0,
                                                         "diagnostics": summ}, indent=2) + "\n")
    print("COMPLETE", a.index, steps, flush=True)


if __name__ == "__main__":
    main()
