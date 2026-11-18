"""X4 (v5, pre-declared): does the critic's fit carry the symmetric-scaling shift?

Supply chain, separate actor and critic Adam without gradient clipping (E1 separate_no_clip / E7), raw rewards, the E1
grid, arms B and S, fresh seeds 122-161, 300k. Three parts:
  D-plain  E7's plain critic (= E1 separate_no_clip/raw) with advantage-quality logging
  D-pop    E7's PopArt critic with the same logging
  M        the plain critic with its output fixed-scaled by the arm's reward scale (B: 1, S: 7), same logging
Logging fits a separate reference critic to standardized Monte-Carlo returns and compares, out of sample, the learner's
values and advantages with the reference's. Everything the logging draws comes from their own
torch generators inside torch.random.fork_rng, with NumPy's global state saved and restored, so D-plain and D-pop
reproduce E1/E7 runs bit-for-bit, and M with c = 1 reproduces D-plain (gate). Existing source files are not modified.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

import optimizer_factorial as of
import e7_critic_scale as e7
from instrumented_ppo import RolloutDiagnostics

DESIGN = json.loads(Path(__file__).with_name("x4_design.json").read_text())
SEEDS, ARMS, PARTS, GRID = DESIGN["seeds"], DESIGN["arms"], DESIGN["parts"], of.GRID
N = len(PARTS) * len(ARMS) * len(GRID) * len(SEEDS)            # 960
REF = DESIGN["reference_critic"]


def ev(pred, target):
    v = float(np.var(target))
    return float("nan") if v == 0 else float(1 - np.var(target - pred) / v)


def ranks(x):
    r = np.empty(len(x)); r[np.argsort(x, kind="stable")] = np.arange(len(x)); return r


def zs(x):
    s = x.std(); return (x - x.mean()) / (s if s > 0 else 1.0)


def gae(rewards, values, starts, last_value, last_start, gamma, lam):
    """SB3 RolloutBuffer.compute_returns_and_advantage, with arbitrary values."""
    T = len(rewards); adv = np.zeros_like(rewards); last = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            nonterm, nxt = 1.0 - last_start, last_value
        else:
            nonterm, nxt = 1.0 - starts[t + 1], values[t + 1]
        delta = rewards[t] + gamma * nxt * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last; adv[t] = last
    return adv


def mc_returns(rewards, starts, last_start, gamma):
    """Discounted return to the episode's end; valid only where that end lies inside the rollout."""
    T = len(rewards); G = np.zeros_like(rewards); ok = np.zeros(rewards.shape, bool)
    for t in reversed(range(T)):
        end = (last_start.astype(bool) if t == T - 1 else starts[t + 1].astype(bool))
        if t == T - 1:
            G[t] = rewards[t]; ok[t] = end
        else:
            G[t] = np.where(end, rewards[t], rewards[t] + gamma * G[t + 1]); ok[t] = np.where(end, True, ok[t + 1])
    return G, ok


class Isolated:
    """Run a block without touching the training RNGs: torch's CPU generator is forked, NumPy's global state restored."""

    def __enter__(self):
        self.np = np.random.get_state(); self.fork = torch.random.fork_rng(devices=[]); self.fork.__enter__()

    def __exit__(self, *exc):
        self.fork.__exit__(*exc); np.random.set_state(self.np); return False


class FixedScaleHead(torch.nn.Module):
    """V(x) = c * linear(x), c fixed (part M). Keeps the wrapped nn.Linear's parameter objects and the 'value_net' prefix."""

    def __init__(self, linear, c):
        super().__init__(); self.linear = linear; self.register_buffer("c", torch.full((1,), float(c)))

    def forward(self, x):
        return self.c * self.linear(x)


class X4Logging:

    def _x4_init(self, seed):
        obs_dim = self.observation_space.shape[0]
        with Isolated():
            torch.manual_seed(REF["torch_seed_offset"] + seed)
            h = REF["hidden"]; layers, d = [], obs_dim
            for k in h:
                layers += [torch.nn.Linear(d, k), torch.nn.Tanh()]; d = k
            self.ref_net = torch.nn.Sequential(*layers, torch.nn.Linear(d, 1)).double()
        self.ref_opt = torch.optim.Adam(self.ref_net.parameters(), lr=REF["lr"])
        self.ref_gen = torch.Generator().manual_seed(REF["torch_seed_offset"] + 1 + seed)
        self.ref_stats = None; self.x4_k = 0

    def _ref_values(self, obs):
        m, s = self.ref_stats
        with torch.no_grad():
            return m + s * self.ref_net(torch.as_tensor(obs, dtype=torch.float64)).flatten().numpy()

    def _learner_values(self, obs):
        with torch.no_grad():
            return self.policy.predict_values(torch.as_tensor(obs, dtype=torch.float32, device=self.device)).flatten().double().numpy()

    def _measure(self):
        buf = self.rollout_buffer; T, E = buf.rewards.shape
        r = buf.rewards.astype(np.float64); st = buf.episode_starts.astype(np.float64); v = buf.values.astype(np.float64)
        ls = np.asarray(self._last_episode_starts, np.float64)
        G, ok = mc_returns(r, st, ls, self.gamma)
        obs = buf.observations.reshape(T, E, -1).astype(np.float64)
        rec = dict(k=self.x4_k, t=int(self.num_timesteps), n_valid=int(ok.sum()),
                   ev_sb3=ev(v.ravel(), buf.returns.astype(np.float64).ravel()), ev_train_mc=ev(v[ok], G[ok]),
                   verr_train=float(np.sqrt(np.mean((v[ok] - G[ok]) ** 2)) / max(G[ok].std(), 1e-12)))
        if self.ref_stats is not None:
            vr = self._ref_values(obs.reshape(T * E, -1)).reshape(T, E)
            vlast = self._ref_values(np.asarray(self._last_obs, np.float64).reshape(E, -1))
            ar = gae(r, vr, st, vlast, ls, self.gamma, self.gae_lambda).ravel(); at = buf.advantages.astype(np.float64).ravel()
            zt, zr = zs(at), zs(ar)
            rec.update(ev_ref_mc=ev(vr[ok], G[ok]), verr_ref=float(np.sqrt(np.mean((vr[ok] - G[ok]) ** 2)) / max(G[ok].std(), 1e-12)),
                       gae_err=float(np.sqrt(np.mean((zt - zr) ** 2))), adv_pearson=float(np.mean(zt * zr)),
                       adv_spearman=float(np.corrcoef(ranks(at), ranks(ar))[0, 1]),
                       sign_dis=float(np.mean(np.sign(zt) != np.sign(zr))), sign_dis_raw=float(np.mean(np.sign(at) != np.sign(ar))))
        # train the reference on this rollout (used from the next rollout on)
        x = torch.as_tensor(obs[ok], dtype=torch.float64); g = G[ok]; m, s = float(g.mean()), float(max(g.std(), 1e-8))
        y = torch.as_tensor((g - m) / s, dtype=torch.float64); n = len(y); bs = REF["batch_size"]
        for _ in range(REF["epochs_per_rollout"]):
            perm = torch.randperm(n, generator=self.ref_gen)
            for i in range(0, n, bs):
                j = perm[i:i + bs]; loss = F.mse_loss(self.ref_net(x[j]).flatten(), y[j])
                self.ref_opt.zero_grad(); loss.backward(); self.ref_opt.step()
        with torch.no_grad():
            rec["ev_ref_insample"] = float(1 - torch.var(y - self.ref_net(x).flatten()) / torch.var(y))
        self.ref_stats = (m, s)
        return rec, obs.reshape(T * E, -1)

    def train(self):
        with Isolated():
            rec, obs = self._measure()
        super().train()
        with Isolated():
            rec["ev_post_insample"] = ev(self._learner_values(obs), self.rollout_buffer.returns.astype(np.float64).ravel())
        self.x4_k += 1
        p = Path(self.diagnostics_dir) / "x4.jsonl.gz"; p.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(p, "at") as fh:
            fh.write(json.dumps(rec, allow_nan=True) + "\n")


class X4Plain(X4Logging, e7.E7Plain):
    pass


class X4Pop(X4Logging, e7.E7PopArt):
    pass


class X4Matched(X4Logging, e7.E7Plain):
    def wrap_critic(self, c):
        self.policy.value_net = FixedScaleHead(self.policy.value_net, c)


def decode(index):
    assert 0 <= index < N
    seed = SEEDS[index % len(SEEDS)]; q = index // len(SEEDS); w = GRID[q % len(GRID)]; q //= len(GRID)
    return dict(part=PARTS[q // len(ARMS)], arm=ARMS[q % len(ARMS)], w=w, seed=seed)


def run(spec, out, budget, index):
    out = Path(out); out.mkdir(parents=True, exist_ok=False); t0 = time.monotonic(); torch.set_num_threads(1)
    part, arm, w, seed = spec["part"], spec["arm"], spec["w"], spec["seed"]
    vec = e7.sc.vector(w, arm, seed, of.arm_scale(arm), of.REFERENCE)
    cls = {"D-plain": X4Plain, "D-pop": X4Pop, "M": X4Matched}[part]
    model = cls("MlpPolicy", vec, n_steps=128, batch_size=256, seed=seed, verbose=0, device="cpu")
    model.part = part
    if part == "M":
        model.wrap_critic(of.arm_scale(arm))
    model._x4_init(seed)
    model.diagnostics_dir = str(out / "diagnostics"); model.output = str(out); model.started = time.monotonic(); model.completed = []
    model.protocol = dict(design="x4-v5", index=index, experiment="X4", part=part, critic_mode={"D-plain": "plain", "D-pop": "popart", "M": "scale_matched"}[part],
                          critic_output_scale=of.arm_scale(arm) if part == "M" else None, optimizer_mode="separate_no_clip", reward_mode="raw", arm=arm,
                          arm_scale=of.arm_scale(arm), w=w, seed=seed, condition="curvature", reference=of.REFERENCE,
                          checkpoints=[budget], total_budget=budget, phase="confirmation")
    model.verify_upstream()
    (out / "protocol.json").write_text(json.dumps(model.protocol, indent=2) + "\n")
    here = Path(__file__)
    srcs = [here, here.with_name("x4_design.json"), here.with_name("e7_critic_scale.py"), here.with_name("e7_design.json"),
            here.with_name("optimizer_factorial.py"), here.with_name("instrumented_ppo.py"), here.with_name("scale_control.py"),
            here.with_name("horizon_supply.py")]
    (out / "manifest.json").write_text(json.dumps({"sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in srcs},
                                                   "torch": torch.__version__, "numpy": np.__version__}, indent=2) + "\n")
    model.learn(total_timesteps=budget, callback=RolloutDiagnostics())
    extra = {}
    if isinstance(model, e7.E7PopArt):
        model.save_popart(out); extra["popart_final"] = model._popart_log[-1] if model._popart_log else None
    (out / "COMPLETE.json").write_text(json.dumps({"requested": budget, "actual_transitions": model.num_timesteps,
                                                   "elapsed_seconds": time.monotonic() - t0, **extra}, indent=2) + "\n")
    print("COMPLETE", index, model.num_timesteps, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--index", type=int)
    ap.add_argument("--budget", type=int, default=300_000)
    ap.add_argument("--part"); ap.add_argument("--arm"); ap.add_argument("--w", type=float); ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    spec = decode(a.index) if a.index is not None else dict(part=a.part, arm=a.arm, w=a.w, seed=a.seed)
    run(spec, a.out, a.budget, a.index if a.index is not None else -1)


if __name__ == "__main__":
    main()
