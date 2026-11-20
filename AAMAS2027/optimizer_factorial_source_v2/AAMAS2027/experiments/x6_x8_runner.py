"""X6, X7, X8 (pre-declared, frozen with PREREGISTRATION_X6_X8.md): three follow-up tests of the symmetric-scaling shift.

X6  Local critics. X5's decentralized execution (one actor network per node on its local view and identifier), but each
    node also has its OWN critic: a separate trunk on the node's local view and identifier and its own output row.
    Nothing in the critic reads the global state. Separate actor and critic Adam, no clipping, raw rewards.
    Parts: plain critic; fixed-scale head (V = c * head, c = 1 for B, 7 for S; X4's part M).
X7  The shared clip under decentralized execution. X5's architecture (per-node actors, centralized critic) trained like
    SB3's default: one Adam step over all parameters, global gradient-norm clip 0.5, raw rewards. Parts:
      plain        plain critic, shared global clip
      normhead     V = c * head, with the gradient reaching V scaled by 1/c^2, so that the value loss is the loss on
                   targets divided by c: the critic regresses targets AND sends gradients of B's size (c = 1 for B)
      normpopart   PopArt head (E7's statistics), with the gradient reaching V scaled by 1/sigma^2: PopArt's
                   normalized loss
      groupclip    plain critic, actor and critic clipped separately to 0.5 under one Adam step (E6/X1's per-group clip)
    X4's fixed-scale head and E7's PopArt both compute the loss on raw returns, so under a shared clip their critic
    gradients still grow with the reward; the normalized-loss variants remove that, which is what X7 tests.
X8  Fit at fixed target scale. The shared controller (X4's setup), separate Adam, no clipping, raw rewards. Arms B, S and
    Bdeg: arm B with the critic's learning rate multiplied by k < 1, so that its critic fits worse without any change in
    the scale of its targets; k is set by a calibration on seeds that are not X8 seeds, before the freeze.

X4's reference-critic logging (x4_runner.X4Logging) is used unchanged in all three. Existing source files are not
modified.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import torch
from torch import nn

import optimizer_factorial as of
import scale_control as sc
import e7_critic_scale as e7
import x5_runner as x5
from x4_runner import X4Logging, FixedScaleHead
from instrumented_ppo import InstrumentedPPO, GroupwiseClipPPO, SeparateNoClipPPO, RolloutDiagnostics

DESIGN = json.loads(Path(__file__).with_name("x6_x8_design.json").read_text())
GRID = of.GRID
HID = x5.HID


# ------------------------------------------------------------------ gradient scaling (X7's normalized losses)
class GradScale(torch.autograd.Function):
    """Identity in the forward pass; multiplies the incoming gradient by a constant in the backward pass."""

    @staticmethod
    def forward(ctx, x, s):
        ctx.s = s
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.s, None


class NormHead(FixedScaleHead):
    """V = c * linear(x), the gradient at V scaled by 1/c^2: identical to regressing linear(x) on targets / c.
    With c = 1 the forward and backward passes are those of the plain head (multiplication by exactly 1.0)."""

    def forward(self, x):
        c = float(self.c)
        return GradScale.apply(self.c * self.linear(x), 1.0 / (c * c))


class NormPopArtHead(e7.PopArtHead):
    """E7's PopArt head with the gradient at V scaled by 1/sigma^2: the loss on normalized targets."""

    def forward(self, x):
        s = float(self.sigma)
        return GradScale.apply(self.sigma * self.linear(x) + self.mu, 1.0 / (s * s))


class PopArtMixin:
    """E7PopArt's rescale-before-update, for any PPO base (E7PopArt itself is bound to SeparateNoClipPPO)."""
    head_class = e7.PopArtHead

    def _setup_model(self):
        super()._setup_model()
        self.policy.value_net = self.head_class(self.policy.value_net)
        self._popart_first = True; self._popart_log = []

    def train(self):
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


# ------------------------------------------------------------------ X6: per-node local critics
class RoutedLinear(nn.Module):
    """One output row per node: input [features | one-hot id], output sum_j id_j * (W_j . f + b_j). Exposes .weight and
    .bias (shapes (n, d) and (n,)) so that X4's FixedScaleHead and E7's PopArtHead can wrap it unchanged."""

    def __init__(self, d, n):
        super().__init__(); self.d, self.n = d, n
        lin = nn.Linear(d, n); self.weight, self.bias = lin.weight, lin.bias

    def forward(self, x):
        f, idv = x[:, :self.d], x[:, self.d:]
        return ((f @ self.weight.T + self.bias) * idv).sum(1, keepdim=True)


class LocalCriticExtractor(x5.DecExtractor):
    """X5's per-node actors, and per-node critic trunks on the local view and id; latent_vf = [trunk output | id]."""

    def __init__(self, n, L, G, act_dim):
        super().__init__(n, L, G, act_dim)
        self.value_net = nn.ModuleList([x5.mlp(L, HID) for _ in range(n)])
        self.latent_dim_vf = HID[-1] + n

    def forward_critic(self, x):
        idv = self.ids(x); loc = x[:, :self.L]; k = idv.argmax(1)
        feat = loc.new_zeros(len(x), HID[-1])
        for j, net in enumerate(self.value_net):
            sel = k == j
            if bool(sel.any()):
                feat = feat.index_put((sel.nonzero(as_tuple=True)[0],), net(loc[sel]))
        return torch.cat([feat, idv], 1)


class LocalCriticPolicy(x5.DecPolicy):
    def _build_mlp_extractor(self):
        self.mlp_extractor = LocalCriticExtractor(self.n_agents, self.L, self.G, int(np.prod(self.action_space.shape)))

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.value_net = RoutedLinear(HID[-1], self.n_agents)
        nn.init.orthogonal_(self.value_net.weight, gain=1.0); nn.init.zeros_(self.value_net.bias)   # SB3's value-head init
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


def independence_check_x6(model):
    """X5's actor checks, and: per-node critic trunks are disjoint, a node's critic receives exactly zero gradient from
    other nodes' samples, and no critic output depends on the global-state entries."""
    pol = model.policy; ex = pol.mlp_extractor; n = ex.n
    sets = [{id(p) for p in net.parameters()} for net in ex.value_net]
    assert all(not (sets[i] & sets[j]) for i in range(n) for j in range(i + 1, n)), "critic trunks are shared"
    obs = torch.as_tensor(np.asarray(model.rollout_buffer.observations).reshape(-1, ex.L + ex.G)[:256], dtype=torch.float32)
    for k in range(n):
        sel = obs[obs[:, ex.L - n + k] > 0.5]
        if not len(sel):
            continue
        pol.zero_grad(); pol.predict_values(sel).sum().backward()
        for j, net in enumerate(ex.value_net):
            g = sum(float(p.grad.abs().sum()) for p in net.parameters() if p.grad is not None)
            assert (g > 0) == (j == k), f"node {k}'s samples move node {j}'s critic (|grad| {g})"
        wg = pol.value_net.linear.weight.grad if hasattr(pol.value_net, "linear") else pol.value_net.weight.grad
        assert all(float(wg[j].abs().sum()) == 0 for j in range(n) if j != k), f"node {k}'s samples move another node's output row"
    pol.zero_grad()
    x = obs[:8].clone().requires_grad_(True); pol.predict_values(x).sum().backward()
    assert float(x.grad[:, ex.L:].abs().sum()) == 0.0, "a critic reads the global state"
    pol.zero_grad()
    groups = model.parameter_groups()
    return dict(n_nodes=n, actor_params_per_node=sum(p.numel() for p in ex.policy_net[0].parameters()),
                critic_params_per_node=sum(p.numel() for p in ex.value_net[0].parameters()) + HID[-1] + 1,
                critic_params=sum(p.numel() for p in groups["critic"]))


def actor_independence(model):
    """X5's actor checks (disjoint per-node actors, zero cross-node actor gradients) without its critic-input check."""
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
    pol.zero_grad()


# ------------------------------------------------------------------ learner classes
class X6Plain(X4Logging, e7.E7Plain):
    pass


class X6Matched(X4Logging, e7.E7Plain):
    def wrap_critic(self, c):
        self.policy.value_net = FixedScaleHead(self.policy.value_net, c)


class X7Global(X4Logging, sc.CheckpointPPO, InstrumentedPPO):
    parameter_groups = GroupwiseClipPPO.parameter_groups      # read-only helper for the independence check; not used in training


class X7NormHead(X7Global):
    def wrap_critic(self, c):
        self.policy.value_net = NormHead(self.policy.value_net, c)


class X7NormPopArt(X4Logging, PopArtMixin, sc.CheckpointPPO, InstrumentedPPO):
    head_class = NormPopArtHead
    parameter_groups = GroupwiseClipPPO.parameter_groups


class X7GroupClip(X4Logging, sc.CheckpointPPO, GroupwiseClipPPO):
    pass


class X8Plain(X4Logging, e7.E7Plain):
    critic_lr_multiplier = 1.0

    def update_optimizers(self):
        super().update_optimizers()
        if self.critic_lr_multiplier != 1.0:
            for g in self.critic_optimizer.param_groups:
                g["lr"] = g["lr"] * self.critic_lr_multiplier


class NoLogMixin:
    """Gate only: the same learner with X4's logging switched off (X4Logging.train is skipped)."""

    def train(self):
        super(X4Logging, self).train()


# ------------------------------------------------------------------ decoding
def spec_of(exp, index):
    d = DESIGN[exp]; seeds, arms, parts = d["seeds"], d["arms"], d["parts"]
    n = len(parts) * len(arms) * len(GRID) * len(seeds); assert 0 <= index < n, (exp, index, n)
    seed = seeds[index % len(seeds)]; q = index // len(seeds); w = GRID[q % len(GRID)]; q //= len(GRID)
    return dict(experiment=exp, part=parts[q // len(arms)], arm=arms[q % len(arms)], w=w, seed=seed)


def n_runs(exp):
    d = DESIGN[exp]; return len(d["parts"]) * len(d["arms"]) * len(GRID) * len(d["seeds"])


# ------------------------------------------------------------------ build and run
def build(spec, log=True):
    exp, part, arm, w, seed = spec["experiment"], spec["part"], spec["arm"], spec["w"], spec["seed"]
    c = of.arm_scale("B" if arm == "Bdeg" else arm)
    if exp in ("X6", "X7"):
        vec, n, L, G = x5.vector(w, arm, seed)
        kw = dict(n_steps=128, batch_size=256, seed=seed, verbose=0, device="cpu", policy_kwargs=dict(n=n, L=L, G=G))
        if exp == "X6":
            cls = {"plain": X6Plain, "M": X6Matched}[part]
            if not log:
                cls = type(cls.__name__ + "NoLog", (NoLogMixin, cls), {})
            model = cls(LocalCriticPolicy, vec, **kw)
            if part == "M":
                model.wrap_critic(c)
        else:
            cls = {"plain": X7Global, "normhead": X7NormHead, "normpopart": X7NormPopArt, "groupclip": X7GroupClip}[part]
            if not log:
                cls = type(cls.__name__ + "NoLog", (NoLogMixin, cls), {})
            model = cls(x5.DecPolicy, vec, **kw)
            if part == "normhead":
                model.wrap_critic(c)
    else:
        vec = e7.sc.vector(w, "B" if arm == "Bdeg" else arm, seed, c, of.REFERENCE)
        cls = X8Plain if log else type("X8PlainNoLog", (NoLogMixin, X8Plain), {})
        model = cls("MlpPolicy", vec, n_steps=128, batch_size=256, seed=seed, verbose=0, device="cpu")
        k = spec.get("critic_lr_multiplier", DESIGN["X8"]["critic_lr_multiplier"] if arm == "Bdeg" else 1.0)
        model.critic_lr_multiplier = float(k)
    return model, c


def run(spec, out, budget, index, log=True, check=False):
    out = Path(out); out.mkdir(parents=True, exist_ok=False); t0 = time.monotonic(); torch.set_num_threads(1)
    exp, part, arm, w, seed = spec["experiment"], spec["part"], spec["arm"], spec["w"], spec["seed"]
    model, c = build(spec, log)
    model._x4_init(seed)
    model.diagnostics_dir = str(out / "diagnostics"); model.output = str(out); model.started = time.monotonic(); model.completed = []
    opt = {"X6": "separate_no_clip", "X8": "separate_no_clip"}.get(exp) or ("groupwise_clip" if part == "groupclip" else "global_clip")
    model.protocol = dict(design=DESIGN["design"], index=index, experiment=exp, part=part, arm=arm, arm_scale=c, w=w, seed=seed,
                          optimizer_mode=opt, reward_mode="raw", max_grad_norm=float(model.max_grad_norm),
                          execution={"X6": "decentralized, local critics", "X7": "decentralized, centralized critic",
                                     "X8": "shared controller"}[exp],
                          critic_mode=part, critic_output_scale=c if part in ("M", "normhead") else None,
                          critic_lr_multiplier=getattr(model, "critic_lr_multiplier", None),
                          condition="curvature", reference=of.REFERENCE, logging=bool(log),
                          checkpoints=[] if exp != "X8" else [budget], total_budget=budget, phase="confirmation")
    model.verify_upstream()
    (out / "protocol.json").write_text(json.dumps(model.protocol, indent=2) + "\n")
    here = Path(__file__)
    srcs = [here, here.with_name("x6_x8_design.json"), here.with_name("x5_runner.py"), here.with_name("x5_design.json"),
            here.with_name("x4_runner.py"), here.with_name("x4_design.json"), here.with_name("e12_localobs.py"),
            here.with_name("e7_critic_scale.py"), here.with_name("e7_design.json"), here.with_name("optimizer_factorial.py"),
            here.with_name("instrumented_ppo.py"), here.with_name("scale_control.py"), here.with_name("horizon_supply.py")]
    (out / "manifest.json").write_text(json.dumps({"sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in srcs},
                                                   "torch": torch.__version__, "numpy": np.__version__}, indent=2) + "\n")
    model.learn(total_timesteps=budget, callback=RolloutDiagnostics())
    extra = {}
    if check and exp == "X6":
        actor_independence(model); extra["independence_check"] = independence_check_x6(model)
    elif check and exp == "X7":
        extra["independence_check"] = x5.independence_check(model)
    if hasattr(model, "_popart_log"):
        model.save_popart(out); extra["popart_final"] = model._popart_log[-1] if model._popart_log else None
    if exp in ("X6", "X7"):
        res = x5.evaluate(model, model.protocol, 10); res.update(requested_checkpoint=budget, actual_agent_transitions=model.num_timesteps)
        (out / f"evaluation_{budget}.json").write_text(json.dumps(res, indent=2))
    (out / "COMPLETE.json").write_text(json.dumps({"requested": budget, "actual_transitions": model.num_timesteps,
                                                   "elapsed_seconds": time.monotonic() - t0, **extra}, indent=2) + "\n")
    print("COMPLETE", exp, index, model.num_timesteps, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--experiment", required=True, choices=["X6", "X7", "X8"])
    ap.add_argument("--out", required=True); ap.add_argument("--index", type=int); ap.add_argument("--budget", type=int, default=300_000)
    ap.add_argument("--part"); ap.add_argument("--arm"); ap.add_argument("--w", type=float); ap.add_argument("--seed", type=int)
    ap.add_argument("--critic-lr-multiplier", type=float, help="X8 calibration only")
    ap.add_argument("--no-log", action="store_true", help="gate only"); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.index is not None:
        spec = spec_of(a.experiment, a.index)
    else:
        spec = dict(experiment=a.experiment, part=a.part, arm=a.arm, w=a.w, seed=a.seed)
    if a.critic_lr_multiplier is not None:
        spec["critic_lr_multiplier"] = a.critic_lr_multiplier
    run(spec, a.out, a.budget, a.index if a.index is not None else -1, log=not a.no_log, check=a.check)


if __name__ == "__main__":
    main()
