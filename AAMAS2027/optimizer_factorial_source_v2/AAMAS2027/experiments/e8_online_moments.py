"""E8 (v4, pre-declared): online moment-matched symmetric control, and an independent-seed replication of E7's P1.

Part A, arm M_on: an increasing affine map r -> a_r * r_B + b_r of B's curvature-only reward, symmetric in gains and
losses, whose constants are re-set before every rollout r so that M_on's training reward tracks A's per-rollout mean
and within-rollout SD under M_on's *own* policy. E7's static map matched A's moments under E1's policies, but M's
policies differ, and its achieved mean missed A's by more than the declared 25%. The targets mu*_r, sd*_r are the
seed-averaged per-rollout moments of A's transformed reward in the same critic mode (plain: E1 separate_no_clip/raw;
popart: E7 popart A), frozen in e8_targets.json before any E8 run. Before rollout r (r >= 1):
    a_r = clip(sd*_r / max(sdhat_B, 1e-3), a_w / 4, 4 a_w),   b_r = mu*_r - a_r * muhat_B,
where muhat_B and sdhat_B are exponential moving averages (weight 0.5) of B's per-rollout reward mean and SD under the
current policy, and a_w is E7's static scale. Rollout 0 uses E7's static constants. An affine map cannot change the
shape of B's reward distribution, so A - M_on compares A with a symmetric control that has the same first two
moments along training; what remains is the asymmetric shape.

Part B: B and S with fresh seeds 52-61 under both critics (the E7 design otherwise), to replicate E7's inconclusive
critic-target attribution (P1) in an independent cohort.

Learner, budget, grid and evaluation are E7's. Existing source files are not modified.
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
from instrumented_ppo import RolloutDiagnostics
import e7_critic_scale as e7

E8 = json.loads(Path(__file__).with_name("e8_design.json").read_text())
TARGETS = json.loads(Path(__file__).with_name("e8_targets.json").read_text())
CRITIC, GRID = e7.CRITIC, of.GRID
N_STEPS = 128
ONLINE = {}


def add_online_affine(vec, w, critic, static=False):
    """Wrap the vector env's step_wait (a closure: the env is cloudpickled at checkpoints). Every N_STEPS calls is one
    PPO rollout; at each rollout boundary the B-reward statistics of the finished rollout update the moving averages
    and the next rollout's (a_r, b_r) are set. static=True keeps E7's constants (used by the pre-launch gate)."""
    c = e7.DESIGN["M_affine"][f"{w:g}"]; a_w, b_w = float(c["scale"]), float(c["shift"])
    tgt = TARGETS[critic][f"{w:g}"]; mu_t, sd_t = tgt["mean"], tgt["sd"]
    st = dict(a=a_w, b=b_w, mu=None, sd=None, steps=0, buf=[], log=[], n=0, s1=0.0, s2=0.0)
    inner = vec.step_wait

    def step_wait():
        obs, rew_b, done, info = inner()
        st["buf"].append(np.asarray(rew_b, float).copy())
        rew = st["a"] * rew_b + st["b"]
        st["n"] += rew.size; st["s1"] += float(rew.sum()); st["s2"] += float((rew ** 2).sum())
        st["steps"] += 1
        if st["steps"] % N_STEPS == 0:
            r = st["steps"] // N_STEPS - 1                         # index of the rollout just finished
            x = np.concatenate(st["buf"]); st["buf"] = []
            m, s = float(x.mean()), float(x.std())
            st["log"].append(dict(rollout=r, a=st["a"], b=st["b"], b_mean=m, b_sd=s, m_mean=st["a"] * m + st["b"], m_sd=st["a"] * s,
                                  target_mean=mu_t[r], target_sd=sd_t[r]))
            st["mu"] = m if st["mu"] is None else 0.5 * st["mu"] + 0.5 * m
            st["sd"] = s if st["sd"] is None else 0.5 * st["sd"] + 0.5 * s
            nxt = r + 1
            if not static and nxt < len(mu_t):
                a = float(np.clip(sd_t[nxt] / max(st["sd"], 1e-3), a_w / 4, 4 * a_w))
                st["a"], st["b"] = a, float(mu_t[nxt] - a * st["mu"])
        return obs, rew, done, info

    vec.step_wait = step_wait
    return st


def decode(index):
    """Part A: 0-159 = critic x w x seed (42-51), arm M_on. Part B: 160-479 = critic x arm (B, S) x w x seed (52-61)."""
    if index < 160:
        seed = of.SEEDS[index % 10]; q = index // 10; w = GRID[q % len(GRID)]; critic = CRITIC[q // len(GRID)]
        return "A", critic, "M_on", w, seed
    i = index - 160
    assert i < 320
    seed = E8["partB_seeds"][i % 10]; q = i // 10; w = GRID[q % len(GRID)]; q //= len(GRID)
    arm = ["B", "S"][q % 2]; critic = CRITIC[q // 2]
    return "B", critic, arm, w, seed


def create(out, index, budget=300_000, static=False):
    part, critic, arm, w, seed = decode(index)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    base_arm = "B" if arm == "M_on" else arm
    vec = sc.vector(w, base_arm, seed, of.arm_scale(base_arm), of.REFERENCE)
    ONLINE["state"] = add_online_affine(vec, w, critic, static=static) if arm == "M_on" else None
    model = e7.CLASSES[critic]("MlpPolicy", vec, n_steps=N_STEPS, batch_size=256, seed=seed, verbose=0, device="cpu")
    model.diagnostics_dir = str(out / "diagnostics"); model.output = str(out); model.started = time.monotonic(); model.completed = []
    model.protocol = dict(design="e8-online-moments-v4", index=index, part=part, optimizer_mode="separate_no_clip", reward_mode="raw",
                          critic_mode=critic, arm=arm, arm_scale=of.arm_scale(base_arm), w=w, seed=seed, static_affine=static,
                          condition="curvature", reference=of.REFERENCE, checkpoints=[budget], total_budget=budget, phase="confirmation")
    model.verify_upstream()
    (out / "protocol.json").write_text(json.dumps(model.protocol, indent=2) + "\n")
    here = Path(__file__)
    srcs = [here, here.with_name("e8_design.json"), here.with_name("e8_targets.json"), here.with_name("e7_critic_scale.py"),
            here.with_name("e7_design.json"), here.with_name("optimizer_factorial.py"), here.with_name("instrumented_ppo.py"),
            here.with_name("scale_control.py")]
    (out / "manifest.json").write_text(json.dumps({"sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in srcs}}, indent=2) + "\n")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--index", required=True, type=int)
    ap.add_argument("--budget", type=int, default=300_000); ap.add_argument("--static", action="store_true")
    a = ap.parse_args()
    model = create(a.out, a.index, a.budget, a.static)
    model.learn(total_timesteps=a.budget, callback=RolloutDiagnostics())
    extra = {}
    if isinstance(model, e7.E7PopArt):
        model.save_popart(a.out); extra["popart_final"] = model._popart_log[-1] if model._popart_log else None
    st = ONLINE.get("state")
    if st is not None:
        n = max(st["n"], 1); m = st["s1"] / n
        extra["M_reward_moments"] = dict(n=st["n"], mean=m, rms=float(np.sqrt(st["s2"] / n)), sd=float(np.sqrt(max(st["s2"] / n - m * m, 0.0))))
        (Path(a.out) / "diagnostics").mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "diagnostics/online_affine.json").write_text(json.dumps(st["log"]) + "\n")
    Path(a.out, "COMPLETE.json").write_text(json.dumps({"requested": a.budget, "actual_transitions": model.num_timesteps, **extra}, indent=2) + "\n")
    print("COMPLETE", a.index, model.num_timesteps, flush=True)


if __name__ == "__main__":
    main()
