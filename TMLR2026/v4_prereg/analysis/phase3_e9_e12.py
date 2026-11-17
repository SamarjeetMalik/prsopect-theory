"""Phase 3 analysis of E9-E12, per PREREGISTRATION_E9_E10.md and PREREGISTRATION_E11_E12.md. Written before any
E9-E12 run. Same estimator as phase3.py.

Usage: phase3_e9_e12.py RUNS_DIR ARTIFACT_DIR OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUNS, ART, OUT = map(Path, sys.argv[1:4]); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ART / "AAMAS2027/analysis")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from boundary import firth  # noqa: E402
from isotonic import isotonic_crossing  # noqa: E402
from labels import check_supply_table  # noqa: E402

GRID = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]
S10, S20, S40 = list(range(42, 52)), list(range(62, 82)), list(range(42, 82))
W10 = np.random.default_rng(20261015).multinomial(10, np.full(10, 0.1), size=2000)
W20 = np.random.default_rng(20261108).multinomial(20, np.full(20, 0.05), size=2000)
W40 = np.random.default_rng(20261109).multinomial(40, np.full(40, 0.025), size=2000)
log = []
def say(s=""): print(s); log.append(s)
def ci(z, level=0.95):
    z = z[np.isfinite(z)]; a = 100 * (1 - level) / 2; return np.percentile(z, [a, 100 - a])
def fmt(d, lo, hi): return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def fit(frame, seeds, W):
    k = frame.assign(w=frame.w.round(4)).pivot(index="seed", columns="w", values="collapsed").reindex(index=seeds, columns=GRID)
    assert not k.isna().any().any(), "missing cells"
    k = k.to_numpy(float); n = np.ones_like(k); w = np.array(GRID)
    p = firth(w, k.sum(0), n.sum(0)); b = firth(w, W @ k, W @ n)
    return dict(w50=p["wcrit"][0], boot=b["wcrit"], in_grid=bool(p["in_grid"][0]), iso=isotonic_crossing(w, k.sum(0), n.sum(0)))


def contrast(fa, fb, label, rows):
    z = fa["boot"] - fb["boot"]; lo, hi = ci(z); lo9, hi9 = ci(z, 0.99); d = fa["w50"] - fb["w50"]
    rows.append(dict(contrast=label, difference=d, ci_low=lo, ci_high=hi, ci_low_99=lo9, ci_high_99=hi9, excludes_zero=bool(lo > 0 or hi < 0)))
    say(f"  {label:44s} {fmt(d, lo, hi)}  99% [{lo9:+.3f}, {hi9:+.3f}]{'  *' if lo > 0 or hi < 0 else ''}")
    return z, d


def ddiff(z1, d1, z2, d2, label, rows):
    z = z1 - z2; lo, hi = ci(z); lo9, hi9 = ci(z, 0.99); d = d1 - d2
    rows.append(dict(contrast=label, difference=d, ci_low=lo, ci_high=hi, ci_low_99=lo9, ci_high_99=hi9, excludes_zero=bool(lo > 0 or hi < 0)))
    say(f"  {label:44s} {fmt(d, lo, hi)}  99% [{lo9:+.3f}, {hi9:+.3f}]{'  *' if lo > 0 or hi < 0 else ''}")
    return d, lo, hi, lo9, hi9


e1 = pd.read_csv(RUNS / "E1_runs.csv"); e7 = pd.read_csv(RUNS / "E7_runs.csv"); e8 = pd.read_csv(RUNS / "E8_runs.csv")
r910 = pd.read_csv(RUNS / "E9E10_runs.csv"); r11 = pd.read_csv(RUNS / "E11_runs.csv"); r12 = pd.read_csv(RUNS / "E12_runs.csv")
for t in (e1, e7, e8, r910, r11, r12): check_supply_table(t)
for t, n in [(r910, 1280), (r11, 720), (r12, 480)]:
    assert len(t) == n and t.complete.all() and (t.actual == 300160).all() and (t.n_eps == 10).all() and t.eval_seeds_ok.all(), n
say("== E9-E12: 1,280 + 720 + 480 runs complete, 300,160 transitions, declared evaluation seeds")

# ---------------------------------------------------------------- E9
e9 = r910[r910.experiment == "E9"]; ref0 = e1[(e1.optimizer_mode == "global_clip") & (e1.reward_mode == "raw")]
refinf = e1[(e1.optimizer_mode == "separate_no_clip") & (e1.reward_mode == "raw")]
levels = [("0.5 (E1)", ref0)] + [(f"{t:g}", e9[e9.max_grad_norm == t]) for t in sorted(e9.max_grad_norm.unique())] + [("inf (E1 no clip)", refinf)]
say("\n== E9 clip-threshold dose-response (default learner, raw rewards)")
rows, zs, mech = [], {}, []
for lab, f in levels:
    fb, fs = fit(f[f.arm == "B"], S10, W10), fit(f[f.arm == "S"], S10, W10)
    zs[lab] = contrast(fs, fb, f"S-B, tau {lab}", rows)
    ratio = f[f.arm == "S"].mb_post_actor_grad_norm.sum() / f[f.arm == "B"].mb_post_actor_grad_norm.sum()
    share = (f.mb_critic_grad_norm ** 2 / (f.mb_critic_grad_norm ** 2 + f.mb_actor_grad_norm ** 2)).mean()
    mech.append(dict(tau=lab, applied_ratio_SB=ratio, critic_share_cellmean=share, S_minus_B=zs[lab][1]))
M = pd.DataFrame(mech); M.to_csv(OUT / "E9_mechanism.csv", index=False); say(M.round(3).to_string(index=False))
r9 = []
p2 = ddiff(*zs["30000"], *zs["0.5 (E1)"], "d(S-B) tau 30000 - tau 0.5", r9)
p3 = ddiff(*zs["30000"], *zs["inf (E1 no clip)"], "d(S-B) tau 30000 - no clip", r9)
pd.DataFrame(rows + r9).to_csv(OUT / "E9_contrasts.csv", index=False)

# ---------------------------------------------------------------- E10
say("\n== E10 critic-target replication, seeds 62-81 (primary) and pooled 42-81 (secondary)")
e10 = r910[r910.experiment == "E10"]
p1x = e1[(e1.optimizer_mode == "separate_no_clip") & (e1.reward_mode == "raw") & e1.arm.isin(["B", "S"])].assign(critic_mode="plain")
pool = pd.concat([p1x, e7[e7.arm.isin(["B", "S"]) & (e7.critic_mode == "popart")], e8[e8.part == "B"], e10], ignore_index=True)
rows = []; res10 = {}
for tag, data, seeds, W in [("primary 62-81", e10, S20, W20), ("pooled 42-81", pool, S40, W40)]:
    F = {(m, a): fit(data[(data.critic_mode == m) & (data.arm == a)], seeds, W) for m in ["plain", "popart"] for a in ["B", "S"]}
    zp = contrast(F[("plain", "S")], F[("plain", "B")], f"{tag}: plain S-B", rows)
    zq = contrast(F[("popart", "S")], F[("popart", "B")], f"{tag}: popart S-B", rows)
    res10[tag] = (zp, zq, ddiff(*zq, *zp, f"{tag}: d(S-B) popart - plain", rows))
pd.DataFrame(rows).to_csv(OUT / "E10_contrasts.csv", index=False)

# ---------------------------------------------------------------- E11
say("\n== E11 CleanRL-style second implementation")
rows = []; F11 = {(c, a): fit(r11[(r11.condition_name == c) & (r11.arm == a)], S10, W10) for c in ["cleanrl_raw", "cleanrl_default", "cleanrl_group"] for a in ["B", "A", "S"]}
z11 = {}
for c in ["cleanrl_raw", "cleanrl_default", "cleanrl_group"]:
    z11[(c, "S-B")] = contrast(F11[(c, "S")], F11[(c, "B")], f"{c}: S-B", rows)
    z11[(c, "A-S")] = contrast(F11[(c, "A")], F11[(c, "S")], f"{c}: A-S", rows)
d11 = ddiff(*z11[("cleanrl_group", "S-B")], *z11[("cleanrl_raw", "S-B")], "d(S-B) group - raw", rows)
pd.DataFrame(rows).to_csv(OUT / "E11_contrasts.csv", index=False)
mech = []
for c in ["cleanrl_raw", "cleanrl_default", "cleanrl_group"]:
    f = r11[r11.condition_name == c]
    mech.append(dict(condition=c, applied_ratio_SB=f[f.arm == "S"].mb_post_actor_grad_norm.sum() / f[f.arm == "B"].mb_post_actor_grad_norm.sum(),
                     critic_share=f.mb_critic_share.mean(), ev_B=f[f.arm == "B"].explained_variance_last.mean(), ev_S=f[f.arm == "S"].explained_variance_last.mean()))
M11 = pd.DataFrame(mech); M11.to_csv(OUT / "E11_mechanism.csv", index=False); say(M11.round(3).to_string(index=False))

# ---------------------------------------------------------------- E12
say("\n== E12 local observations with agent identity")
rows = []; F12 = {(l, a): fit(r12[(r12.learner == l) & (r12.arm == a)], S10, W10) for l in ["default", "decoupled"] for a in ["B", "A", "S"]}
z12 = {}
for l in ["default", "decoupled"]:
    z12[(l, "S-B")] = contrast(F12[(l, "S")], F12[(l, "B")], f"{l}: S-B", rows)
    z12[(l, "A-S")] = contrast(F12[(l, "A")], F12[(l, "S")], f"{l}: A-S", rows)
pd.DataFrame(rows).to_csv(OUT / "E12_contrasts.csv", index=False)
cr = []
for (l, a), x in F12.items():
    ref = e1[(e1.optimizer_mode == ("global_clip" if l == "default" else "separate_no_clip")) & (e1.reward_mode == ("raw" if l == "default" else "vecnormalize")) & (e1.arm == a)]
    cr.append(dict(learner=l, arm=a, w50_local=x["w50"], w50_global_E1=fit(ref, S10, W10)["w50"]))
pd.DataFrame(cr).to_csv(OUT / "E12_crossings_vs_E1.csv", index=False); say(pd.DataFrame(cr).round(3).to_string(index=False))

# ---------------------------------------------------------------- predictions
def g(rows_csv, lab):
    t = pd.read_csv(OUT / rows_csv); return t[t.contrast == lab].iloc[0]
say("\n== Pre-declared predictions")
ratios = list(M.applied_ratio_SB)
say(f"E9 P1 applied S/B ratio monotone in tau and > 0.5 at 30000: {[round(r, 3) for r in ratios]} -> "
    f"{'MET' if all(np.diff(ratios[:-1]) > 0) and M.set_index('tau').loc['30000'].applied_ratio_SB > 0.5 else 'NOT MET'}")
say(f"E9 P2 d(S-B) 30000 - 0.5 < 0: {fmt(p2[0], p2[1], p2[2])} -> {'MET' if p2[2] < 0 else 'NOT MET'}")
say(f"E9 P3 d(S-B) 30000 - no clip includes 0: {fmt(p3[0], p3[1], p3[2])} -> {'MET' if p3[1] <= 0 <= p3[2] else 'NOT MET'}")
zp, zq, dd = res10["primary 62-81"]; pq = g("E10_contrasts.csv", "primary 62-81: popart S-B"); pp = g("E10_contrasts.csv", "primary 62-81: plain S-B")
if pq.ci_low > 0: v5 = "NOT MET: target normalization does not remove the residual"
elif dd[4] < 0: v5 = "MET at the 99% level"
elif dd[2] < 0: v5 = "supported at the declared 95% level only"
else: v5 = "NOT MET"
say(f"E10 P5 primary: popart S-B {fmt(pq.difference, pq.ci_low, pq.ci_high)}; change {fmt(dd[0], dd[1], dd[2])} (99% [{dd[3]:+.3f}, {dd[4]:+.3f}]) -> {v5}")
say(f"E10 P6 plain S-B > 0: {fmt(pp.difference, pp.ci_low, pp.ci_high)} -> {'MET' if pp.ci_low > 0 else 'NOT MET'}")
a = g("E11_contrasts.csv", "cleanrl_raw: S-B"); b = g("E11_contrasts.csv", "cleanrl_default: S-B"); c = g("E11_contrasts.csv", "cleanrl_default: A-S")
say(f"E11 P1 raw S-B > 0: {fmt(a.difference, a.ci_low, a.ci_high)} -> {'MET' if a.ci_low > 0 else 'NOT MET'}")
say(f"E11 P2 d(S-B) group - raw < 0: {fmt(d11[0], d11[1], d11[2])} -> {'MET' if d11[2] < 0 else 'NOT MET'}")
say(f"E11 P3 default S-B includes 0: {fmt(b.difference, b.ci_low, b.ci_high)} -> {'MET' if b.ci_low <= 0 <= b.ci_high else 'NOT MET'}")
say(f"E11 P4 default A-S > 0: {fmt(c.difference, c.ci_low, c.ci_high)} -> {'MET' if c.ci_low > 0 else 'NOT MET'}")
a = g("E12_contrasts.csv", "default: S-B"); b = g("E12_contrasts.csv", "decoupled: S-B"); c = g("E12_contrasts.csv", "decoupled: A-S")
say(f"E12 P5 default S-B > 0: {fmt(a.difference, a.ci_low, a.ci_high)} -> {'MET' if a.ci_low > 0 else 'NOT MET'}")
say(f"E12 P6 decoupled S-B includes 0: {fmt(b.difference, b.ci_low, b.ci_high)} -> {'MET' if b.ci_low <= 0 <= b.ci_high else 'NOT MET'}")
say(f"E12 P7 decoupled A-S > 0: {fmt(c.difference, c.ci_low, c.ci_high)} -> {'MET' if c.ci_low > 0 else 'NOT MET'}")
(OUT / "phase3_e9_e12_report.txt").write_text("\n".join(log) + "\n")
