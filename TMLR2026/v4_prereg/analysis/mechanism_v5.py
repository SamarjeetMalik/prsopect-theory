"""Six-panel optimizer-mechanism figure at the training-run level (revision v5; descriptive, uses only stored per-run
records and stored contrasts; no number in it is new evidence of causation, which comes from the interventions).

A  critic pre-clip gradient norm against the reward multiplier (E1 default cell: B x1, R x3.451, S x7)
B  per-run actor attenuation (applied / pre-clip actor norm) against the critic's pre-clip norm, for every clip
   threshold tau (E1, E9), with the shared-clip curve min(1, tau / ||g||)
C  seed-paired S/B applied-actor ratio against the S/B critic-norm ratio (same seed and mixture cell), by tau
D  seed-paired S/B ratio of the actor's actual parameter update, by clipping mode (E6)
E  S-B crossing shift against the measured applied S/B ratio across tau (E9, E1)
F  S-B under each intervention (E1, E6, E8, E10, E11), with paired 95% seed-block intervals

Usage: mechanism_v5.py RUNS_DIR RES_DIR OUT_PDF
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RUNS, RES, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
COL = {"B": "#0072B2", "R": "#E69F00", "S": "#009E73", "A": "#D55E00"}
e1 = pd.read_csv(RUNS / "E1_runs.csv"); e6 = pd.read_csv(RUNS / "E6_runs.csv"); r910 = pd.read_csv(RUNS / "E9E10_runs.csv")
d1 = e1[(e1.optimizer_mode == "global_clip") & (e1.reward_mode == "raw")].assign(tau=0.5)
e9 = r910[r910.experiment == "E9"].assign(tau=lambda d: d.max_grad_norm)
clip = pd.concat([d1[d1.arm.isin(["B", "S"])], e9], ignore_index=True)
TAUS = [0.5, 300.0, 3000.0, 10000.0, 30000.0]
TCOL = dict(zip(TAUS, plt.cm.viridis(np.linspace(0.05, 0.85, len(TAUS)))))
rng = np.random.default_rng(0)
fig, ax = plt.subplots(2, 3, figsize=(7.2, 5.0))

# A
a = ax[0, 0]
for arm, s in [("B", 1.0), ("R", 3.451), ("S", 7.0)]:
    y = d1[d1.arm == arm].mb_critic_grad_norm
    a.scatter(s * np.exp(rng.normal(0, 0.04, len(y))), y, s=4, alpha=.35, color=COL[arm], lw=0)
    a.plot(s, y.median(), "o", color="black", ms=3.5)
mB = d1[d1.arm == "B"].mb_critic_grad_norm.median(); xs = np.array([0.9, 8])
a.plot(xs, mB * xs, ls="--", color="0.4", lw=.8, label=r"$\propto s$")
a.set_xscale("log"); a.set_yscale("log"); a.set_xticks([1, 3.451, 7]); a.set_xticklabels(["1\n(B)", "3.45\n(R)", "7\n(S)"]); a.minorticks_off(); a.tick_params(axis="x", which="minor", bottom=False)
a.set_xlabel("reward multiplier $s$"); a.set_ylabel(r"critic pre-clip $\|g_c\|$ (run mean)"); a.set_title("A  critic gradient vs. $s$")
a.legend(frameon=False, loc="upper left")

# B
b = ax[0, 1]
for t in TAUS:
    d = clip[clip.tau == t]; k = d.mb_post_actor_grad_norm / d.mb_actor_grad_norm
    b.scatter(d.mb_total_grad_norm, k, s=4, alpha=.4, color=TCOL[t], lw=0, label=fr"$\tau={t:g}$")
    xx = np.logspace(2.5, 5, 50); b.plot(xx, np.minimum(1, t / xx), color=TCOL[t], lw=.8)
b.set_xscale("log"); b.set_yscale("log"); b.set_xlabel(r"joint pre-clip $\|g\|$ (run mean)")
b.set_ylabel(r"actor applied / pre-clip (run mean)"); b.set_title(r"B  $\kappa\approx\min(1,\tau/\|g\|)$")
b.legend(frameon=False, ncol=1, loc="lower left", markerscale=2)

# C
c = ax[0, 2]
for t in TAUS:
    d = clip[clip.tau == t]
    p = d.pivot_table(index=["seed", "w"], columns="arm", values=["mb_post_actor_grad_norm", "mb_critic_grad_norm"])
    rc = p[("mb_critic_grad_norm", "S")] / p[("mb_critic_grad_norm", "B")]; ra = p[("mb_post_actor_grad_norm", "S")] / p[("mb_post_actor_grad_norm", "B")]
    c.scatter(rc, ra, s=5, alpha=.5, color=TCOL[t], lw=0)
xx = np.logspace(-0.3, 1.6, 50); c.plot(xx, 1 / xx, ls="--", color="0.4", lw=.8, label=r"$1/x$")
c.axhline(1, color="0.7", lw=.6)
c.set_xscale("log"); c.set_yscale("log"); c.set_xlabel(r"S/B critic $\|g_c\|$ (paired)")
c.set_ylabel("S/B applied actor gradient (paired)"); c.set_title("C  seed-paired attenuation"); c.legend(frameon=False, loc="lower left")

# D
dd = ax[1, 0]
MODES = [("global_clip", "shared clip"), ("groupwise_clip", "per-group clip"), ("separate_no_clip", "no clip")]
for i, (mo, lab) in enumerate(MODES):
    d = e6[e6.optimizer_mode == mo]
    for j, (q, mk, off) in enumerate([("pu_applied_actor_grad_norm", "s", -0.15), ("pu_actor_update_norm", "o", 0.15)]):
        p = d.pivot_table(index=["seed", "w"], columns="arm", values=q); r = (p.S / p.B).dropna()
        dd.scatter(i + off + rng.normal(0, 0.03, len(r)), r, s=4, alpha=.35, color=["#7f7f7f", "black"][j], lw=0)
        dd.plot(i + off, np.median(r), mk, color=["#7f7f7f", "black"][j], ms=4, mfc="white" if j == 0 else "black",
                label=["applied gradient", "parameter update"][j] if i == 0 else None)
dd.axhline(1, color="0.7", lw=.6); dd.set_yscale("log"); dd.set_xticks(range(3)); dd.set_xticklabels(["shared", "per-group", "none"]); dd.set_xlabel("clipping")
dd.set_ylabel("S/B, same seed and $w$"); dd.set_title("D  update vs. gradient (E6)"); dd.legend(frameon=False, loc="lower right", fontsize=5.5)

# E
e = ax[1, 1]
k9 = pd.read_csv(RES / "E9_contrasts.csv").set_index("contrast"); m9 = pd.read_csv(RES / "E9_mechanism.csv")
m9 = m9.assign(tau=m9.tau.astype(str)).set_index("tau")
LAB = [("0.5 (E1)", "S-B, tau 0.5 (E1)", "0.5"), ("300", "S-B, tau 300", "300"), ("3000", "S-B, tau 3000", "3000"),
       ("10000", "S-B, tau 10000", "$10^4$"), ("30000", "S-B, tau 30000", r"$3{\cdot}10^4$"), ("inf (E1 no clip)", "S-B, tau inf (E1 no clip)", "none")]
for t, cn, lab in LAB:
    x = m9.loc[t].applied_ratio_SB; r = k9.loc[cn]
    e.errorbar(x, r.difference, yerr=[[r.difference - r.ci_low], [r.ci_high - r.difference]], fmt="o", ms=3.5, color="#009E73", capsize=1.5, lw=.9)
    e.annotate(lab, (x, r.difference), textcoords="offset points", xytext=(4, -9) if lab == "none" else (4, 3), ha="left", fontsize=6)
e.axhline(0, color="0.6", lw=.6); e.set_xlabel("measured S/B applied actor gradient"); e.set_ylabel(r"S$-$B $\Delta w_{50}$ (95% CI)")
e.set_title(r"E  clip threshold (E9)"); e.set_xlim(0, 1.25)

# F
f = ax[1, 2]
g = lambda fn, **kw: pd.read_csv(RES / fn).pipe(lambda d: d.loc[np.logical_and.reduce([d[k] == v for k, v in kw.items()])].iloc[0])
k10 = pd.read_csv(RES / "E10_contrasts.csv").set_index("contrast"); k11 = pd.read_csv(RES / "E11_contrasts.csv").set_index("contrast")
k8 = pd.read_csv(RES / "E8B_contrasts.csv")
ROWS = [("SB3: shared clip (default)", g("E1E2_contrasts.csv", cell="global_clip/raw", contrast="S-B")),
        ("SB3: per-group clip", g("E6_contrasts.csv", mode="groupwise_clip", contrast="S-B")),
        ("SB3: no clip", g("E1E2_contrasts.csv", cell="separate_no_clip/raw", contrast="S-B")),
        ("SB3: no clip + reward norm.", g("E1E2_contrasts.csv", cell="separate_no_clip/vecnormalize", contrast="S-B")),
        ("SB3: no clip + PopArt (E8)", k8[k8.cohort.str.startswith("seeds") & (k8.quantity == "popart S-B")].iloc[0]),
        ("SB3: no clip + PopArt (E10)", k10.loc["primary 62-81: popart S-B"]),
        ("CleanRL-style: shared clip", k11.loc["cleanrl_raw: S-B"]),
        ("CleanRL-style: per-group clip", k11.loc["cleanrl_group: S-B"]),
        ("CleanRL-style: reward norm.", k11.loc["cleanrl_default: S-B"])]
for i, (lab, r) in enumerate(ROWS):
    y = len(ROWS) - 1 - i; col = "#009E73" if (r.ci_low > 0 or r.ci_high < 0) else "0.5"
    f.plot([r.ci_low, r.ci_high], [y, y], color=col, lw=2); f.plot(r.difference, y, "o", color="black", ms=3)
f.axvline(0, color="0.6", lw=.6, ls="--"); f.set_yticks(range(len(ROWS))); f.set_yticklabels([r[0] for r in ROWS][::-1], fontsize=6)
f.set_xlabel(r"S$-$B $\Delta w_{50}$ (95% CI)"); f.set_title("F  interventions")
fig.tight_layout(); fig.savefig(OUT); print("wrote", OUT)
