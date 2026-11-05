"""Displays added in the 5 November revision. Every value comes from stored results or per-run tables; no new
experiment and no change to any pre-declared estimate.

- fig_summary.pdf       principal contrasts across E1, E4-E7 with declared 95% and coverage-calibrated intervals
- fig_e7_moments.pdf    E7 achieved training-reward mean and SD of A and M by mixture weight, both critics
- fig_lbf_sensitivity.pdf  E4 A-S and per-arm crossings across label thresholds, Firth and isotonic
- results_table.tex     compact table of the principal contrasts
- coverage_table.tex    simulated coverage of the seed-block percentile intervals
- revision_numbers.tex  macros for the text (coverage range, calibrated level, mechanism check, information structure)

Usage: revision_figures.py RUNS_DIR RES_DIR
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RUNS, RES = map(Path, sys.argv[1:3])
OUT = Path(__file__).resolve().parents[2] / "generated/v4"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})
CAL = pd.read_csv(RES / "stat_calibrated_intervals.csv"); COV = pd.read_csv(RES / "stat_coverage.csv"); MC = pd.read_csv(RES / "stat_bootstrap_stability.csv")
LEVEL = float(CAL.level.iloc[0])

# ---------------------------------------------------------------- principal contrasts
META = {  # contrast label -> (display, environment, learner/intervention, reading)
    "E1 global_clip/raw S-B": (r"S$-$B", "Supply", "Shared clip, raw rewards (default)", "symmetric shift"),
    "E6 groupwise_clip S-B": (r"S$-$B", "Supply", "Per-group clip, raw rewards", "symmetric shift reduced"),
    "E1 separate_no_clip/vecnormalize S-B": (r"S$-$B", "Supply", "Separate optimizers, normalized rewards", "none detectable"),
    "E7 popart S-B": (r"S$-$B", "Supply", "Separate optimizers, PopArt critic, raw rewards", "none detectable"),
    "E1 separate_no_clip/vecnormalize A-S": (r"A$-$S", "Supply", "Separate optimizers, normalized rewards", "asymmetric shift"),
    "E1 separate_no_clip/vecnormalize A-B": (r"A$-$B", "Supply", "Separate optimizers, normalized rewards", "loss-only shift"),
    "E6 groupwise_clip A-S": (r"A$-$S", "Supply", "Per-group clip, raw rewards", "asymmetric shift"),
    "E5 3.6M A-S": (r"A$-$S", "Supply", "Separate optimizers, normalized rewards, 3.6M", "persists at 3.6M"),
    "E7 popart A-S": (r"A$-$S", "Supply", "PopArt critic, raw rewards", "asymmetric shift"),
    "E7 plain A-M": (r"A$-$M", "Supply", "Plain critic, raw rewards", "not separable from moments"),
    "E7 popart A-M": (r"A$-$M", "Supply", "PopArt critic, raw rewards", "exceeds scale-matched control"),
    "E4 global_clip A-S": (r"A$-$S", "LBF", "MAPPO, shared clip", "asymmetric shift"),
    "E4 separate_no_clip A-S": (r"A$-$S", "LBF", "MAPPO, separate optimizers", "asymmetric shift"),
}
ORDER = ["E1 global_clip/raw S-B", "E6 groupwise_clip S-B", "E1 separate_no_clip/vecnormalize S-B", "E7 popart S-B",
         "E1 separate_no_clip/vecnormalize A-S", "E1 separate_no_clip/vecnormalize A-B", "E6 groupwise_clip A-S", "E5 3.6M A-S",
         "E7 popart A-S", "E7 plain A-M", "E7 popart A-M", "E4 global_clip A-S", "E4 separate_no_clip A-S"]
c = CAL.set_index("contrast").loc[ORDER].reset_index()
fig, ax = plt.subplots(figsize=(6.6, 4.1))
y = np.arange(len(c))[::-1]
for yi, r in zip(y, c.itertuples()):
    sym = META[r.contrast][0].startswith("S")
    col = "#009E73" if sym else ("#CC79A7" if "M" in META[r.contrast][0] else "#D55E00")
    ax.plot([r.ci_low_cal, r.ci_high_cal], [yi, yi], color=col, lw=1, alpha=.55)
    ax.plot([r.ci_low_95, r.ci_high_95], [yi, yi], color=col, lw=3)
    ax.plot(r.difference, yi, "o", color="black", ms=3.5, zorder=3)
ax.axvline(0, color="0.4", lw=.7, ls="--")
ax.set_yticks(y); ax.set_yticklabels([f"{r.contrast.split()[0]}  {META[r.contrast][0]}: {META[r.contrast][2]}" for r in c.itertuples()])
ax.set_xlabel(r"Crossing difference $\Delta w_{50}$")
for yb in [len(c) - 4.5, len(c) - 9.5, len(c) - 11.5]:
    ax.axhline(yb, color="0.85", lw=.6)
fig.tight_layout(); fig.savefig(OUT / "fig_summary.pdf"); plt.close(fig)

SEEDS = {"E1": 10, "E4": 10, "E5": 10, "E6": 10, "E7": 10}
rows = []
for r in c.itertuples():
    disp, env, setting, reading = META[r.contrast]; exp = r.contrast.split()[0]
    est = f"${r.difference:+.3f}$"; i95 = f"$[{r.ci_low_95:+.3f},\\,{r.ci_high_95:+.3f}]$"; ical = f"$[{r.ci_low_cal:+.3f},\\,{r.ci_high_cal:+.3f}]$"
    rows.append(" & ".join([exp, env, setting, disp, est, i95, ical, reading]) + r" \\")
(OUT / "results_table.tex").write_text(r"""\begin{table}[!htbp]
\caption{Principal crossing contrasts. Each contrast pairs two arms over the same ten training seeds (seeds 42--51) on the experiment's mixture grid; intervals resample complete training-seed blocks (2,000 draws). The 95\% interval is the pre-declared one. The """ + f"{LEVEL:.0%}".replace("%", r"\%") + r""" interval is the coverage-calibrated sensitivity of Section~\ref{sec:stats}, whose simulated coverage is at least 0.94. No principal verdict differs between the two.}
\label{tab:results}
\centering\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}llp{.23\textwidth}lcccp{.13\textwidth}@{}}
\toprule
Exp. & Env. & Learner / intervention & Contrast & Estimate & 95\% interval & """ + f"{LEVEL:.0%}".replace("%", r"\%") + r""" interval & Reading \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- coverage table
lv = [0.95, 0.975, 0.99]
rows = []
for r in COV.itertuples():
    rows.append(f"{r.scenario.replace(', contrast', ',').replace(' contrast', '')} & {r.seed_sd:.0f} & ${r.true_w50:.3f}$ & ${r.true_contrast:+.2f}$ & " +
                " & ".join(f"{COV.loc[r.Index, f'cover_w50_{l}']:.3f}" for l in lv) + " & " +
                " & ".join(f"{COV.loc[r.Index, f'cover_contrast_{l}']:.3f}" for l in lv) + " & " +
                " & ".join(f"{COV.loc[r.Index, f'excludes_zero_{l}']:.3f}" for l in lv) + r" \\")
(OUT / "coverage_table.tex").write_text(r"""\begin{table}[H]
\caption{Simulated behaviour of the seed-block percentile intervals at the paper's design: ten seeds, the E1 grid, one label per seed and cell, 1,000 replicates per row, 2,000 bootstrap draws. Each scenario has two arms; ``seed SD'' is the standard deviation of a per-seed logit intercept shared by both arms. Coverage is of the true crossing ($w_{50}$) and of the true paired contrast; the last block is the fraction of replicates whose contrast interval excludes zero, which is the false-positive rate under a null contrast and power otherwise.}
\label{tab:coverage}
\centering\scriptsize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lcccccccccccc@{}}
\toprule
& & & & \multicolumn{3}{c}{Coverage of $w_{50}$} & \multicolumn{3}{c}{Coverage of contrast} & \multicolumn{3}{c}{Interval excludes 0} \\
\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(l){11-13}
Scenario & Seed SD & True $w_{50}$ & True contrast & 95 & 97.5 & 99 & 95 & 97.5 & 99 & 95 & 97.5 & 99 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- E7 achieved moments
e7 = pd.read_csv(RUNS / "E7_runs.csv"); e1 = pd.read_csv(RUNS / "E1_runs.csv")
pa = e1[(e1.optimizer_mode == "separate_no_clip") & (e1.reward_mode == "raw") & (e1.arm == "A")].assign(critic_mode="plain")
D = pd.concat([pa, e7[e7.arm.isin(["A", "M"])]])
D = D.assign(mean=np.where(D.arm == "M", D.M_mean, D.reward_transformed_mean), rms=np.where(D.arm == "M", D.M_rms, D.reward_transformed_rms))
D = D.assign(sd=np.sqrt(np.maximum(D.rms ** 2 - D["mean"] ** 2, 0)))
dm = e7[e7.arm == "M"].assign(w=lambda d: d.w.round(4))
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3), sharex=True)
for ax, q, lab in zip(axes, ["mean", "sd"], ["Training-reward mean", "Training-reward SD (within run)"]):
    for mode, ls in [("plain", "-"), ("popart", "--")]:
        for arm, col, mk in [("A", "#D55E00", "^"), ("M", "#CC79A7", "v")]:
            g = D[(D.critic_mode == mode) & (D.arm == arm)].groupby(D.w.round(4))[q]
            m, s = g.mean(), g.std() / np.sqrt(g.count())
            ax.errorbar(m.index, m.values, yerr=1.96 * s.values, color=col, ls=ls, marker=mk, ms=3, lw=1, capsize=1.5,
                        label=f"{arm}, {'plain' if mode == 'plain' else 'PopArt'} critic")
    ax.set_xlabel("Mixture weight $w$"); ax.set_title(lab)
axes[0].axhline(0, color="0.5", lw=.6)
axes[0].legend(frameon=False, ncol=2, loc="lower left")
fig.tight_layout(); fig.savefig(OUT / "fig_e7_moments.pdf"); plt.close(fig)

# ---------------------------------------------------------------- LBF estimator and threshold sensitivity
ts = pd.read_csv(RES / "E4_theta_sensitivity.csv")
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=True)
STY = {"linear": ("#7f7f7f", "D"), "curvature": ("#0072B2", "o"), "A": ("#D55E00", "^"), "S": ("#009E73", "s")}
for ax, opt, title in zip(axes, ["global_clip", "separate_no_clip"], ["MAPPO, shared clip", "MAPPO, separate optimizers"]):
    for j, th in enumerate([0.5, 0.626, 0.75]):
        for k, arm in enumerate(["linear", "curvature", "A", "S"]):
            r = ts[(ts.theta == th) & (ts.optimizer == opt) & (ts.quantity == arm)].iloc[0]
            x = j + (k - 1.5) * 0.17; col, mk = STY[arm]
            est = np.clip(r.estimate, -0.25, 1.1)
            ax.plot([x, x], np.clip([r.ci_low, r.ci_high], -0.25, 1.1), color=col, lw=1.2)
            ax.plot(x, est, mk, color=col, ms=3.5, mfc=col if r.in_grid else "white", label=arm if j == 0 and opt == "global_clip" else None)
            if np.isfinite(r.isotonic):
                ax.plot(x + 0.06, r.isotonic, "x", color=col, ms=3.5)
    ax.axhspan(0, 0.95, color="0.95", zorder=0)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels([r"$\theta=0.5$", r"$\theta=0.626$ (declared)", r"$\theta=0.75$"]); ax.set_title(title)
axes[0].set_ylabel(r"Crossing $w_{50}$"); axes[0].set_ylim(-0.27, 1.12)
axes[0].legend(frameon=False, ncol=4, loc="upper left", fontsize=6.5)
fig.tight_layout(); fig.savefig(OUT / "fig_lbf_sensitivity.pdf"); plt.close(fig)

# ---------------------------------------------------------------- macros
def mac(n, v): return f"\\newcommand{{\\{n}}}{{{v}}}\n"
m = ""
c95 = [COV[f"cover_contrast_0.95"].min(), COV[f"cover_contrast_0.95"].max()]; w95 = [COV["cover_w50_0.95"].min(), COV["cover_w50_0.95"].max()]
allcov = [min(c95[0], w95[0]), max(c95[1], w95[1])]
m += mac("VrvCovRange", f"{allcov[0]:.2f}--{allcov[1]:.2f}")
nulls = COV[COV.true_contrast.abs() < 1e-12]
m += mac("VrvFalsePos", f"{nulls['excludes_zero_0.95'].min():.3f}--{nulls['excludes_zero_0.95'].max():.3f}")
m += mac("VrvCalLevel", f"{LEVEL:.0%}".replace("%", r"\%"))
cc = [COV[f"cover_contrast_{LEVEL}"].min(), COV[f"cover_w50_{LEVEL}"].min()]
m += mac("VrvCalCovMin", f"{min(cc):.2f}")
m += mac("VrvMCShift", f"{np.abs(MC[['shift_low', 'shift_high']].to_numpy()).max():.3f}")
m += mac("VrvNContrasts", f"{len(CAL)}")
m += mac("VrvVerdictsChanged", f"{int((CAL.excludes_zero_95 != CAL.excludes_zero_cal).sum())}")
e1d = pd.read_csv(RES / "E1E2_diagnostics.csv"); ps = e1d[e1d.cell == "global_clip/raw"].set_index("arm")
m += mac("VrvCriticRatio", f"{ps.loc['B'].mb_critic_grad_norm / ps.loc['S'].mb_critic_grad_norm:.3f}")
m += mac("VrvActorRatioPre", f"{ps.loc['S'].mb_actor_grad_norm / ps.loc['B'].mb_actor_grad_norm:.2f}")
pred = lambda r: 0.5 * r.mb_actor_grad_norm / np.hypot(r.mb_actor_grad_norm, r.mb_critic_grad_norm)   # first-order shared-clip formula at cell means
m += mac("VrvPredRatio", f"{pred(ps.loc['S']) / pred(ps.loc['B']):.3f}")
m += mac("VrvMeasRatio", f"{ps.loc['S'].mb_post_actor_grad_norm / ps.loc['B'].mb_post_actor_grad_norm:.3f}")
x = CAL.set_index("contrast")
m += mac("VrvPlainAMcal", f"$[{x.loc['E7 plain A-M'].ci_low_cal:+.3f},\\,{x.loc['E7 plain A-M'].ci_high_cal:+.3f}]$")
(OUT / "revision_numbers.tex").write_text(m)
print("wrote fig_summary, fig_e7_moments, fig_lbf_sensitivity, results_table, coverage_table, revision_numbers")
