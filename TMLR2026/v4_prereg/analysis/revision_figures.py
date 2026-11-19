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
HOLM = pd.read_csv(RES / "stat_holm.csv").set_index("contrast"); CAL = pd.read_csv(RES / "stat_calibrated_intervals.csv"); COV = pd.read_csv(RES / "stat_coverage.csv"); MC = pd.read_csv(RES / "stat_bootstrap_stability.csv")
LEVEL = float(CAL.level.iloc[0])

# ---------------------------------------------------------------- principal contrasts
META = {  # contrast label -> (display, environment, learner/intervention, reading)
    "E1 global_clip/raw S-B": (r"S$-$B", "Supply", "Shared clip, raw rewards (default)", "symmetric shift"),
    "E6 groupwise_clip S-B": (r"S$-$B", "Supply", "Per-group clip, raw rewards", "shift reduced; not replicated (X1)"),
    "E1 separate_no_clip/vecnormalize S-B": (r"S$-$B", "Supply", "Separate optimizers, normalized rewards", "none detectable"),
    "E7 popart S-B": (r"S$-$B", "Supply", "Separate optimizers, PopArt critic, raw rewards", "none detectable"),
    "E1 separate_no_clip/vecnormalize A-S": (r"A$-$S", "Supply", "Separate optimizers, normalized rewards", "asymmetric shift"),
    "E1 separate_no_clip/vecnormalize A-B": (r"A$-$B", "Supply", "Separate optimizers, normalized rewards", "loss-only shift"),
    "E6 groupwise_clip A-S": (r"A$-$S", "Supply", "Per-group clip, raw rewards", "asymmetric shift"),
    "E5 3.6M A-S": (r"A$-$S", "Supply", "Separate optimizers, normalized rewards, 3.6M", "persists at 3.6M"),
    "E7 popart A-S": (r"A$-$S", "Supply", "PopArt critic, raw rewards", "asymmetric shift"),
    "E7 plain A-M": (r"A$-$M", "Supply", "Plain critic, raw rewards", "not separable from moments"),
    "E7 popart A-M": (r"A$-$M", "Supply", "PopArt critic, raw rewards", "exceeds scale-matched control"),
    "E8 plain A-M_on": (r"A$-$M$_{\mathrm{on}}$", "Supply", "Plain critic, raw rewards", "small, beyond moments"),
    "E8 popart A-M_on": (r"A$-$M$_{\mathrm{on}}$", "Supply", "PopArt critic, raw rewards", "exceeds moment-matched control"),
    "E8B popart S-B": (r"S$-$B", "Supply", "PopArt critic, raw rewards, seeds 52--61", "none detectable"),
    "E4 global_clip A-S": (r"A$-$S", "LBF", "MAPPO, shared clip", "asymmetric shift"),
    "E4 separate_no_clip A-S": (r"A$-$S", "LBF", "MAPPO, separate optimizers", "asymmetric shift"),
    "E10 plain S-B": (r"S$-$B", "Supply", "Plain critic, raw rewards, seeds 62--81", "symmetric shift replicates"),
    "E10 popart S-B": (r"S$-$B", "Supply", "PopArt critic, raw rewards, seeds 62--81", "none detectable"),
    "E11 cleanrl_raw S-B": (r"S$-$B", "Supply", "CleanRL-style, global clip, raw rewards", "symmetric shift"),
    "E11 cleanrl_group S-B": (r"S$-$B", "Supply", "CleanRL-style, per-group clip, raw rewards", "shift persists; reduction not established (X2)"),
    "E11 cleanrl_default S-B": (r"S$-$B", "Supply", "CleanRL-style, normalized rewards", "none detectable"),
    "E11 cleanrl_default A-S": (r"A$-$S", "Supply", "CleanRL-style, normalized rewards", "asymmetric shift"),
    "E12 default S-B": (r"S$-$B", "Supply", "Local observations, default learner", "none detectable (predicted nonzero)"),
    "E12 decoupled A-S": (r"A$-$S", "Supply", "Local observations, decoupled learner", "asymmetric shift"),
}
ORDER = ["E1 global_clip/raw S-B", "E6 groupwise_clip S-B", "E1 separate_no_clip/vecnormalize S-B", "E7 popart S-B", "E8B popart S-B",
         "E10 plain S-B", "E10 popart S-B", "E11 cleanrl_raw S-B", "E11 cleanrl_group S-B", "E11 cleanrl_default S-B", "E12 default S-B",
         "E1 separate_no_clip/vecnormalize A-S", "E1 separate_no_clip/vecnormalize A-B", "E6 groupwise_clip A-S", "E5 3.6M A-S",
         "E7 popart A-S", "E11 cleanrl_default A-S", "E12 decoupled A-S",
         "E7 plain A-M", "E7 popart A-M", "E8 plain A-M_on", "E8 popart A-M_on", "E4 global_clip A-S", "E4 separate_no_clip A-S"]
JK = pd.read_csv(RES / "jackknife_v5.csv"); JKP = JK[JK.kind == "principal"].set_index("contrast")
c = CAL.set_index("contrast").loc[ORDER].join(JKP[["jt_low_95", "jt_high_95", "jt_excludes_zero_95", "holm_jt_reject"]]).reset_index()
fig, ax = plt.subplots(figsize=(6.6, 5.4))
y = np.arange(len(c))[::-1]
for yi, r in zip(y, c.itertuples()):
    sym = META[r.contrast][0].startswith("S")
    col = "#009E73" if sym else ("#CC79A7" if "M" in META[r.contrast][0] else "#D55E00")
    ax.plot([r.ci_low_95, r.ci_high_95], [yi + .18, yi + .18], color=col, lw=1, alpha=.6)
    ax.plot([r.jt_low_95, r.jt_high_95], [yi, yi], color=col, lw=3)
    ax.plot(r.difference, yi, "o", color="black", ms=3.5, zorder=3)
ax.axvline(0, color="0.4", lw=.7, ls="--")
ax.set_yticks(y); ax.set_yticklabels([f"{r.contrast.split()[0]}  {META[r.contrast][0]}: {META[r.contrast][2].replace('--', chr(8211))}" for r in c.itertuples()])
ax.set_xlabel(r"Crossing difference $\Delta w_{50}$")
for yb in [len(c) - 11.5, len(c) - 18.5, len(c) - 22.5]:
    ax.axhline(yb, color="0.85", lw=.6)
fig.tight_layout(); fig.savefig(OUT / "fig_summary.pdf"); plt.close(fig)

assert (c.jt_excludes_zero_95 == ((c.ci_low_95 > 0) | (c.ci_high_95 < 0))).all(), "caption claims the two 95% intervals agree"
rows = []
for r in c.itertuples():
    disp, env, setting, reading = META[r.contrast]; exp = r.contrast.split()[0]
    est = f"${r.difference:+.3f}$"; i95 = f"$[{r.ci_low_95:+.3f},\\,{r.ci_high_95:+.3f}]$"; ijt = f"$[{r.jt_low_95:+.3f},\\,{r.jt_high_95:+.3f}]$"
    hm = "yes" if r.holm_jt_reject else "no"
    if r.jt_excludes_zero_95 and not r.holm_jt_reject:
        reading += "; not after Holm"
    rows.append(" & ".join([exp, env, setting, disp, est, i95, ijt, hm, reading]) + r" \\")
(OUT / "results_table.tex").write_text(r"""\begin{table}[!htbp]
\caption{Principal crossing contrasts. Each contrast pairs two arms over the same training seeds on the experiment's mixture grid (ten seeds, 42--51, except E8's 52--61 and E10's twenty, 62--81). ``Seed-block'': the pre-declared 95\% percentile interval from 2,000 resamples of complete training seeds, which under-covers at this sample size. ``Jackknife-$t$'': the calibrated 95\% interval of Section~\ref{sec:stats} (leave-one-seed-out standard error, $t_{n-1}$ quantiles), whose simulated coverage is near nominal. ``Holm'': whether the contrast remains nonzero after a Holm adjustment of the jackknife-$t$ $p$-values across all """ + f"{len(JKP)}" + r""" principal contrasts (family-wise $\alpha=0.05$); E1's and E6's shared-clip S$-$B are the same runs, shown once but counted twice, which is conservative. The two 95\% intervals agree on every zero-exclusion verdict. The fresh-seed experiments X1--X5 are reported in Tables~\ref{tab:x1x2}, \ref{tab:x3}, \ref{tab:x4} and~\ref{tab:x5} and are not in this family.}
\label{tab:results}
\centering\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}ll>{\raggedright\arraybackslash}p{.21\textwidth}lcccc >{\raggedright\arraybackslash}p{.12\textwidth}@{}}
\toprule
Exp. & Env. & Learner / intervention & Contrast & Estimate & Seed-block 95\% & Jackknife-$t$ 95\% & Holm & Reading \\
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

# ---------------------------------------------------------------- calibration study table (calibration_v5.py)
CAL5 = pd.read_csv(RES.parent / "results_v5/calibration_summary.csv")
def scen(nm):
    parts = nm.split("-"); out = []
    for q in parts:
        out.append({"iid": "no seed effect", "u1": "seed SD 1", "u2": "seed SD 2", "v": "+ random slope", "rho0.5": r"$\rho=0.5$", "rho0": r"$\rho=0$",
                    "het2": "arm-2 SD $\\times2$", "cloglog": "cloglog link", "n20": "", "n30": "", "low": "low", "high": "high"}.get(q, q))
    return ", ".join(x for x in out if x)
rows = []
for r in CAL5.itertuples():
    rows.append(" & ".join([scen(r.name), f"{r.n}", f"${r.true_d:+.2f}$"] + [f"{getattr(r, mm + '_d95'):.3f}" for mm in ["pct", "bca", "jt", "glmm", "pboot"]]
                           + [f"{getattr(r, mm + '_x95'):.3f}" for mm in ["pct", "jt"]]) + r" \\")
(OUT / "calibration_table.tex").write_text(r"""\begin{table}[H]
\caption{Calibration study of interval procedures for a paired crossing contrast (\texttt{calibration\_v5.py}): 10,000 replicates per row, nominal 95\%. ``Seed SD'' is the SD of a per-seed logit intercept; $\rho$ is its correlation between the two arms (1 unless stated); ``edge'' rows put the crossing near $w=0.1$ or $0.95$. Coverage is of the true contrast $\Delta$; ``excludes 0'' is the false-positive rate when $\Delta=0$ and the power otherwise. Pct: pre-declared seed-block percentile; BCa: bias-corrected and accelerated; J-$t$: seed-level jackknife-$t$; HLM: hierarchical logistic model, Wald; PB: parametric bootstrap from it.}
\label{tab:calibration}
\centering\scriptsize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lcccccccccc@{}}
\toprule
& & & \multicolumn{5}{c}{Coverage of $\Delta$} & \multicolumn{2}{c}{Excludes 0} \\
\cmidrule(lr){4-8}\cmidrule(l){9-10}
Scenario & Seeds & $\Delta$ & Pct & BCa & J-$t$ & HLM & PB & Pct & J-$t$ \\
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
ts = pd.read_csv(RES.parent / "results_jt/E4_theta_sensitivity.csv")   # jackknife-t (revision v6)
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
m += mac("VrvHolmKept", f"{int(HOLM.holm_reject.sum())}"); m += mac("VrvHolmMaxP", f"{HOLM[HOLM.holm_reject].p_boot.max():.3f}")
m += mac("VrvVerdictsChanged", f"{int((CAL.excludes_zero_95 != CAL.excludes_zero_cal).sum())}")
e1d = pd.read_csv(RES / "E1E2_diagnostics.csv"); ps = e1d[e1d.cell == "global_clip/raw"].set_index("arm")
m += mac("VrvCriticRatio", f"{ps.loc['B'].mb_critic_grad_norm / ps.loc['S'].mb_critic_grad_norm:.3f}")
m += mac("VrvActorRatioPre", f"{ps.loc['S'].mb_actor_grad_norm / ps.loc['B'].mb_actor_grad_norm:.2f}")
pred = lambda r: 0.5 * r.mb_actor_grad_norm / np.hypot(r.mb_actor_grad_norm, r.mb_critic_grad_norm)   # first-order shared-clip formula at cell means
m += mac("VrvPredRatio", f"{pred(ps.loc['S']) / pred(ps.loc['B']):.3f}")
m += mac("VrvMeasRatio", f"{ps.loc['S'].mb_post_actor_grad_norm / ps.loc['B'].mb_post_actor_grad_norm:.3f}")
x = CAL.set_index("contrast")
m += mac("VrvPlainAMcal", f"$[{x.loc['E7 plain A-M'].ci_low_cal:+.3f},\\,{x.loc['E7 plain A-M'].ci_high_cal:+.3f}]$")
import json
root = Path(__file__).resolve().parents[3]
des = next(q for q in [root / "AAMAS2027/optimizer_factorial_source_v2/AAMAS2027/experiments/e7_design.json",
                       root / "v4/executed_source/optimizer_factorial_source_v2/AAMAS2027/experiments/e7_design.json"] if q.exists())
aff = json.loads(des.read_text())["M_affine"]; e8 = pd.read_csv(RUNS / "E8_runs.csv"); pa8 = e8[e8.part == "A"]
lo = pa8.apply(lambda r: r.online_a_min <= aff[f"{r.w:g}"]["scale"] / 4 + 1e-9, axis=1)
m += mac("VeightClamp", f"{int(lo.sum())}"); m += mac("VeightNA", f"{len(pa8)}")
# E12 (post hoc, descriptive): S's applied actor gradient relative to B's under local observation, default learner
e12 = pd.read_csv(RUNS / "E12_runs.csv"); g12 = e12[e12.learner == "default"].groupby("arm").mb_post_actor_grad_norm.mean()
m += mac("VtwelveAppliedRatio", f"{g12['S'] / g12['B']:.2f}")
# ---------------------------------------------------------------- calibration study (calibration_v5.py) and jackknife-t
CS = pd.read_csv(RES.parent / "results_v5/calibration_summary.csv"); c10 = CS[CS.n == 10]; nul = c10[c10.true_d.abs() < 1e-9]
rng_ = lambda col, d=c10: f"{d[col].min():.2f}--{d[col].max():.2f}"
m += mac("VcalScen", f"{len(CS)}"); m += mac("VcalScenTen", f"{len(c10)}")
m += mac("VcalPctCov", rng_("pct_d95")); m += mac("VcalPctCovNine", rng_("pct_d99")); m += mac("VcalPctFP", rng_("pct_x95", nul))
m += mac("VcalBcaCov", rng_("bca_d95")); m += mac("VcalJtCov", rng_("jt_d95")); m += mac("VcalJtFP", rng_("jt_x95", nul))
m += mac("VcalJtCovW", rng_("jt_w95")); m += mac("VcalPctCovWMin", f"{c10.pct_w95.min():.2f}")
m += mac("VcalGlmmCov", rng_("glmm_d95")); m += mac("VcalGlmmFPMax", f"{nul.glmm_x95.max():.3f}"); m += mac("VcalPbootFPMax", f"{nul.pboot_x95.max():.3f}")
m += mac("VcalWidthRatio", f"{(c10.jt_wid / c10.pct_wid).median():.2f}")
pw = lambda n, nm: CS[(CS.n == n) & (CS.name.str.startswith(nm))].jt_x95.iloc[0]
m += mac("VcalPowTen", f"{c10[c10.name == 'steep-effect-u1'].jt_x95.iloc[0]:.2f}"); m += mac("VcalPowTwenty", f"{pw(20, 'steep-effect-u1'):.2f}"); m += mac("VcalPowThirty", f"{pw(30, 'steep-effect-u1'):.2f}")
JC = JK[JK.kind == "change"].set_index("contrast")
for lab, nm in [("E6 per-group minus shared clip, S-B", "Six"), ("E9 tau 30000 minus tau 0.5, S-B", "NineDose"), ("E9 tau 30000 minus no clip, S-B", "NineEnd"),
                ("E7 PopArt minus plain, S-B", "Seven"), ("E8 PopArt minus plain, S-B (seeds 52-61)", "Eight"), ("E10 PopArt minus plain, S-B (seeds 62-81)", "Ten"),
                ("PopArt minus plain, S-B, pooled seeds 42-81", "Pool"), ("E11 per-group minus shared clip, S-B", "Eleven")]:
    x = JC.loc[lab]
    m += mac(f"VjtCh{nm}", f"${x.difference:+.3f}$ $[{x.jt_low_95:+.3f},\\,{x.jt_high_95:+.3f}]$"); m += mac(f"VjtCh{nm}Nine", f"$[{x.jt_low_99:+.3f},\\,{x.jt_high_99:+.3f}]$")
for lab, nm in [("E8 plain A-M_on", "EightAMon"), ("E10 plain S-B", "TenPlain"), ("E5 3.6M A-S", "FiveEnd"), ("E1 global_clip/raw S-B", "OneSym")]:
    x = JKP.loc[lab]; m += mac(f"Vjt{nm}", f"$[{x.jt_low_95:+.3f},\\,{x.jt_high_95:+.3f}]$")
for lab, nm in [("E1 global_clip/raw S-B", "Std"), ("E1 separate_no_clip/vecnormalize S-B", "SymBoth"), ("E1 separate_no_clip/vecnormalize A-S", "AsymBoth"),
                ("E11 cleanrl_raw S-B", "ElevenRaw"), ("E11 cleanrl_default S-B", "ElevenDefault"), ("E11 cleanrl_default A-S", "ElevenAsym"),
                ("E8 plain A-M_on", "AMonPlain"), ("E8 popart A-M_on", "AMonPop"), ("E12 decoupled A-S", "TwelveAsym")]:
    x = JKP.loc[lab]; m += mac(f"VjtP{nm}", f"${x.difference:+.3f}$ $[{x.jt_low_95:+.3f},\\,{x.jt_high_95:+.3f}]$")
m += mac("VjtHolmKept", f"{int(JKP.holm_jt_reject.sum())}"); m += mac("VjtNExcl", f"{int(JKP.jt_excludes_zero_95.sum())}")
m += mac("VjtHolmLost", ", ".join(sorted(JKP[JKP.jt_excludes_zero_95 & ~JKP.holm_jt_reject.astype(bool)].index)).replace("_", r"\_"))
(OUT / "revision_numbers.tex").write_text(m)
print("wrote fig_summary, fig_e7_moments, fig_lbf_sensitivity, results_table, coverage_table, revision_numbers")
