"""Render v4 tables, figures and macros from v4_prereg/results/*.csv into TMLR2026/generated/v4/.

Usage (from the repository root): python TMLR2026/v4_prereg/analysis/render_v4.py
No number in the v4 manuscript sections is typed by hand; they come from here.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES, OUT = ROOT / "v4_prereg/results", ROOT / "generated/v4"
OUT.mkdir(parents=True, exist_ok=True)
CELLS = ["global_clip/raw", "global_clip/vecnormalize", "separate_no_clip/raw", "separate_no_clip/vecnormalize", "eps1e-8/raw"]
CELL_TEX = {"global_clip/raw": "Global clip, raw (default)", "global_clip/vecnormalize": "Global clip, normalized reward",
            "separate_no_clip/raw": "Separate optimizers, raw", "separate_no_clip/vecnormalize": "Separate optimizers, normalized",
            "eps1e-8/raw": r"Global clip, raw, Adam $\epsilon{=}10^{-8}$"}


def num(x, sign=False):
    return f"{x:+.3f}" if sign else f"{x:.3f}"


def iv(lo, hi, sign=False):
    return f"[{num(lo, sign)},\\,{num(hi, sign)}]"


def star(r):
    return r"$^{\ast}$" if (r.ci_low > 0 or r.ci_high < 0) else ""


def table(path, caption, label, colspec, header, rows, size=r"\small", fit=False):
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    open_, close_ = (r"\resizebox{\linewidth}{!}{%" + "\n", "}\n") if fit else ("", "")
    path.write_text(f"""\\begin{{table}}[t]
\\caption{{{caption}}}
\\label{{{label}}}
\\centering{size}
{open_}\\begin{{tabular}}{{{colspec}}}
\\toprule
{header} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
{close_}\\end{{table}}
""")


# ---------------------------------------------------------------- E1/E2
cr = pd.read_csv(RES / "E1E2_crossings.csv"); co = pd.read_csv(RES / "E1E2_contrasts.csv")
it = pd.read_csv(RES / "E1E2_interactions.csv"); dg = pd.read_csv(RES / "E1E2_diagnostics.csv")
rows = []
for c in CELLS:
    r = [CELL_TEX[c]]
    for a in ["B", "A", "S", "R"]:
        x = cr[(cr.cell == c) & (cr.arm == a)].iloc[0]; r.append(f"${num(x.w50)}$ ${iv(x.ci_low, x.ci_high)}$")
    rows.append(r)
table(OUT / "factorial_crossings.tex",
      r"Optimizer $\times$ reward-normalization factorial (E1) and Adam-$\epsilon$ addendum (E2): crossings $w_{50}$ at 300k with 95\% seed-block intervals. B: curvature only; A: losses $\times7$; S: both branches $\times7$; R: both branches $\times3.451$ (A's pooled RMS ratio). Ten seeds per cell, eight mixture cells, one platform.",
      "tab:factorial_crossings", "@{}lcccc@{}", r"Optimizer / reward & B & A & S & R", rows, size=r"\scriptsize", fit=True)
rows = []
for c in CELLS:
    r = [CELL_TEX[c]]
    for name in ["A-S", "A-B", "S-B", "R-B"]:
        x = co[(co.cell == c) & (co.contrast == name)].iloc[0]
        r.append(f"${num(x.difference, True)}${star(x)} ${iv(x.ci_low, x.ci_high, True)}$")
    rows.append(r)
table(OUT / "factorial_contrasts.tex",
      r"Paired seed-block contrasts within each optimizer cell. $^{\ast}$: 95\% interval excludes zero. A$-$S is the pre-declared asymmetry contrast; S$-$B and R$-$B are symmetric-scaling effects.",
      "tab:factorial_contrasts", "@{}lcccc@{}", r"Optimizer / reward & A$-$S & A$-$B & S$-$B & R$-$B", rows, size=r"\scriptsize", fit=True)
rows = []
for c in CELLS[1:]:
    r = [CELL_TEX[c]]
    for name in ["A-S", "A-B", "S-B", "R-B"]:
        x = it[(it.cell == c) & (it.contrast == name)].iloc[0]
        r.append(f"${num(x.change_vs_standard, True)}${star(x)} ${iv(x.ci_low, x.ci_high, True)}$")
    rows.append(r)
table(OUT / "factorial_interactions.tex",
      r"Change in each contrast relative to the default cell (global clip, raw reward), paired by seed. $^{\ast}$: interval excludes zero.",
      "tab:factorial_interactions", "@{}lcccc@{}", r"Intervention & $\Delta$(A$-$S) & $\Delta$(A$-$B) & $\Delta$(S$-$B) & $\Delta$(R$-$B)", rows, size=r"\scriptsize", fit=True)
rows = []
for c in CELLS:
    for a in ["B", "A", "S"]:
        x = dg[(dg.cell == c) & (dg.arm == a)].iloc[0]
        rows.append([CELL_TEX[c] if a == "B" else "", a, f"{x.mb_critic_grad_norm:,.2f}", f"{x.mb_actor_grad_norm:.2f}", f"{x.mb_post_actor_grad_norm:.4f}",
                     f"{x.mb_actor_grad_energy_share:.3f}", f"{x.mb_clip_freq:.3f}", f"{x.reward_transformed_rms:.1f}"])
table(OUT / "factorial_diagnostics.tex",
      r"Optimizer diagnostics (means over runs of per-minibatch means). Pre-clip critic and actor gradient norms, post-clip (applied) actor norm, actor share of squared gradient norm, fraction of minibatches whose norm exceeds 0.5, and transformed-reward RMS. With separate optimizers no clipping is applied, so the applied actor norm equals the pre-clip norm.",
      "tab:factorial_diagnostics", "@{}llrrrrrr@{}", r"Cell & Arm & Critic & Actor & Applied actor & Actor share & Over 0.5 & Reward RMS", rows, size=r"\scriptsize", fit=True)

# Figure: A-S and S-B by cell
fig, axes = plt.subplots(1, 2, figsize=(9.2, 2.9), sharey=True)
for ax, name, title in [(axes[0], "S-B", "Symmetric $\\times7$ minus curvature (S$-$B)"), (axes[1], "A-S", "Losses $\\times7$ minus symmetric $\\times7$ (A$-$S)")]:
    sub = co[co.contrast == name].set_index("cell").loc[CELLS]
    y = np.arange(len(CELLS))[::-1]
    ax.errorbar(sub.difference, y, xerr=[sub.difference - sub.ci_low, sub.ci_high - sub.difference], fmt="o", color="k", ms=4, capsize=2)
    ax.axvline(0, color="0.6", lw=0.8); ax.set_title(title, fontsize=9); ax.set_xlabel("crossing difference at 300k", fontsize=8)
    ax.tick_params(labelsize=7)
axes[0].set_yticks(np.arange(len(CELLS))[::-1]); axes[0].set_yticklabels([CELL_TEX[c].replace("$\\epsilon{=}10^{-8}$", "eps=1e-8").replace("$", "") for c in CELLS], fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "factorial_contrasts.pdf"); plt.close(fig)

# ---------------------------------------------------------------- E3
h = pd.read_csv(RES / "E3_crossings.csv", keep_default_na=False, na_values=[""]); hc = pd.read_csv(RES / "E3_contrasts.csv")
st = pd.read_csv(RES / "E3_stability.csv", keep_default_na=False, na_values=[""]); rc = pd.read_csv(RES / "E3_recovery.csv", keep_default_na=False, na_values=[""])
COND_TEX = {"null": "Linear", "curvature": "Curvature only", "lambda7": r"$\lambda=7$"}
PRIMARY = [300000, 900000, 1800000, 3600000]; BUD = {300000: "300k", 900000: "900k", 1800000: "1.8M", 3600000: "3.6M"}
rows = []
for cond in ["null", "curvature", "lambda7"]:
    r = [COND_TEX[cond]]
    for ck in PRIMARY:
        x = h[(h.condition == cond) & (h.checkpoint == ck)].iloc[0]; r.append(f"${num(x.w50)}$ ${iv(x.ci_low, x.ci_high)}$")
    rows.append(r)
r = [r"$\lambda7-$curvature"]
for ck in PRIMARY:
    x = hc[(hc.checkpoint == ck) & (hc.contrast == "lambda7-curvature")].iloc[0]; r.append(f"${num(x.difference, True)}${star(x)} ${iv(x.ci_low, x.ci_high, True)}$")
rows.append(r)
table(OUT / "horizon_table.tex",
      r"Refined single-platform horizon cohort (E3): nine mixture cells $\{.2,.4,.6,.65,.7,.75,.8,.85,1\}$, ten seeds, 270 trajectories trained to 3.6M and evaluated every 150k. Crossings with 95\% seed-block intervals; last row is the paired contrast.",
      "tab:horizon", "@{}lcccc@{}", r"Condition & 300k & 900k & 1.8M & 3.6M", rows, size=r"\scriptsize", fit=True)
rows = []
for _, x in st.iterrows():
    y = rc[rc.condition == x.condition].iloc[0]
    rows.append([COND_TEX[x.condition], f"${num(x.change_3p6M_minus_1p8M, True)}$ ${iv(x.ci_low, x.ci_high, True)}$", "yes" if x.stable_pm05 else "no",
                 f"{y.recovered_3p6M}/{y.collapsed_300k}", f"{y.relapsed_3p6M}/{y.noncollapsed_300k}"])
table(OUT / "horizon_stability.tex",
      r"Late-budget change and within-policy outcomes (E3). Stability requires the 3.6M$-$1.8M interval to lie within $\pm.05$ (pre-declared for v4). Recovery: collapsed at 300k, not collapsed at 3.6M; relapse: the reverse.",
      "tab:horizon_stability", "@{}lcccc@{}", r"Condition & 3.6M$-$1.8M & Stable & Recovered & Relapsed", rows)
fig, ax = plt.subplots(figsize=(6.2, 3.0))
for cond, m in [("null", "o"), ("curvature", "s"), ("lambda7", "^")]:
    sub = h[h.condition == cond].sort_values("checkpoint"); t = sub.checkpoint / 1e6
    ax.plot(t, sub.w50, marker=m, ms=3, lw=1, label=COND_TEX[cond].replace("$", "")); ax.fill_between(t, sub.ci_low, sub.ci_high, alpha=.15)
ax.set_xlabel("requested agent transitions (millions)", fontsize=8); ax.set_ylabel("$w_{50}$", fontsize=9); ax.tick_params(labelsize=7); ax.legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(OUT / "horizon_curves.pdf"); plt.close(fig)

# ---------------------------------------------------------------- E4
c4 = pd.read_csv(RES / "E4_crossings.csv"); k4 = pd.read_csv(RES / "E4_contrasts.csv"); d4 = pd.read_csv(RES / "E4_diagnostics.csv")
OPT_TEX = {"global_clip": "Global clip", "separate_no_clip": "Separate, no clip"}
ARM4 = {"linear": "Linear", "curvature": "Curvature", "A": r"A ($\lambda{=}4$)", "S": r"S ($\times4$)"}
rows = []
for opt in ["global_clip", "separate_no_clip"]:
    r = [OPT_TEX[opt]]
    for a in ["linear", "curvature", "A", "S"]:
        x = c4[(c4.optimizer == opt) & (c4.arm == a)].iloc[0]
        r.append(f"${num(x.w50)}$ ${iv(x.ci_low, x.ci_high)}$" + ("$^{\\dagger}$" if not x.in_grid else ""))
    for name in ["A-S", "A-linear"]:
        x = k4[(k4.optimizer == opt) & (k4.contrast == name)].iloc[0]; r.append(f"${num(x.difference, True)}${star(x)} ${iv(x.ci_low, x.ci_high, True)}$")
    rows.append(r)
table(OUT / "lbf_table.tex",
      r"MAPPO on Level-Based Foraging (E4): crossings and paired contrasts. Ten seeds, ten mixture cells, 2M timesteps, 800 trajectories. $^{\dagger}$: point estimate outside the sampled grid (linear and curvature rarely collapse, so their crossings are not localized). $^{\ast}$: interval excludes zero.",
      "tab:lbf", "@{}lcccccc@{}", r"Optimizer & Linear & Curvature & A & S & A$-$S & A$-$linear", rows, size=r"\scriptsize", fit=True)

# ---------------------------------------------------------------- cell-count tables
def cell_header(cells_str):
    return [c.split(":")[0] for c in cells_str.split()]
g1 = cell_header(cr.iloc[0].cells)
rows = [[CELL_TEX[c] if a == "B" else "", a] + [c2.split(":")[1].split("/")[0] for c2 in cr[(cr.cell == c) & (cr.arm == a)].iloc[0].cells.split()]
        for c in CELLS for a in ["B", "A", "S", "R"]]
table(OUT / "factorial_cells.tex", r"E1/E2 observed collapsed runs out of ten seeds per mixture cell at 300k.", "tab:factorial_cells",
      "@{}ll" + "r" * len(g1) + "@{}", "Cell & Arm & " + " & ".join(f"$w{{=}}{x}$" for x in g1), rows, size=r"\scriptsize", fit=True)
g3 = cell_header(h.iloc[0].cells)
rows = [[COND_TEX[cond] if ck == PRIMARY[0] else "", BUD[ck]] + [c2.split(":")[1].split("/")[0] for c2 in h[(h.condition == cond) & (h.checkpoint == ck)].iloc[0].cells.split()]
        for cond in ["null", "curvature", "lambda7"] for ck in PRIMARY]
table(OUT / "horizon_cells.tex", r"E3 observed collapsed trajectories out of ten seeds per mixture cell at the four primary budgets. Full checkpoint records are in the archive.", "tab:horizon_cells",
      "@{}ll" + "r" * len(g3) + "@{}", "Condition & Budget & " + " & ".join(f"$w{{=}}{x}$" for x in g3), rows, size=r"\scriptsize")
g4 = [c.split(":")[0] for c in c4.iloc[0].cells.split()]
rows = [[OPT_TEX[opt] if a == "linear" else "", ARM4[a]] + [c2.split(":")[1] for c2 in c4[(c4.optimizer == opt) & (c4.arm == a)].iloc[0].cells.split()]
        for opt in ["global_clip", "separate_no_clip"] for a in ["linear", "curvature", "A", "S"]]
table(OUT / "lbf_cells.tex", r"E4 observed collapsed runs out of ten seeds per mixture cell (2M timesteps).", "tab:lbf_cells",
      "@{}ll" + "r" * len(g4) + "@{}", "Optimizer & Arm & " + " & ".join(f"$w{{=}}{x}$" for x in g4), rows, size=r"\scriptsize")

# ---------------------------------------------------------------- macros
def mac(name, v): return f"\\newcommand{{\\{name}}}{{{v}}}\n"
def cv(df, **kw):
    q = df
    for k, v in kw.items(): q = q[q[k] == v]
    x = q.iloc[0]; d = x.difference if "difference" in x else x.w50
    return f"${num(d, True)}$ ${iv(x.ci_low, x.ci_high, True)}$"
e1 = pd.read_csv(RES / "E1E2_diagnostics.csv")
m = ""
m += mac("VfourSymStd", cv(co, cell="global_clip/raw", contrast="S-B"))
m += mac("VfourAsymStd", cv(co, cell="global_clip/raw", contrast="A-S"))
m += mac("VfourAsymBoth", cv(co, cell="separate_no_clip/vecnormalize", contrast="A-S"))
m += mac("VfourSymBoth", cv(co, cell="separate_no_clip/vecnormalize", contrast="S-B"))
m += mac("VfourLossBoth", cv(co, cell="separate_no_clip/vecnormalize", contrast="A-B"))
m += mac("VfourAsymNorm", cv(co, cell="global_clip/vecnormalize", contrast="A-S"))
m += mac("VfourAsymSep", cv(co, cell="separate_no_clip/raw", contrast="A-S"))
m += mac("VfourSymSep", cv(co, cell="separate_no_clip/raw", contrast="S-B"))
m += mac("VfourSymNorm", cv(co, cell="global_clip/vecnormalize", contrast="S-B"))
m += mac("VfourSymEps", cv(co, cell="eps1e-8/raw", contrast="S-B"))
m += mac("VfourSymEpsChange", f"${num(it[(it.cell == 'eps1e-8/raw') & (it.contrast == 'S-B')].iloc[0].change_vs_standard, True)}$ ${iv(*it[(it.cell == 'eps1e-8/raw') & (it.contrast == 'S-B')].iloc[0][['ci_low', 'ci_high']], True)}$")
m += mac("VfourLBFAsymGlobal", cv(k4, optimizer="global_clip", contrast="A-S"))
m += mac("VfourLBFAsymSep", cv(k4, optimizer="separate_no_clip", contrast="A-S"))
m += mac("VfourHorizonEnd", cv(hc, checkpoint=3600000, contrast="lambda7-curvature"))
m += mac("VfourHorizonStart", cv(hc, checkpoint=300000, contrast="lambda7-curvature"))
bstd = cr[(cr.cell == "global_clip/raw") & (cr.arm == "B")].iloc[0]; bothers = cr[(cr.cell != "global_clip/raw") & (cr.arm == "B")].w50
m += mac("VfourBaselineStd", f"${num(bstd.w50)}$"); m += mac("VfourBaselineRange", f"${bothers.min():.2f}$--${bothers.max():.2f}$")
ps = e1[(e1.cell == "global_clip/raw")].set_index("arm")
m += mac("VfourAppliedActorB", f"{ps.loc['B'].mb_post_actor_grad_norm:.4f}"); m += mac("VfourAppliedActorS", f"{ps.loc['S'].mb_post_actor_grad_norm:.4f}")
m += mac("VfourCriticNormB", f"{ps.loc['B'].mb_critic_grad_norm:,.0f}"); m += mac("VfourCriticNormS", f"{ps.loc['S'].mb_critic_grad_norm:,.0f}")
lbf_share = d4[d4.optimizer_mode == "global_clip"].mb_actor_grad_energy_share
m += mac("VfourLBFActorShare", f"{lbf_share.min():.2f}--{lbf_share.max():.2f}")
xp = pd.read_csv(RES / "crossplatform.csv")
a, b = xp.iloc[0], xp.iloc[1]
m += mac("VfourAgreeControl", f"{a.agree}/{a.total} ({100 * a.agree / a.total:.1f}\\%)")
m += mac("VfourBitControl", f"{int(a.bit_identical_profit)}/{a.total}")
m += mac("VfourAgreeHorizon", f"{b.agree:,}/{b.total:,} ({100 * b.agree / b.total:.1f}\\%)")
cp = pd.read_csv(RES / "compute_v4.csv")
m += mac("VfourTrajectories", f"{cp.trajectories.sum():,}"); m += mac("VfourCoreHours", f"{cp.single_core_hours.sum():,.0f}")
DESC = {"E1": "Optimizer $\\times$ normalization factorial, 300k", "E2": "Adam-$\\epsilon$ addendum, 300k",
        "E3": "Refined horizon cohort to 3.6M", "E4": "MAPPO on LBF, 2M timesteps"}
table(OUT / "compute_table.tex",
      r"Measured v4 compute: completed Slurm array tasks, one CPU core each (single-thread PyTorch), summed elapsed time. Runs were executed at most eight concurrently.",
      "tab:compute_v4", "@{}llrrr@{}", r"ID & Experiment & Trajectories & Core-hours & Mean minutes per run",
      [[r.experiment, DESC[r.experiment], f"{r.trajectories:,}", f"{r.single_core_hours:,.1f}", f"{r.mean_minutes:.1f}"] for r in cp.itertuples()]
      + [["", "Total", f"{cp.trajectories.sum():,}", f"{cp.single_core_hours.sum():,.1f}", ""]])
(OUT / "v4_numbers.tex").write_text(m)
print("wrote", sorted(p.name for p in OUT.iterdir()))
