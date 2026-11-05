"""Figures and sensitivity tables added in response to the 5 November external review. Exploratory presentation of
existing E1, E3-E6 runs: no new experiment, no change to any pre-declared estimate.

- fig_e5_horizon.pdf     E5 crossings and paired contrasts at all 24 checkpoints (E3 default-optimizer contrast for reference)
- fig_e6_mechanism.pdf   E1 pre-clip gradient norms under the shared clip; E6 S/B update ratios; E6 crossing contrasts
- fig_e4_curves.pdf      E4 observed collapse proportions (Wilson), Firth curves and isotonic crossings, both optimizers
- fig_label.pdf          E1 evaluation outcomes against the two-part collapse rule
- fig_reward_moments.pdf E1 transformed-reward mean, RMS and loss-branch fraction by arm and w
- loso_table.tex         leave-one-seed-out range of the principal contrasts (point estimates refitted on 9 seeds)
Writes CSVs to RES and figures/tables to TMLR2026/generated/v4/.

Usage: review_figures.py RUNS_DIR ARTIFACT_DIR RES_DIR
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RUNS, ART, RES = map(Path, sys.argv[1:4])
OUT = Path(__file__).resolve().parents[2] / "generated/v4"
sys.path.insert(0, str(ART / "AAMAS2027/analysis"))
from boundary import firth, wilson  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from isotonic import isotonic_crossing  # noqa: E402

SEEDS = list(range(42, 52))
WEIGHTS = np.random.default_rng(20261015).multinomial(10, np.full(10, 0.1), size=2000)   # as phase3.py
G1 = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]
G3 = [0.2, 0.4, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 1.0]
G4 = [0.0, 0.1, 0.2, 0.3, 0.5, 0.65, 0.75, 0.85, 0.9, 0.95]
THETA4 = 0.626
# One colour and marker per arm in every figure (Okabe-Ito; markers keep them apart in greyscale).
STYLE = {"linear": ("#7f7f7f", "D", "Linear"), "B": ("#0072B2", "o", "B: curvature"), "curvature": ("#0072B2", "o", "Curvature"),
         "A": ("#D55E00", "^", r"A: losses $\times\lambda$"), "S": ("#009E73", "s", r"S: both $\times\lambda$"),
         "R": ("#CC79A7", "v", r"R: both $\times3.451$")}
CON_STYLE = {"A-S": ("#D55E00", "^"), "A-B": ("#E69F00", "D"), "S-B": ("#009E73", "s")}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})


def counts(frame, grid, col="collapsed"):
    k = frame.assign(w=frame.w.round(4)).pivot(index="seed", columns="w", values=col)
    k = k.reindex(index=SEEDS, columns=[round(x, 4) for x in grid]).to_numpy(float)
    assert not np.isnan(k).any(), "missing cells"
    return k


def fit(k, grid, boot=True):
    w = np.array(grid); n = np.ones_like(k)
    p = firth(w, k.sum(0), n.sum(0))
    out = dict(w50=p["wcrit"][0], b0=p["b0"][0], b1=p["b1"][0], in_grid=bool(p["in_grid"][0]), cells=k.sum(0))
    if boot:
        out["boot"] = firth(w, WEIGHTS @ k, WEIGHTS @ n)["wcrit"]
    return out


def ci(z):
    ok = np.isfinite(z)
    return (np.nan, np.nan) if not ok.any() else tuple(np.percentile(z[ok], [2.5, 97.5]))


def loso(ka, kb, grid):
    """Contrast of Firth point estimates with each training seed left out in turn."""
    out = []
    for j in range(len(SEEDS)):
        keep = np.arange(len(SEEDS)) != j
        out.append(fit(ka[keep], grid, False)["w50"] - fit(kb[keep], grid, False)["w50"])
    return np.array(out)


e1 = pd.read_csv(RUNS / "E1_runs.csv"); e1["cell"] = e1.optimizer_mode + "/" + e1.reward_mode
e3 = pd.read_csv(RUNS / "E3_checkpoints.csv", keep_default_na=False, na_values=[""])
e4 = pd.read_csv(RUNS / "E4_runs.csv").assign(c=lambda d: (d.team_return < THETA4).astype(int))
e5 = pd.read_csv(RUNS / "E5_checkpoints.csv"); e6 = pd.read_csv(RUNS / "E6_runs.csv")
assert len(e1) == 1280 and len(e4) == 800 and len(e5) == 270 * 24 and len(e6) == 720 and e5.present.all()

# ------------------------------------------------------------------ E5 at every checkpoint
CK = sorted(e5.checkpoint.unique()); rows = []
for ck in CK:
    g = e5[e5.checkpoint == ck]; f = {a: fit(counts(g[g.arm == a], G3), G3) for a in "BAS"}
    for a in "BAS":
        lo, hi = ci(f[a]["boot"]); rows.append(dict(kind="crossing", name=a, checkpoint=ck, value=f[a]["w50"], ci_low=lo, ci_high=hi))
    for name, x, y in [("A-S", "A", "S"), ("A-B", "A", "B"), ("S-B", "S", "B")]:
        lo, hi = ci(f[x]["boot"] - f[y]["boot"])
        rows.append(dict(kind="contrast", name=name, checkpoint=ck, value=f[x]["w50"] - f[y]["w50"], ci_low=lo, ci_high=hi))
e3rows = []
for ck in CK:
    g = e3[e3.checkpoint == ck]; f = {c: fit(counts(g[g.condition == c], G3), G3) for c in ("lambda7", "curvature")}
    lo, hi = ci(f["lambda7"]["boot"] - f["curvature"]["boot"])
    e3rows.append(dict(checkpoint=ck, value=f["lambda7"]["w50"] - f["curvature"]["w50"], ci_low=lo, ci_high=hi))
r5 = pd.DataFrame(rows); r5.to_csv(RES / "E5_all_checkpoints.csv", index=False)
r3 = pd.DataFrame(e3rows); r3.to_csv(RES / "E3_all_checkpoints_contrast.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
for a in "BAS":
    s = r5[(r5.kind == "crossing") & (r5.name == a)]; t = s.checkpoint / 1e6; c, m, lab = STYLE[a]
    ax[0].plot(t, s.value, color=c, marker=m, ms=2.8, lw=1.1, label=lab.replace(r"\lambda", "7")); ax[0].fill_between(t, s.ci_low, s.ci_high, color=c, alpha=.13, lw=0)
ax[0].set(xlabel="requested agent transitions (millions)", ylabel=r"crossing $w_{50}$", title="E5 crossings (decoupled, normalized)")
for name in ("A-S", "A-B", "S-B"):
    s = r5[(r5.kind == "contrast") & (r5.name == name)]; t = s.checkpoint / 1e6; c, m = CON_STYLE[name]
    ax[1].plot(t, s.value, color=c, marker=m, ms=2.8, lw=1.1, label=name.replace("-", "$-$")); ax[1].fill_between(t, s.ci_low, s.ci_high, color=c, alpha=.13, lw=0)
ax[1].plot(r3.checkpoint / 1e6, r3.value, color="0.35", ls="--", lw=1.0, label=r"E3 default: $\lambda7-$curvature")
ax[1].axhline(0, color="0.5", lw=.7); ax[1].set(xlabel="requested agent transitions (millions)", ylabel="paired crossing difference", title="E5 contrasts")
for a in ax:
    a.legend(frameon=False, loc="best")
fig.tight_layout(); fig.savefig(OUT / "fig_e5_horizon.pdf"); plt.close(fig)

# ------------------------------------------------------------------ E6 mechanism
ur = pd.read_csv(RES / "E6_update_ratios.csv"); k6 = pd.read_csv(RES / "E6_contrasts.csv")
MODES = [("global_clip", "Shared clip"), ("groupwise_clip", "Per-group clip"), ("separate_no_clip", "No clip")]
d = e1[e1.cell == "global_clip/raw"].groupby("arm")[["mb_critic_grad_norm", "mb_actor_grad_norm", "mb_post_actor_grad_norm"]].mean()
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.5), gridspec_kw=dict(width_ratios=[1.0, 1.15, 1.15]))
x = np.arange(2)
for j, (col, lab) in enumerate([("mb_critic_grad_norm", "critic, pre-clip"), ("mb_actor_grad_norm", "actor, pre-clip"), ("mb_post_actor_grad_norm", "actor, applied")]):
    ax[0].bar(x + (j - 1) * .26, d.loc[["B", "S"], col], .26, color=["0.25", "0.55", "0.85"][j], edgecolor="k", lw=.4, label=lab)
ax[0].set_yscale("log"); ax[0].set_xticks(x, ["B", "S"]); ax[0].axhline(0.5, color="#D55E00", lw=.8, ls=":")
ax[0].text(-0.45, 0.6, "clip 0.5", color="#D55E00", fontsize=6, va="bottom", ha="left")
ax[0].set(ylabel="gradient norm (per-minibatch mean)", title="E1 default: shared clip"); ax[0].legend(frameon=False, fontsize=6, loc="upper left")
y = np.arange(len(MODES))[::-1]
for off, q, lab, c in [(.12, "pu_actor_update_norm", "actor parameter update", "k"), (-.12, "pu_applied_actor_grad_norm", "applied actor gradient", "0.55")]:
    s = ur[(ur.ratio == "S/B") & (ur.quantity == q)].set_index("mode").loc[[m for m, _ in MODES]]
    ax[1].errorbar(s.point, y + off, xerr=[s.point - s.ci_low, s.ci_high - s.point], fmt="o", color=c, ms=3.5, capsize=2, label=lab)
ax[1].axvline(1, color="0.5", lw=.7); ax[1].set_yticks(y, [t for _, t in MODES]); ax[1].set(xlabel="S/B ratio (E6, seed-paired 95% CI)", title="Actor update shrinks only\nunder the shared clip")
ax[1].legend(frameon=False, fontsize=6, loc="lower right")
for off, name in [(.12, "S-B"), (-.12, "A-S")]:
    s = k6[k6.contrast == name].set_index("mode").loc[[m for m, _ in MODES]]; c, m = CON_STYLE[name]
    ax[2].errorbar(s.difference, y + off, xerr=[s.difference - s.ci_low, s.ci_high - s.difference], fmt=m, color=c, ms=3.5, capsize=2, label=name.replace("-", "$-$"))
ax[2].axvline(0, color="0.5", lw=.7); ax[2].set_yticks(y, ["", "", ""]); ax[2].set(xlabel="crossing difference at 300k", title="E6 crossing contrasts")
ax[2].legend(frameon=False, fontsize=6, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "fig_e6_mechanism.pdf"); plt.close(fig)

# ------------------------------------------------------------------ E4 response curves
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
wg = np.linspace(0, 1, 201)
for a_, (opt, title) in zip(ax, [("global_clip", "Global clip"), ("separate_no_clip", "Separate optimizers, no clip")]):
    for arm in ("linear", "curvature", "A", "S"):
        g = e4[(e4.optimizer_mode == opt) & (e4.arm == arm)]; k = counts(g, G4, "c"); f = fit(k, G4, False)
        c, m, lab = STYLE[arm]; lab = lab.replace(r"\lambda", "4"); p = k.sum(0) / 10; lo, hi = wilson(k.sum(0), 10)
        jit = {"linear": -.009, "curvature": -.003, "A": .003, "S": .009}[arm]
        a_.errorbar(np.array(G4) + jit, p, yerr=[p - lo, hi - p], fmt=m, color=c, ms=3, capsize=1.2, lw=.7, label=lab)
        a_.plot(wg, 1 / (1 + np.exp(-(f["b0"] + f["b1"] * wg))), color=c, lw=1.0)
        iso = isotonic_crossing(np.array(G4), k.sum(0), np.full(len(G4), 10.0))
        if np.isfinite(iso):
            a_.plot([iso], [0.5], marker="|", color=c, ms=9, mew=1.5)
    a_.axhline(.5, color="0.6", lw=.6, ls=":"); a_.set(xlabel="mixture weight $w$", title=title)
ax[0].set_ylabel(r"collapsed fraction ($\theta=0.626$)")
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, ncol=4, loc="lower center")
fig.tight_layout(rect=(0, .08, 1, 1)); fig.savefig(OUT / "fig_e4_curves.pdf"); plt.close(fig)

# ------------------------------------------------------------------ label validation
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for lab, c, m, sz in [(0, "#0072B2", "o", 5), (1, "#D55E00", "x", 14)]:
    s = e1[e1.collapsed == lab]
    ax.scatter(s.mean_S, s.profit, s=sz, c=c, marker=m, lw=.7, alpha=.55 if lab == 0 else .9, label=f"collapsed ({len(s)})" if lab else f"not collapsed ({len(s)})")
ax.axvline(10, color="0.3", lw=.7, ls="--"); ax.axhline(-127.1, color="0.3", lw=.7, ls="--")
ax.set(xlabel="evaluation base-stock target $S$", ylabel="evaluation profit", title=f"E1, all {len(e1):,} runs")
ax.legend(frameon=False, loc="center right"); fig.tight_layout(); fig.savefig(OUT / "fig_label.pdf"); plt.close(fig)
lab_stats = dict(n=len(e1), collapsed=int(e1.collapsed.sum()),
                 near_S=int(((e1.mean_S - 10).abs() < 1).sum()), near_profit=int(((e1.profit + 127.1).abs() < 1).sum()))
pd.DataFrame([lab_stats]).to_csv(RES / "E1_label_margins.csv", index=False)

# ------------------------------------------------------------------ reward moments
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.4))
for j, (col, ttl) in enumerate([("reward_transformed_mean", "transformed-reward mean"), ("reward_transformed_rms", "transformed-reward RMS"), ("reward_centered_neg", "loss-branch fraction")]):
    for arm in "BASR":
        for cell, ls in [("global_clip/raw", "-"), ("separate_no_clip/vecnormalize", "--")]:
            s = e1[(e1.cell == cell) & (e1.arm == arm)].groupby(e1.w.round(4))[col].mean(); c, m, lab = STYLE[arm]
            ax[j].plot(s.index, s.values, color=c, marker=m, ms=2.5, lw=.9, ls=ls, label=lab.replace(r"\lambda", "7") if cell == "global_clip/raw" else None)
    ax[j].set(xlabel="mixture weight $w$", title=ttl)
ax[0].legend(frameon=False, fontsize=6); ax[2].plot([], [], "k--", lw=.9, label="decoupled cell"); ax[2].plot([], [], "k-", lw=.9, label="default cell")
ax[2].legend(frameon=False, fontsize=6, loc="best"); fig.tight_layout(); fig.savefig(OUT / "fig_reward_moments.pdf"); plt.close(fig)

# ------------------------------------------------------------------ leave-one-seed-out
def e1k(cell, arm): return counts(e1[(e1.cell == cell) & (e1.arm == arm)], G1)
def e5k(arm): return counts(e5[(e5.checkpoint == 3600000) & (e5.arm == arm)], G3)
def e6k(mode, arm): return counts(e6[(e6.optimizer_mode == mode) & (e6.arm == arm)], G1)
def e4k(opt, arm): return counts(e4[(e4.optimizer_mode == opt) & (e4.arm == arm)], G4, "c")
SPECS = [("E1 default", "S-B", e1k("global_clip/raw", "S"), e1k("global_clip/raw", "B"), G1),
         ("E1 default", "A-S", e1k("global_clip/raw", "A"), e1k("global_clip/raw", "S"), G1),
         ("E1 decoupled", "S-B", e1k("separate_no_clip/vecnormalize", "S"), e1k("separate_no_clip/vecnormalize", "B"), G1),
         ("E1 decoupled", "A-S", e1k("separate_no_clip/vecnormalize", "A"), e1k("separate_no_clip/vecnormalize", "S"), G1),
         ("E5, 3.6M", "S-B", e5k("S"), e5k("B"), G3), ("E5, 3.6M", "A-S", e5k("A"), e5k("S"), G3),
         ("E6 per-group clip", "S-B", e6k("groupwise_clip", "S"), e6k("groupwise_clip", "B"), G1),
         ("E6 per-group clip", "A-S", e6k("groupwise_clip", "A"), e6k("groupwise_clip", "S"), G1),
         ("E4 global clip", "A-S", e4k("global_clip", "A"), e4k("global_clip", "S"), G4),
         ("E4 separate", "A-S", e4k("separate_no_clip", "A"), e4k("separate_no_clip", "S"), G4)]
rows = []
for exp, name, ka, kb, grid in SPECS:
    full = fit(ka, grid, False)["w50"] - fit(kb, grid, False)["w50"]; z = loso(ka, kb, grid)
    rows.append(dict(experiment=exp, contrast=name, full=full, loso_min=np.nanmin(z), loso_max=np.nanmax(z),
                     same_sign=int(np.sum(np.sign(z) == np.sign(full))), finite=int(np.isfinite(z).sum())))
lo_df = pd.DataFrame(rows); lo_df.to_csv(RES / "loso.csv", index=False)
body = "\n".join(f"{r.experiment} & {r.contrast.replace('-', '$-$')} & ${r.full:+.3f}$ & $[{r.loso_min:+.3f},\\,{r.loso_max:+.3f}]$ & {r.same_sign}/{r.finite} \\\\"
                 for r in lo_df.itertuples())
(OUT / "loso_table.tex").write_text(f"""\\begin{{table}}[H]
\\caption{{Leave-one-seed-out sensitivity of the principal contrasts (exploratory). Each contrast is refitted ten times on nine training seeds; the range of the ten point estimates and the number that keep the full-data sign are shown. E4 uses the pre-declared $\\theta=0.626$.}}
\\label{{tab:loso}}
\\centering\\small
\\begin{{tabular}}{{@{{}}llrcr@{{}}}}
\\toprule
Experiment & Contrast & All seeds & Leave-one-out range & Same sign \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")

# ------------------------------------------------------------------ macros
def mac(n, v): return f"\\newcommand{{\\{n}}}{{{v}}}\n"
s = r5[r5.kind == "contrast"]
asm = s[s.name == "A-S"]; sbm = s[s.name == "S-B"]
m = mac("VrevAsymPos", f"{int((asm.ci_low > 0).sum())}/{len(asm)}") + mac("VrevSymExcl", f"{int(((sbm.ci_low > 0) | (sbm.ci_high < 0)).sum())}/{len(sbm)}")
neg = sbm[sbm.ci_high < 0]
m += mac("VrevSymNeg", f"{len(neg)}") + mac("VrevSymNegSpan", f"{neg.checkpoint.min() / 1e6:.2f}--{neg.checkpoint.max() / 1e6:.1f}M" if len(neg) else "--")
m += mac("VrevSymPos", f"{int((sbm.ci_low > 0).sum())}")
m += mac("VrevAsymMin", f"{asm.value.min():+.3f}") + mac("VrevAsymMax", f"{asm.value.max():+.3f}")
m += mac("VrevCritB", f"{d.loc['B'].mb_critic_grad_norm:,.0f}") + mac("VrevCritS", f"{d.loc['S'].mb_critic_grad_norm:,.0f}")
m += mac("VrevLabelN", f"{lab_stats['n']:,}") + mac("VrevLabelNearS", str(lab_stats["near_S"])) + mac("VrevLabelNearP", str(lab_stats["near_profit"]))
flip = lo_df[lo_df.same_sign < lo_df.finite]
m += mac("VrevLosoFlips", "none" if flip.empty else ", ".join(f"{r.experiment} {r.contrast.replace('-', '$-$')} ({r.finite - r.same_sign}/{r.finite})" for r in flip.itertuples()))
(OUT / "review_numbers.tex").write_text(m)
print(lo_df.round(3).to_string(index=False)); print(lab_stats); print(m)
