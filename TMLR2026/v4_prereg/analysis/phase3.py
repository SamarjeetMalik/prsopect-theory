"""Phase 3 analysis of v4 E1 (optimizer factorial), E2 (Adam-eps addendum) and E3 (refined horizon).

Implements PREREGISTRATION_v4.md and optimizer_factorial_v2/PROTOCOL.md exactly:
- Firth logistic midpoint via the artifact's boundary.firth;
- 2,000 paired training-seed-block multinomial bootstrap draws, bootstrap seed 20261015;
- the same draws for every condition, so all contrasts are seed-paired;
- non-finite draws are dropped from percentiles and counted.

Usage: phase3.py EXTRACTED_DIR ARTIFACT_DIR OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXT, ART, OUT = map(Path, sys.argv[1:4])
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ART / "AAMAS2027/analysis"))
from boundary import firth  # noqa: E402

SEEDS = list(range(42, 52))
GRID = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]
ARMS = ["B", "A", "S", "R"]
WEIGHTS = np.random.default_rng(20261015).multinomial(10, np.full(10, 0.1), size=2000)
EQUIV = 0.05
log = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)


def fit(frame, grid, value="collapsed"):
    """frame: rows for one condition with columns seed, w, collapsed. Returns point, boot draws, cell counts."""
    frame = frame.assign(w=frame.w.round(4)); grid = [round(x, 4) for x in grid]
    k = frame.pivot(index="seed", columns="w", values=value).reindex(index=SEEDS, columns=grid)
    assert not k.isna().any().any(), "missing cells"
    k = k.to_numpy(float); n = np.ones_like(k); w = np.array(grid)
    point = firth(w, k.sum(0), n.sum(0)); boot = firth(w, WEIGHTS @ k, WEIGHTS @ n)
    return dict(w50=point["wcrit"][0], status=point["status"][0], in_grid=bool(point["in_grid"][0]),
                slope=point["b1"][0], boot=boot["wcrit"], cells=k.sum(0).astype(int))


def ci(z):
    ok = np.isfinite(z)
    if ok.sum() == 0: return (np.nan, np.nan, 0)
    lo, hi = np.percentile(z[ok], [2.5, 97.5]); return (lo, hi, int(ok.sum()))


def summarize_fit(name, f):
    lo, hi, u = ci(f["boot"])
    return dict(condition=name, w50=f["w50"], ci_low=lo, ci_high=hi, usable=u, status=f["status"], in_grid=f["in_grid"],
                zero_width=bool(np.isfinite(lo) and abs(hi - lo) < 1e-12), cells=" ".join(f"{w:g}:{c}/10" for w, c in zip(f.get("grid", GRID), f["cells"])))


def contrast(name, a, b):
    d = a["boot"] - b["boot"]; lo, hi, u = ci(d)
    return dict(contrast=name, difference=a["w50"] - b["w50"], ci_low=lo, ci_high=hi, usable=u,
                excludes_zero=bool(lo > 0 or hi < 0), within_equiv=bool(lo >= -EQUIV and hi <= EQUIV))


# ------------------------------------------------------------------ E1 / E2
e1 = pd.read_csv(EXT / "E1_runs.csv"); e2 = pd.read_csv(EXT / "E2_runs.csv")
say("== Validation")
for name, f, n_exp, key in [("E1", e1, 1280, ["optimizer_mode", "reward_mode", "arm", "w", "seed"]), ("E2", e2, 320, ["arm", "w", "seed"])]:
    assert len(f) == n_exp and f.complete.all(), name
    assert not f.duplicated(key).any(), name
    assert (f.actual == 300160).all() and (f.n_eps == 10).all() and f.eval_seeds_ok.all(), name
    assert set(f.w.round(4)) == set(GRID) and set(f.seed) == set(SEEDS) and set(f.arm) == set(ARMS), name
    say(f"{name}: {len(f)} runs complete, unique, 300,160 transitions, 10 evaluation episodes with the declared seeds")
assert (e2.adam_eps == 1e-8).all()
assert set(e1.design) == {"optimizer-factorial-v2"} and set(e2.design) == {"eps-addendum-v4"}

cells = {}
for (opt, rew), g in e1.groupby(["optimizer_mode", "reward_mode"]):
    for arm in ARMS: cells[(f"{opt}/{rew}", arm)] = fit(g[g.arm == arm], GRID)
for arm in ARMS: cells[("eps1e-8/raw", arm)] = fit(e2[e2.arm == arm], GRID)
CELLS = ["global_clip/raw", "global_clip/vecnormalize", "separate_no_clip/raw", "separate_no_clip/vecnormalize", "eps1e-8/raw"]

fits = pd.DataFrame([dict(cell=c, arm=a, **summarize_fit(f"{c}:{a}", cells[(c, a)])) for c in CELLS for a in ARMS])
fits.to_csv(OUT / "E1E2_crossings.csv", index=False)

CONTRASTS = [("A-S", "A", "S"), ("A-B", "A", "B"), ("S-B", "S", "B"), ("R-B", "R", "B")]
rows = []
for c in CELLS:
    for name, hi, lo in CONTRASTS:
        rows.append(dict(cell=c, **contrast(name, cells[(c, hi)], cells[(c, lo)])))
con = pd.DataFrame(rows); con.to_csv(OUT / "E1E2_contrasts.csv", index=False)

# Interactions: change in each contrast relative to the standard cell, paired by seed.
base = "global_clip/raw"; rows = []
for c in CELLS[1:]:
    for name, hi, lo in CONTRASTS:
        d_cell = cells[(c, hi)]["boot"] - cells[(c, lo)]["boot"]; d_base = cells[(base, hi)]["boot"] - cells[(base, lo)]["boot"]
        z = d_cell - d_base; lo_, hi_, u = ci(z)
        pt = (cells[(c, hi)]["w50"] - cells[(c, lo)]["w50"]) - (cells[(base, hi)]["w50"] - cells[(base, lo)]["w50"])
        rows.append(dict(cell=c, contrast=name, change_vs_standard=pt, ci_low=lo_, ci_high=hi_, usable=u, excludes_zero=bool(lo_ > 0 or hi_ < 0)))
inter = pd.DataFrame(rows); inter.to_csv(OUT / "E1E2_interactions.csv", index=False)

# Optimizer diagnostics per cell x arm (means over runs of per-run minibatch means).
e1["cell"] = e1.optimizer_mode + "/" + e1.reward_mode; e2["cell"] = "eps1e-8/raw"
both = pd.concat([e1, e2], ignore_index=True)
dcols = [c for c in ["mb_actor_grad_norm", "mb_critic_grad_norm", "mb_post_actor_grad_norm", "mb_actor_grad_energy_share", "mb_clip_freq"] if c in both]
rcols = [c for c in both.columns if c.startswith("reward_") and c.endswith("_rms")]
diag = both.groupby(["cell", "arm"])[dcols + rcols].mean().reindex([(c, a) for c in CELLS for a in ARMS])
diag.to_csv(OUT / "E1E2_diagnostics.csv")

# Cross-platform agreement: E1 global_clip/raw B/A/S vs archived direct control (authors' cluster).
arch = []
for p in sorted((ART / "AAMAS2027/new_runs/deadline_confirmation_v1/control").glob("run_*")):
    pr = json.loads((p / "protocol.json").read_text()); ev = json.loads((p / "evaluation_300000.json").read_text())
    arch.append(dict(arm=pr["arm"], w=round(pr["w"], 4), seed=pr["seed"], arch_collapsed=int(ev["summary"]["collapsed"]), arch_profit=ev["summary"]["profit"]))
arch = pd.DataFrame(arch)
mine = e1[(e1.cell == base) & e1.arm.isin(["B", "A", "S"])].assign(w=lambda d: d.w.round(4))
m = mine.merge(arch, on=["arm", "w", "seed"], how="inner"); assert len(m) == 240
agree = int((m.collapsed == m.arch_collapsed).sum())
exact = int(np.isclose(m.profit, m.arch_profit, atol=1e-9).sum())

# ------------------------------------------------------------------ E3
e3 = pd.read_csv(EXT / "E3_checkpoints.csv", keep_default_na=False, na_values=[""])  # 'null' is a condition name
G3 = [0.2, 0.4, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 1.0]; CK = sorted(e3.checkpoint.unique()); CONDS = ["null", "curvature", "lambda7"]
assert e3.present.all() and len(e3) == 270 * 24 and (e3.n_eps == 10).all(), "E3 incomplete"
assert not e3.duplicated(["w", "condition", "seed", "checkpoint"]).any()
say(f"E3: 270 trajectories x {len(CK)} checkpoints present, unique, 10 evaluation episodes each")
f3 = {(cond, ck): fit(e3[(e3.condition == cond) & (e3.checkpoint == ck)], G3) for cond in CONDS for ck in CK}
for v in f3.values(): v["grid"] = G3
h_rows = [dict(condition=cond, checkpoint=ck, **{k: v for k, v in summarize_fit(f"{cond}@{ck}", f3[(cond, ck)]).items() if k != "condition"}) for cond in CONDS for ck in CK]
pd.DataFrame(h_rows).to_csv(OUT / "E3_crossings.csv", index=False)
PRIMARY = [300000, 900000, 1800000, 3600000]
hc = []
for ck in PRIMARY:
    hc.append(dict(checkpoint=ck, **contrast("lambda7-curvature", f3[("lambda7", ck)], f3[("curvature", ck)])))
    hc.append(dict(checkpoint=ck, **contrast("lambda7-null", f3[("lambda7", ck)], f3[("null", ck)])))
    hc.append(dict(checkpoint=ck, **contrast("curvature-null", f3[("curvature", ck)], f3[("null", ck)])))
hc = pd.DataFrame(hc); hc.to_csv(OUT / "E3_contrasts.csv", index=False)
stab = []
for cond in CONDS:
    z = f3[(cond, 3600000)]["boot"] - f3[(cond, 1800000)]["boot"]; lo, hi, u = ci(z)
    stab.append(dict(condition=cond, change_3p6M_minus_1p8M=f3[(cond, 3600000)]["w50"] - f3[(cond, 1800000)]["w50"], ci_low=lo, ci_high=hi, usable=u,
                     stable_pm05=bool(lo >= -.05 and hi <= .05), stable_pm02=bool(lo >= -.02 and hi <= .02)))
stab = pd.DataFrame(stab); stab.to_csv(OUT / "E3_stability.csv", index=False)
lab = e3.pivot_table(index=["condition", "w", "seed"], columns="checkpoint", values="collapsed")
rec = []
for cond in CONDS:
    L = lab.loc[cond]; c300 = L[300000] == 1; c36 = L[3600000] == 1
    rec.append(dict(condition=cond, collapsed_300k=int(c300.sum()), recovered_3p6M=int((c300 & ~c36).sum()),
                    noncollapsed_300k=int((~c300).sum()), relapsed_3p6M=int((~c300 & c36).sum())))
rec = pd.DataFrame(rec); rec.to_csv(OUT / "E3_recovery.csv", index=False)
# Cross-platform agreement with the archived horizon cohort on the 5 common cells.
ah = []
for p in sorted((ART / "AAMAS2027/new_runs/deadline_confirmation_v1/horizon").glob("run_*")):
    pr = json.loads((p / "protocol.json").read_text())
    for ck in pr["checkpoints"]:
        ev = json.loads((p / f"evaluation_{ck}.json").read_text())
        ah.append(dict(condition=pr["condition"], w=round(pr["w"], 4), seed=pr["seed"], checkpoint=ck, arch=int(ev["summary"]["collapsed"])))
ah = pd.DataFrame(ah); mh = e3.assign(w=e3.w.round(4)).merge(ah, on=["condition", "w", "seed", "checkpoint"]); assert len(mh) == 150 * 24
h_agree = int((mh.collapsed == mh.arch).sum())
pd.DataFrame([dict(comparison="E1 default cell B/A/S vs archived direct control", agree=agree, bit_identical_profit=exact, total=240),
              dict(comparison="E3 vs archived horizon cohort (5 common cells, all checkpoints)", agree=h_agree, bit_identical_profit=np.nan, total=len(mh))]
             ).to_csv(OUT / "crossplatform.csv", index=False)

# ------------------------------------------------------------------ report
def fmt(x): return "nan" if not np.isfinite(x) else f"{x:+.3f}"
say("\n== E1/E2 crossings (w50 [95% CI], cells k/10)")
for _, r in fits.iterrows():
    say(f"{r.cell:30s} {r.arm}: {r.w50:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}] usable {r.usable} {r.status}{' ZERO-WIDTH' if r.zero_width else ''}  {r.cells}")
say("\n== E1/E2 contrasts")
for _, r in con.iterrows():
    say(f"{r.cell:30s} {r.contrast}: {fmt(r.difference)} [{fmt(r.ci_low)}, {fmt(r.ci_high)}]{'  *' if r.excludes_zero else ''}{'  equiv(+-.05)' if r.within_equiv else ''}")
say("\n== Change in contrast vs global_clip/raw (paired)")
for _, r in inter.iterrows():
    say(f"{r.cell:30s} {r.contrast}: {fmt(r.change_vs_standard)} [{fmt(r.ci_low)}, {fmt(r.ci_high)}]{'  *' if r.excludes_zero else ''}")
say("\n== Diagnostics (mean per run)\n" + diag.round(4).to_string())
say(f"\n== Cross-platform labels (E1 global_clip/raw B/A/S vs archive): {agree}/240 agree; {exact}/240 bit-identical profit")
say(f"== Cross-platform labels (E3 vs archived horizon, 5 cells x 3 conditions x 10 seeds x 24 checkpoints): {h_agree}/{len(mh)} agree")
say("\n== E3 primary-budget crossings")
for cond in CONDS:
    for ck in PRIMARY:
        r = summarize_fit("", f3[(cond, ck)]); say(f"{cond:10s} {ck:>8d}: {r['w50']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]{' ZERO-WIDTH' if r['zero_width'] else ''}  {r['cells']}")
say("\n== E3 contrasts\n" + hc.round(3).to_string(index=False))
say("\n== E3 stability (3.6M - 1.8M)\n" + stab.round(3).to_string(index=False))
say("\n== E3 recovery/relapse (conditional on 300k label, state at 3.6M)\n" + rec.to_string(index=False))
(OUT / "phase3_report.txt").write_text("\n".join(log) + "\n")
