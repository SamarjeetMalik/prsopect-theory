"""Phase 3 analysis of v4 E4 (MAPPO on LBF; optimizer x transform), per PREREGISTRATION_v4.md section E4.

Usage: phase3_e4.py E4_RUNS_CSV ARTIFACT_DIR OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CSV, ART, OUT = map(Path, sys.argv[1:4]); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ART / "AAMAS2027/analysis"))
from boundary import firth  # noqa: E402

SEEDS = list(range(42, 52)); GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.65, 0.75, 0.85, 0.9, 0.95]
ARMS = ["linear", "curvature", "A", "S"]; OPTS = ["global_clip", "separate_no_clip"]
WEIGHTS = np.random.default_rng(20261015).multinomial(10, np.full(10, 0.1), size=2000)
log = []
def say(s=""): print(s); log.append(s)
def ci(z):
    ok = np.isfinite(z)
    return (np.nan, np.nan, 0) if not ok.any() else (*np.percentile(z[ok], [2.5, 97.5]), int(ok.sum()))

f = pd.read_csv(CSV).assign(w=lambda d: d.w.round(4))
assert len(f) == 800 and f.complete.all() and (f.n_eps == 10).all() and (f.actual >= 2_000_000).all()
assert not f.duplicated(["optimizer_mode", "arm", "w", "seed"]).any()
assert set(f.design) == {"e4-lbf-v4"} and set(f.seed) == set(SEEDS) and set(f.w) == set(round(x, 4) for x in GRID)
say("E4: 800 runs complete, unique, >= 2M transitions, 10 evaluation episodes each")

fits = {}
for opt in OPTS:
    for arm in ARMS:
        g = f[(f.optimizer_mode == opt) & (f.arm == arm)]
        k = g.pivot(index="seed", columns="w", values="collapsed").reindex(index=SEEDS, columns=[round(x, 4) for x in GRID]).to_numpy(float)
        assert not np.isnan(k).any(); n = np.ones_like(k); w = np.array(GRID)
        p = firth(w, k.sum(0), n.sum(0)); b = firth(w, WEIGHTS @ k, WEIGHTS @ n)
        fits[(opt, arm)] = dict(w50=p["wcrit"][0], status=p["status"][0], in_grid=bool(p["in_grid"][0]), boot=b["wcrit"], cells=k.sum(0).astype(int))

rows = []
say("\n== E4 crossings (w50 [95% CI], collapsed k/10 per cell)")
for opt in OPTS:
    for arm in ARMS:
        x = fits[(opt, arm)]; lo, hi, u = ci(x["boot"])
        cells = " ".join(f"{w:g}:{c}" for w, c in zip(GRID, x["cells"]))
        rows.append(dict(optimizer=opt, arm=arm, w50=x["w50"], ci_low=lo, ci_high=hi, usable=u, status=x["status"], in_grid=x["in_grid"], cells=cells))
        say(f"{opt:17s} {arm:9s} {x['w50']:.3f} [{lo:.3f}, {hi:.3f}] usable {u} {x['status']}{'' if x['in_grid'] else ' OUT-OF-GRID'}  {cells}")
pd.DataFrame(rows).to_csv(OUT / "E4_crossings.csv", index=False)

CON = [("curvature-linear", "curvature", "linear"), ("A-linear", "A", "linear"), ("A-curvature", "A", "curvature"),
       ("S-curvature", "S", "curvature"), ("A-S", "A", "S")]
rows = []; say("\n== E4 contrasts (paired seed-block)")
for opt in OPTS:
    for name, a, b in CON:
        d = fits[(opt, a)]["boot"] - fits[(opt, b)]["boot"]; lo, hi, u = ci(d); pt = fits[(opt, a)]["w50"] - fits[(opt, b)]["w50"]
        rows.append(dict(optimizer=opt, contrast=name, difference=pt, ci_low=lo, ci_high=hi, usable=u, excludes_zero=bool(lo > 0 or hi < 0)))
        say(f"{opt:17s} {name:17s} {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]{'  *' if lo > 0 or hi < 0 else ''}")
con = pd.DataFrame(rows); con.to_csv(OUT / "E4_contrasts.csv", index=False)

rows = []; say("\n== Change separate_no_clip minus global_clip (paired)")
for name, a, b in CON:
    z = (fits[("separate_no_clip", a)]["boot"] - fits[("separate_no_clip", b)]["boot"]) - (fits[("global_clip", a)]["boot"] - fits[("global_clip", b)]["boot"])
    lo, hi, u = ci(z)
    pt = (fits[("separate_no_clip", a)]["w50"] - fits[("separate_no_clip", b)]["w50"]) - (fits[("global_clip", a)]["w50"] - fits[("global_clip", b)]["w50"])
    rows.append(dict(contrast=name, change=pt, ci_low=lo, ci_high=hi, usable=u, excludes_zero=bool(lo > 0 or hi < 0)))
    say(f"{name:17s} {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]{'  *' if lo > 0 or hi < 0 else ''}")
inter = pd.DataFrame(rows); inter.to_csv(OUT / "E4_interactions.csv", index=False)

dcols = [c for c in ["mb_actor_grad_norm", "mb_critic_grad_norm", "mb_post_actor_grad_norm", "mb_actor_grad_energy_share", "mb_clip_freq",
                     "reward_blend_rms", "reward_blend_neg", "reward_transformed_rms", "reward_transformed_neg", "elapsed"] if c in f]
diag = f.groupby(["optimizer_mode", "arm"])[dcols].mean().reindex([(o, a) for o in OPTS for a in ARMS])
diag.to_csv(OUT / "E4_diagnostics.csv"); say("\n== E4 diagnostics (mean per run)\n" + diag.round(4).to_string())

def get(opt, name): return con[(con.optimizer == opt) & (con.contrast == name)].iloc[0]
def getI(name): return inter[inter.contrast == name].iloc[0]
p1 = get("global_clip", "curvature-linear"); p2g = get("global_clip", "A-linear"); p2s = get("separate_no_clip", "A-linear")
i_c = getI("curvature-linear"); i_s = getI("S-curvature")
say("\n== Pre-declared predictions")
say(f"P1 global_clip curvature-linear > 0: {p1.difference:+.3f} [{p1.ci_low:+.3f}, {p1.ci_high:+.3f}] -> {'MET' if p1.ci_low > 0 else 'NOT MET'}")
say(f"P2 A-linear > 0: global {p2g.difference:+.3f} [{p2g.ci_low:+.3f}, {p2g.ci_high:+.3f}], separate {p2s.difference:+.3f} [{p2s.ci_low:+.3f}, {p2s.ci_high:+.3f}] -> "
    f"{'MET (both)' if p2g.ci_low > 0 and p2s.ci_low > 0 else ('MET (global only)' if p2g.ci_low > 0 else 'NOT MET')}")
say(f"P3 separate shrinks curvature-linear: change {i_c.change:+.3f} [{i_c.ci_low:+.3f}, {i_c.ci_high:+.3f}] -> {'MET' if i_c.ci_high < 0 else 'NOT MET'}; "
    f"shrinks S-curvature: change {i_s.change:+.3f} [{i_s.ci_low:+.3f}, {i_s.ci_high:+.3f}] -> {'MET' if i_s.ci_high < 0 else 'NOT MET'}")
(OUT / "phase3_e4_report.txt").write_text("\n".join(log) + "\n")
