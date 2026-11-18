"""EXPLORATORY (not pre-registered; written after X3's results were seen). Pools the two fresh cohorts X1 (seeds 82-101)
and X3 (seeds 102-121), forty seeds, for the configurations they share:
  no clipping       X1 separate_no_clip + X3 neither   (Adam is per-parameter, so one or two Adam groups are equivalent)
  per-group clip    X1 groupwise_clip  + X3 both
and reports S-B for each and their difference, with seed-level jackknife-t (39 df) on pooled Firth crossings.

Usage: explore_x13_pooled.py X1_runs.csv X3_runs.csv ARTIFACT_DIR OUT_DIR
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

X1, X3, ART, OUT = map(Path, sys.argv[1:5])
for d in [str(ART / "AAMAS2027/analysis"), os.environ.get("BOUNDARY_DIR", "")]:
    if d and (Path(d) / "boundary.py").exists():
        sys.path.insert(0, d)
from boundary import firth  # noqa: E402

GRID = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]; SEEDS = list(range(82, 122)); n = len(SEEDS); J = 1 - np.eye(n)
a, b = pd.read_csv(X1), pd.read_csv(X3)
sets = {"noclip": pd.concat([a[a["mode"] == "separate_no_clip"], b[b["mode"] == "neither"]]),
        "groupclip": pd.concat([a[a["mode"] == "groupwise_clip"], b[b["mode"] == "both"]]),
        "globalclip_X1only": a[a["mode"] == "global_clip"]}


def cross(f, seeds):
    k = f.assign(w=f.w.round(4)).pivot(index="seed", columns="w", values="collapsed").reindex(index=seeds, columns=GRID)
    assert not k.isna().any().any(); k = k.to_numpy(float); one = np.ones_like(k); w = np.array(GRID); Jn = 1 - np.eye(len(seeds))
    return firth(w, k.sum(0), one.sum(0))["wcrit"][0], firth(w, Jn @ k, Jn @ one)["wcrit"]


def sb(f, seeds):
    s, b_ = cross(f[f.arm == "S"], seeds), cross(f[f.arm == "B"], seeds)
    return s[0] - b_[0], s[1] - b_[1]


def jt(d, j):
    j = j[np.isfinite(j)]; m = len(j); se = np.sqrt((m - 1) / m * ((j - j.mean()) ** 2).sum()); t = stats.t.ppf(.975, m - 1)
    return d - t * se, d + t * se


rows = []
R = {k: sb(v, SEEDS if k != "globalclip_X1only" else SEEDS[:20]) for k, v in sets.items()}
for k, (d, j) in R.items():
    lo, hi = jt(d, j); rows.append(dict(quantity=f"S-B {k}", difference=d, jt_low_95=lo, jt_high_95=hi))
d = R["groupclip"][0] - R["noclip"][0]; j = R["groupclip"][1] - R["noclip"][1]; lo, hi = jt(d, j)
rows.append(dict(quantity="groupclip minus noclip", difference=d, jt_low_95=lo, jt_high_95=hi))
out = pd.DataFrame(rows); out.to_csv(OUT / "explore_x13_pooled.csv", index=False); print(out.to_string(index=False))
