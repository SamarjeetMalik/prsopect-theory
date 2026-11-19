"""Seed-level jackknife-t intervals for every interval the frozen E1-E12 analyses report (revision v6; not pre-registered).

Each frozen analysis script draws its seed-block bootstrap weights as
    np.random.default_rng(SEED).multinomial(N, ..., size=2000)
and turns the draws into intervals with np.percentile. This script re-executes each script's source unchanged except for
two substitutions:
  * the weight matrix becomes the N x N leave-one-seed-out matrix 1 - I, so every "draw" is a leave-one-seed-out
    estimate, paired across conditions exactly as the bootstrap draws were;
  * np.percentile(z, [a, 100 - a]) returns the jackknife-t half-width, -/+ t_{n-1}(1 - a/100) * se_jack, around zero.
The outputs go to a separate directory. Every interval column is then re-centred on its row's full-sample estimate
(est +/- t * se, as in jackknife_v5.py), and the pre-declared percentile endpoints are kept beside it as pct_low/pct_high.
Zero-exclusion flags are recomputed; verdict text in the re-executed reports is not meaningful and is discarded.
The pre-declared outputs in results/ are not touched.

Usage: jackknife_all.py RUNS_DIR ARTIFACT_DIR RES_DIR JT_DIR
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DRAWS = re.compile(r"np\.random\.default_rng\(\d+\)\.multinomial\((\d+),\s*np\.full\(\d+,\s*[0-9./ ]+\),\s*size=2000\)")


def jk_percentile(z, q):
    """Stand-in for np.percentile on a vector of leave-one-seed-out values: returns -/+ the jackknife-t half-width."""
    v = np.asarray(z, float); v = v[np.isfinite(v)]; n = len(v)
    if n < 3:
        return np.array([np.nan, np.nan])
    se = np.sqrt((n - 1) / n * ((v - v.mean()) ** 2).sum())
    a = min(q) / 100; h = stats.t.ppf(1 - a, n - 1) * se
    return np.array([-h, h])


def run_one(script, args):
    src = (HERE / script).read_text()
    new, k = DRAWS.subn(lambda m: f"(1 - np.eye({m.group(1)}))", src)
    assert k > 0, f"{script}: no bootstrap weight matrix found"
    new = new.replace("np.percentile(", "_jk_percentile(")
    sys.argv = [str(HERE / script)] + args
    ns = {"__file__": str(HERE / script), "__name__": "__main__", "_jk_percentile": jk_percentile}
    exec(compile(new, script, "exec"), ns)


# estimate column per output file, and the interval column pairs it carries
EST = ["difference", "change_3p6M_minus_1p8M", "change_vs_standard", "change_vs_global", "change", "w50", "value", "estimate", "point"]
PAIRS = [("ci_low", "ci_high"), ("ci_low_99", "ci_high_99")]
SCRIPTS = [  # (script, args builder, outputs)
    ("phase3.py", lambda R, A, J: [R, A, J], ["E1E2_crossings", "E1E2_contrasts", "E1E2_interactions", "E3_crossings", "E3_contrasts", "E3_stability"]),
    ("phase3_e4.py", lambda R, A, J: [R / "E4_runs.csv", A, J], ["E4_crossings", "E4_contrasts", "E4_interactions", "E4_theta_sensitivity"]),
    ("phase3_e5e6.py", lambda R, A, J: [R, R / "E1_runs.csv", A, J], ["E5_crossings", "E5_contrasts", "E5_stability", "E6_crossings", "E6_contrasts", "E6_interactions", "E6_update_ratios"]),
    ("phase3_e7.py", lambda R, A, J: [R / "E7_runs.csv", R / "E1_runs.csv", A, J], ["E7_crossings", "E7_contrasts", "E7_interactions"]),
    ("phase3_e8.py", lambda R, A, J: [R / "E8_runs.csv", R / "E7_runs.csv", R / "E1_runs.csv", A, J], ["E8A_contrasts", "E8B_contrasts"]),
    ("phase3_e9_e12.py", lambda R, A, J: [R, A, J], ["E9_contrasts", "E10_contrasts", "E11_contrasts", "E12_contrasts"]),
]


def recentre(name, res, work, jt):
    rd = lambda f: pd.read_csv(f, keep_default_na=False, na_values=[""])   # E3's condition "null" is a label, not a missing value
    p, j = rd(res / f"{name}.csv"), rd(work / f"{name}.csv")
    assert len(p) == len(j), name
    est = next(c for c in EST if c in j.columns)
    assert np.allclose(p[est].to_numpy(float), j[est].to_numpy(float), equal_nan=True), f"{name}: estimates differ from the pre-declared run"
    out = p.copy()
    for lo, hi in PAIRS:
        if lo in j.columns:
            out[f"pct_{lo}"], out[f"pct_{hi}"] = p[lo], p[hi]
            out[lo], out[hi] = j[est] + j[lo], j[est] + j[hi]
    if "excludes_zero" in out.columns:
        out["excludes_zero"] = (out.ci_low > 0) | (out.ci_high < 0)
    if "excludes_one" in out.columns:  # ratio rows: a jackknife-t interval on the ratio scale
        out["excludes_one"] = (out.ci_low > 1) | (out.ci_high < 1)
    for col, m in [("stable_pm05", 0.05), ("stable_pm02", 0.02)]:
        if col in out.columns:
            out[col] = (out.ci_low >= -m) & (out.ci_high <= m)
    if "within_equiv" in out.columns:
        out["within_equiv"] = (out.ci_low >= -0.05) & (out.ci_high <= 0.05)
    if "usable" in out.columns:
        out = out.rename(columns={"usable": "pct_usable"})
    out.insert(0, "interval", "jackknife-t")
    out.to_csv(jt / f"{name}.csv", index=False)


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        run_one(sys.argv[2], sys.argv[3:]); sys.exit(0)
    import shutil
    R, A, RES, JT = map(Path, sys.argv[1:5]); JT.mkdir(parents=True, exist_ok=True); WORK = JT / "_work"
    for script, args, outs in SCRIPTS:
        print("+", script, flush=True)
        shutil.rmtree(WORK, ignore_errors=True); WORK.mkdir()   # everything a script writes here holds un-centred offsets
        subprocess.run([sys.executable, __file__, "--one", script] + [str(x) for x in args(R, A, WORK)], check=True, stdout=subprocess.DEVNULL)
        for name in outs:
            recentre(name, RES, WORK, JT)
    shutil.rmtree(WORK)
    print("wrote", sorted(p.name for p in JT.glob("*.csv")))
