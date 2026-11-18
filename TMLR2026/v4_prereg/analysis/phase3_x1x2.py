"""Phase 3 analysis of X1 and X2, per PREREGISTRATION_X1_X2.md. Written before any X1 or X2 run.

Primary inference: seed-level jackknife-t (t with n-1 df) on pooled Firth crossings, chosen by calibration_v5.py.
Secondary: seed-block percentile bootstrap (2,000 multinomial draws, seed 20261119), and the pooled analysis with the
earlier seeds 42-51 (E6 for X1, E11 for X2).

Usage: phase3_x1x2.py RUNS_DIR ARTIFACT_DIR OUT_DIR
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RUNS, ART, OUT = map(Path, sys.argv[1:4]); OUT.mkdir(parents=True, exist_ok=True)
for d in [str(ART / "AAMAS2027/analysis"), os.environ.get("BOUNDARY_DIR", "")]:
    if d and (Path(d) / "boundary.py").exists():
        sys.path.insert(0, d)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from boundary import firth  # noqa: E402
from labels import check_supply_table  # noqa: E402

GRID = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]
FRESH = list(range(82, 102)); OLD = list(range(42, 52))
log = []
def say(s=""): print(s); log.append(s)


def table(frame, seeds):
    k = frame.assign(w=frame.w.round(4)).pivot(index="seed", columns="w", values="collapsed").reindex(index=seeds, columns=GRID)
    assert not k.isna().any().any(), "missing cells"
    return k.to_numpy(float)


def est(frame, seeds, W):
    """Pooled crossing, leave-one-seed-out crossings, bootstrap crossings."""
    k = table(frame, seeds); one = np.ones_like(k); w = np.array(GRID); n = len(seeds)
    J = 1 - np.eye(n)
    return (firth(w, k.sum(0), one.sum(0))["wcrit"][0], firth(w, J @ k, J @ one)["wcrit"], firth(w, W @ k, W @ one)["wcrit"])


def jt(d, j, level):
    j = j[np.isfinite(j)]; n = len(j); se = np.sqrt((n - 1) / n * ((j - j.mean()) ** 2).sum()); t = stats.t.ppf(1 - (1 - level) / 2, n - 1)
    return d - t * se, d + t * se


def pc(z, level):
    z = z[np.isfinite(z)]; a = 100 * (1 - level) / 2; return np.percentile(z, [a, 100 - a])


rows = []
def report(label, d, j, z, cohort):
    l95, h95 = jt(d, j, .95); l99, h99 = jt(d, j, .99); p95 = pc(z, .95)
    rows.append(dict(cohort=cohort, quantity=label, difference=d, jt_low_95=l95, jt_high_95=h95, jt_low_99=l99, jt_high_99=h99,
                     pct_low_95=p95[0], pct_high_95=p95[1], jt_excludes_zero_95=bool(l95 > 0 or h95 < 0)))
    say(f"  {cohort:22s} {label:40s} {d:+.3f}  jt95 [{l95:+.3f}, {h95:+.3f}]  jt99 [{l99:+.3f}, {h99:+.3f}]  pct95 [{p95[0]:+.3f}, {p95[1]:+.3f}]"
        f"{'  *' if (l95 > 0 or h95 < 0) else ''}")
    return l95, h95


def sb(frame, seeds, W):
    s, b = frame[frame.arm == "S"], frame[frame.arm == "B"]
    es, js, zs = est(s, seeds, W); eb, jb, zb = est(b, seeds, W)
    return es - eb, js - jb, zs - zb


x1 = pd.read_csv(RUNS / "X1_runs.csv"); x2 = pd.read_csv(RUNS / "X2_runs.csv")
e6 = pd.read_csv(RUNS / "E6_runs.csv"); e11 = pd.read_csv(RUNS / "E11_runs.csv")
for t in (x1, e6):
    check_supply_table(t)
for t, n in [(x1, 960), (x2, 640)]:
    assert len(t) == n and t.complete.all() and (t.actual == 300160).all() and (t.n_eps == 10).all() and t.eval_seeds_ok.all(), n
say("== X1/X2: 960 + 640 runs complete, 300,160 transitions, declared evaluation seeds")
W20 = np.random.default_rng(20261119).multinomial(20, np.full(20, 1 / 20), size=2000)
W30 = np.random.default_rng(20261120).multinomial(30, np.full(30, 1 / 30), size=2000)
res = {}

say("\n== X1 (SB3; fresh seeds 82-101 primary; with E6 seeds 42-51 secondary)")
x1m = x1.rename(columns={"mode": "m"}); e6m = e6[e6.arm.isin(["B", "S"])].assign(m=e6.optimizer_mode)
for cohort, data, seeds, W in [("fresh 82-101", x1m, FRESH, W20), ("pooled 42-51 + 82-101", pd.concat([e6m, x1m]), OLD + FRESH, W30)]:
    r = {m: sb(data[data.m == m], seeds, W) for m in ["global_clip", "groupwise_clip", "separate_no_clip"]}
    for m in r:
        report(f"S-B {m}", *r[m], cohort)
    ch = tuple(a - b for a, b in zip(r["groupwise_clip"], r["global_clip"]))
    res[("X1", cohort)] = (r, report("change S-B, groupwise minus global", *ch, cohort))

say("\n== X2 (CleanRL-style; fresh seeds 82-101 primary; with E11 seeds 42-51 secondary)")
x2m = x2.assign(m=x2.condition_name); e11m = e11[e11.arm.isin(["B", "S"]) & e11.condition_name.isin(["cleanrl_raw", "cleanrl_group"])].assign(m=e11.condition_name)
for cohort, data, seeds, W in [("fresh 82-101", x2m, FRESH, W20), ("pooled 42-51 + 82-101", pd.concat([e11m, x2m]), OLD + FRESH, W30)]:
    r = {m: sb(data[data.m == m], seeds, W) for m in ["cleanrl_raw", "cleanrl_group"]}
    for m in r:
        report(f"S-B {m}", *r[m], cohort)
    ch = tuple(a - b for a, b in zip(r["cleanrl_group"], r["cleanrl_raw"]))
    res[("X2", cohort)] = (r, report("change S-B, group minus raw", *ch, cohort))
pd.DataFrame(rows).to_csv(OUT / "X1X2_contrasts.csv", index=False)

# ---------------------------------------------------------------- X1 optimizer diagnostics (seed-paired S/B ratios)
say("\n== X1 optimizer state, seed-paired S/B ratios over the 160 (seed, w) pairs per mode (geometric mean, jackknife-t over seeds)")
Q = ["pu_applied_actor_grad_norm", "pu_actor_update_norm", "pu_actor_adam_m_norm", "pu_actor_adam_sqrt_v_norm", "pu_actor_adam_direction_norm", "kappa_mean"]
drows = []
for m in ["global_clip", "groupwise_clip", "separate_no_clip"]:
    d = x1[x1["mode"] == m]
    for q in Q:
        if q not in d or d[q].isna().all():
            continue
        p = d.pivot_table(index=["seed", "w"], columns="arm", values=q); lr = np.log(p.S / p.B).rename("lr").reset_index()
        per_seed = lr.groupby("seed").lr.mean().reindex(FRESH).to_numpy(); g = per_seed.mean(); n = len(per_seed)
        jk = np.array([np.delete(per_seed, i).mean() for i in range(n)]); lo, hi = jt(g, jk, .95)
        drows.append(dict(mode=m, quantity=q, ratio=float(np.exp(g)), low_95=float(np.exp(lo)), high_95=float(np.exp(hi))))
        say(f"  {m:17s} {q:32s} S/B {np.exp(g):.3f} [{np.exp(lo):.3f}, {np.exp(hi):.3f}]")
    k = d.groupby("arm")[["kappa_mean", "kappa_median", "kappa_p10", "kappa_p90", "kappa_cv"]].median() if "kappa_mean" in d else None
    if k is not None and m == "global_clip":
        say(f"  global_clip kappa by arm (median over runs):\n{k.round(6).to_string()}")
pd.DataFrame(drows).to_csv(OUT / "X1_optimizer_state.csv", index=False)

# ---------------------------------------------------------------- pre-declared predictions
g = lambda c, q: next(r for r in rows if r["cohort"] == c and r["quantity"] == q)
F = "fresh 82-101"
def ex(r, sign): return (r["jt_low_95"] > 0) if sign > 0 else (r["jt_high_95"] < 0)
say("\n== Pre-declared predictions (primary: fresh seeds, jackknife-t 95%)")
r = g(F, "S-B global_clip"); say(f"P1 X1 global S-B > 0: {r['difference']:+.3f} -> {'MET' if ex(r, 1) else 'NOT MET'}")
r = g(F, "change S-B, groupwise minus global"); say(f"P2 X1 change groupwise - global < 0: {r['difference']:+.3f} -> {'MET' if ex(r, -1) else 'NOT MET'}")
r = g(F, "S-B separate_no_clip"); say(f"P3 X1 no-clip S-B > 0: {r['difference']:+.3f} -> {'MET' if ex(r, 1) else 'NOT MET'}")
dd = {(x["mode"], x["quantity"]): x for x in drows}
a = dd.get(("global_clip", "pu_actor_adam_direction_norm")); b = dd.get(("groupwise_clip", "pu_actor_adam_direction_norm"))
if a and b:
    say(f"P4 X1 Adam direction S/B below 1 under the global clip and above it under groupwise: {a['ratio']:.3f} [{a['low_95']:.3f}, {a['high_95']:.3f}], "
        f"{b['ratio']:.3f} [{b['low_95']:.3f}, {b['high_95']:.3f}] -> {'MET' if a['high_95'] < 1 and b['low_95'] > a['high_95'] else 'NOT MET'}")
r = g(F, "S-B cleanrl_raw"); say(f"P5 X2 raw S-B > 0: {r['difference']:+.3f} -> {'MET' if ex(r, 1) else 'NOT MET'}")
r = g(F, "change S-B, group minus raw"); say(f"P6 X2 change group - raw < 0: {r['difference']:+.3f} -> {'MET' if ex(r, -1) else 'NOT MET'}")
(OUT / "phase3_x1x2_report.txt").write_text("\n".join(log) + "\n")
