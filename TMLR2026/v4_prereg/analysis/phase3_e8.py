"""Phase 3 analysis of E8 (online moment-matched control; independent-seed replication of E7 P1), per
PREREGISTRATION_E8.md. Written before any E8 run. Same estimator as phase3_e7.py.

Usage: phase3_e8.py E8_RUNS_CSV E7_RUNS_CSV E1_RUNS_CSV ARTIFACT_DIR OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

E8CSV, E7CSV, E1CSV, ART, OUT = map(Path, sys.argv[1:6]); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ART / "AAMAS2027/analysis")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from boundary import firth  # noqa: E402
from isotonic import isotonic_crossing  # noqa: E402
from labels import check_supply_table  # noqa: E402

GRID = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]; MODES = ["plain", "popart"]
S_A, S_B = list(range(42, 52)), list(range(52, 62))
W_A = np.random.default_rng(20261015).multinomial(10, np.full(10, 0.1), size=2000)
W_POOL = np.random.default_rng(20261107).multinomial(20, np.full(20, 0.05), size=2000)
log = []
def say(s=""): print(s); log.append(s)
def ci(z, level=0.95):
    z = z[np.isfinite(z)]; a = 100 * (1 - level) / 2
    return (np.nan, np.nan, 0) if not len(z) else (*np.percentile(z, [a, 100 - a]), len(z))
def fmt(d, lo, hi): return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def fit(frame, seeds, W):
    k = frame.assign(w=frame.w.round(4)).pivot(index="seed", columns="w", values="collapsed").reindex(index=seeds, columns=GRID)
    assert not k.isna().any().any(), "missing cells"
    k = k.to_numpy(float); n = np.ones_like(k); w = np.array(GRID)
    p = firth(w, k.sum(0), n.sum(0)); b = firth(w, W @ k, W @ n)
    return dict(w50=p["wcrit"][0], in_grid=bool(p["in_grid"][0]), boot=b["wcrit"], cells=k.sum(0).astype(int), iso=isotonic_crossing(w, k.sum(0), n.sum(0)))


def con(F, h, l):
    d = F[h]["boot"] - F[l]["boot"]; lo, hi, _ = ci(d); lo99, hi99, _ = ci(d, 0.99)
    return dict(difference=F[h]["w50"] - F[l]["w50"], ci_low=lo, ci_high=hi, ci_low_99=lo99, ci_high_99=hi99,
                excludes_zero=bool(lo > 0 or hi < 0), iso=F[h]["iso"] - F[l]["iso"])


e8, e7, e1 = pd.read_csv(E8CSV), pd.read_csv(E7CSV), pd.read_csv(E1CSV)
for t in (e8, e7, e1): check_supply_table(t)
assert len(e8) == 480 and e8.complete.all() and (e8.actual == 300160).all() and (e8.n_eps == 10).all() and e8.eval_seeds_ok.all()
assert set(e8.design) == {"e8-online-moments-v4"} and not e8.static_affine.any() and not e8.duplicated(["critic_mode", "arm", "w", "seed"]).any()
pa_ = e8[e8.part == "A"]; pb_ = e8[e8.part == "B"]
assert len(pa_) == 160 and set(pa_.arm) == {"M_on"} and set(pa_.seed) == set(S_A) and (pa_.online_rollouts == 469).all()
assert len(pb_) == 320 and set(pb_.arm) == {"B", "S"} and set(pb_.seed) == set(S_B)
say("== E8: 480 runs complete, unique, 300,160 transitions; part A 160 (M_on, seeds 42-51), part B 320 (B/S, seeds 52-61)")
e1r = e1[(e1.optimizer_mode == "separate_no_clip") & (e1.reward_mode == "raw") & e1.arm.isin(["B", "A", "S"])].assign(critic_mode="plain")
D = pd.concat([e1r, e7, pa_], ignore_index=True)

# ---------------------------------------------------------------- part A
F = {(m, a): fit(D[(D.critic_mode == m) & (D.arm == a)], S_A, W_A) for m in MODES for a in ["B", "A", "S", "M", "M_on"]}
rows = []
for m in MODES:
    for name, h, l in [("A-M_on", "A", "M_on"), ("M_on-B", "M_on", "B"), ("A-S", "A", "S"), ("M_on-M", "M_on", "M"), ("A-M", "A", "M")]:
        rows.append(dict(mode=m, contrast=name, **con({k[1]: v for k, v in F.items() if k[0] == m}, h, l)))
CA = pd.DataFrame(rows); CA.to_csv(OUT / "E8A_contrasts.csv", index=False)
say("\n== E8 part A crossings")
for m in MODES:
    say("  " + m + ": " + "; ".join(f"{a} {F[(m, a)]['w50']:.3f}" for a in ["B", "A", "S", "M", "M_on"]))
say("\n== E8 part A contrasts (95% primary; 99% sensitivity)")
for _, r in CA.iterrows():
    say(f"  {r['mode']:6s} {r.contrast:7s} {fmt(r.difference, r.ci_low, r.ci_high)}  99% [{r.ci_low_99:+.3f}, {r.ci_high_99:+.3f}]{'  *' if r.excludes_zero else ''}")
z = (F[("popart", "A")]["boot"] - F[("popart", "M_on")]["boot"]) - (F[("plain", "A")]["boot"] - F[("plain", "M_on")]["boot"]); lo, hi, _ = ci(z)
dAM = (F[("popart", "A")]["w50"] - F[("popart", "M_on")]["w50"]) - (F[("plain", "A")]["w50"] - F[("plain", "M_on")]["w50"])
say(f"  d(A-M_on) popart - plain: {fmt(dAM, lo, hi)}")


def pooled(x, mean_col, rms_col):
    mean = x[mean_col].mean(); return mean, float(np.sqrt((x[rms_col] ** 2).mean() - mean ** 2))


match = []
for m in MODES:
    am, asd = pooled(D[(D.critic_mode == m) & (D.arm == "A")], "reward_transformed_mean", "reward_transformed_rms")
    x = pa_[pa_.critic_mode == m]; mm, msd = pooled(x, "M_mean", "M_rms")
    match.append(dict(mode=m, A_mean=am, A_sd=asd, Mon_mean=mm, Mon_sd=msd, rel_mean=abs(mm - am) / abs(am), rel_sd=abs(msd - asd) / asd,
                      track_mean_mae=x.track_mean_mae.mean(), track_sd_mae=x.track_sd_mae.mean(), a_min=x.online_a_min.min(), a_max=x.online_a_max.max()))
MT = pd.DataFrame(match); MT.to_csv(OUT / "E8A_matching.csv", index=False)
say("\n== E8 part A matching (pooled as in E7 P5)\n" + MT.round(3).to_string(index=False))

# ---------------------------------------------------------------- part B
FB = {(m, a): fit(pb_[(pb_.critic_mode == m) & (pb_.arm == a)], S_B, W_A) for m in MODES for a in ["B", "S"]}
Dp = pd.concat([e1r[e1r.arm.isin(["B", "S"])], e7[e7.arm.isin(["B", "S"])], pb_], ignore_index=True)
FP = {(m, a): fit(Dp[(Dp.critic_mode == m) & (Dp.arm == a)], S_A + S_B, W_POOL) for m in MODES for a in ["B", "S"]}
rows = []
for tag, FF in [("seeds 52-61 (primary)", FB), ("pooled 42-61 (secondary)", FP)]:
    for m in MODES:
        rows.append(dict(cohort=tag, quantity=f"{m} S-B", **con({k[1]: v for k, v in FF.items() if k[0] == m}, "S", "B")))
    z = (FF[("popart", "S")]["boot"] - FF[("popart", "B")]["boot"]) - (FF[("plain", "S")]["boot"] - FF[("plain", "B")]["boot"])
    lo, hi, _ = ci(z); lo99, hi99, _ = ci(z, 0.99)
    pt = (FF[("popart", "S")]["w50"] - FF[("popart", "B")]["w50"]) - (FF[("plain", "S")]["w50"] - FF[("plain", "B")]["w50"])
    rows.append(dict(cohort=tag, quantity="d(S-B) popart - plain", difference=pt, ci_low=lo, ci_high=hi, ci_low_99=lo99, ci_high_99=hi99,
                     excludes_zero=bool(lo > 0 or hi < 0), iso=np.nan))
CB = pd.DataFrame(rows); CB.to_csv(OUT / "E8B_contrasts.csv", index=False)
say("\n== E8 part B")
for _, r in CB.iterrows():
    say(f"  {r.cohort:26s} {r.quantity:22s} {fmt(r.difference, r.ci_low, r.ci_high)}  99% [{r.ci_low_99:+.3f}, {r.ci_high_99:+.3f}]{'  *' if r.excludes_zero else ''}")

# ---------------------------------------------------------------- predictions
g = lambda m, n: CA[(CA["mode"] == m) & (CA.contrast == n)].iloc[0]
say("\n== E8 pre-declared predictions")
ok1 = bool((MT.rel_mean <= 0.10).all() and (MT.rel_sd <= 0.10).all())
say(f"P1 matching (pooled mean and SD within 10% under both critics): {'MET' if ok1 else 'NOT MET'} "
    + "; ".join(f"{r['mode']}: mean {r.rel_mean:.0%}, SD {r.rel_sd:.0%}" for _, r in MT.iterrows()))
r2 = g("popart", "A-M_on"); say(f"P2 popart A-M_on > 0: {fmt(r2.difference, r2.ci_low, r2.ci_high)} -> {'MET' if r2.ci_low > 0 else 'NOT MET'}")
r3 = g("plain", "A-M_on")
v3 = ("positive, excludes 0: plain asymmetric contrast not explained by the first two moments" if r3.ci_low > 0 else
      ("includes 0: a moment effect under the raw-target critic remains possible" if r3.ci_high > 0 else "negative, excludes 0 (undeclared direction)"))
say(f"P3 plain A-M_on: {fmt(r3.difference, r3.ci_low, r3.ci_high)} -> {v3}")
say("P4 M_on-M: " + "; ".join(f"{m} {fmt(g(m, 'M_on-M').difference, g(m, 'M_on-M').ci_low, g(m, 'M_on-M').ci_high)}" for m in MODES))
pb1 = CB[(CB.cohort.str.startswith("seeds")) & (CB.quantity == "popart S-B")].iloc[0]; pb2 = CB[(CB.cohort.str.startswith("seeds")) & (CB.quantity.str.startswith("d("))].iloc[0]
if pb1.ci_low > 0: v5 = "NOT MET: popart S-B positive -> output normalization of targets is not the channel"
elif not pb1.excludes_zero and pb2.ci_high < 0: v5 = "MET: critic-target-scale attribution supported in an independent cohort"
else: v5 = "INCONCLUSIVE"
say(f"P5 part B primary: popart S-B {fmt(pb1.difference, pb1.ci_low, pb1.ci_high)}; d(S-B) {fmt(pb2.difference, pb2.ci_low, pb2.ci_high)} -> {v5}")
if not ok1: say("   (P2-P3 carry the matching qualification of P1)")
(OUT / "phase3_e8_report.txt").write_text("\n".join(log) + "\n")
