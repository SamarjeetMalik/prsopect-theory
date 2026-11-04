"""Phase 3 analysis of E5 (decoupled horizon cohort) and E6 (clipping isolation), per PREREGISTRATION_E5.md and
PREREGISTRATION_E6.md. Same estimator and bootstrap as phase3.py.

Usage: phase3_e5e6.py EXTRACTED_DIR E1_RUNS_CSV ARTIFACT_DIR OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXT, E1CSV, ART, OUT = map(Path, sys.argv[1:5]); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ART / "AAMAS2027/analysis")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from boundary import firth  # noqa: E402
from isotonic import isotonic_crossing  # noqa: E402

SEEDS = list(range(42, 52))
WEIGHTS = np.random.default_rng(20261015).multinomial(10, np.full(10, 0.1), size=2000)
EQUIV = 0.05
log = []
def say(s=""): print(s); log.append(s)
def ci(z):
    ok = np.isfinite(z)
    return (np.nan, np.nan, 0) if not ok.any() else (*np.percentile(z[ok], [2.5, 97.5]), int(ok.sum()))
def fmt(d, lo, hi): return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def fit(frame, grid):
    frame = frame.assign(w=frame.w.round(4)); grid = [round(x, 4) for x in grid]
    k = frame.pivot(index="seed", columns="w", values="collapsed").reindex(index=SEEDS, columns=grid)
    assert not k.isna().any().any(), "missing cells"
    k = k.to_numpy(float); n = np.ones_like(k); w = np.array(grid)
    p = firth(w, k.sum(0), n.sum(0)); b = firth(w, WEIGHTS @ k, WEIGHTS @ n)
    return dict(w50=p["wcrit"][0], in_grid=bool(p["in_grid"][0]), status=p["status"][0], boot=b["wcrit"], cells=k.sum(0).astype(int),
                iso=isotonic_crossing(w, k.sum(0), n.sum(0)))


def contrast(a, b):
    d = a["boot"] - b["boot"]; lo, hi, u = ci(d)
    return dict(difference=a["w50"] - b["w50"], ci_low=lo, ci_high=hi, usable=u, excludes_zero=bool(lo > 0 or hi < 0),
                within_equiv=bool(lo >= -EQUIV and hi <= EQUIV), identified=a["in_grid"] and b["in_grid"])


# ================================================================== E5
e5 = pd.read_csv(EXT / "E5_checkpoints.csv")
G5 = [0.2, 0.4, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 1.0]; CK = sorted(e5.checkpoint.unique()); ARMS5 = ["B", "A", "S"]
assert e5.present.all() and len(e5) == 270 * 24 and (e5.n_eps == 10).all() and e5.complete.all()
assert set(e5.design) == {"e5-decoupled-horizon-v4"} and set(e5.optimizer_mode) == {"separate_no_clip"} and set(e5.reward_mode) == {"vecnormalize"}
assert not e5.duplicated(["arm", "w", "seed", "checkpoint"]).any()
say("== E5: 270 trajectories x 24 checkpoints present, unique, 10 evaluation episodes each")
f5 = {(a, ck): fit(e5[(e5.arm == a) & (e5.checkpoint == ck)], G5) for a in ARMS5 for ck in CK}
rows = []
for a in ARMS5:
    for ck in CK:
        x = f5[(a, ck)]; lo, hi, u = ci(x["boot"])
        rows.append(dict(arm=a, checkpoint=ck, w50=x["w50"], ci_low=lo, ci_high=hi, usable=u, in_grid=x["in_grid"], isotonic=x["iso"],
                         cells=" ".join(f"{w:g}:{c}/10" for w, c in zip(G5, x["cells"]))))
pd.DataFrame(rows).to_csv(OUT / "E5_crossings.csv", index=False)
PRIMARY = [300000, 900000, 1800000, 3600000]
rows = []
for ck in PRIMARY:
    for name, hi_, lo_ in [("A-S", "A", "S"), ("A-B", "A", "B"), ("S-B", "S", "B")]:
        rows.append(dict(checkpoint=ck, contrast=name, **contrast(f5[(hi_, ck)], f5[(lo_, ck)])))
c5 = pd.DataFrame(rows); c5.to_csv(OUT / "E5_contrasts.csv", index=False)
say("\n== E5 crossings at primary budgets")
for a in ARMS5:
    say("  " + a + ": " + "; ".join(f"{ck // 1000}k {f5[(a, ck)]['w50']:.3f}" for ck in PRIMARY))
say("\n== E5 contrasts")
for _, r in c5.iterrows():
    say(f"  {r.checkpoint // 1000:>5}k {r.contrast}: {fmt(r.difference, r.ci_low, r.ci_high)}{'  *' if r.excludes_zero else ''}{'' if r.identified else '  (not identified)'}")
st = []
for a in ARMS5:
    z = f5[(a, 3600000)]["boot"] - f5[(a, 1800000)]["boot"]; lo, hi, u = ci(z)
    st.append(dict(arm=a, change=f5[(a, 3600000)]["w50"] - f5[(a, 1800000)]["w50"], ci_low=lo, ci_high=hi, stable_pm05=bool(lo >= -.05 and hi <= .05)))
st = pd.DataFrame(st); st.to_csv(OUT / "E5_stability.csv", index=False)
lab = e5.pivot_table(index=["arm", "w", "seed"], columns="checkpoint", values="collapsed")
rec = []
for a in ARMS5:
    L = lab.loc[a]; c300 = L[300000] == 1; c36 = L[3600000] == 1
    rec.append(dict(arm=a, collapsed_300k=int(c300.sum()), recovered_3p6M=int((c300 & ~c36).sum()), noncollapsed_300k=int((~c300).sum()), relapsed_3p6M=int((~c300 & c36).sum())))
rec = pd.DataFrame(rec); rec.to_csv(OUT / "E5_recovery.csv", index=False)
say("\n== E5 stability (3.6M - 1.8M)\n" + st.round(3).to_string(index=False)); say("\n== E5 recovery\n" + rec.to_string(index=False))
g = lambda ck, n: c5[(c5.checkpoint == ck) & (c5.contrast == n)].iloc[0]
p1 = g(300000, "A-S").ci_low > 0 and not g(300000, "S-B").excludes_zero
p2 = all(g(ck, "A-S").ci_low > 0 for ck in PRIMARY)
p2fade = g(300000, "A-S").ci_low > 0 and not g(3600000, "A-S").ci_low > 0
p3 = all(not g(ck, "S-B").excludes_zero for ck in PRIMARY)
say("\n== E5 pre-declared predictions")
say(f"P1 300k: A-S > 0 and S-B includes 0 -> {'MET' if p1 else 'NOT MET'}")
say(f"P2 persistence: A-S excludes 0 at every primary budget -> {'MET' if p2 else ('FADES (300k yes, 3.6M no)' if p2fade else 'NOT MET')}")
say(f"P3 symmetric: S-B includes 0 at every primary budget -> {'MET' if p3 else 'NOT MET: ' + ', '.join(f'{ck // 1000}k' for ck in PRIMARY if g(ck, 'S-B').excludes_zero)}")

# ================================================================== E6
e6 = pd.read_csv(EXT / "E6_runs.csv"); G6 = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]
MODES = ["global_clip", "groupwise_clip", "separate_no_clip"]; ARMS6 = ["B", "A", "S"]
assert len(e6) == 720 and e6.complete.all() and (e6.actual == 300160).all() and (e6.n_eps == 10).all() and e6.eval_seeds_ok.all()
assert set(e6.design) == {"e6-clip-isolation-v4"} and not e6.duplicated(["optimizer_mode", "arm", "w", "seed"]).any()
say("\n== E6: 720 runs complete, unique, 300,160 transitions, declared evaluation seeds")
f6 = {(m, a): fit(e6[(e6.optimizer_mode == m) & (e6.arm == a)], G6) for m in MODES for a in ARMS6}
rows = []
for m in MODES:
    for a in ARMS6:
        x = f6[(m, a)]; lo, hi, u = ci(x["boot"])
        rows.append(dict(mode=m, arm=a, w50=x["w50"], ci_low=lo, ci_high=hi, usable=u, in_grid=x["in_grid"], isotonic=x["iso"],
                         cells=" ".join(f"{w:g}:{c}/10" for w, c in zip(G6, x["cells"]))))
pd.DataFrame(rows).to_csv(OUT / "E6_crossings.csv", index=False)
rows = []
for m in MODES:
    for name, hi_, lo_ in [("S-B", "S", "B"), ("A-B", "A", "B"), ("A-S", "A", "S")]:
        rows.append(dict(mode=m, contrast=name, **contrast(f6[(m, hi_)], f6[(m, lo_)])))
c6 = pd.DataFrame(rows); c6.to_csv(OUT / "E6_contrasts.csv", index=False)
rows = []
for m in MODES[1:]:
    for name, hi_, lo_ in [("S-B", "S", "B"), ("A-S", "A", "S")]:
        z = (f6[(m, hi_)]["boot"] - f6[(m, lo_)]["boot"]) - (f6[("global_clip", hi_)]["boot"] - f6[("global_clip", lo_)]["boot"]); lo, hi, u = ci(z)
        pt = (f6[(m, hi_)]["w50"] - f6[(m, lo_)]["w50"]) - (f6[("global_clip", hi_)]["w50"] - f6[("global_clip", lo_)]["w50"])
        rows.append(dict(mode=m, contrast=name, change_vs_global=pt, ci_low=lo, ci_high=hi, usable=u, excludes_zero=bool(lo > 0 or hi < 0)))
i6 = pd.DataFrame(rows); i6.to_csv(OUT / "E6_interactions.csv", index=False)
say("\n== E6 crossings")
for m in MODES:
    say("  " + m + ": " + "; ".join(f"{a} {f6[(m, a)]['w50']:.3f}" for a in ARMS6))
say("\n== E6 contrasts")
for _, r in c6.iterrows():
    say(f"  {r['mode']:17s} {r.contrast}: {fmt(r.difference, r.ci_low, r.ci_high)}{'  *' if r.excludes_zero else ''}")
say("\n== E6 change relative to global_clip")
for _, r in i6.iterrows():
    say(f"  {r['mode']:17s} d({r.contrast}): {fmt(r.change_vs_global, r.ci_low, r.ci_high)}{'  *' if r.excludes_zero else ''}")

# parameter-update diagnostics: seed-paired ratio of total actor update norm, S/B and A/B, per mode
def ratio(m, num, den, col="pu_actor_update_norm"):
    piv = lambda a: e6[(e6.optimizer_mode == m) & (e6.arm == a)].assign(w=lambda d: d.w.round(4)).pivot(index="seed", columns="w", values=col).reindex(index=SEEDS).to_numpy(float).sum(1)
    x, y = piv(num), piv(den); point = x.sum() / y.sum(); boot = (WEIGHTS @ x) / (WEIGHTS @ y)
    lo, hi = np.percentile(boot, [2.5, 97.5]); return point, lo, hi
rows = []
for m in MODES:
    for num in ["S", "A"]:
        for col in ["pu_actor_update_norm", "pu_critic_update_norm", "pu_applied_actor_grad_norm"]:
            pt, lo, hi = ratio(m, num, "B", col)
            rows.append(dict(mode=m, ratio=f"{num}/B", quantity=col, point=pt, ci_low=lo, ci_high=hi, excludes_one=bool(lo > 1 or hi < 1)))
ru = pd.DataFrame(rows); ru.to_csv(OUT / "E6_update_ratios.csv", index=False)
ud = e6.groupby(["optimizer_mode", "arm"])[[c for c in e6.columns if c.startswith("pu_") or c in ("mb_clip_freq", "explained_variance_last")]].mean()
ud = ud.reindex([(m, a) for m in MODES for a in ARMS6]); ud.to_csv(OUT / "E6_update_diagnostics.csv")
say("\n== E6 seed-paired update ratios (actor parameter-update norm unless stated)")
for _, r in ru.iterrows():
    say(f"  {r['mode']:17s} {r.ratio} {r.quantity:28s} {r.point:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]{'  (excludes 1)' if r.excludes_one else ''}")
say("\n== E6 update diagnostics (means)\n" + ud[["pu_actor_update_norm", "pu_critic_update_norm", "pu_applied_actor_grad_norm", "pu_actor_update_per_grad", "explained_variance_last"]].round(5).to_string())

# reproduction of E1 labels for the two re-run modes
e1 = pd.read_csv(E1CSV); e1 = e1[(e1.reward_mode == "raw") & e1.optimizer_mode.isin(["global_clip", "separate_no_clip"]) & e1.arm.isin(ARMS6)]
mrg = e6[e6.optimizer_mode.isin(["global_clip", "separate_no_clip"])].assign(w=lambda d: d.w.round(4)).merge(
    e1.assign(w=e1.w.round(4)), on=["optimizer_mode", "arm", "w", "seed"], suffixes=("", "_e1"))
agree = int((mrg.collapsed == mrg.collapsed_e1).sum()); exact = int(np.isclose(mrg.profit, mrg.profit_e1, atol=1e-12, rtol=0).sum())
say(f"\n== E6 reproduction of E1 (global_clip and separate_no_clip, raw): labels {agree}/{len(mrg)}; bit-identical profit {exact}/{len(mrg)}")
pd.DataFrame([dict(comparison="E6 vs E1 (two re-run modes)", agree=agree, bit_identical_profit=exact, total=len(mrg))]).to_csv(OUT / "E6_reproduction.csv", index=False)

gc = lambda m, n: c6[(c6["mode"] == m) & (c6.contrast == n)].iloc[0]
gi = lambda m, n: i6[(i6["mode"] == m) & (i6.contrast == n)].iloc[0]
gr = lambda m: ru[(ru["mode"] == m) & (ru.ratio == "S/B") & (ru.quantity == "pu_actor_update_norm")].iloc[0]
p1 = gr("global_clip").ci_high < 1 and not gr("groupwise_clip").excludes_one and not gr("separate_no_clip").excludes_one
p2_shared = (not gc("groupwise_clip", "S-B").excludes_zero) and gi("groupwise_clip", "S-B").ci_high < 0
say("\n== E6 pre-declared predictions")
say(f"P1 actor update S/B < 1 under global clip, ~1 otherwise -> {'MET' if p1 else 'NOT MET'} "
    f"(global {gr('global_clip').point:.3f}, groupwise {gr('groupwise_clip').point:.3f}, separate {gr('separate_no_clip').point:.3f})")
say(f"P2 clipping isolation: {'artifact attributed to the shared clip' if p2_shared else 'a channel other than the shared clip remains'} "
    f"(groupwise S-B {fmt(gc('groupwise_clip', 'S-B').difference, gc('groupwise_clip', 'S-B').ci_low, gc('groupwise_clip', 'S-B').ci_high)}; "
    f"d(S-B) vs global {fmt(gi('groupwise_clip', 'S-B').change_vs_global, gi('groupwise_clip', 'S-B').ci_low, gi('groupwise_clip', 'S-B').ci_high)})")
say(f"P3 A-S under groupwise clip > 0 -> {'MET' if gc('groupwise_clip', 'A-S').ci_low > 0 else 'NOT MET'} "
    f"({fmt(gc('groupwise_clip', 'A-S').difference, gc('groupwise_clip', 'A-S').ci_low, gc('groupwise_clip', 'A-S').ci_high)})")
say(f"P4 reproduction: {agree}/{len(mrg)} labels equal E1")
(OUT / "phase3_e5e6_report.txt").write_text("\n".join(log) + "\n")
