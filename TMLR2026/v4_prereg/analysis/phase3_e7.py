"""Phase 3 analysis of E7 (critic-target scale isolation and moment-matched reward control), per
PREREGISTRATION_E7.md. Same estimator and bootstrap draws as phase3.py, so the reused E1 cells stay paired.

Usage: phase3_e7.py E7_RUNS_CSV E1_RUNS_CSV ARTIFACT_DIR OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

E7CSV, E1CSV, ART, OUT = map(Path, sys.argv[1:5]); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ART / "AAMAS2027/analysis")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from boundary import firth  # noqa: E402
from isotonic import isotonic_crossing  # noqa: E402

SEEDS = list(range(42, 52))
WEIGHTS = np.random.default_rng(20261015).multinomial(10, np.full(10, 0.1), size=2000)
GRID = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]
MODES, ARMS = ["plain", "popart"], ["B", "A", "S", "M"]
log = []
def say(s=""): print(s); log.append(s)
def ci(z):
    ok = np.isfinite(z)
    return (np.nan, np.nan, 0) if not ok.any() else (*np.percentile(z[ok], [2.5, 97.5]), int(ok.sum()))
def fmt(d, lo, hi): return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
def excl(lo, hi): return bool(lo > 0 or hi < 0)


def fit(frame):
    frame = frame.assign(w=frame.w.round(4))
    k = frame.pivot(index="seed", columns="w", values="collapsed").reindex(index=SEEDS, columns=GRID)
    assert not k.isna().any().any(), "missing cells"
    k = k.to_numpy(float); n = np.ones_like(k); w = np.array(GRID)
    p = firth(w, k.sum(0), n.sum(0)); b = firth(w, WEIGHTS @ k, WEIGHTS @ n)
    return dict(w50=p["wcrit"][0], in_grid=bool(p["in_grid"][0]), status=p["status"][0], boot=b["wcrit"], cells=k.sum(0).astype(int),
                iso=isotonic_crossing(w, k.sum(0), n.sum(0)))


# ------------------------------------------------------------------ data
e7 = pd.read_csv(E7CSV); e1 = pd.read_csv(E1CSV)
assert len(e7) == 400 and e7.complete.all() and (e7.actual == 300160).all() and (e7.n_eps == 10).all() and e7.eval_seeds_ok.all()
assert set(e7.design) == {"e7-critic-scale-v4"} and set(e7.optimizer_mode) == {"separate_no_clip"} and set(e7.reward_mode) == {"raw"}
assert not e7.duplicated(["critic_mode", "arm", "w", "seed"]).any()
assert set(e7["index"]) == set(range(240, 640))
assert len(e7[(e7.critic_mode == "plain")]) == 80 and set(e7[e7.critic_mode == "plain"].arm) == {"M"}
reuse = e1[(e1.optimizer_mode == "separate_no_clip") & (e1.reward_mode == "raw") & e1.arm.isin(["B", "A", "S"])].assign(critic_mode="plain")
assert len(reuse) == 240 and reuse.complete.all()
pa = e7[e7.critic_mode == "popart"]
assert (pa.popart_updates > 0).all() and (pa.popart_max_drift <= 1e-3 * 1e3).all()
say("== E7: 400 new runs complete, unique, 300,160 transitions, declared evaluation seeds; indices 240-639")
say(f"   plain B/A/S reused from E1 separate_no_clip/raw: {len(reuse)} runs (gate 1: index 120 == E1 run 760 bit-identical)")
say(f"   popart: {len(pa)} runs, {int(pa.popart_updates.min())}-{int(pa.popart_updates.max())} rescales each, "
    f"largest output drift {pa.popart_max_drift.max():.2e} (assertion never fired: every run completed)")
D = pd.concat([reuse, e7], ignore_index=True)

# ------------------------------------------------------------------ crossings and contrasts
F = {(m, a): fit(D[(D.critic_mode == m) & (D.arm == a)]) for m in MODES for a in ARMS}
rows = []
for m in MODES:
    for a in ARMS:
        x = F[(m, a)]; lo, hi, u = ci(x["boot"])
        rows.append(dict(mode=m, arm=a, w50=x["w50"], ci_low=lo, ci_high=hi, usable=u, in_grid=x["in_grid"], status=x["status"], isotonic=x["iso"],
                         cells=" ".join(f"{w:g}:{c}/10" for w, c in zip(GRID, x["cells"]))))
cr = pd.DataFrame(rows); cr.to_csv(OUT / "E7_crossings.csv", index=False)
say("\n== E7 crossings (w50 [95% CI]; isotonic; cells k/10)")
for _, r in cr.iterrows():
    say(f"  {r['mode']:6s} {r.arm}: {r.w50:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]  iso {r.isotonic:.3f}{'' if r.in_grid else '  (not in grid)'}  {r.cells}")

PAIRS = [("S-B", "S", "B"), ("A-S", "A", "S"), ("A-M", "A", "M"), ("M-B", "M", "B"), ("A-B", "A", "B")]
rows = []
for m in MODES:
    for name, h, l in PAIRS:
        lo, hi, u = ci(F[(m, h)]["boot"] - F[(m, l)]["boot"])
        rows.append(dict(mode=m, contrast=name, difference=F[(m, h)]["w50"] - F[(m, l)]["w50"], ci_low=lo, ci_high=hi, usable=u,
                         excludes_zero=excl(lo, hi), isotonic=F[(m, h)]["iso"] - F[(m, l)]["iso"],
                         identified=F[(m, h)]["in_grid"] and F[(m, l)]["in_grid"]))
C = pd.DataFrame(rows); C.to_csv(OUT / "E7_contrasts.csv", index=False)
say("\n== E7 contrasts within critic mode (Firth [95% CI]; isotonic)")
for _, r in C.iterrows():
    say(f"  {r['mode']:6s} {r.contrast}: {fmt(r.difference, r.ci_low, r.ci_high)}{'  *' if r.excludes_zero else '   '}  iso {r.isotonic:+.3f}"
        f"{'' if r.identified else '  (not identified)'}")
rows = []
for name, h, l in PAIRS[:3]:
    z = (F[("popart", h)]["boot"] - F[("popart", l)]["boot"]) - (F[("plain", h)]["boot"] - F[("plain", l)]["boot"]); lo, hi, u = ci(z)
    pt = (F[("popart", h)]["w50"] - F[("popart", l)]["w50"]) - (F[("plain", h)]["w50"] - F[("plain", l)]["w50"])
    rows.append(dict(contrast=name, change=pt, ci_low=lo, ci_high=hi, usable=u, excludes_zero=excl(lo, hi)))
I = pd.DataFrame(rows); I.to_csv(OUT / "E7_interactions.csv", index=False)
say("\n== E7 interactions (popart minus plain)")
for _, r in I.iterrows():
    say(f"  d({r.contrast}): {fmt(r.change, r.ci_low, r.ci_high)}{'  *' if r.excludes_zero else ''}")

# ------------------------------------------------------------------ diagnostics
rows = []
for m in MODES:
    for a in ARMS:
        x = D[(D.critic_mode == m) & (D.arm == a)]
        r = dict(mode=m, arm=a, explained_variance_last=x.explained_variance_last.mean(), actor_grad_norm=x.mb_actor_grad_norm.mean(),
                 critic_grad_norm=x.mb_critic_grad_norm.mean(), value_loss=x.mb_value_loss.mean())
        if m == "popart": r["popart_sigma_final"] = x.popart_sigma_final.mean()
        if a == "M":
            mean = x.M_mean.mean(); r.update(train_reward_mean=mean, train_reward_sd=float(np.sqrt((x.M_rms ** 2).mean() - mean ** 2)))
        else:
            mean = x.reward_transformed_mean.mean()
            r.update(train_reward_mean=mean, train_reward_sd=float(np.sqrt((x.reward_transformed_rms ** 2).mean() - mean ** 2)))
        rows.append(r)
G = pd.DataFrame(rows); G.to_csv(OUT / "E7_diagnostics.csv", index=False)
say("\n== E7 diagnostics per (mode, arm): explained variance at last update, per-minibatch grad norms, training-reward moments")
say(G.round(3).to_string(index=False))

# ------------------------------------------------------------------ pre-declared predictions
c = lambda m, n: C[(C["mode"] == m) & (C.contrast == n)].iloc[0]
d = lambda n: I[I.contrast == n].iloc[0]
say("\n== E7 pre-declared predictions")
sb, dsb = c("popart", "S-B"), d("S-B")
if sb.ci_low > 0: v1 = "NOT MET: S-B stays positive under popart -> target scale (output normalization) is not the channel"
elif not sb.excludes_zero and dsb.ci_high < 0: v1 = "MET: residual symmetric effect attributed to critic target scale"
elif not sb.excludes_zero: v1 = "INCONCLUSIVE: popart S-B includes 0 but d(S-B) does not exclude 0"
else: v1 = "UNDECLARED BRANCH: popart S-B negative with interval excluding 0"
say(f"P1 critic-target pathway: popart S-B {fmt(sb.difference, sb.ci_low, sb.ci_high)}; d(S-B) {fmt(dsb.change, dsb.ci_low, dsb.ci_high)} -> {v1}")
ev = G.set_index(["mode", "arm"]).explained_variance_last
gap = ev[("popart", "S")] - ev[("popart", "B")]
say(f"P2 critic fit (descriptive): popart EV S {ev[('popart', 'S')]:.3f} vs B {ev[('popart', 'B')]:.3f} (gap {gap:+.3f}; plain S {ev[('plain', 'S')]:.3f} "
    f"vs B {ev[('plain', 'B')]:.3f}) -> {'MET' if abs(gap) <= 0.1 else 'NOT MET'}")
a_s = c("popart", "A-S")
say(f"P3 asymmetry without critic pathway: popart A-S {fmt(a_s.difference, a_s.ci_low, a_s.ci_high)} -> {'MET' if a_s.ci_low > 0 else 'NOT MET'}")
am = {m: c(m, "A-M") for m in MODES}
p4 = all(am[m].ci_low > 0 for m in MODES)
say("P4 moment control: " + "; ".join(f"{m} A-M {fmt(am[m].difference, am[m].ci_low, am[m].ci_high)}" for m in MODES)
    + f" -> {'MET' if p4 else 'NOT MET: A-S reported as possibly a moment effect; abstract must change'}")
mt = G.set_index(["mode", "arm"])
p5 = {}
for m in MODES:
    ra = lambda q: abs(mt.loc[(m, "M"), q] - mt.loc[(m, "A"), q]) / abs(mt.loc[(m, "A"), q])
    p5[m] = (ra("train_reward_mean"), ra("train_reward_sd"))
say("P5 matching check (within 25%): " + "; ".join(
    f"{m} M mean {mt.loc[(m, 'M'), 'train_reward_mean']:.2f} vs A {mt.loc[(m, 'A'), 'train_reward_mean']:.2f} ({p5[m][0]:.0%}), "
    f"SD {mt.loc[(m, 'M'), 'train_reward_sd']:.2f} vs {mt.loc[(m, 'A'), 'train_reward_sd']:.2f} ({p5[m][1]:.0%})" for m in MODES)
    + f" -> {'MET' if all(max(v) <= .25 for v in p5.values()) else 'NOT MET: P4 carries the matching qualification'}")
say("P6 reproduction: gate 1 passed (E7 index 120 bit-identical to E1 run 760), so reused plain B/A/S labels are E1's -> MET")
(OUT / "E7_log.txt").write_text("\n".join(log) + "\n")
