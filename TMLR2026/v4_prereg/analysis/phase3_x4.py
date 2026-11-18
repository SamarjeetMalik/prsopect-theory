"""Phase 3 analysis of X4, per PREREGISTRATION_X4.md. Written and dry-run on synthetic data before any X4 run.

Arms B and S, fresh seeds 122-161, three parts (D-plain, D-pop, M = scale-matched critic). Crossing contrasts S-B of the pooled Firth crossing;
advantage-quality metrics are run means of the X4 log (rollouts 1..end), seed-paired: per seed, the mean over the eight
cells of log(S/B) (ratios) or S-B (differences). Primary inference: seed-level jackknife-t (t, 39 df). Equivalence: the
90% jackknife-t interval within the declared margin. Secondary: seed-block percentile bootstrap (2,000 draws, seed
20261122).

Usage: phase3_x4.py RUNS_DIR ARTIFACT_DIR OUT_DIR
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

GRID = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]; SEEDS = list(range(122, 162)); PARTS = ["D-plain", "D-pop", "M"]; NS = len(SEEDS)
W = np.random.default_rng(20261122).multinomial(NS, np.full(NS, 1 / NS), size=2000); J = 1 - np.eye(NS)
RATIO = ["verr_train", "gae_err", "sign_dis"]; DIFF = ["adv_spearman"]; MARGIN_SB, MARGIN_EV, MARGIN_RATIO = 0.10, 0.05, 0.10
log = []
def say(s=""): print(s); log.append(s)


def crossing(frame):
    k = frame.assign(w=frame.w.round(4)).pivot(index="seed", columns="w", values="collapsed").reindex(index=SEEDS, columns=GRID)
    assert not k.isna().any().any(), "missing cells"
    k = k.to_numpy(float); one = np.ones_like(k); w = np.array(GRID)
    return np.array([firth(w, k.sum(0), one.sum(0))["wcrit"][0]]), firth(w, J @ k, J @ one)["wcrit"], firth(w, W @ k, W @ one)["wcrit"]


def jt(d, j, level):
    j = j[np.isfinite(j)]; n = len(j); se = np.sqrt((n - 1) / n * ((j - j.mean()) ** 2).sum()); t = stats.t.ppf(1 - (1 - level) / 2, n - 1)
    return d - t * se, d + t * se


def summarize(name, est):
    d, j, z = est; d = float(d[0]); l95, h95 = jt(d, j, .95); l90, h90 = jt(d, j, .90); zz = z[np.isfinite(z)]; p = np.percentile(zz, [2.5, 97.5])
    say(f"  {name:28s} {d:+.3f}  jt95 [{l95:+.3f}, {h95:+.3f}]  jt90 [{l90:+.3f}, {h90:+.3f}]  pct95 [{p[0]:+.3f}, {p[1]:+.3f}]")
    return dict(quantity=name, difference=d, jt_low_95=l95, jt_high_95=h95, jt_low_90=l90, jt_high_90=h90, pct_low_95=p[0], pct_high_95=p[1])


def paired(frame, q, how):
    """Per-seed mean over cells of log(S/B) or S-B; returns the mean, its jackknife values, and bootstrap means."""
    p = frame.pivot_table(index=["seed", "w"], columns="arm", values=q)
    cell = np.log(p.S / p.B) if how == "ratio" else p.S - p.B
    ps = cell.groupby(level="seed").mean().reindex(SEEDS).to_numpy(); assert np.isfinite(ps).all(), q
    return np.array([ps.mean()]), (J @ ps) / (NS - 1), (W @ ps) / NS


x = pd.read_csv(RUNS / "X4_runs.csv"); check_supply_table(x)
assert len(x) == 1920 and x.complete.all() and (x.actual == 300160).all() and (x.n_eps == 10).all() and x.eval_seeds_ok.all()
assert (x.x4_records == 469).all() and set(x.seed) == set(SEEDS)
say("== X4: 1,920 runs complete, 300,160 transitions, declared evaluation seeds, 469 X4 records each")
by = {p: x[x.part == p] for p in PARTS}

say("\n== crossing contrasts")
SB = {p: tuple(a - b for a, b in zip(crossing(by[p][by[p].arm == "S"]), crossing(by[p][by[p].arm == "B"]))) for p in PARTS}
rows = [summarize(f"S-B {p}", SB[p]) for p in PARTS]
CH = tuple(a - b for a, b in zip(SB["M"], SB["D-plain"])); rows.append(summarize("change M - D-plain", CH))
CP = tuple(a - b for a, b in zip(SB["D-pop"], SB["D-plain"])); rows.append(summarize("change D-pop - D-plain", CP))
R = pd.DataFrame(rows).set_index("quantity")

say("\n== advantage quality, seed-paired S vs B (ratios: geometric mean S/B; differences: S-B)")
M = []
for p in PARTS:
    for q, how in [(q, "ratio") for q in RATIO] + [("adv_spearman", "diff"), ("adv_spearman", "ratio"), ("ev_sb3", "diff"),
                                                    ("ev_train_mc", "diff"), ("ev_ref_mc", "diff"), ("ev_post_insample", "diff")]:
        d, j, z = paired(by[p], "x4_" + q, how); g = float(d[0]); l95, h95 = jt(g, j, .95); l90, h90 = jt(g, j, .90); pc = np.percentile(z, [2.5, 97.5])
        f = np.exp if how == "ratio" else (lambda v: v)
        M.append(dict(part=p, metric=q, scale=how, value=float(f(g)), low_95=float(f(l95)), high_95=float(f(h95)), low_90=float(f(l90)),
                      high_90=float(f(h90)), pct_low_95=float(f(pc[0])), pct_high_95=float(f(pc[1]))))
        say(f"  {p:8s} {q:18s} {'S/B' if how == 'ratio' else 'S-B'} {f(g):+.3f}  95% [{f(l95):+.3f}, {f(h95):+.3f}]  90% [{f(l90):+.3f}, {f(h90):+.3f}]")
M = pd.DataFrame(M); m = lambda p, q, how: M[(M.part == p) & (M.metric == q) & (M.scale == how)].iloc[0]

say("\n== descriptive: run means by arm")
for p in PARTS:
    for a in ["B", "S"]:
        d = by[p][by[p].arm == a]
        say(f"  {p:8s} {a}  " + "  ".join(f"{q} {d['x4_' + q].mean():.3f}" for q in ["ev_sb3", "ev_train_mc", "ev_ref_mc", "verr_train", "gae_err", "adv_spearman", "sign_dis"])
            )

say("\n== Pre-declared predictions (jackknife-t)")
P = {}
r = R.loc["S-B D-plain"]; P[1] = r.jt_low_95 > 0
say(f"P1 replication, S-B D-plain > 0 (95%): {r.difference:+.3f} [{r.jt_low_95:+.3f}, {r.jt_high_95:+.3f}] -> {'MET' if P[1] else 'NOT MET'}")
sub = []
for q in RATIO:
    e = m("D-plain", q, "ratio"); ok = e.low_95 > 1; sub.append(ok); say(f"P2 {q:12s} S/B > 1: {e.value:.3f} [{e.low_95:.3f}, {e.high_95:.3f}] -> {'met' if ok else 'not met'}")
e = m("D-plain", "adv_spearman", "diff"); ok = e.high_95 < 0; sub.append(ok); say(f"P2 adv_spearman S-B < 0: {e.value:+.4f} [{e.low_95:+.4f}, {e.high_95:+.4f}] -> {'met' if ok else 'not met'}")
P[2] = all(sub); say(f"P2 advantage degradation under S (all four): {'MET' if P[2] else 'NOT MET'} ({sum(sub)}/4)")
sub = []
for q in RATIO + ["adv_spearman"]:
    e = m("D-pop", q, "ratio"); ok = e.low_90 > 1 - MARGIN_RATIO and e.high_90 < 1 + MARGIN_RATIO; sub.append(ok)
    say(f"P3 {q:12s} D-pop S/B within 1 +/- {MARGIN_RATIO} (90%): {e.value:.3f} [{e.low_90:.3f}, {e.high_90:.3f}] -> {'met' if ok else 'not met'}")
P[3] = all(sub); say(f"P3 PopArt equalizes advantage quality: {'MET' if P[3] else 'NOT MET'}")
e = m("M", "ev_train_mc", "diff"); P[4] = e.low_90 > -MARGIN_EV and e.high_90 < MARGIN_EV
d0 = m("D-plain", "ev_train_mc", "diff")
say(f"P4 manipulation check, M critic-fit gap S-B (EV vs MC returns) within +/-{MARGIN_EV} (90%): {e.value:+.3f} [{e.low_90:+.3f}, {e.high_90:+.3f}] (D-plain {d0.value:+.3f}) -> {'MET' if P[4] else 'NOT MET'}")
c, f_ = R.loc["change M - D-plain"], R.loc["S-B M"]
dec = c.jt_high_95 < 0; eq = f_.jt_low_90 > -MARGIN_SB and f_.jt_high_90 < MARGIN_SB; P[5] = dec and eq
say(f"P5a change M - D-plain < 0 (95%): {c.difference:+.3f} [{c.jt_low_95:+.3f}, {c.jt_high_95:+.3f}] -> {'met' if dec else 'not met'}")
say(f"P5b S-B under M within +/-{MARGIN_SB} (90%): {f_.difference:+.3f} [{f_.jt_low_90:+.3f}, {f_.jt_high_90:+.3f}] -> {'met' if eq else 'not met'}")
say(f"P5 channel test (scale-matched critic removes the shift): {'MET' if P[5] else 'NOT MET'}{'' if P[4] else ' (not interpretable: P4 not met)'}")
pf = m("D-pop", "ev_train_mc", "diff"); fit_eq = pf.low_90 > -MARGIN_EV and pf.high_90 < MARGIN_EV; pp = R.loc["S-B D-pop"]
pop_eq = pp.jt_low_90 > -MARGIN_SB and pp.jt_high_90 < MARGIN_SB; pop_pos = pp.jt_low_95 > 0; P[6] = fit_eq
say(f"P6 H-fit test (D-pop): fit gap S-B within +/-{MARGIN_EV} (90%): {pf.value:+.3f} [{pf.low_90:+.3f}, {pf.high_90:+.3f}] -> {'met' if fit_eq else 'not met'};"
    f" S-B D-pop {pp.difference:+.3f}, 90% [{pp.jt_low_90:+.3f}, {pp.jt_high_90:+.3f}], 95% lower {pp.jt_low_95:+.3f}")

say("\n== Declared interpretation")
if not P[1]:
    say("P1 not met: the shift did not replicate on the fresh seeds; X4 cannot test the critic account (reported first).")
if not P[4]:
    ch = "Channel (M): the manipulation failed to equalize critic fit; no claim from M."
elif P[5]:
    ch = "Channel (M): fixing the critic's output scale removes the shift; the critic's target scale carries it at this budget."
elif f_.jt_low_95 > 0:
    ch = "Channel (M): the shift persists with the critic's target scale matched; a channel other than the critic's target scale exists."
else:
    ch = "Channel (M): inconclusive (neither removed nor shown to persist)."
if not fit_eq:
    fit = "H-fit (D-pop): PopArt did not equalize critic fit; no test of H-fit."
elif pop_pos and R.loc["change D-pop - D-plain"].jt_high_95 < 0:
    fit = "H-fit (D-pop): with critic fit equalized by PopArt the shift is reduced but persists (95% above zero): fit carries at most part of it."
elif pop_pos:
    fit = "H-fit (D-pop): H-fit is falsified as stated: with critic fit equalized by PopArt the shift persists (95% above zero) and is not shown reduced."
elif pop_eq:
    fit = "H-fit (D-pop): consistent with H-fit: with critic fit equalized, the shift is equivalent to zero within +/-0.10."
else:
    fit = "H-fit (D-pop): inconclusive: fit equalized, shift neither shown above zero nor shown within +/-0.10."
reading = ch + " " + fit
say(ch); say(fit)
R.to_csv(OUT / "X4_contrasts.csv"); M.to_csv(OUT / "X4_advantage_quality.csv", index=False)
pd.DataFrame([dict(prediction=k, met=bool(v)) for k, v in P.items()] + [dict(prediction="reading", met=reading)]).to_csv(OUT / "X4_predictions.csv", index=False)
(OUT / "phase3_x4_report.txt").write_text("\n".join(log) + "\n")
