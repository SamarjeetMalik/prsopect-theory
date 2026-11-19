"""Phase 3 analysis of X5, per PREREGISTRATION_X5.md. Written and dry-run on synthetic data before any X5 run.

Decentralized execution (per-node actors on local views, a centralized critic). Arms B and S, fresh seeds 162-201, three
parts (plain, popart, M = fixed-scale head). Crossing contrasts S-B of the pooled Firth crossing;
advantage-quality metrics are run means of the X5 (X4-style) log (rollouts 1..end), seed-paired: per seed, the mean over the eight
cells of log(S/B) (ratios) or S-B (differences). Primary inference: seed-level jackknife-t (t, 39 df). Equivalence: the
90% jackknife-t interval within the declared margin. Secondary: seed-block percentile bootstrap (2,000 draws, seed
20261123).

Usage: phase3_x5.py RUNS_DIR ARTIFACT_DIR OUT_DIR
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

GRID = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]; SEEDS = list(range(162, 202)); PARTS = ["plain", "popart", "M"]; NS = len(SEEDS)
W = np.random.default_rng(20261123).multinomial(NS, np.full(NS, 1 / NS), size=2000); J = 1 - np.eye(NS)
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


x = pd.read_csv(RUNS / "X5_runs.csv"); check_supply_table(x)
assert len(x) == 1920 and x.complete.all() and (x.actual == 300160).all() and (x.n_eps == 10).all() and x.eval_seeds_ok.all()
assert (x.x5_records == 469).all() and set(x.seed) == set(SEEDS) and set(x.execution) == {"decentralized"}
say("== X5: 1,920 runs complete, 300,160 transitions, declared evaluation seeds, 469 X5 records each, decentralized execution")
by = {p: x[x.part == p] for p in PARTS}

say("\n== crossing contrasts")
SB = {p: tuple(a - b for a, b in zip(crossing(by[p][by[p].arm == "S"]), crossing(by[p][by[p].arm == "B"]))) for p in PARTS}
rows = [summarize(f"S-B {p}", SB[p]) for p in PARTS]
CH = tuple(a - b for a, b in zip(SB["M"], SB["plain"])); rows.append(summarize("change M - plain", CH))
CP = tuple(a - b for a, b in zip(SB["popart"], SB["plain"])); rows.append(summarize("change popart - plain", CP))
R = pd.DataFrame(rows).set_index("quantity")

say("\n== advantage quality, seed-paired S vs B (ratios: geometric mean S/B; differences: S-B)")
M = []
for p in PARTS:
    for q, how in [(q, "ratio") for q in RATIO] + [("adv_spearman", "diff"), ("adv_spearman", "ratio"), ("ev_sb3", "diff"),
                                                    ("ev_train_mc", "diff"), ("ev_ref_mc", "diff"), ("ev_post_insample", "diff")]:
        d, j, z = paired(by[p], "x5_" + q, how); g = float(d[0]); l95, h95 = jt(g, j, .95); l90, h90 = jt(g, j, .90); pc = np.percentile(z, [2.5, 97.5])
        f = np.exp if how == "ratio" else (lambda v: v)
        M.append(dict(part=p, metric=q, scale=how, value=float(f(g)), low_95=float(f(l95)), high_95=float(f(h95)), low_90=float(f(l90)),
                      high_90=float(f(h90)), pct_low_95=float(f(pc[0])), pct_high_95=float(f(pc[1]))))
        say(f"  {p:8s} {q:18s} {'S/B' if how == 'ratio' else 'S-B'} {f(g):+.3f}  95% [{f(l95):+.3f}, {f(h95):+.3f}]  90% [{f(l90):+.3f}, {f(h90):+.3f}]")
M = pd.DataFrame(M); m = lambda p, q, how: M[(M.part == p) & (M.metric == q) & (M.scale == how)].iloc[0]

say("\n== descriptive: run means by arm")
for p in PARTS:
    for a in ["B", "S"]:
        d = by[p][by[p].arm == a]
        say(f"  {p:8s} {a}  " + "  ".join(f"{q} {d['x5_' + q].mean():.3f}" for q in ["ev_sb3", "ev_train_mc", "ev_ref_mc", "verr_train", "gae_err", "adv_spearman", "sign_dis"]))

say("\n== Pre-declared predictions (jackknife-t)")
P = {}
r = R.loc["S-B plain"]; P[1] = r.jt_low_95 > 0; absent = r.jt_low_90 > -MARGIN_SB and r.jt_high_90 < MARGIN_SB
say(f"P1 symmetric shift under decentralized execution, S-B plain > 0 (95%): {r.difference:+.3f} [{r.jt_low_95:+.3f}, {r.jt_high_95:+.3f}] -> {'MET' if P[1] else 'NOT MET'}"
    f"; 90% [{r.jt_low_90:+.3f}, {r.jt_high_90:+.3f}] within +/-{MARGIN_SB}: {absent}")
sub = []
for q in RATIO:
    e = m("plain", q, "ratio"); ok = e.low_95 > 1; sub.append(ok); say(f"P2 {q:12s} S/B > 1: {e.value:.3f} [{e.low_95:.3f}, {e.high_95:.3f}] -> {'met' if ok else 'not met'}")
e = m("plain", "adv_spearman", "diff"); ok = e.high_95 < 0; sub.append(ok); say(f"P2 adv_spearman S-B < 0: {e.value:+.4f} [{e.low_95:+.4f}, {e.high_95:+.4f}] -> {'met' if ok else 'not met'}")
P[2] = all(sub); say(f"P2 advantage degradation under S, plain critic (all four): {'MET' if P[2] else 'NOT MET'} ({sum(sub)}/4)")
pf = m("popart", "ev_train_mc", "diff"); P[3] = pf.low_90 > -MARGIN_EV and pf.high_90 < MARGIN_EV
say(f"P3 PopArt equalizes critic fit, S-B within +/-{MARGIN_EV} (90%): {pf.value:+.3f} [{pf.low_90:+.3f}, {pf.high_90:+.3f}] -> {'MET' if P[3] else 'NOT MET'}")
mf = m("M", "ev_train_mc", "diff"); P[4] = mf.low_90 > -MARGIN_EV and mf.high_90 < MARGIN_EV
say(f"P4 fixed-scale head equalizes critic fit, S-B within +/-{MARGIN_EV} (90%): {mf.value:+.3f} [{mf.low_90:+.3f}, {mf.high_90:+.3f}] -> {'MET' if P[4] else 'NOT MET'}")
c, f_ = R.loc["change M - plain"], R.loc["S-B M"]; dec = c.jt_high_95 < 0; eq = f_.jt_low_90 > -MARGIN_SB and f_.jt_high_90 < MARGIN_SB; P[5] = dec and eq
say(f"P5a change M - plain < 0 (95%): {c.difference:+.3f} [{c.jt_low_95:+.3f}, {c.jt_high_95:+.3f}] -> {'met' if dec else 'not met'}")
say(f"P5b S-B under M within +/-{MARGIN_SB} (90%): {f_.difference:+.3f} [{f_.jt_low_90:+.3f}, {f_.jt_high_90:+.3f}] -> {'met' if eq else 'not met'}")
say(f"P5 channel test (fixed-scale head removes the shift): {'MET' if P[5] else 'NOT MET'}")
c2, pp = R.loc["change popart - plain"], R.loc["S-B popart"]; dec2 = c2.jt_high_95 < 0; eq2 = pp.jt_low_90 > -MARGIN_SB and pp.jt_high_90 < MARGIN_SB; P[6] = dec2 and eq2
say(f"P6a change popart - plain < 0 (95%): {c2.difference:+.3f} [{c2.jt_low_95:+.3f}, {c2.jt_high_95:+.3f}] -> {'met' if dec2 else 'not met'}")
say(f"P6b S-B under PopArt within +/-{MARGIN_SB} (90%): {pp.difference:+.3f} [{pp.jt_low_90:+.3f}, {pp.jt_high_90:+.3f}] -> {'met' if eq2 else 'not met'}")
say(f"P6 PopArt removes the shift: {'MET' if P[6] else 'NOT MET'}")

say("\n== Declared interpretation")
if P[1]:
    sc_ = "Scope: a symmetric-scaling shift is present under decentralized execution (per-node actors on local views, centralized critic)."
elif absent:
    sc_ = ("Scope: no symmetric-scaling shift under decentralized execution: S-B is equivalent to zero within +/-0.10 (90%). The shift is "
           "established for the shared controller only, and P5-P6 cannot test a channel for an absent shift.")
else:
    sc_ = ("Scope: not detected and not shown absent under decentralized execution (inconclusive at this power); the shift is established for "
           "the shared controller only, and P5-P6 cannot test a channel.")
if not P[1]:
    ch = "Channel (M): not tested (P1 not met)."; po = "PopArt: not tested (P1 not met)."
else:
    if not P[4]:
        ch = "Channel (M): the fixed-scale head did not equalize critic fit; no claim from M."
    elif P[5]:
        ch = "Channel (M): fixing the critic's output scale removes the shift; the critic's target scale carries it under decentralized execution at this budget."
    elif f_.jt_low_95 > 0:
        ch = "Channel (M): the shift persists with the critic's target scale fixed; a channel other than the critic's target scale exists in this setting."
    else:
        ch = "Channel (M): inconclusive (neither removed nor shown to persist)."
    if not P[3]:
        po = "PopArt: did not equalize critic fit; no test of the fit account."
    elif P[6]:
        po = "PopArt: with critic fit equalized the shift is equivalent to zero within +/-0.10 and reduced; consistent with the critic-target account (not proof)."
    elif pp.jt_low_95 > 0 and not dec2:
        po = "PopArt: with critic fit equalized the shift persists (95% above zero) and is not shown reduced: the fit account is falsified as stated in this setting."
    elif pp.jt_low_95 > 0:
        po = "PopArt: the shift is reduced but persists with critic fit equalized: the fit carries at most part of it in this setting."
    else:
        po = "PopArt: inconclusive (fit equalized; shift neither shown removed nor shown to persist)."
reading = " ".join([sc_, ch, po])
for s_ in (sc_, ch, po):
    say(s_)
R.to_csv(OUT / "X5_contrasts.csv"); M.to_csv(OUT / "X5_advantage_quality.csv", index=False)
pd.DataFrame([dict(prediction=k, met=bool(v)) for k, v in P.items()] + [dict(prediction="reading", met=reading)]).to_csv(OUT / "X5_predictions.csv", index=False)
(OUT / "phase3_x5_report.txt").write_text("\n".join(log) + "\n")
