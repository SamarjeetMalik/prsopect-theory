"""Phase 3 analysis of X6, X7 and X8, per PREREGISTRATION_X6_X8.md. Written and dry-run on relabelled earlier data before
any X6-X8 run.

Crossing contrasts are differences of the pooled Firth crossing. Advantage-quality and gradient metrics are run means
(or run medians for gradient norms), seed-paired: per seed, the mean over the eight cells of log(ratio) or of the
difference. Primary inference: seed-level jackknife-t (t, 39 df). Equivalence: the 90% jackknife-t interval within the
declared margin. Secondary: seed-block percentile bootstrap (2,000 draws; seeds 20261124/5/6).

Usage: phase3_x6_x8.py RUNS_DIR ARTIFACT_DIR OUT_DIR [X6|X7|X8 ...]
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RUNS, ART, OUT = map(Path, sys.argv[1:4]); OUT.mkdir(parents=True, exist_ok=True)
WHICH = sys.argv[4:] or ["X6", "X7", "X8"]
for d in [str(ART / "AAMAS2027/analysis"), os.environ.get("BOUNDARY_DIR", "")]:
    if d and (Path(d) / "boundary.py").exists():
        sys.path.insert(0, d)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from boundary import firth  # noqa: E402
from labels import check_supply_table  # noqa: E402

GRID = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]
CFG = {"X6": dict(seeds=list(range(202, 242)), parts=["plain", "M"], arms=["B", "S"], boot=20261124, n=1280),
       "X7": dict(seeds=list(range(242, 282)), parts=["plain", "normhead", "normpopart", "groupclip"], arms=["B", "S"], boot=20261125, n=2560),
       "X8": dict(seeds=list(range(282, 322)), parts=["plain"], arms=["B", "S", "Bdeg"], boot=20261126, n=960)}
RATIO = ["verr_train", "gae_err", "sign_dis"]; MSB, MEV = 0.10, 0.05; GRAD_EQ = (0.8, 1.25)


def jt(d, j, level):
    j = j[np.isfinite(j)]; n = len(j); se = np.sqrt((n - 1) / n * ((j - j.mean()) ** 2).sum()); t = stats.t.ppf(1 - (1 - level) / 2, n - 1)
    return d - t * se, d + t * se


def analyse(exp):
    c = CFG[exp]; SEEDS = c["seeds"]; NS = len(SEEDS); log = []
    W = np.random.default_rng(c["boot"]).multinomial(NS, np.full(NS, 1 / NS), size=2000); J = 1 - np.eye(NS)

    def say(s=""):
        print(s); log.append(s)

    def crossing(frame):
        k = frame.assign(w=frame.w.round(4)).pivot(index="seed", columns="w", values="collapsed").reindex(index=SEEDS, columns=GRID)
        assert not k.isna().any().any(), "missing cells"
        k = k.to_numpy(float); one = np.ones_like(k); w = np.array(GRID)
        return np.array([firth(w, k.sum(0), one.sum(0))["wcrit"][0]]), firth(w, J @ k, J @ one)["wcrit"], firth(w, W @ k, W @ one)["wcrit"]

    def summarize(name, est):
        d, j, z = est; d = float(d[0]); l95, h95 = jt(d, j, .95); l90, h90 = jt(d, j, .90); zz = z[np.isfinite(z)]; p = np.percentile(zz, [2.5, 97.5])
        say(f"  {name:34s} {d:+.3f}  jt95 [{l95:+.3f}, {h95:+.3f}]  jt90 [{l90:+.3f}, {h90:+.3f}]  pct95 [{p[0]:+.3f}, {p[1]:+.3f}]")
        return dict(quantity=name, difference=d, jt_low_95=l95, jt_high_95=h95, jt_low_90=l90, jt_high_90=h90, pct_low_95=p[0], pct_high_95=p[1])

    def paired(frame, q, how, a="S", b="B"):
        p = frame.pivot_table(index=["seed", "w"], columns="arm", values=q)
        cell = np.log(p[a] / p[b]) if how == "ratio" else p[a] - p[b]
        ps = cell.groupby(level="seed").mean().reindex(SEEDS).to_numpy(); assert np.isfinite(ps).all(), q
        return np.array([ps.mean()]), (J @ ps) / (NS - 1), (W @ ps) / NS

    def metric(frame, q, how, a="S", b="B", part=None):
        d, j, z = paired(frame, q, how, a, b); g = float(d[0]); l95, h95 = jt(g, j, .95); l90, h90 = jt(g, j, .90); pc = np.percentile(z, [2.5, 97.5])
        f = np.exp if how == "ratio" else (lambda v: v)
        r = dict(part=part, contrast=f"{a}{'/' if how == 'ratio' else '-'}{b}", metric=q.replace("x_", "").replace("med_", "median "), scale=how,
                 value=float(f(g)), low_95=float(f(l95)), high_95=float(f(h95)), low_90=float(f(l90)), high_90=float(f(h90)),
                 pct_low_95=float(f(pc[0])), pct_high_95=float(f(pc[1])))
        say(f"  {str(part):10s} {r['contrast']:7s} {r['metric']:24s} {r['value']:+.3f}  95% [{r['low_95']:+.3f}, {r['high_95']:+.3f}]  90% [{r['low_90']:+.3f}, {r['high_90']:+.3f}]")
        return r

    within = lambda r, m: r["low_90"] > -m and r["high_90"] < m
    within_q = lambda r, m: r.jt_low_90 > -m and r.jt_high_90 < m

    x = pd.read_csv(RUNS / f"{exp}_runs.csv"); x = x.assign(arm=x.x_arm); check_supply_table(x.assign(arm=x.arm.replace({"Bdeg": "B"})))
    assert len(x) == c["n"] and x.complete.all() and (x.actual == 300160).all() and (x.n_eps == 10).all() and x.eval_seeds_ok.all()
    assert (x.x_records == 469).all() and set(x.seed) == set(SEEDS) and set(x.experiment) == {exp}
    say(f"== {exp}: {len(x):,} runs complete, 300,160 transitions, declared evaluation seeds, 469 X4-style records each")
    by = {p: x[x.part == p] for p in c["parts"]}
    say("\n== crossing contrasts"); rows = []; SB = {}
    for p in c["parts"]:
        cr = {a: crossing(by[p][by[p].arm == a]) for a in c["arms"]}
        SB[p] = tuple(u - v for u, v in zip(cr["S"], cr["B"])); rows.append(summarize(f"S-B {p}", SB[p]))
        if exp == "X8":
            DB = tuple(u - v for u, v in zip(cr["Bdeg"], cr["B"])); rows.append(summarize("Bdeg-B", DB))
            rows.append(summarize("(S-B) - (Bdeg-B)", tuple(u - v for u, v in zip(SB[p], DB))))
    for p in c["parts"][1:]:
        rows.append(summarize(f"change {p} - plain", tuple(u - v for u, v in zip(SB[p], SB["plain"]))))
    R = pd.DataFrame(rows).set_index("quantity")

    say("\n== advantage quality and gradients, seed-paired")
    M = []; pairs = [("S", "B")] + ([("Bdeg", "B")] if exp == "X8" else [])
    for p in c["parts"]:
        for a, b in pairs:
            for q in RATIO:
                M.append(metric(by[p], "x_" + q, "ratio", a, b, p))
            for q in ["adv_spearman", "ev_train_mc", "ev_ref_mc"]:
                M.append(metric(by[p], "x_" + q, "diff", a, b, p))
            for q in ["med_critic_grad_norm", "med_actor_grad_norm", "med_post_actor_grad_norm"]:
                M.append(metric(by[p], q, "ratio", a, b, p))
    if exp == "X8":
        fd = by["plain"].pivot_table(index=["seed", "w"], columns="arm", values="x_ev_train_mc")
        cell = (fd.Bdeg - fd.B) - (fd.S - fd.B); ps = cell.groupby(level="seed").mean().reindex(SEEDS).to_numpy()
        g = float(ps.mean()); j = (J @ ps) / (NS - 1); l95, h95 = jt(g, j, .95); l90, h90 = jt(g, j, .90)
        M.append(dict(part="plain", contrast="(Bdeg-B)-(S-B)", metric="ev_train_mc", scale="diff", value=g, low_95=l95, high_95=h95, low_90=l90, high_90=h90,
                      pct_low_95=np.nan, pct_high_95=np.nan))
        say(f"  fit-gap difference (Bdeg-B)-(S-B) ev_train_mc {g:+.3f}  90% [{l90:+.3f}, {h90:+.3f}]")
    M = pd.DataFrame(M)
    m = lambda p, ct, q, how: M[(M.part == p) & (M.contrast == ct) & (M.metric == q) & (M.scale == how)].iloc[0]

    say("\n== Pre-declared predictions (jackknife-t)"); P = {}; reading = []
    r1 = R.loc["S-B plain"]; P[1] = r1.jt_low_95 > 0; absent = within_q(r1, MSB)
    say(f"P1 S-B plain > 0 (95%): {r1.difference:+.3f} [{r1.jt_low_95:+.3f}, {r1.jt_high_95:+.3f}] -> {'MET' if P[1] else 'NOT MET'}; 90% within +/-{MSB}: {absent}")
    if exp == "X6":
        sub = [m("plain", "S/B", q, "ratio")["low_95"] > 1 for q in RATIO] + [m("plain", "S-B", "adv_spearman", "diff")["high_95"] < 0]
        P[2] = all(sub); say(f"P2 advantage degradation (plain, all four): {'MET' if P[2] else 'NOT MET'} ({sum(sub)}/4)")
        f = m("M", "S-B", "ev_train_mc", "diff"); P[3] = within(f, MEV); say(f"P3 head equalizes fit within +/-{MEV} (90%): {f['value']:+.3f} [{f['low_90']:+.3f}, {f['high_90']:+.3f}] -> {'MET' if P[3] else 'NOT MET'}")
        ch, sm = R.loc["change M - plain"], R.loc["S-B M"]; P[4] = bool(ch.jt_high_95 < 0 and within_q(sm, MSB))
        say(f"P4 change M - plain < 0 (95%) and S-B under M within +/-{MSB} (90%): {ch.difference:+.3f} [{ch.jt_low_95:+.3f}, {ch.jt_high_95:+.3f}]; "
            f"{sm.difference:+.3f} [{sm.jt_low_90:+.3f}, {sm.jt_high_90:+.3f}] -> {'MET' if P[4] else 'NOT MET'}{'' if P[1] else ' (descriptive only: P1 not met)'}")
        if P[1]:
            reading.append("The shift is present with local critics; the claim extends to fully decentralized actor-critic training on the supply chain.")
            if P[3] and P[4]:
                reading.append("The critic's target scale carries it there, at this budget, without clipping.")
            elif P[3] and sm.jt_low_95 > 0:
                reading.append("With the head's fit equalized the shift persists: a channel other than the critic's target scale exists with local critics.")
            else:
                reading.append("Channel: inconclusive.")
        elif absent:
            reading.append("No shift at this margin with local critics: the confound, as measured, needs a critic that sees the global state.")
        else:
            reading.append("Local critics: not detected and not shown absent (inconclusive).")
    if exp == "X7":
        ok = []
        for p in ["normhead", "normpopart"]:
            g = m(p, "S/B", "median critic_grad_norm", "ratio"); ok.append(g["low_90"] > GRAD_EQ[0] and g["high_90"] < GRAD_EQ[1])
            say(f"P2 {p}: S/B median pre-clip critic norm {g['value']:.3f} 90% [{g['low_90']:.3f}, {g['high_90']:.3f}] within {GRAD_EQ}: {ok[-1]}")
        gp = m("plain", "S/B", "median critic_grad_norm", "ratio"); ok.append(gp["low_95"] > 1)
        say(f"P2 plain: S/B median pre-clip critic norm {gp['value']:.3f} 95% [{gp['low_95']:.3f}, {gp['high_95']:.3f}] > 1: {ok[-1]}")
        P[2] = all(ok); say(f"P2 manipulation check: {'MET' if P[2] else 'NOT MET'}")
        for k, p in [(3, "normhead"), (4, "normpopart")]:
            ch, sp = R.loc[f"change {p} - plain"], R.loc[f"S-B {p}"]; P[k] = bool(ch.jt_high_95 < 0 and within_q(sp, MSB))
            say(f"P{k} {p}: change {ch.difference:+.3f} [{ch.jt_low_95:+.3f}, {ch.jt_high_95:+.3f}]; S-B {sp.difference:+.3f} 90% [{sp.jt_low_90:+.3f}, {sp.jt_high_90:+.3f}] -> "
                f"{'MET' if P[k] else 'NOT MET'}{'' if P[1] else ' (descriptive only: P1 not met)'}")
        q5 = R.loc["change groupclip - plain"]; q5eq = within_q(q5, MSB); q5neg = q5.jt_high_95 < 0
        say(f"Q5 change groupclip - plain: {q5.difference:+.3f} 95% [{q5.jt_low_95:+.3f}, {q5.jt_high_95:+.3f}] 90% [{q5.jt_low_90:+.3f}, {q5.jt_high_90:+.3f}]; "
            f"within +/-{MSB}: {q5eq}; below zero: {q5neg}")
        if not P[1]:
            reading.append("P1 not met: the default shift is not shown under decentralized execution; P3-Q5 descriptive only.")
        else:
            reading.append("The shift under SB3's default shared clip transfers to decentralized execution.")
            if not P[2]:
                reading.append("The manipulation failed (P2); P3 and P4 descriptive only.")
            else:
                for k, p in [(3, "normhead"), (4, "normpopart")]:
                    sp = R.loc[f"S-B {p}"]
                    if P[k] and q5eq:
                        reading.append(f"{p}: making the critic's targets and gradients B-sized removes the clipped shift and per-group clipping does not detectably reduce it: the clipped shift runs through the critic's target scale, not the clip's starving of the actor.")
                    elif P[k] and q5neg:
                        reading.append(f"{p}: removes the clipped shift, and per-group clipping also reduces it: both routes contribute; not apportioned.")
                    elif P[k]:
                        reading.append(f"{p}: removes the clipped shift; the per-group clipping estimate is inconclusive.")
                    elif sp.jt_low_95 > 0:
                        reading.append(f"{p}: the shift persists with B-sized critic targets and gradients: a channel other than the critic's target and gradient scale exists under the shared clip.")
                    else:
                        reading.append(f"{p}: inconclusive.")
    if exp == "X8":
        fg = M[M.contrast == "(Bdeg-B)-(S-B)"].iloc[0]; P[2] = bool(fg.low_90 > -MEV and fg.high_90 < MEV)
        say(f"P2 fit match: (Bdeg-B)-(S-B) ev_train_mc {fg.value:+.3f} 90% [{fg.low_90:+.3f}, {fg.high_90:+.3f}] within +/-{MEV}: {'MET' if P[2] else 'NOT MET'}")
        db, dd = R.loc["Bdeg-B"], R.loc["(S-B) - (Bdeg-B)"]; P[3] = bool(db.jt_low_95 > 0)
        say(f"P3 Bdeg-B > 0 (95%): {db.difference:+.3f} [{db.jt_low_95:+.3f}, {db.jt_high_95:+.3f}] -> {'MET' if P[3] else 'NOT MET'}")
        say(f"   (S-B) - (Bdeg-B): {dd.difference:+.3f} 90% [{dd.jt_low_90:+.3f}, {dd.jt_high_90:+.3f}] within +/-{MSB}: {within_q(dd, MSB)}")
        if not P[2]:
            reading.append("The fit manipulation failed (P2); descriptive only.")
        elif P[3] and within_q(dd, MSB):
            reading.append("A critic degraded to S's fit at B's target scale reproduces the shift: a worse fit is sufficient, consistent with fit mediating the target-scale effect.")
        elif within_q(db, MSB):
            reading.append("Degrading the fit this way does not reproduce the shift: the target-scale effect is not explained by the fit as measured.")
        else:
            reading.append("Partial or inconclusive.")
    say("\n== Declared interpretation")
    for s_ in reading:
        say(s_)
    R.to_csv(OUT / f"{exp}_contrasts.csv"); M.to_csv(OUT / f"{exp}_advantage_quality.csv", index=False)
    pd.DataFrame([dict(prediction=k, met=bool(v)) for k, v in P.items()] + [dict(prediction="reading", met=" ".join(reading))]).to_csv(OUT / f"{exp}_predictions.csv", index=False)
    (OUT / f"phase3_{exp.lower()}_report.txt").write_text("\n".join(log) + "\n")


for e in WHICH:
    analyse(e)
