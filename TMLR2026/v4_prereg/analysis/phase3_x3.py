"""Phase 3 analysis of X3, per PREREGISTRATION_X3.md. Written before any X3 run.

2 x 2 factorial (actor clipped or not) x (critic clipped or not), arms B and S, fresh seeds 102-121. The outcome is the
symmetric-scaling contrast S-B of the pooled Firth crossing; the factorial effects are
  critic effect = 1/2 [(SB_critic_only - SB_neither) + (SB_both - SB_actor_only)]
  actor effect  = 1/2 [(SB_actor_only - SB_neither) + (SB_both - SB_critic_only)]
  interaction   = (SB_both - SB_actor_only) - (SB_critic_only - SB_neither)
Primary inference: seed-level jackknife-t (t, 19 df). Equivalence of the actor effect: 90% jackknife-t interval within
+/-0.10. Secondary: seed-block percentile bootstrap (2,000 draws, seed 20261121).

Usage: phase3_x3.py RUNS_DIR ARTIFACT_DIR OUT_DIR
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

GRID = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]; SEEDS = list(range(102, 122)); MODES = ["both", "actor_only", "critic_only", "neither"]
W = np.random.default_rng(20261121).multinomial(20, np.full(20, 1 / 20), size=2000); J = 1 - np.eye(20)
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


x = pd.read_csv(RUNS / "X3_runs.csv"); check_supply_table(x)
assert len(x) == 1280 and x.complete.all() and (x.actual == 300160).all() and (x.n_eps == 10).all() and x.eval_seeds_ok.all()
say("== X3: 1,280 runs complete, 300,160 transitions, declared evaluation seeds")
SB = {}
for m in MODES:
    d = x[x["mode"] == m]; s, b = crossing(d[d.arm == "S"]), crossing(d[d.arm == "B"])
    SB[m] = tuple(a - c for a, c in zip(s, b))
E = {"S-B both": SB["both"], "S-B actor_only": SB["actor_only"], "S-B critic_only": SB["critic_only"], "S-B neither": SB["neither"]}
comb = lambda f: tuple(f(*(SB[m][i] for m in MODES)) for i in range(3))
E["critic effect"] = comb(lambda bo, ao, co, ne: 0.5 * ((co - ne) + (bo - ao)))
E["actor effect"] = comb(lambda bo, ao, co, ne: 0.5 * ((ao - ne) + (bo - co)))
E["interaction"] = comb(lambda bo, ao, co, ne: (bo - ao) - (co - ne))
rows = []
for k, (d, j, z) in E.items():
    d = float(d[0]); l95, h95 = jt(d, j, .95); l90, h90 = jt(d, j, .90); l99, h99 = jt(d, j, .99); zz = z[np.isfinite(z)]
    p = np.percentile(zz, [2.5, 97.5])
    rows.append(dict(quantity=k, difference=d, jt_low_95=l95, jt_high_95=h95, jt_low_90=l90, jt_high_90=h90, jt_low_99=l99, jt_high_99=h99,
                     pct_low_95=p[0], pct_high_95=p[1], jt_excludes_zero_95=bool(l95 > 0 or h95 < 0)))
    say(f"  {k:16s} {d:+.3f}  jt95 [{l95:+.3f}, {h95:+.3f}]  jt90 [{l90:+.3f}, {h90:+.3f}]  jt99 [{l99:+.3f}, {h99:+.3f}]  pct95 [{p[0]:+.3f}, {p[1]:+.3f}]"
        f"{'  *' if (l95 > 0 or h95 < 0) else ''}")
R = pd.DataFrame(rows).set_index("quantity"); R.to_csv(OUT / "X3_contrasts.csv")

say("\n== optimizer state, seed-paired S/B ratios (geometric mean over cells, jackknife-t over seeds)")
Q = ["pu_applied_actor_grad_norm", "pu_actor_update_norm", "pu_applied_critic_grad_norm", "pu_critic_update_norm",
     "pu_actor_adam_direction_norm", "pu_critic_adam_direction_norm", "explained_variance_last"]
drows = []
for m in MODES:
    d = x[x["mode"] == m]
    for q in Q:
        if q not in d:
            continue
        p = d.pivot_table(index=["seed", "w"], columns="arm", values=q)
        if q == "explained_variance_last":
            ps = (p.S - p.B).groupby(level="seed").mean().reindex(SEEDS).to_numpy(); g = ps.mean()
            jk = np.array([np.delete(ps, i).mean() for i in range(20)]); lo, hi = jt(g, jk, .95)
            drows.append(dict(mode=m, quantity="EV S-B", value=g, low_95=lo, high_95=hi)); say(f"  {m:12s} {'EV S-B':30s} {g:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            continue
        ps = np.log(p.S / p.B).groupby(level="seed").mean().reindex(SEEDS).to_numpy(); g = ps.mean()
        jk = np.array([np.delete(ps, i).mean() for i in range(20)]); lo, hi = jt(g, jk, .95)
        drows.append(dict(mode=m, quantity=q, value=float(np.exp(g)), low_95=float(np.exp(lo)), high_95=float(np.exp(hi))))
        say(f"  {m:12s} {q:30s} S/B {np.exp(g):.3f} [{np.exp(lo):.3f}, {np.exp(hi):.3f}]")
pd.DataFrame(drows).to_csv(OUT / "X3_optimizer_state.csv", index=False)

say("\n== Pre-declared predictions (jackknife-t)")
c, a, i, b = R.loc["critic effect"], R.loc["actor effect"], R.loc["interaction"], R.loc["S-B both"]
say(f"P1 critic-clipping effect on S-B > 0 (95%): {c.difference:+.3f} [{c.jt_low_95:+.3f}, {c.jt_high_95:+.3f}] -> {'MET' if c.jt_low_95 > 0 else 'NOT MET'}")
say(f"P2 actor-clipping effect within +/-0.10 (90% interval): {a.difference:+.3f} [{a.jt_low_90:+.3f}, {a.jt_high_90:+.3f}] -> "
    f"{'MET' if a.jt_low_90 > -0.10 and a.jt_high_90 < 0.10 else 'NOT MET'}")
say(f"P3 S-B with both groups clipped > 0 (95%): {b.difference:+.3f} [{b.jt_low_95:+.3f}, {b.jt_high_95:+.3f}] -> {'MET' if b.jt_low_95 > 0 else 'NOT MET'}")
say(f"   interaction (reported, no prediction): {i.difference:+.3f} [{i.jt_low_95:+.3f}, {i.jt_high_95:+.3f}]")
(OUT / "phase3_x3_report.txt").write_text("\n".join(log) + "\n")
