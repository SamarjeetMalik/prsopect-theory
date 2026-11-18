"""Seed-level jackknife-t intervals for every principal contrast and every between-condition change (revision v5;
not pre-registered). calibration_v5.py found this the only procedure among five whose simulated coverage stayed near
nominal in every scenario. The pre-declared seed-block percentile intervals are kept and reported alongside.

The contrast list is the one stat_checks.py defines (its definition block is executed verbatim, so the two cannot
drift apart); the changes are defined below from the same frames.

Usage: jackknife_v5.py RUNS_DIR ARTIFACT_DIR RES_DIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
src = (HERE / "stat_checks.py").read_text()
head = src[: src.index("\nrows = []\n")]            # imports, helpers and the CON list, up to the first analysis step
ns = {"__file__": str(HERE / "stat_checks.py")}
argv = sys.argv; sys.argv = [str(HERE / "stat_checks.py")] + argv[1:4]
exec(compile(head, "stat_checks.py", "exec"), ns)
sys.argv = argv
firth, CON, G1 = ns["firth"], ns["CON"], ns["G1"]
RES = Path(argv[3])


def table(frame, grid):
    seeds = sorted(frame.seed.unique())
    k = frame.assign(w=frame.w.round(4)).pivot(index="seed", columns="w", values="collapsed").reindex(index=seeds, columns=[round(x, 4) for x in grid])
    assert not k.isna().any().any(), "missing cells"
    return seeds, k.to_numpy(float)


def est_jack(frame, grid):
    """Pooled Firth crossing and its leave-one-seed-out values, keyed by seed."""
    seeds, k = table(frame, grid); n = len(seeds); w = np.array(grid); one = np.ones_like(k)
    full = firth(w, k.sum(0), one.sum(0))["wcrit"][0]
    J = 1 - np.eye(n); jk = firth(w, J @ k, J @ one)["wcrit"]
    return full, dict(zip(seeds, jk))


def jt(est, jvals, level):
    v = np.asarray(jvals, float); v = v[np.isfinite(v)]; n = len(v)
    se = np.sqrt((n - 1) / n * ((v - v.mean()) ** 2).sum()); t = stats.t.ppf(1 - (1 - level) / 2, n - 1)
    return est - t * se, est + t * se, se, n


def contrast(fa, fb, grid):
    ea, ja = est_jack(fa, grid); eb, jb = est_jack(fb, grid); assert ja.keys() == jb.keys(), "arms not paired by seed"
    return ea - eb, {s: ja[s] - jb[s] for s in ja}


def change(c1, c2):
    (d1, j1), (d2, j2) = c1, c2; assert j1.keys() == j2.keys(), "changes must be paired by seed"
    return d1 - d2, {s: j1[s] - j2[s] for s in j1}


rows = []
def record(kind, label, d, j, stored=None):
    l95, h95, se, n = jt(d, list(j.values()), 0.95); l99, h99, _, _ = jt(d, list(j.values()), 0.99)
    r = dict(kind=kind, contrast=label, n_seeds=n, difference=d, se_jack=se, jt_low_95=l95, jt_high_95=h95, jt_low_99=l99, jt_high_99=h99,
             jt_excludes_zero_95=bool(l95 > 0 or h95 < 0), jt_excludes_zero_99=bool(l99 > 0 or h99 < 0))
    if stored is not None:
        r.update(pct_low_95=stored.ci_low, pct_high_95=stored.ci_high, pct_excludes_zero_95=bool(stored.ci_low > 0 or stored.ci_high < 0))
        assert abs(stored.difference - d) < 1e-9, label
    rows.append(r)


for label, grid, a, b, stored in CON:
    d, j = contrast(a, b, grid); record("principal", label, d, j, stored)

# ---------------------------------------------------------------- between-condition changes (paired by seed)
R = Path(argv[1])
e1, e6, e7all, e8 = ns["e1"], ns["e6"], ns["e7all"], ns["e8"]
r910, r11 = pd.read_csv(R / "E9E10_runs.csv"), pd.read_csv(R / "E11_runs.csv")
SB = lambda d: contrast(d[d.arm == "S"], d[d.arm == "B"], G1)
cell = lambda o, r_: e1[(e1.optimizer_mode == o) & (e1.reward_mode == r_)]
glob = SB(cell("global_clip", "raw"))
record("change", "E6 per-group minus shared clip, S-B", *change(SB(e6[e6.optimizer_mode == "groupwise_clip"]), glob))
e9 = r910[r910.experiment == "E9"]
record("change", "E9 tau 30000 minus tau 0.5, S-B", *change(SB(e9[e9.max_grad_norm == 30000]), glob))
record("change", "E9 tau 30000 minus no clip, S-B", *change(SB(e9[e9.max_grad_norm == 30000]), SB(cell("separate_no_clip", "raw"))))
record("change", "E7 PopArt minus plain, S-B", *change(SB(e7all[e7all.critic_mode == "popart"]), SB(e7all[e7all.critic_mode == "plain"])))
p8 = e8[e8.part == "B"]
record("change", "E8 PopArt minus plain, S-B (seeds 52-61)", *change(SB(p8[p8.critic_mode == "popart"]), SB(p8[p8.critic_mode == "plain"])))
e10 = r910[r910.experiment == "E10"]
record("change", "E10 PopArt minus plain, S-B (seeds 62-81)", *change(SB(e10[e10.critic_mode == "popart"]), SB(e10[e10.critic_mode == "plain"])))
plain = pd.concat([cell("separate_no_clip", "raw").assign(critic_mode="plain"), p8, e10]); plain = plain[plain.critic_mode == "plain"]
pop = pd.concat([e7all[e7all.critic_mode == "popart"], p8[p8.critic_mode == "popart"], e10[e10.critic_mode == "popart"]])
record("change", "PopArt minus plain, S-B, pooled seeds 42-81", *change(SB(pop), SB(plain)))
c11 = lambda n: r11[r11.condition_name == n]
record("change", "E11 per-group minus shared clip, S-B", *change(SB(c11("cleanrl_group")), SB(c11("cleanrl_raw"))))

out = pd.DataFrame(rows)
# two-sided jackknife-t p-values and a Holm adjustment across the principal contrasts
out["p_jt"] = 2 * stats.t.sf(np.abs(out.difference / out.se_jack), out.n_seeds - 1)
pr = out[out.kind == "principal"].sort_values("p_jt"); m_ = len(pr); still = True; rej = {}
for i, (ix, r) in enumerate(pr.iterrows()):
    still = still and r.p_jt <= 0.05 / (m_ - i); rej[ix] = still
out["holm_jt_reject"] = out.index.map(rej)
out.to_csv(RES / "jackknife_v5.csv", index=False)
pd.set_option("display.width", 220)
show = out.assign(jt95=lambda d: d.apply(lambda r: f"[{r.jt_low_95:+.3f}, {r.jt_high_95:+.3f}]", axis=1),
                  jt99=lambda d: d.apply(lambda r: f"[{r.jt_low_99:+.3f}, {r.jt_high_99:+.3f}]", axis=1))
print(show[["kind", "contrast", "n_seeds", "difference", "jt95", "jt99", "jt_excludes_zero_95", "pct_excludes_zero_95"]].to_string(index=False))
ch = out[(out.kind == "principal") & (out.jt_excludes_zero_95 != out.pct_excludes_zero_95)]
print(f"\nprincipal verdicts that differ between percentile and jackknife-t at 95%: {len(ch)}")
