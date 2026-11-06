"""Methodological checks on the v4 estimator (added in revision; not pre-registered, and changing no reported number).

1. Output consistency: re-derives the headline contrasts from the per-run tables with the declared 2,000 seed-block
   draws (seed 20261015) and asserts that they equal the stored results CSVs.
2. Bootstrap Monte Carlo stability: recomputes the same contrasts with an independent set of 10,000 draws
   (seed 20261105) and reports the shift in each interval endpoint.
3. Interval coverage: simulates the design (ten seeds, the E1 grid, one evaluation label per seed and cell) from known
   logistic curves, with and without a per-seed random intercept shared across arms, and reports how often the
   seed-block percentile interval covers the true crossing and the true paired contrast.

Usage: stat_checks.py RUNS_DIR ARTIFACT_DIR RES_DIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUNS, ART, RES = map(Path, sys.argv[1:4])
sys.path.insert(0, str(ART / "AAMAS2027/analysis")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from boundary import firth  # noqa: E402
from labels import lbf_label, E4_THETA  # noqa: E402

SEEDS = list(range(42, 52))
W2K = np.random.default_rng(20261015).multinomial(10, np.full(10, 0.1), size=2000)
W10K = np.random.default_rng(20261105).multinomial(10, np.full(10, 0.1), size=10000)
G1 = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]
G5 = [0.2, 0.4, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 1.0]
G4 = [0.0, 0.1, 0.2, 0.3, 0.5, 0.65, 0.75, 0.85, 0.9, 0.95]
log = []
def say(s=""): print(s); log.append(s)


def matrix(frame, grid, col="collapsed"):
    seeds = sorted(frame.seed.unique()) if frame.seed.min() >= 52 else SEEDS   # E8 part B uses fresh seeds 52-61
    assert len(seeds) == 10
    k = frame.assign(w=frame.w.round(4)).pivot(index="seed", columns="w", values=col).reindex(index=seeds, columns=[round(x, 4) for x in grid])
    assert not k.isna().any().any(), "missing cells"
    return k.to_numpy(float)


def fit(k, grid, W):
    n = np.ones_like(k); w = np.array(grid)
    return firth(w, k.sum(0), n.sum(0))["wcrit"][0], firth(w, W @ k, W @ n)["wcrit"]


LEVELS = (0.95, 0.975, 0.99)


def pct(z, level=0.95):
    z = z[np.isfinite(z)]; a = 100 * (1 - level) / 2; return np.percentile(z, [a, 100 - a])


# ---------------------------------------------------------------- contrasts: (label, grid, frames for a and b, stored row)
e1 = pd.read_csv(RUNS / "E1_runs.csv"); e5 = pd.read_csv(RUNS / "E5_checkpoints.csv"); e6 = pd.read_csv(RUNS / "E6_runs.csv")
e7 = pd.read_csv(RUNS / "E7_runs.csv"); e4 = pd.read_csv(RUNS / "E4_runs.csv")
e4 = e4.assign(collapsed=lbf_label(e4.team_return, E4_THETA))
e7all = pd.concat([e1[(e1.optimizer_mode == "separate_no_clip") & (e1.reward_mode == "raw") & e1.arm.isin(["B", "A", "S"])].assign(critic_mode="plain"), e7])
c1, c5, c6, c7, c4 = (pd.read_csv(RES / f) for f in ["E1E2_contrasts.csv", "E5_contrasts.csv", "E6_contrasts.csv", "E7_contrasts.csv", "E4_contrasts.csv"])
i7 = pd.read_csv(RES / "E7_interactions.csv")
cell = lambda o, r: e1[(e1.optimizer_mode == o) & (e1.reward_mode == r)]
CON = []
for o, r, name in [("global_clip", "raw", "S-B"), ("separate_no_clip", "vecnormalize", "S-B"), ("separate_no_clip", "vecnormalize", "A-S"),
                   ("separate_no_clip", "vecnormalize", "A-B")]:
    hi, lo = name.split("-"); d = cell(o, r)
    CON.append((f"E1 {o}/{r} {name}", G1, d[d.arm == hi], d[d.arm == lo], c1[(c1.cell == f"{o}/{r}") & (c1.contrast == name)].iloc[0]))
for m, name in [("global_clip", "S-B"), ("groupwise_clip", "S-B"), ("groupwise_clip", "A-S")]:
    hi, lo = name.split("-"); d = e6[e6.optimizer_mode == m]
    CON.append((f"E6 {m} {name}", G1, d[d.arm == hi], d[d.arm == lo], c6[(c6["mode"] == m) & (c6.contrast == name)].iloc[0]))
for m, name in [("plain", "A-M"), ("popart", "S-B"), ("popart", "A-S"), ("popart", "A-M")]:
    hi, lo = name.split("-"); d = e7all[e7all.critic_mode == m]
    CON.append((f"E7 {m} {name}", G1, d[d.arm == hi], d[d.arm == lo], c7[(c7["mode"] == m) & (c7.contrast == name)].iloc[0]))
e8 = pd.read_csv(RUNS / "E8_runs.csv"); c8a = pd.read_csv(RES / "E8A_contrasts.csv"); c8b = pd.read_csv(RES / "E8B_contrasts.csv")
for m in ["plain", "popart"]:
    d = pd.concat([e7all[(e7all.critic_mode == m) & (e7all.arm == "A")], e8[(e8.critic_mode == m) & (e8.arm == "M_on")]])
    CON.append((f"E8 {m} A-M_on", G1, d[d.arm == "A"], d[d.arm == "M_on"], c8a[(c8a["mode"] == m) & (c8a.contrast == "A-M_on")].iloc[0]))
d = e8[(e8.part == "B") & (e8.critic_mode == "popart")]
CON.append(("E8B popart S-B", G1, d[d.arm == "S"], d[d.arm == "B"], c8b[c8b.cohort.str.startswith("seeds") & (c8b.quantity == "popart S-B")].iloc[0]))
d = e5[e5.checkpoint == 3600000]
CON.append(("E5 3.6M A-S", G5, d[d.arm == "A"], d[d.arm == "S"], c5[(c5.checkpoint == 3600000) & (c5.contrast == "A-S")].iloc[0]))
for o in ["global_clip", "separate_no_clip"]:
    d = e4[e4.optimizer_mode == o]
    CON.append((f"E4 {o} A-S", G4, d[d.arm == "A"], d[d.arm == "S"], c4[(c4.optimizer == o) & (c4.contrast == "A-S")].iloc[0]))

rows = []
say("== 1-2. Output consistency (declared 2,000 draws) and Monte Carlo stability (independent 10,000 draws)")
for label, grid, a, b, stored in CON:
    ka, kb = matrix(a, grid), matrix(b, grid)
    pa, ba = fit(ka, grid, W2K); pb, bb = fit(kb, grid, W2K)
    lo, hi = pct(ba - bb)
    assert abs((pa - pb) - stored.difference) < 1e-9 and abs(lo - stored.ci_low) < 1e-9 and abs(hi - stored.ci_high) < 1e-9, label
    _, ba10 = fit(ka, grid, W10K); _, bb10 = fit(kb, grid, W10K); lo10, hi10 = pct(ba10 - bb10)
    batches = np.array([pct(ba10[j::5] - bb10[j::5]) for j in range(5)])
    rows.append(dict(contrast=label, difference=pa - pb, ci_low_2k=lo, ci_high_2k=hi, ci_low_10k=lo10, ci_high_10k=hi10,
                     shift_low=lo10 - lo, shift_high=hi10 - hi, mcse_low=batches[:, 0].std(ddof=1) / np.sqrt(5),
                     mcse_high=batches[:, 1].std(ddof=1) / np.sqrt(5), sign_unchanged=bool((lo > 0) == (lo10 > 0) and (hi < 0) == (hi10 < 0))))
    say(f"  {label:38s} {pa - pb:+.3f}  2k [{lo:+.3f}, {hi:+.3f}]  10k [{lo10:+.3f}, {hi10:+.3f}]  reproduces stored: yes")
mc = pd.DataFrame(rows); mc.to_csv(RES / "stat_bootstrap_stability.csv", index=False)
say(f"  largest endpoint shift {np.abs(mc[['shift_low', 'shift_high']].to_numpy()).max():.3f}; "
    f"zero-exclusion verdict unchanged in {int(mc.sign_unchanged.sum())}/{len(mc)}")

# ---------------------------------------------------------------- 3. coverage
def sim(b0, b1, d0, d1, sd_seed, reps, rng):
    """Two arms on G1, ten seeds, one label per seed and cell; arm 2 = arm 1 + (d0, d1); a per-seed logit intercept
    (SD sd_seed) is shared by both arms, as a seed's initialisation is shared by paired runs."""
    w = np.array(G1); out = []
    t1, t2 = -b0 / b1, -(b0 + d0) / (b1 + d1)
    for _ in range(reps):
        u = rng.normal(0, sd_seed, size=(10, 1))
        k1 = (rng.random((10, len(w))) < 1 / (1 + np.exp(-(b0 + u + b1 * w)))).astype(float)
        k2 = (rng.random((10, len(w))) < 1 / (1 + np.exp(-(b0 + d0 + u + (b1 + d1) * w)))).astype(float)
        p1, q1 = fit(k1, G1, W2K); p2, q2 = fit(k2, G1, W2K)
        row = [np.isfinite(p1) and 0 <= p1 <= 1]
        for lev in LEVELS:
            l1, h1 = pct(q1, lev); lc, hc = pct(q2 - q1, lev)
            row += [l1 <= t1 <= h1, lc <= t2 - t1 <= hc, lc > 0 or hc < 0]
        out.append(row)
    o = np.array(out, float).mean(0); r = dict(true_w50=t1, true_contrast=t2 - t1, in_grid=o[0])
    for j, lev in enumerate(LEVELS):
        r.update({f"cover_w50_{lev}": o[1 + 3 * j], f"cover_contrast_{lev}": o[2 + 3 * j], f"excludes_zero_{lev}": o[3 + 3 * j]})
    return r


say("\n== 3. Coverage of seed-block percentile intervals (nominal 95%), 1,000 replicates per row")
rng = np.random.default_rng(20261106); rows = []
# slopes and intercepts in the range of the fitted E1/E7 curves (steep: b1 ~ -20; shallow: b1 ~ -8)
for name, b0, b1, d0, d1 in [("steep, null contrast", 8.0, -20.0, 0.0, 0.0), ("steep, contrast +0.10", 8.0, -20.0, 2.0, 0.0),
                              ("shallow, null contrast", 3.5, -8.0, 0.0, 0.0), ("shallow, contrast +0.20", 3.5, -8.0, 1.6, 0.0)]:
    for sd in [0.0, 1.0]:
        r = sim(b0, b1, d0, d1, sd, 1000, rng); rows.append(dict(scenario=name, seed_sd=sd, **r))
        say(f"  {name:24s} seed SD {sd:.0f}: true w50 {r['true_w50']:.3f}, contrast {r['true_contrast']:+.3f}; " + "; ".join(
            f"nominal {lev:.3f}: cover w50 {r[f'cover_w50_{lev}']:.3f}, contrast {r[f'cover_contrast_{lev}']:.3f}, excl. 0 {r[f'excludes_zero_{lev}']:.3f}" for lev in LEVELS))
cov = pd.DataFrame(rows); cov.to_csv(RES / "stat_coverage.csv", index=False)
# calibrated level: the smallest nominal level whose simulated contrast coverage is at least 0.94 in every scenario
CAL = next((lev for lev in LEVELS if cov[f"cover_contrast_{lev}"].min() >= 0.94), LEVELS[-1])
say(f"\n== 4. Key contrasts at the calibrated nominal level {CAL} (simulated contrast coverage >= 0.94 in every scenario)")
rows = []
for label, grid, a, b, stored in CON:
    pa, ba = fit(matrix(a, grid), grid, W2K); pb, bb = fit(matrix(b, grid), grid, W2K); lo, hi = pct(ba - bb, CAL)
    ex95, exc = bool(stored.ci_low > 0 or stored.ci_high < 0), bool(lo > 0 or hi < 0)
    rows.append(dict(contrast=label, difference=pa - pb, ci_low_95=stored.ci_low, ci_high_95=stored.ci_high, level=CAL, ci_low_cal=lo, ci_high_cal=hi,
                     excludes_zero_95=ex95, excludes_zero_cal=exc))
    say(f"  {label:38s} {pa - pb:+.3f}  95% [{stored.ci_low:+.3f}, {stored.ci_high:+.3f}]  {CAL:.1%} [{lo:+.3f}, {hi:+.3f}]"
        f"{'  verdict changes' if ex95 != exc else ''}")
pd.DataFrame(rows).to_csv(RES / "stat_calibrated_intervals.csv", index=False)

# ---------------------------------------------------------------- 5. multiplicity (Holm across the principal contrasts)
say("\n== 5. Holm adjustment across the principal contrasts (two-sided seed-block bootstrap p-values, alpha 0.05)")
rows = []
for label, grid, a_, b_, stored in CON:
    _, ba = fit(matrix(a_, grid), grid, W2K); _, bb = fit(matrix(b_, grid), grid, W2K); z = (ba - bb)[np.isfinite(ba - bb)]
    p_ = max(2 * min(np.mean(z <= 0), np.mean(z >= 0)), 1 / len(W2K))
    rows.append(dict(contrast=label, difference=stored.difference, p_boot=min(p_, 1.0)))
H = pd.DataFrame(rows).sort_values("p_boot").reset_index(drop=True); m_ = len(H)
H["holm_threshold"] = [0.05 / (m_ - i) for i in range(m_)]
rej = []; still = True
for r in H.itertuples():
    still = still and r.p_boot <= r.holm_threshold; rej.append(still)
H["holm_reject"] = rej
H["unadjusted_reject"] = H.p_boot <= 0.05
H.to_csv(RES / "stat_holm.csv", index=False)
for r in H.itertuples():
    say(f"  {r.contrast:38s} {r.difference:+.3f}  p {r.p_boot:.4f}  Holm threshold {r.holm_threshold:.4f}  {'reject' if r.holm_reject else 'retain'}")
(RES / "stat_checks_report.txt").write_text("\n".join(log) + "\n")
