"""Expanded calibration study of interval procedures for the w50 crossing and paired crossing contrasts (revision v5;
not pre-registered, changes no stored estimate).

The design simulated is the paper's: n training seeds, one binary label per seed and mixture cell, the E1 grid (or a
stated grid), two arms paired by seed. Data-generating scenarios vary curve steepness, link (logistic or a monotone
non-logistic link), per-seed random intercepts and slopes, their correlation across arms, arm-specific heterogeneity,
contrast size, and crossings near the grid edge. The truth for each arm is the 50% crossing of the population-averaged
(marginal) collapse curve, which is what the pooled fit estimates; it is computed numerically.

Interval procedures compared, each at nominal 95% and 99%:
  pct   seed-block percentile bootstrap (the paper's procedure; Firth fits, multinomial seed weights)
  bca   bias-corrected and accelerated seed-block bootstrap (acceleration from the leave-one-seed-out jackknife)
  jt    jackknife-t: leave-one-seed-out standard error with t(n-1) quantiles
  glmm  hierarchical logistic model (arm-specific intercept and slope, a per-seed random intercept shared by both
        arms; Gauss-Hermite marginal likelihood, weak ridge penalty for separation); Wald interval by the delta method
  pboot parametric bootstrap from the fitted hierarchical model, re-estimated with the paper's pooled Firth fit;
        percentile interval

Usage: calibration_v5.py SCENARIO_INDEX OUT_DIR [--reps R] [--boot B] [--pboot P] [--seed S]
       calibration_v5.py --list
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import optimize, special, stats

import os
for _d in [os.environ.get("BOUNDARY_DIR", ""), str(Path(__file__).resolve().parents[3] / "AAMAS2027/analysis")]:
    if _d and (Path(_d) / "boundary.py").exists():
        sys.path.insert(0, _d)
from boundary import firth  # noqa: E402

G1 = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]
LEVELS = (0.95, 0.99)


def scenarios():
    """(name, dict). b0/b1: arm-1 logit intercept and slope on w; d0/d1: arm-2 shifts; sd_u/sd_v: per-seed intercept and
    slope SDs (arm 1); het: arm-2 SD multiplier; rho: cross-arm correlation of seed effects; link; n seeds."""
    S = []
    base = dict(b0=8.0, b1=-20.0, d0=0.0, d1=0.0, sd_u=0.0, sd_v=0.0, het=1.0, rho=1.0, link="logit", n=10, grid=G1)
    def add(name, **kw):
        s = dict(base); s.update(kw); S.append((name, s))
    for shape, b0, b1, d0 in [("steep", 8.0, -20.0, 2.0), ("shallow", 3.5, -8.0, 1.6)]:
        for eff, dd in [("null", 0.0), ("effect", d0)]:
            add(f"{shape}-{eff}-iid", b0=b0, b1=b1, d0=dd)
            add(f"{shape}-{eff}-u1", b0=b0, b1=b1, d0=dd, sd_u=1.0)
            add(f"{shape}-{eff}-u2", b0=b0, b1=b1, d0=dd, sd_u=2.0)
    add("steep-null-u1-v", sd_u=1.0, sd_v=4.0)
    add("steep-effect-u1-v", d0=2.0, sd_u=1.0, sd_v=4.0)
    add("steep-null-u1-rho0.5", sd_u=1.0, rho=0.5)
    add("steep-null-u1-rho0", sd_u=1.0, rho=0.0)
    add("steep-null-u1-het2", sd_u=1.0, het=2.0)
    add("steep-null-cloglog-u1", link="cloglog", sd_u=1.0)
    add("steep-effect-cloglog-u1", link="cloglog", d0=2.0, sd_u=1.0)
    add("edge-low-null-u1", b0=2.0, b1=-20.0, sd_u=1.0)          # crossing near w = 0.1
    add("edge-high-null-u1", b0=19.0, b1=-20.0, sd_u=1.0)        # crossing near w = 0.95
    for n in (20, 30):
        add(f"steep-null-u1-n{n}", sd_u=1.0, n=n)
        add(f"steep-effect-u1-n{n}", d0=2.0, sd_u=1.0, n=n)
        add(f"shallow-effect-u1-n{n}", b0=3.5, b1=-8.0, d0=1.6, sd_u=1.0, n=n)
    return S


def inv_link(eta, link):
    return special.expit(eta) if link == "logit" else -np.expm1(-np.exp(eta))


def marginal_w50(b0, b1, sd_u, sd_v, link, rng):
    u = rng.normal(0, sd_u, 400_000) if sd_u > 0 else np.zeros(1); v = rng.normal(0, sd_v, len(u)) if sd_v > 0 else np.zeros(len(u))
    f = lambda w: inv_link(b0 + u + (b1 + v) * w, link).mean() - 0.5
    return optimize.brentq(f, -2.0, 3.0, xtol=1e-7)


def simulate(s, rng):
    n, w = s["n"], np.array(s["grid"])
    u1 = rng.normal(0, s["sd_u"], (n, 1)); v1 = rng.normal(0, s["sd_v"], (n, 1))
    z_u, z_v = rng.normal(0, 1, (n, 1)), rng.normal(0, 1, (n, 1))
    r = s["rho"]; c = np.sqrt(max(1 - r * r, 0))
    u2 = s["het"] * (r * u1 + c * s["sd_u"] * z_u); v2 = s["het"] * (r * v1 + c * s["sd_v"] * z_v)
    p1 = inv_link(s["b0"] + u1 + (s["b1"] + v1) * w, s["link"])
    p2 = inv_link(s["b0"] + s["d0"] + u2 + (s["b1"] + s["d1"] + v2) * w, s["link"])
    return (rng.random((n, len(w))) < p1).astype(float), (rng.random((n, len(w))) < p2).astype(float)


def pooled(k, w, W=None):
    """Firth crossing from the pooled table, and from seed-weighted tables if W is given."""
    n = np.ones_like(k)
    if W is None:
        return firth(w, k.sum(0), n.sum(0))["wcrit"][0]
    return firth(w, W @ k, W @ n)["wcrit"]


def pct(z, level):
    z = z[np.isfinite(z)]
    if len(z) < 20:
        return np.nan, np.nan
    a = 100 * (1 - level) / 2
    return tuple(np.percentile(z, [a, 100 - a]))


def bca(z, est, jack, level):
    z = z[np.isfinite(z)]; jack = jack[np.isfinite(jack)]
    if len(z) < 20 or len(jack) < 3 or not np.isfinite(est):
        return np.nan, np.nan
    prop = np.clip(np.mean(z < est) + 0.5 * np.mean(z == est), 1 / len(z), 1 - 1 / len(z)); z0 = stats.norm.ppf(prop)
    m = jack.mean(); num = ((m - jack) ** 3).sum(); den = 6 * (((m - jack) ** 2).sum()) ** 1.5
    a = num / den if den > 0 else 0.0
    out = []
    for q in [(1 - level) / 2, 1 - (1 - level) / 2]:
        zq = stats.norm.ppf(q); adj = stats.norm.cdf(z0 + (z0 + zq) / (1 - a * (z0 + zq)))
        out.append(np.percentile(z, 100 * np.clip(adj, 0, 1)))
    return tuple(out)


def jack_t(est, jack, level):
    jack = jack[np.isfinite(jack)]; n = len(jack)
    if n < 3 or not np.isfinite(est):
        return np.nan, np.nan
    se = np.sqrt((n - 1) / n * ((jack - jack.mean()) ** 2).sum()); t = stats.t.ppf(1 - (1 - level) / 2, n - 1)
    return est - t * se, est + t * se


GH_X, GH_W = np.polynomial.hermite_e.hermegauss(15)
GH_W = GH_W / GH_W.sum()


def glmm_fit(k1, k2, w):
    """Two arms, logit p = a_j + b_j * zw + u_s, u_s ~ N(0, s^2) shared by the seed's two arms; zw is standardized w.
    Ridge penalty N(0, 10^2) on (a_j, b_j) for separation. Returns params, covariance (inverse Hessian) or None."""
    mu, sd = w.mean(), w.std(); zw = (w - mu) / sd
    def nll(th):
        a1, b1, a2, b2, ls = th; s = np.exp(ls)
        e1 = a1 + b1 * zw; e2 = a2 + b2 * zw
        uu = s * GH_X[:, None, None]
        l1 = k1[None] * (e1 + uu) - np.logaddexp(0, e1 + uu); l2 = k2[None] * (e2 + uu) - np.logaddexp(0, e2 + uu)
        per_seed = l1.sum(2) + l2.sum(2)                       # nodes x seeds
        ll = special.logsumexp(per_seed + np.log(GH_W)[:, None], axis=0).sum()
        return -ll + ((th[:4] ** 2).sum()) / 200.0 + (ls - np.log(0.5)) ** 2 / 8.0
    th0 = np.array([0.0, -2.0, 0.0, -2.0, np.log(0.5)])
    r = optimize.minimize(nll, th0, method="L-BFGS-B")
    th = r.x; h = 1e-4; H = np.zeros((5, 5))
    for i in range(5):
        for j in range(i, 5):
            ei, ej = np.eye(5)[i] * h, np.eye(5)[j] * h
            H[i, j] = H[j, i] = (nll(th + ei + ej) - nll(th + ei - ej) - nll(th - ei + ej) + nll(th - ei - ej)) / (4 * h * h)
    try:
        cov = np.linalg.inv(H)
        if not np.all(np.isfinite(cov)) or np.any(np.diag(cov) <= 0):
            cov = None
    except np.linalg.LinAlgError:
        cov = None
    return th, cov, mu, sd


def glmm_contrast(th, cov, mu, sd, level):
    """Delta-method Wald interval for the conditional (u = 0) crossing contrast, which equals the marginal one under a
    symmetric random intercept and logit link."""
    a1, b1, a2, b2, _ = th
    if cov is None or abs(b1) < 1e-6 or abs(b2) < 1e-6:
        return np.nan, np.nan, np.nan
    w1, w2 = mu - sd * a1 / b1, mu - sd * a2 / b2
    g = np.array([-sd / b1, sd * a1 / b1 ** 2, sd / b2, -sd * a2 / b2 ** 2, 0.0])
    se = np.sqrt(max(g @ cov @ g, 0)); zq = stats.norm.ppf(1 - (1 - level) / 2); d = w2 - w1
    return d, d - zq * se, d + zq * se


def glmm_w1(th, cov, mu, sd, level):
    a1, b1 = th[0], th[1]
    if cov is None or abs(b1) < 1e-6:
        return np.nan, np.nan, np.nan
    w1 = mu - sd * a1 / b1; g = np.array([-sd / b1, sd * a1 / b1 ** 2, 0, 0, 0]); se = np.sqrt(max(g @ cov @ g, 0))
    zq = stats.norm.ppf(1 - (1 - level) / 2)
    return w1, w1 - zq * se, w1 + zq * se


def pboot(th, mu, sd, w, n, P, rng):
    a1, b1, a2, b2, ls = th; s = np.exp(ls); zw = (w - mu) / sd
    u = rng.normal(0, s, (P, n, 1))
    k1 = (rng.random((P, n, len(w))) < special.expit(a1 + b1 * zw + u)).astype(float)
    k2 = (rng.random((P, n, len(w))) < special.expit(a2 + b2 * zw + u)).astype(float)
    nn = np.full((P, len(w)), float(n))
    f1 = firth(w, k1.sum(1), nn)["wcrit"]; f2 = firth(w, k2.sum(1), nn)["wcrit"]
    return f1, f2 - f1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("index", nargs="?", type=int); ap.add_argument("out", nargs="?")
    ap.add_argument("--reps", type=int, default=10000); ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--pboot", type=int, default=500); ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--list", action="store_true"); a = ap.parse_args()
    S = scenarios()
    if a.list:
        for i, (nm, s) in enumerate(S):
            print(i, nm, s["n"])
        return
    name, s = S[a.index]; rng = np.random.default_rng([a.seed, a.index]); w = np.array(s["grid"]); n = s["n"]
    t1 = marginal_w50(s["b0"], s["b1"], s["sd_u"], s["sd_v"], s["link"], rng)
    t2 = marginal_w50(s["b0"] + s["d0"], s["b1"] + s["d1"], s["sd_u"] * s["het"], s["sd_v"] * s["het"], s["link"], rng)
    W = np.random.default_rng([a.seed, 7]).multinomial(n, np.full(n, 1 / n), size=a.boot)
    J = 1 - np.eye(n)                                          # leave-one-seed-out weights
    rows = []; t0 = time.monotonic()
    for rep in range(a.reps):
        k1, k2 = simulate(s, rng)
        e1, e2 = pooled(k1, w), pooled(k2, w); d = e2 - e1
        b1_, b2_ = pooled(k1, w, W), pooled(k2, w, W); bd = b2_ - b1_
        j1, j2 = pooled(k1, w, J), pooled(k2, w, J); jd = j2 - j1
        row = dict(rep=rep, est_w1=e1, est_d=d, in_grid_w1=bool(np.isfinite(e1) and w.min() <= e1 <= w.max()),
                   fail_w1=bool(not np.isfinite(e1)), fail_d=bool(not np.isfinite(d)), boot_finite=float(np.isfinite(bd).mean()))
        th, cov, mu, sdw = glmm_fit(k1, k2, w); pf1, pfd = pboot(th, mu, sdw, w, n, a.pboot, rng)
        row["glmm_fail"] = cov is None
        for lev in LEVELS:
            L = f"{int(round(lev * 100))}"
            iv = dict(pct=(pct(b1_, lev), pct(bd, lev)), bca=(bca(b1_, e1, j1, lev), bca(bd, d, jd, lev)),
                      jt=(jack_t(e1, j1, lev), jack_t(d, jd, lev)),
                      glmm=(glmm_w1(th, cov, mu, sdw, lev)[1:], glmm_contrast(th, cov, mu, sdw, lev)[1:]),
                      pboot=(pct(pf1, lev), pct(pfd, lev)))
            for m, ((l1, h1), (lc, hc)) in iv.items():
                row[f"{m}_{L}_cov_w1"] = bool(l1 <= t1 <= h1) if np.isfinite(l1) else np.nan
                row[f"{m}_{L}_cov_d"] = bool(lc <= t2 - t1 <= hc) if np.isfinite(lc) else np.nan
                row[f"{m}_{L}_excl0"] = bool(lc > 0 or hc < 0) if np.isfinite(lc) else np.nan
                row[f"{m}_{L}_width_d"] = hc - lc
        rows.append(row)
    import pandas as pd
    df = pd.DataFrame(rows); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"cal_{a.index:02d}.csv.gz", index=False)
    meta = dict(index=a.index, name=name, scenario={k: v for k, v in s.items()}, true_w1=t1, true_w2=t2, true_d=t2 - t1,
                reps=a.reps, boot=a.boot, pboot=a.pboot, seed=a.seed, seconds=time.monotonic() - t0)
    (out / f"cal_{a.index:02d}.json").write_text(json.dumps(meta, indent=2, default=float))
    print(json.dumps(meta, default=float))


if __name__ == "__main__":
    main()
