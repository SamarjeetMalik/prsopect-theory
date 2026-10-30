"""Decreasing isotonic 50% crossing, using the convention of the legacy reanalysis (reanalyze.py).

Weighted pool-adjacent-violators fit of the observed collapse proportions; the crossing is linear interpolation
between the last cell with fitted probability above .5 and the first below it; a plateau at exactly .5 returns the
midpoint of the plateau. Returns -inf when every fitted value is below .5 (crossing below the grid), +inf when
every fitted value is at least .5 (crossing above the grid).
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression


def isotonic_crossing(w, k, n):
    w = np.asarray(w, float); k = np.asarray(k, float); n = np.asarray(n, float)
    p = IsotonicRegression(increasing=False).fit_transform(w, k / n, sample_weight=n)
    eq = np.isclose(p, .5, atol=1e-12)
    if eq.any():
        return float((w[eq].min() + w[eq].max()) / 2)
    if (p < .5).all():
        return -np.inf
    if (p > .5).all():
        return np.inf
    j = int(np.flatnonzero(p < .5)[0])
    return float(w[j - 1] + (.5 - p[j - 1]) * (w[j] - w[j - 1]) / (p[j] - p[j - 1]))
