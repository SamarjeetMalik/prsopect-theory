"""Collapse labels, with the declared thresholds asserted against the pre-registration text.

Added after deviation D4: the E4 runner read its LBF threshold from a calibration file that was missing from the
frozen source and silently fell back to 0.5. Analysis code no longer trusts a stored label or a default. Every label
is recomputed here from stored evaluation statistics. Each threshold is a module constant checked against the
sentence in PREREGISTRATION_v4.md that declares it, and any non-finite input or out-of-range threshold aborts.
"""
from pathlib import Path
import re

import numpy as np

PREREG = Path(__file__).resolve().parents[1] / "PREREGISTRATION_v4.md"
E4_THETA = 0.626                  # LBF: collapsed iff native team return (mean of ten evaluation episodes) < theta
E4_THETA_SENSITIVITY = (0.5, 0.626, 0.75)
SUPPLY_S, SUPPLY_PROFIT = 10.0, -127.1   # supply chain: collapsed iff mean_S < 10 and profit <= -127.1


def _declared_lbf_theta(text):
    m = re.findall(r"θ = ([0-9.]+) native team return", text)
    assert len(m) == 1, f"expected one declared LBF threshold in {PREREG.name}, found {m}"
    return float(m[0])


def check_declared():
    """Abort unless the module constants equal the pre-registered values."""
    text = PREREG.read_text(encoding="utf-8")
    assert _declared_lbf_theta(text) == E4_THETA, "E4_THETA differs from the pre-registered threshold"
    assert E4_THETA in E4_THETA_SENSITIVITY


def lbf_label(team_return, theta=E4_THETA):
    r = np.asarray(team_return, dtype=float)
    assert np.isfinite(r).all(), "non-finite team return"
    assert (r >= 0).all() and (r <= 1.0 + 1e-9).all(), "native LBF team return must lie in [0, 1]"
    assert theta in E4_THETA_SENSITIVITY, f"theta {theta} is not a declared or sensitivity threshold"
    return (r < theta).astype(int)


def supply_label(mean_S, profit):
    s, p = np.asarray(mean_S, dtype=float), np.asarray(profit, dtype=float)
    assert np.isfinite(s).all() and np.isfinite(p).all(), "non-finite evaluation statistic"
    return ((s < SUPPLY_S) & (p <= SUPPLY_PROFIT)).astype(int)


def check_supply_table(frame):
    """Stored supply-chain labels must equal the declared rule applied to the stored statistics."""
    got = supply_label(frame.mean_S, frame.profit)
    bad = int((got != frame.collapsed.astype(int).to_numpy()).sum())
    assert bad == 0, f"{bad} stored supply-chain labels disagree with the declared rule"


check_declared()
