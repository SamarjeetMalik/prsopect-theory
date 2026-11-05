"""Label-generation tests (run: python -m pytest TMLR2026/v4_prereg/tests from the archive root)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import labels  # noqa: E402

RUNS = Path(__file__).resolve().parents[3] / "v4/runs"


def test_declared_threshold_matches_preregistration():
    labels.check_declared()
    assert labels.E4_THETA == 0.626


def test_lbf_rule_is_strict_less_than():
    assert labels.lbf_label([0.625, 0.626, 0.627]).tolist() == [1, 0, 0]


def test_lbf_rejects_undeclared_threshold_and_bad_input():
    with pytest.raises(AssertionError):
        labels.lbf_label([0.6], theta=0.55)
    with pytest.raises(AssertionError):
        labels.lbf_label([np.nan])
    with pytest.raises(AssertionError):
        labels.lbf_label([1.5])


def test_supply_rule_requires_both_clauses():
    assert labels.supply_label([9.9, 9.9, 10.0], [-127.1, -127.0, -128.0]).tolist() == [1, 0, 0]


@pytest.mark.skipif(not (RUNS / "E4_runs.csv").exists(), reason="per-run tables not present")
def test_e4_stored_labels_are_the_placeholder_and_relabel_counts():
    f = pd.read_csv(RUNS / "E4_runs.csv")
    assert (f.collapsed.astype(int) == labels.lbf_label(f.team_return, 0.5)).all()      # deviation D4
    counts = {t: int(labels.lbf_label(f.team_return, t).sum()) for t in labels.E4_THETA_SENSITIVITY}
    assert counts == {0.5: 176, 0.626: 290, 0.75: 393}


@pytest.mark.skipif(not (RUNS / "E1_runs.csv").exists(), reason="per-run tables not present")
@pytest.mark.parametrize("name", ["E1_runs.csv", "E2_runs.csv", "E6_runs.csv", "E7_runs.csv", "E5_checkpoints.csv"])
def test_supply_stored_labels_follow_declared_rule(name):
    labels.check_supply_table(pd.read_csv(RUNS / name))
