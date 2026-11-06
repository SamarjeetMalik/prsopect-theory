#!/usr/bin/env python3
"""Rebuild every v4 analysis output and the manuscript from the anonymous artifact.

Run from the artifact root:  python TMLR2026/v4_prereg/analysis/rebuild_v4.py
Steps: (1) E1-E3, (2) E4 and (3) E5-E8 analysis from the per-run tables in v4/runs/, (4) the review figures and
leave-one-seed-out table, (5) estimator checks (output consistency, bootstrap stability, interval coverage) and the
revision figures, (6) render the v4 tables, figures and macros, (7) the legacy rebuild (re-analyses the legacy sweeps and compiles the PDF).
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
A = ROOT / "TMLR2026/v4_prereg/analysis"
RES = ROOT / "TMLR2026/v4_prereg/results"
RUNS = ROOT / "v4/runs"


def run(*args):
    print("+", " ".join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True)


run(sys.executable, A / "phase3.py", RUNS, ROOT, RES)
run(sys.executable, A / "phase3_e4.py", RUNS / "E4_runs.csv", ROOT, RES)
run(sys.executable, A / "phase3_e5e6.py", RUNS, RUNS / "E1_runs.csv", ROOT, RES)
run(sys.executable, A / "phase3_e7.py", RUNS / "E7_runs.csv", RUNS / "E1_runs.csv", ROOT, RES)
run(sys.executable, A / "phase3_e8.py", RUNS / "E8_runs.csv", RUNS / "E7_runs.csv", RUNS / "E1_runs.csv", ROOT, RES)
run(sys.executable, A / "review_figures.py", RUNS, ROOT, RES)
run(sys.executable, A / "stat_checks.py", RUNS, ROOT, RES)
run(sys.executable, A / "dist_audit.py", "analyse", RUNS, RES)
run(sys.executable, A / "eq2_check.py", RUNS, RES)
run(sys.executable, A / "revision_figures.py", RUNS, RES)
run(sys.executable, A / "render_v4.py")
run(sys.executable, ROOT / "AAMAS2027/analysis/rebuild_tmlr.py")
print("Rebuilt", ROOT / "TMLR2026/build/main.pdf")
