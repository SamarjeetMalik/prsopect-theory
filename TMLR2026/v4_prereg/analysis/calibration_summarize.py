"""Summarize the calibration study (calibration_v5.py) from its per-replicate outputs.

Usage: calibration_summarize.py CAL_DIR OUT_DIR

CAL_DIR holds cal_NN.json (scenario parameters, true crossings, run settings) and cal_NN.csv.gz (one row per replicate)
for every scenario, as written by `calibration_v5.py NN CAL_DIR`. Writes, in OUT_DIR:
  calibration_summary.csv    per scenario and interval procedure: coverage of the true contrast at 95% (_d95) and 99%
                             (_d99), coverage of arm 1's crossing at 95% (_w95), the rate at which the 95% interval
                             excludes zero (_x95; false-positive rate under a null contrast, power otherwise) and the
                             median 95% width (_wid); fail = share of replicates whose contrast is not finite, gfail = share
                             whose hierarchical-model fit failed. Rates are over replicates with a finite interval.
  calibration_scenarios.csv  each scenario's data-generating parameters and its true marginal crossings.
Added in revision (v7) to make the summary reproducible; it reproduces the summary used since v5 exactly.
"""
import glob
import json
import sys
from pathlib import Path

import pandas as pd

METHODS = ["pct", "bca", "jt", "glmm", "pboot"]


def main(cal, out):
    rows, scen = [], []
    for j in sorted(glob.glob(str(Path(cal) / "cal_*.json"))):
        m = json.load(open(j)); d = pd.read_csv(j.replace(".json", ".csv.gz")); s = m["scenario"]
        r = dict(i=m["index"], name=m["name"], n=s["n"], true_d=round(m["true_d"], 3), fail=round(d.fail_d.mean(), 3),
                 gfail=round(d.glmm_fail.mean(), 3))
        for meth in METHODS:
            r[meth + "_d95"] = round(d[f"{meth}_95_cov_d"].mean(), 3); r[meth + "_x95"] = round(d[f"{meth}_95_excl0"].mean(), 3)
            r[meth + "_w95"] = round(d[f"{meth}_95_cov_w1"].mean(), 3); r[meth + "_d99"] = round(d[f"{meth}_99_cov_d"].mean(), 3)
            r[meth + "_wid"] = round(d[f"{meth}_95_width_d"].median(), 3)
        rows.append(r)
        scen.append(dict(i=m["index"], name=m["name"], n=s["n"], link=s["link"], b0=s["b0"], b1=s["b1"], d0=s["d0"], d1=s["d1"],
                         sd_u=s["sd_u"], sd_v=s["sd_v"], rho=s["rho"], het=s["het"], true_w1=m["true_w1"], true_w2=m["true_w2"],
                         true_d=m["true_d"], reps=m["reps"], boot=m["boot"], pboot=m["pboot"], seed=m["seed"]))
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "calibration_summary.csv", index=False)
    pd.DataFrame(scen).to_csv(out / "calibration_scenarios.csv", index=False)
    print(f"summarized {len(rows)} scenarios")


if __name__ == "__main__":
    main(*sys.argv[1:3])
