"""Build e8_targets.json from the per-rollout distribution extraction, before any E8 run.

For each critic mode and mixture weight, the target at rollout r is the seed average of A's per-rollout mean
training reward, and the root-mean-square over seeds of A's within-rollout SD. plain: E1 separate_no_clip/raw A;
popart: E7 popart A.

Usage: e8_build_targets.py E1E7_rollout_distributions.csv.gz OUT_JSON
"""
import json
import sys

import numpy as np
import pandas as pd

d = pd.read_csv(sys.argv[1]); a = d[d.arm == "A"]
out = {}
for mode, src in [("plain", "E1"), ("popart", "E7")]:
    x = a[(a.critic_mode == mode) & (a.src == src)]
    out[mode] = {}
    for w, g in x.groupby("w"):
        assert g.seed.nunique() == 10 and g.groupby("seed").rollout.count().eq(469).all(), (mode, w)
        t = g.groupby("rollout").agg(mean=("r_mean", "mean"), sd=("r_std", lambda s: float(np.sqrt(np.mean(np.square(s))))))
        out[mode][f"{w:g}"] = dict(mean=t["mean"].round(8).tolist(), sd=t["sd"].round(8).tolist())
assert all(len(out[m]) == 8 for m in out)
open(sys.argv[2], "w").write(json.dumps(out) + "\n")
print("targets written", {m: sorted(out[m]) for m in out})
