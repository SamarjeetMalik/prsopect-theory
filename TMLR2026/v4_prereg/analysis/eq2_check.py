"""Per-minibatch check of the shared-clip formula (Eq. 2) against logged applied actor norms. Added in revision;
descriptive, not pre-registered; changes no reported estimate.

Input: v4/runs/E1E6_eq2_minibatch.csv. Each row is one run of E1 global_clip/raw (arms B, S) or E6 (all modes and
arms), summarizing its per-minibatch pre-clip actor, critic and total norms and its logged post-clip actor norm. The
predicted applied norm is evaluated per minibatch, then averaged. Ratios S/B are ratios of sums over the 80 paired
runs of each arm.

Usage: eq2_check.py RUNS_DIR RES_DIR
"""
import sys
from pathlib import Path

import pandas as pd

RUNS, RES = Path(sys.argv[1]), Path(sys.argv[2])
OUT = Path(__file__).resolve().parents[2] / "generated/v4"
d = pd.read_csv(RUNS / "E1E6_eq2_minibatch.csv")
assert len(d) == 880 and (d.n_mb > 0).all()
rows = []
for (exp, mode), g in d.groupby(["exp", "optimizer_mode"]):
    s, b = g[g.arm == "S"], g[g.arm == "B"]
    rows.append(dict(exp=exp, mode=mode, n_runs_B=len(b), n_runs_S=len(s),
                     measured_ratio_SB=s.measured_post_actor.sum() / b.measured_post_actor.sum(),
                     predicted_ratio_SB=s.predicted_post_actor.sum() / b.predicted_post_actor.sum(),
                     max_rel_err_minibatch=g.max_abs_rel_err.max(),
                     critic_share_B=b.critic_share_mean.mean(), critic_share_S=s.critic_share_mean.mean(),
                     clip_frac_B=b.clip_frac.mean(), clip_frac_S=s.clip_frac.mean()))
R = pd.DataFrame(rows); R.to_csv(RES / "eq2_check.csv", index=False)
print(R.round(4).to_string(index=False))
g = R[(R.exp == "E1") & (R["mode"] == "global_clip")].iloc[0]; gw = R[(R.exp == "E6") & (R["mode"] == "groupwise_clip")].iloc[0]
def mac(n, v): return f"\\newcommand{{\\{n}}}{{{v}}}\n"
m = mac("VeqPred", f"{g.predicted_ratio_SB:.3f}") + mac("VeqMeas", f"{g.measured_ratio_SB:.3f}")
m += mac("VeqMaxErr", f"{R[R['mode'] != 'separate_no_clip'].max_rel_err_minibatch.max():.0e}".replace("e-0", r"\times10^{-").replace("e-", r"\times10^{-") + "}")
m += mac("VeqShareB", f"{g.critic_share_B:.4f}") + mac("VeqShareS", f"{g.critic_share_S:.5f}")
m += mac("VeqGroupPred", f"{gw.predicted_ratio_SB:.3f}") + mac("VeqGroupMeas", f"{gw.measured_ratio_SB:.3f}")
(OUT / "eq2_numbers.tex").write_text(m); print(m)
