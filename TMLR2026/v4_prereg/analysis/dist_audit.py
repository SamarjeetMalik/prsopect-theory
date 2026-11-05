"""Training-reward and critic-target distribution audit for E7 (and the reused E1 plain-critic arms), and the E4
per-episode label audit. Added in revision; descriptive, not pre-registered; changes no reported estimate.

Reward quantiles of arm M are reconstructed exactly from the logged B reward and M's frozen affine map
(a > 0, so quantiles map monotonically). M's rollout-buffer returns are already in M's units, because the rollout
buffer stores the wrapped reward. The fraction of negative rewards is not recoverable for M from logged quantiles
and is reported only for B, A and S.

Stage 1 (extraction; needs the 45 MB per-rollout file, which is not archived):
    dist_audit.py reduce E1E7_rollout_distributions.csv.gz E7_DESIGN_JSON RUNS_DIR   -> RUNS_DIR/E7_reward_distributions.csv
Stage 2 (part of the one-command rebuild):
    dist_audit.py analyse RUNS_DIR RES_DIR   (reads E7_reward_distributions.csv and E4_label_audit.csv)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODE = sys.argv[1]
Q = ["p01", "p10", "p25", "p50", "p75", "p90", "p99"]
log = []
def say(s=""): print(s); log.append(s)

if MODE == "reduce":
    DIST, DESIGN, RUNS = sys.argv[2], sys.argv[3], Path(sys.argv[4])
    d = pd.read_csv(DIST)
    aff = json.loads(Path(DESIGN).read_text())["M_affine"]
    m = d.arm == "M"
    a = d.loc[m, "w"].map(lambda w: aff[f"{w:g}"]["scale"]); b = d.loc[m, "w"].map(lambda w: aff[f"{w:g}"]["shift"])
    d.loc[m, "r_mean"] = a * d.loc[m, "r_mean"] + b; d.loc[m, "r_std"] = a * d.loc[m, "r_std"]
    for q in Q:
        d.loc[m, "r_" + q] = a * d.loc[m, "r_" + q] + b
    d.loc[m, "r_neg"] = np.nan
    d["qskew"] = (d.r_p90 + d.r_p10 - 2 * d.r_p50) / (d.r_p90 - d.r_p10).replace(0, np.nan)   # Bowley-type, in [-1, 1]
    d["ret_cv"] = d.ret_std / d.ret_mean.abs().replace(0, np.nan)
    assert d.groupby(["critic_mode", "arm", "w", "seed"]).rollout.count().eq(469).all(), "every run must have 469 rollouts"
    cols = ["r_mean", "r_std", "r_p10", "r_p50", "r_p90", "qskew", "r_neg", "ret_mean", "ret_std"]
    d.groupby(["critic_mode", "arm", "w", "seed"])[cols].mean().reset_index().to_csv(RUNS / "E7_reward_distributions.csv", index=False)
    sys.exit(0)

RUNS, RES = Path(sys.argv[2]), Path(sys.argv[3]); E4A = RUNS / "E4_label_audit.csv"
OUT = Path(__file__).resolve().parents[2] / "generated/v4"
cols = ["r_mean", "r_std", "r_p10", "r_p50", "r_p90", "qskew", "r_neg", "ret_mean", "ret_std"]
per_run = pd.read_csv(RUNS / "E7_reward_distributions.csv")
assert len(per_run) == 640
pooled = per_run.groupby(["critic_mode", "arm"])[cols].mean()
pooled.to_csv(RES / "dist_audit_pooled.csv")
per_w = per_run.groupby(["critic_mode", "arm", "w"])[cols].mean()
per_w.to_csv(RES / "dist_audit_per_w.csv")
say("== Training-reward and critic-target distributions (per-rollout statistics averaged over 469 rollouts, then over runs)")
say(pooled.round(3).to_string())
say("\n== A vs M by critic: what an affine (scale-and-shift) control can and cannot match")
for mode in ["plain", "popart"]:
    A, M = pooled.loc[(mode, "A")], pooled.loc[(mode, "M")]
    say(f"  {mode:6s} mean A {A.r_mean:+.2f} M {M.r_mean:+.2f} | SD A {A.r_std:.2f} M {M.r_std:.2f} | p10 A {A.r_p10:+.2f} M {M.r_p10:+.2f} | "
        f"p90 A {A.r_p90:+.2f} M {M.r_p90:+.2f} | quantile skew A {A.qskew:+.3f} M {M.qskew:+.3f} | return SD A {A.ret_std:.1f} M {M.ret_std:.1f}")

e4 = pd.read_csv(E4A)
say(f"\n== E4 per-episode audit ({len(e4)} runs)")
say(f"  episodes per run: {sorted(e4.n_eps.unique())}; max |stored team_return - mean(stored episodes)| = {e4.abs_diff.max():.3e}")
say(f"  stored label == (stored team_return < 0.5): {int((e4.stored_collapsed == (e4.stored_team_return < 0.5).astype(int)).sum())}/{len(e4)}")
say(f"  per-episode native returns in [{e4.ep_min.min():.3f}, {e4.ep_max.max():.3f}]")
for th in [0.5, 0.626, 0.75]:
    say(f"  theta {th}: collapsed {int((e4.recomputed_mean < th).sum())} (from recomputed per-episode means)")
assert e4.abs_diff.max() < 1e-9 and (e4.n_eps == 10).all()
ties = int(((e4.stored_team_return < 0.75) != (e4.recomputed_mean < 0.75)).sum())
say(f"  runs whose theta-0.75 label depends on floating-point summation order (return exactly at 0.75): {ties}")
(RES / "dist_audit_report.txt").write_text("\n".join(log) + "\n")

ARM = {"B": "B (curvature)", "A": r"A (losses $\times7$)", "S": r"S (both $\times7$)", "M": "M (affine of B)"}
rows = []
for mode in ["plain", "popart"]:
    for arm in ["B", "A", "S", "M"]:
        x = pooled.loc[(mode, arm)]
        neg = "--" if np.isnan(x.r_neg) else f"{x.r_neg:.2f}"
        rows.append(f"{'Plain' if mode == 'plain' else 'PopArt'} & {ARM[arm]} & ${x.r_mean:+.1f}$ & {x.r_std:.1f} & ${x.r_p10:+.1f}$ & ${x.r_p50:+.1f}$ & "
                    f"${x.r_p90:+.1f}$ & ${x.qskew:+.2f}$ & {neg} & ${x.ret_mean:+.0f}$ & {x.ret_std:.0f} \\\\")
(OUT / "dist_table.tex").write_text(r"""\begin{table}[H]
\caption{E7 training-reward and critic-target distributions. Per-rollout statistics of the transformed training reward (pooled over the five nodes) and of the rollout-buffer returns that the critic regresses, averaged over the 469 rollouts of each run and then over the 80 runs of each arm (ten seeds, eight cells). Quantile skewness is $(p_{90}+p_{10}-2p_{50})/(p_{90}-p_{10})$. M's reward statistics are reconstructed exactly from the logged B reward and M's frozen affine map; its fraction of negative rewards is not recoverable from the logs. Descriptive; not pre-registered.}
\label{tab:dist}
\centering\scriptsize\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}llrrrrrrrrr@{}}
\toprule
& & \multicolumn{7}{c}{Training reward (per rollout)} & \multicolumn{2}{c}{Critic targets} \\
\cmidrule(lr){3-9}\cmidrule(l){10-11}
Critic & Arm & Mean & SD & $p_{10}$ & $p_{50}$ & $p_{90}$ & Q-skew & Frac.\ $<0$ & Mean & SD \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
def mac(n, v): return f"\\newcommand{{\\{n}}}{{{v}}}\n"
P, Q_ = pooled.loc["plain"], pooled.loc["popart"]
m = mac("VdaSkewA", f"${P.loc['A'].qskew:+.2f}$ (plain) and ${Q_.loc['A'].qskew:+.2f}$ (PopArt)")
m += mac("VdaSkewM", f"${P.loc['M'].qskew:+.2f}$ and ${Q_.loc['M'].qskew:+.2f}$")
m += mac("VdaRetMeanA", f"${P.loc['A'].ret_mean:+.0f}$"); m += mac("VdaRetMeanM", f"${P.loc['M'].ret_mean:+.0f}$")
m += mac("VdaRetSdA", f"{P.loc['A'].ret_std:.0f}"); m += mac("VdaRetSdM", f"{P.loc['M'].ret_std:.0f}")
m += mac("VdaETies", f"{ties}"); m += mac("VdaEMaxDiff", f"{e4.abs_diff.max():.0e}".replace("e-", "\\times10^{-") + "}")
(OUT / "dist_numbers.tex").write_text(m)
