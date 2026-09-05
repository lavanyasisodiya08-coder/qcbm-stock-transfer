"""
Significance testing on the multi-window results - now each (ticker,
window) combination is an observation, giving 30 paired samples per
method instead of 10, more statistical power to detect a real effect
if one exists.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from scipy import stats

import config


if __name__ == "__main__":
    df = pd.read_csv(config.RESULTS / "multi_window_results.csv")

    pivot = df.pivot_table(index=["ticker", "window"], columns="method", values="accuracy")
    baseline = pivot["no_augmentation"]
    methods = [c for c in pivot.columns if c != "no_augmentation"]

    rows = []
    for method in methods:
        treatment = pivot[method]
        diff = treatment - baseline
        t_stat, t_p = stats.ttest_rel(treatment, baseline)
        try:
            w_stat, w_p = stats.wilcoxon(treatment, baseline)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        rows.append({
            "method": method,
            "mean_diff_vs_baseline": round(diff.mean(), 4),
            "n_observations": len(diff),
            "paired_t_stat": round(t_stat, 4),
            "paired_t_pvalue": round(t_p, 4),
            "wilcoxon_pvalue": round(w_p, 4) if w_p == w_p else None,
            "significant_at_0.05": bool(t_p < 0.05),
        })

    sig_df = pd.DataFrame(rows)
    sig_df.to_csv(config.RESULTS / "multi_window_significance.csv", index=False)
    print(sig_df.to_string(index=False))
