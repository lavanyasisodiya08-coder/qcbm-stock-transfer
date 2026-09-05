"""
Multi-window robustness check: repeats the augmentation comparison across
3 different chronological train/test cutoffs per stock (not just one
80/20 split), to see whether the earlier null result holds across time
periods or was specific to one split. The QCBM and classical-Markov
source model are trained once on the full pooled large-cap sequence and
reused across windows (a simplification - a stricter walk-forward design
would retrain per-window using only large-cap data available up to that
point; noted as a limitation for the paper).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

import config
from src.encode import encode_dataframe, bucket_sequence
from src.baseline_model import LogisticBaseline
from src.evaluate import score_logistic
from src.qcbm_model import QCBM
from src.classical_augment import bootstrap_augment, markov_cross_augment

WINDOWS = [0.15, 0.20, 0.25]  # different test-set fractions = different chronological cutoffs


def load_sequence(group_name, name):
    path = config.DATA_RAW / f"{group_name}_{name}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Date"])
    return bucket_sequence(encode_dataframe(df))


def split_at(seq, test_fraction):
    n = len(seq)
    idx = int(n * (1 - test_fraction))
    return seq[:idx], seq[idx:]


def build_large_cap_pool():
    pooled = []
    for name in config.LARGE_CAP.keys():
        seq = load_sequence("large_cap", name)
        if seq:
            pooled.extend(seq)
    return pooled


def evaluate(train_seq, test_seq, augmented_train_seq, lookback=3):
    model = LogisticBaseline(lookback=lookback)
    return score_logistic(model, augmented_train_seq, test_seq)


if __name__ == "__main__":
    print("Building pooled large-cap source and training QCBM once...")
    source_seq = build_large_cap_pool()
    qcbm = QCBM(n_layers=4, seed=config.RANDOM_SEED)
    qcbm.fit(source_seq, n_steps=150, verbose=False)
    print(f"  Source pool size: {len(source_seq)} tokens. QCBM trained.")

    all_rows = []
    for window_frac in WINDOWS:
        print(f"\n--- Window: test_fraction={window_frac} ---")
        for name in config.SMALL_CAP.keys():
            seq = load_sequence("small_cap", name)
            if seq is None or len(seq) < 20:
                continue
            train_seq, test_seq = split_at(seq, window_frac)
            n_synth = len(train_seq)

            try:
                r = evaluate(train_seq, test_seq, train_seq)
                all_rows.append({"ticker": name, "window": window_frac, "method": "no_augmentation", **r})
            except ValueError:
                continue

            boot = bootstrap_augment(train_seq, n_synth, seed=config.RANDOM_SEED)
            r = evaluate(train_seq, test_seq, train_seq + boot)
            all_rows.append({"ticker": name, "window": window_frac, "method": "bootstrap", **r})

            mk = markov_cross_augment(source_seq, n_synth, start_bucket=train_seq[-1], seed=config.RANDOM_SEED)
            r = evaluate(train_seq, test_seq, train_seq + mk)
            all_rows.append({"ticker": name, "window": window_frac, "method": "markov_cross_transfer", **r})

            qc = qcbm.generate_sequence(n_synth, start_bucket=train_seq[-1], seed=config.RANDOM_SEED)
            r = evaluate(train_seq, test_seq, train_seq + qc)
            all_rows.append({"ticker": name, "window": window_frac, "method": "qcbm_cross_transfer", **r})

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(config.RESULTS / "multi_window_results.csv", index=False)

    # Average each stock's accuracy across the 3 windows per method,
    # then summarize - this is the "does it hold up over time" view.
    per_stock_avg = results_df.groupby(["ticker", "method"])["accuracy"].mean().reset_index()
    per_stock_avg.to_csv(config.RESULTS / "multi_window_per_stock_avg.csv", index=False)

    summary = results_df.groupby(["window", "method"])[["accuracy", "macro_f1"]].mean().round(4).reset_index()
    summary.to_csv(config.RESULTS / "multi_window_summary.csv", index=False)

    print("\n=== Mean accuracy by window and method ===")
    print(summary.to_string(index=False))
