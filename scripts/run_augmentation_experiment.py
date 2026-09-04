"""
Stage 2 experiment: does augmenting a data-scarce (small-cap) stock's
training sequence with synthetic data from a data-rich (large-cap) source
improve next-bucket prediction? Compares no augmentation, bootstrap,
classical Markov cross-transfer, and QCBM cross-transfer.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

import config
from src.encode import encode_dataframe, bucket_sequence
from src.baseline_model import LogisticBaseline
from src.evaluate import time_ordered_split, score_logistic
from src.qcbm_model import QCBM
from src.classical_augment import bootstrap_augment, markov_cross_augment


def load_sequence(group_name: str, name: str):
    path = config.DATA_RAW / f"{group_name}_{name}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Date"])
    encoded = encode_dataframe(df)
    return bucket_sequence(encoded)


def build_pooled_source_sequence(max_len_per_stock: int = 1000):
    """Pool large-cap sequences into one source sequence for the QCBM /
    classical Markov cross-transfer to learn from. Simple concatenation —
    small spurious transitions at stock boundaries are an accepted
    simplification for this MVP."""
    pooled = []
    for name in config.LARGE_CAP.keys():
        seq = load_sequence("large_cap", name)
        if seq is None:
            print(f"[SKIP] large_cap/{name} not found — run data pipeline first.")
            continue
        pooled.extend(seq[:max_len_per_stock])
    return pooled


def evaluate_augmented(train_seq, test_seq, augmented_train_seq, lookback=3):
    """Fit LogisticBaseline on augmented_train_seq, evaluate on test_seq
    using train_seq's tail for lookback context (so the test split itself
    never contains synthetic data)."""
    model = LogisticBaseline(lookback=lookback)
    return score_logistic(model, augmented_train_seq, test_seq)


def run_for_smallcap(name: str, source_seq: list, n_synthetic: int = 300):
    seq = load_sequence("small_cap", name)
    if seq is None:
        print(f"[SKIP] small_cap/{name} not found.")
        return []
    if len(seq) < 20:
        print(f"[SKIP] small_cap/{name} too short ({len(seq)} days).")
        return []

    train_seq, test_seq = time_ordered_split(seq)
    rows = []

    # --- Baseline: no augmentation ---
    try:
        result = evaluate_augmented(train_seq, test_seq, train_seq)
        rows.append({"ticker": name, "method": "no_augmentation", **result})
    except ValueError as e:
        print(f"[WARN] {name} no_augmentation skipped: {e}")

    # --- Bootstrap (resamples the target's own short history) ---
    synth = bootstrap_augment(train_seq, n_synthetic, seed=config.RANDOM_SEED)
    aug_seq = train_seq + synth
    try:
        result = evaluate_augmented(train_seq, test_seq, aug_seq)
        rows.append({"ticker": name, "method": "bootstrap", **result})
    except ValueError as e:
        print(f"[WARN] {name} bootstrap skipped: {e}")

    # --- Classical Markov cross-transfer (fit on large-cap, generate) ---
    synth = markov_cross_augment(source_seq, n_synthetic, seed=config.RANDOM_SEED)
    aug_seq = train_seq + synth
    try:
        result = evaluate_augmented(train_seq, test_seq, aug_seq)
        rows.append({"ticker": name, "method": "markov_cross_transfer", **result})
    except ValueError as e:
        print(f"[WARN] {name} markov_cross_transfer skipped: {e}")

    # --- QCBM cross-transfer (train quantum circuit on large-cap, generate) ---
    print(f"  Training QCBM for {name}...")
    qcbm = QCBM(n_layers=4, seed=config.RANDOM_SEED)
    qcbm.fit(source_seq, n_steps=150, verbose=False)
    synth = qcbm.generate_sequence(n_synthetic, seed=config.RANDOM_SEED)
    aug_seq = train_seq + synth
    try:
        result = evaluate_augmented(train_seq, test_seq, aug_seq)
        rows.append({"ticker": name, "method": "qcbm_cross_transfer", **result})
    except ValueError as e:
        print(f"[WARN] {name} qcbm_cross_transfer skipped: {e}")

    return rows


if __name__ == "__main__":
    print("Building pooled large-cap source sequence...")
    source_seq = build_pooled_source_sequence()
    print(f"Pooled source sequence length: {len(source_seq)}")

    all_rows = []
    for name in config.SMALL_CAP.keys():
        print(f"\n=== {name} ===")
        all_rows += run_for_smallcap(name, source_seq)

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(config.RESULTS / "augmentation_results.csv", index=False)

    print("\n=== Per-stock, per-method results ===")
    print(results_df.to_string(index=False))

    print("\n=== Mean by method (across small-cap stocks) ===")
    summary = results_df.groupby("method")[["accuracy", "macro_f1"]].agg(["mean", "std"]).round(4)
    print(summary)
    summary.to_csv(config.RESULTS / "augmentation_summary.csv")
