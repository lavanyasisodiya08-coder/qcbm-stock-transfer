"""
Conditions analysis: instead of pooling all large-cap sources together,
train QCBM and classical Markov transfer on EACH large-cap source
individually, then check whether transfer helps more when the source
is sector-matched or volatility-matched to the target small-cap.
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
from src.classical_augment import markov_cross_augment
from src.conditions import (
    sector_match,
    compute_volatility,
    compute_volatility_thresholds,
    volatility_bucket,
)

N_SYNTHETIC = 300
QCBM_STEPS = 100


def load_encoded(group_name: str, name: str):
    path = config.DATA_RAW / f"{group_name}_{name}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Date"])
    return encode_dataframe(df)


def evaluate_augmented(train_seq, test_seq, augmented_train_seq, lookback=3):
    model = LogisticBaseline(lookback=lookback)
    return score_logistic(model, augmented_train_seq, test_seq)


if __name__ == "__main__":
    sources = {}
    for name in config.LARGE_CAP.keys():
        enc = load_encoded("large_cap", name)
        if enc is None:
            print(f"[SKIP] large_cap/{name} not found.")
            continue
        sources[name] = {
            "seq": bucket_sequence(enc),
            "volatility": compute_volatility(enc["pct_return"]),
        }

    low_cut, high_cut = compute_volatility_thresholds(
        [v["volatility"] for v in sources.values()]
    )
    for name, info in sources.items():
        info["vol_bucket"] = volatility_bucket(info["volatility"], low_cut, high_cut)

    print("Source volatility buckets:")
    for name, info in sources.items():
        print(f"  {name}: std={info['volatility']:.4f} -> {info['vol_bucket']}")

    rows = []
    for target_name in config.SMALL_CAP.keys():
        enc = load_encoded("small_cap", target_name)
        if enc is None:
            print(f"[SKIP] small_cap/{target_name} not found.")
            continue
        seq = bucket_sequence(enc)
        if len(seq) < 20:
            print(f"[SKIP] small_cap/{target_name} too short ({len(seq)} days).")
            continue
        train_seq, test_seq = time_ordered_split(seq)

        try:
            base_result = evaluate_augmented(train_seq, test_seq, train_seq)
            rows.append({
                "source": "NONE", "target": target_name, "method": "no_augmentation",
                "sector_match": "n/a", "source_vol_bucket": "n/a",
                **base_result,
            })
        except ValueError as e:
            print(f"[WARN] {target_name} no_augmentation skipped: {e}")

        for source_name, info in sources.items():
            print(f"  {source_name} -> {target_name} ...")
            sec = sector_match(source_name, target_name)
            vol_bucket = info["vol_bucket"]

            synth = markov_cross_augment(info["seq"], N_SYNTHETIC, seed=config.RANDOM_SEED)
            aug_seq = train_seq + synth
            try:
                result = evaluate_augmented(train_seq, test_seq, aug_seq)
                rows.append({
                    "source": source_name, "target": target_name,
                    "method": "markov_cross_transfer",
                    "sector_match": sec, "source_vol_bucket": vol_bucket,
                    **result,
                })
            except ValueError as e:
                print(f"[WARN] {source_name}->{target_name} markov skipped: {e}")

            qcbm = QCBM(n_layers=4, seed=config.RANDOM_SEED)
            qcbm.fit(info["seq"], n_steps=QCBM_STEPS, verbose=False)
            synth = qcbm.generate_sequence(N_SYNTHETIC, seed=config.RANDOM_SEED)
            aug_seq = train_seq + synth
            try:
                result = evaluate_augmented(train_seq, test_seq, aug_seq)
                rows.append({
                    "source": source_name, "target": target_name,
                    "method": "qcbm_cross_transfer",
                    "sector_match": sec, "source_vol_bucket": vol_bucket,
                    **result,
                })
            except ValueError as e:
                print(f"[WARN] {source_name}->{target_name} qcbm skipped: {e}")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(config.RESULTS / "conditions_results.csv", index=False)

    print("\n=== All source/target/method results ===")
    print(results_df.to_string(index=False))

    print("\n=== Mean accuracy by sector_match (transfer methods only) ===")
    transfer_only = results_df[results_df["method"] != "no_augmentation"]
    print(transfer_only.groupby(["method", "sector_match"])["accuracy"].agg(["mean", "std", "count"]).round(4))

    print("\n=== Mean accuracy by source volatility bucket (transfer methods only) ===")
    print(transfer_only.groupby(["method", "source_vol_bucket"])["accuracy"].agg(["mean", "std", "count"]).round(4))
