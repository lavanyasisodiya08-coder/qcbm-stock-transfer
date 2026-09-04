import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

import config
from src.encode import encode_dataframe, bucket_sequence
from src.baseline_model import MarkovBaseline, LogisticBaseline
from src.evaluate import time_ordered_split, score_markov, score_logistic, summarize_by_group


def run_for_group(group_name: str, tickers: dict) -> list[dict]:
    rows = []
    for name in tickers.keys():
        path = config.DATA_RAW / f"{group_name}_{name}.csv"
        if not path.exists():
            print(f"[SKIP] {name} - no cached data at {path}")
            continue

        df = pd.read_csv(path, parse_dates=["Date"])
        encoded = encode_dataframe(df)
        seq = bucket_sequence(encoded)

        if len(seq) < 20:
            print(f"[SKIP] {name} - too short ({len(seq)} days) for a meaningful split")
            continue

        train_seq, test_seq = time_ordered_split(seq)

        markov_result = score_markov(MarkovBaseline(), train_seq, test_seq)
        rows.append({"ticker": name, "group": group_name, "model": "Markov", **markov_result})

        try:
            logistic_result = score_logistic(LogisticBaseline(lookback=3), train_seq, test_seq)
            rows.append({"ticker": name, "group": group_name, "model": "Logistic", **logistic_result})
        except ValueError as e:
            print(f"[WARN] Logistic skipped for {name}: {e}")

    return rows


if __name__ == "__main__":
    all_rows = []
    all_rows += run_for_group("large_cap", config.LARGE_CAP)
    all_rows += run_for_group("small_cap", config.SMALL_CAP)

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(config.RESULTS / "baseline_results.csv", index=False)

    summary = summarize_by_group(results_df)
    summary.to_csv(config.RESULTS / "summary_by_group.csv")

    print()
    print("=== Per-stock results ===")
    print(results_df.to_string(index=False))
    print()
    print("=== Summary by group ===")
    print(summary)
