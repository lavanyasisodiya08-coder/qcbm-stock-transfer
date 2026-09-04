"""
Time-ordered train/test split (never shuffled — that would leak the future
into training), accuracy + macro-F1, grouped summaries.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import config


def time_ordered_split(seq: list[str], test_fraction: float = None):
    """Split a bucket sequence into train/test, preserving order."""
    if test_fraction is None:
        test_fraction = config.TEST_FRACTION
    n = len(seq)
    split_idx = int(n * (1 - test_fraction))
    return seq[:split_idx], seq[split_idx:]


def score_markov(model, train_seq: list[str], test_seq: list[str]) -> dict:
    model.fit(train_seq)
    preds = model.predict_sequence(test_seq)
    true = test_seq[1:]  # predict_sequence predicts t+1 for each t in test_seq[:-1]
    return {
        "accuracy": accuracy_score(true, preds),
        "macro_f1": f1_score(true, preds, average="macro", zero_division=0),
        "n_test": len(true),
    }


def score_logistic(model, train_seq: list[str], test_seq: list[str]) -> dict:
    model.fit(train_seq)
    # need lookback context from train_seq to predict the start of test_seq
    combined = train_seq[-model.lookback:] + test_seq
    preds = model.predict_sequence(combined)
    true = model.true_labels_for_sequence(combined)
    return {
        "accuracy": accuracy_score(true, preds),
        "macro_f1": f1_score(true, preds, average="macro", zero_division=0),
        "n_test": len(true),
    }


def summarize_by_group(results_df: pd.DataFrame) -> pd.DataFrame:
    """results_df must have columns: ticker, group, model, accuracy, macro_f1"""
    return (
        results_df.groupby(["group", "model"])[["accuracy", "macro_f1"]]
        .agg(["mean", "std"])
        .round(4)
    )