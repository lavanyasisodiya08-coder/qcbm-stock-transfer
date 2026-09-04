"""
Turn daily % returns into discrete return-bucket tokens.
Same idea as turning a melody into a sequence of note tokens:
each day's price move becomes one of 5 symbols.
"""

import numpy as np
import pandas as pd

import config


def compute_returns(df: pd.DataFrame, price_col: str = "Close") -> pd.Series:
    """Daily % returns from a price column."""
    return df[price_col].pct_change() * 100.0


def bucket_return(pct_return: float) -> str:
    """Map a single % return to one of the 5 bucket labels."""
    t = config.BUCKET_THRESHOLDS
    if pd.isna(pct_return):
        return None
    if pct_return < t["big_down"]:
        return "BIG_DOWN"
    elif pct_return < t["down"]:
        return "DOWN"
    elif pct_return < t["up"]:
        return "FLAT"
    elif pct_return < t["big_up"]:
        return "UP"
    else:
        return "BIG_UP"


def encode_dataframe(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """
    Adds 'pct_return' and 'bucket' columns to a raw OHLCV dataframe.
    Drops the first row (no return defined).
    """
    out = df.copy()
    out["pct_return"] = compute_returns(out, price_col)
    out["bucket"] = out["pct_return"].apply(bucket_return)
    out = out.dropna(subset=["bucket"]).reset_index(drop=True)
    return out


def bucket_sequence(df_encoded: pd.DataFrame) -> list[str]:
    """Extract just the ordered list of bucket tokens."""
    return df_encoded["bucket"].tolist()


def bucket_to_index(seq: list[str]) -> np.ndarray:
    """Convert bucket-label sequence to integer indices (0-4) for modeling."""
    label_to_idx = {label: i for i, label in enumerate(config.BUCKET_LABELS)}
    return np.array([label_to_idx[b] for b in seq])


def index_to_bucket(indices) -> list[str]:
    """Inverse of bucket_to_index."""
    return [config.BUCKET_LABELS[i] for i in indices]


def bigram_counts(seq: list[str]) -> pd.DataFrame:
    """Count transitions bucket[t] -> bucket[t+1]. Used by the Markov baseline
    and as the target distribution the QCBM will later learn (Phase 2)."""
    labels = config.BUCKET_LABELS
    counts = pd.DataFrame(0, index=labels, columns=labels)
    for a, b in zip(seq[:-1], seq[1:]):
        counts.loc[a, b] += 1
    return counts