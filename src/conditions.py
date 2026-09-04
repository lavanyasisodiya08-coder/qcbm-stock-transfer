"""
Conditions analysis utilities: sector similarity and volatility-regime
labels for each (source, target) stock pair, used to check whether
cross-stock transfer helps more under some conditions than others.
"""

import numpy as np
import pandas as pd

import config


def sector_match(source_name: str, target_name: str) -> str:
    """'same' or 'different' sector label for a source/target pair."""
    s = config.SECTOR.get(source_name)
    t = config.SECTOR.get(target_name)
    if s is None or t is None:
        return "unknown"
    return "same" if s == t else "different"


def compute_volatility(pct_returns: pd.Series) -> float:
    """Std dev of daily % returns - simple volatility measure."""
    return float(pct_returns.std())


def compute_volatility_thresholds(vols: list) -> tuple:
    """Tertile cutoffs across a list of source volatilities."""
    arr = np.array(vols)
    return float(np.quantile(arr, 1 / 3)), float(np.quantile(arr, 2 / 3))


def volatility_bucket(vol: float, low_cut: float, high_cut: float) -> str:
    if vol < low_cut:
        return "low_vol"
    elif vol < high_cut:
        return "medium_vol"
    else:
        return "high_vol"
