import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

import config


def generate_fake_ohlcv(name: str, n_days: int, seed: int, drift: float = 0.0005, vol: float = 0.02) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(drift, vol, n_days)
    prices = 100 * np.exp(np.cumsum(log_returns))
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
    df = pd.DataFrame({
        "Date": dates,
        "Open": prices * (1 - rng.uniform(0, 0.005, n_days)),
        "High": prices * (1 + rng.uniform(0, 0.01, n_days)),
        "Low": prices * (1 - rng.uniform(0, 0.01, n_days)),
        "Close": prices,
        "Volume": rng.integers(1_000_000, 10_000_000, n_days),
        "ticker": name,
    })
    return df


if __name__ == "__main__":
    for i, name in enumerate(config.LARGE_CAP.keys()):
        df = generate_fake_ohlcv(name, n_days=3800, seed=100 + i)
        df.to_csv(config.DATA_RAW / f"large_cap_{name}.csv", index=False)

    for i, name in enumerate(config.SMALL_CAP.keys()):
        df = generate_fake_ohlcv(name, n_days=190, seed=200 + i, vol=0.035)
        df.to_csv(config.DATA_RAW / f"small_cap_{name}.csv", index=False)

    print("Synthetic demo data written to data/raw/")
