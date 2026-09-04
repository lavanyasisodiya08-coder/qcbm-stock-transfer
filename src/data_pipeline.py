"""
Fetch and cache daily OHLCV data via yfinance.
Gracefully skips any ticker that fails (bad symbol, delisted, no data).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yfinance as yf

import config


def fetch_ticker(ticker: str, start: str, end: str, retries: int = 2) -> pd.DataFrame | None:
    """Fetch OHLCV for one ticker. Returns None (with a warning) on failure."""
    for attempt in range(retries + 1):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty:
                print(f"[WARN] No data returned for {ticker} — skipping.")
                return None
            df = df.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df["ticker"] = ticker
            return df
        except Exception as e:
            print(f"[WARN] Attempt {attempt + 1} failed for {ticker}: {e}")
            time.sleep(1.5)
    print(f"[WARN] Giving up on {ticker} after {retries + 1} attempts — skipping.")
    return None


def fetch_group(group: dict[str, str], group_name: str) -> dict[str, pd.DataFrame]:
    out = {}
    for name, ticker in group.items():
        cache_path = config.DATA_RAW / f"{group_name}_{name}.csv"
        if cache_path.exists():
            print(f"[CACHE] {name} ({ticker}) — loading from disk.")
            out[name] = pd.read_csv(cache_path, parse_dates=["Date"])
            continue

        print(f"[FETCH] {name} ({ticker}) from Yahoo Finance...")
        df = fetch_ticker(ticker, config.START_DATE, config.END_DATE)
        if df is None:
            continue
        df.to_csv(cache_path, index=False)
        out[name] = df

    print(f"[{group_name}] fetched {len(out)}/{len(group)} tickers successfully.")
    return out


if __name__ == "__main__":
    large = fetch_group(config.LARGE_CAP, "large_cap")
    small = fetch_group(config.SMALL_CAP, "small_cap")
    print("Large-cap tickers loaded:", list(large.keys()))
    print("Small-cap tickers loaded:", list(small.keys()))
