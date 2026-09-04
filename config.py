"""
Central config: tickers, date ranges, bucket thresholds, paths.
"""

from pathlib import Path

ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

for p in (DATA_RAW, DATA_PROCESSED, RESULTS):
    p.mkdir(parents=True, exist_ok=True)

LARGE_CAP = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "INFY": "INFY.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "SBIN": "SBIN.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
}

SMALL_CAP = {
    "LEELAHOTELS": "THELEELA.NS",
    "AEGISVOPAK": "AEGISVOPAK.NS",
}

# ---- Sector map (broad parent categories so sector_match can register
# a "same" when two tickers are in a related, not identical, sub-sector) ----
SECTOR = {
    "RELIANCE": "Energy",
    "TCS": "IT",
    "HDFCBANK": "Banking",
    "INFY": "IT",
    "ICICIBANK": "Banking",
    "HINDUNILVR": "FMCG",
    "SBIN": "Banking",
    "BHARTIARTL": "Telecom",
    "LEELAHOTELS": "Hospitality",
    "AEGISVOPAK": "Energy",
}

START_DATE = "2010-01-01"
END_DATE = "2026-09-03"

BUCKET_THRESHOLDS = {
    "big_down": -2.0,
    "down": -0.3,
    "up": 0.3,
    "big_up": 2.0,
}
BUCKET_LABELS = ["BIG_DOWN", "DOWN", "FLAT", "UP", "BIG_UP"]

TEST_FRACTION = 0.2
RANDOM_SEED = 42
