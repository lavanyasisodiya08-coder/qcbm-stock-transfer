"""
Pure unit tests for src/encode.py — no network required.
Run: python tests/test_encode.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.encode import bucket_return, encode_dataframe, bucket_to_index, index_to_bucket


def test_bucket_return_boundaries():
    assert bucket_return(-5.0) == "BIG_DOWN"
    assert bucket_return(-2.0) == "DOWN"       # boundary: exactly -2.0 -> DOWN (not < -2.0)
    assert bucket_return(-1.0) == "DOWN"
    assert bucket_return(-0.3) == "FLAT"        # boundary
    assert bucket_return(0.0) == "FLAT"
    assert bucket_return(0.3) == "UP"           # boundary
    assert bucket_return(1.5) == "UP"
    assert bucket_return(2.0) == "BIG_UP"       # boundary
    assert bucket_return(10.0) == "BIG_UP"
    print("test_bucket_return_boundaries PASSED")


def test_encode_dataframe_drops_first_row():
    df = pd.DataFrame({"Close": [100, 101, 99, 105]})
    encoded = encode_dataframe(df)
    assert len(encoded) == 3  # first row (no return) dropped
    assert "bucket" in encoded.columns
    print("test_encode_dataframe_drops_first_row PASSED")


def test_index_roundtrip():
    seq = ["BIG_DOWN", "FLAT", "BIG_UP", "UP", "DOWN"]
    idx = bucket_to_index(seq)
    back = index_to_bucket(idx)
    assert back == seq
    print("test_index_roundtrip PASSED")


if __name__ == "__main__":
    test_bucket_return_boundaries()
    test_encode_dataframe_drops_first_row()
    test_index_roundtrip()
    print("\nAll tests passed.")