import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

import config


class MarkovBaseline:
    def __init__(self):
        self.transition_probs = None

    def fit(self, seq: list[str]):
        labels = config.BUCKET_LABELS
        counts = pd.DataFrame(0, index=labels, columns=labels, dtype=float)
        for a, b in zip(seq[:-1], seq[1:]):
            counts.loc[a, b] += 1
        row_sums = counts.sum(axis=1)
        row_sums[row_sums == 0] = 1
        self.transition_probs = counts.div(row_sums, axis=0)
        return self

    def predict(self, current_bucket: str) -> str:
        if self.transition_probs is None:
            raise RuntimeError("Call fit() first.")
        if current_bucket not in self.transition_probs.index:
            return "FLAT"
        return self.transition_probs.loc[current_bucket].idxmax()

    def predict_sequence(self, seq: list[str]) -> list[str]:
        return [self.predict(b) for b in seq[:-1]]


class LogisticBaseline:
    def __init__(self, lookback: int = 3):
        self.lookback = lookback
        self.model = LogisticRegression(max_iter=1000)
        self.encoder = OneHotEncoder(
            categories=[config.BUCKET_LABELS] * lookback,
            sparse_output=False,
            handle_unknown="ignore",
        )
        self._fitted = False

    def _make_features(self, seq: list[str]):
        X_raw, y = [], []
        for i in range(self.lookback, len(seq)):
            window = seq[i - self.lookback:i]
            X_raw.append(window)
            y.append(seq[i])
        return X_raw, y

    def fit(self, seq: list[str]):
        X_raw, y = self._make_features(seq)
        if len(X_raw) == 0:
            raise ValueError("Sequence too short for the chosen lookback.")
        X = self.encoder.fit_transform(X_raw)
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict_sequence(self, seq: list[str]) -> list[str]:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        X_raw, _ = self._make_features(seq)
        X = self.encoder.transform(X_raw)
        return self.model.predict(X).tolist()

    def true_labels_for_sequence(self, seq: list[str]) -> list[str]:
        _, y = self._make_features(seq)
        return y
