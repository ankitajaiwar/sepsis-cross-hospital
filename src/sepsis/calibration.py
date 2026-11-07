"""Platt scaling that can only rescale scores, never reverse their order."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class Platt:
    def fit(self, score: np.ndarray, y: np.ndarray) -> "Platt":
        score, y = np.asarray(score, float), np.asarray(y, float)
        lr = LogisticRegression(C=1e6).fit(score[:, None], y)
        a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        if a <= 0:
            # A flat or negative slope would scramble the ranking (this happens with a
            # weak model on a small validation set). Keep the order and only shift the
            # level so the average prediction matches the observed rate.
            a, lo, hi, target = 1.0, -50.0, 50.0, y.mean()
            for _ in range(60):
                b = (lo + hi) / 2
                if (1 / (1 + np.exp(-(score + b)))).mean() < target:
                    lo = b
                else:
                    hi = b
        self.a, self.b = a, b
        return self

    def __call__(self, score) -> np.ndarray:
        return 1 / (1 + np.exp(-(self.a * np.asarray(score, float) + self.b)))
