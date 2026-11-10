"""Gradient-boosted trees on hourly features, with optional importance weighting
(a label-free domain-adaptation method) and Platt calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from .calibration import Platt

from .features import subsample_negatives


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class GBMModel:
    def __init__(self, neg_rate: float = 0.3, max_iter: int = 300, seed: int = 0):
        self.neg_rate = neg_rate
        # early_stopping is off on purpose: scikit-learn's default ("auto") would hold out a
        # random 10% of training *hours*, so hours of the same patient would sit on both sides
        # and the number of trees used would vary from run to run. Every model uses max_iter trees.
        self.clf = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=max_iter, max_leaf_nodes=31, min_samples_leaf=200,
            l2_regularization=1.0, early_stopping=False, random_state=seed)
        self.platt = None
        self.seed = seed

    def fit(self, X: pd.DataFrame, meta: pd.DataFrame, weight=None):
        idx = subsample_negatives(meta, self.neg_rate, self.seed)
        w = None if weight is None else pd.Series(weight, index=X.index).loc[idx].to_numpy()
        # drop measurements never recorded at the training hospital (e.g. EtCO2 at B)
        self.columns = [c for c in X.columns if X.loc[idx, c].notna().any()]
        self.clf.fit(X.loc[idx, self.columns], meta.loc[idx, "SepsisLabel"].to_numpy(), sample_weight=w)
        return self

    def raw_score(self, X):
        p = self.clf.predict_proba(X[self.columns])[:, 1]
        # undo the effect of having dropped some negative hours during training
        odds = p / np.clip(1 - p, 1e-9, None) * self.neg_rate
        return odds / (1 + odds)

    def calibrate(self, X, y):
        self.platt = Platt().fit(_logit(self.raw_score(X)), y)
        return self

    def predict(self, X):
        p = self.raw_score(X)
        if self.platt is None:
            return p
        return self.platt(_logit(p))


def importance_weights(X_src: pd.DataFrame, X_tgt: pd.DataFrame, n: int = 200_000,
                       clip=(0.1, 10.0), seed: int = 0) -> np.ndarray:
    """Weight source-hospital hours by how much they look like target-hospital hours.
    Uses no target labels: a classifier learns to tell the hospitals apart, and its
    odds become the weights."""
    rng = np.random.default_rng(seed)
    s = X_src.iloc[rng.choice(len(X_src), min(n, len(X_src)), replace=False)]
    t = X_tgt.iloc[rng.choice(len(X_tgt), min(n, len(X_tgt)), replace=False)]
    dom = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, min_samples_leaf=500,
                                         early_stopping=False, random_state=seed)
    cols = [c for c in s.columns if s[c].notna().any() and t[c].notna().any()]
    dom.fit(pd.concat([s[cols], t[cols]]), np.r_[np.zeros(len(s)), np.ones(len(t))])
    p = dom.predict_proba(X_src[cols])[:, 1]
    w = np.clip(p / np.clip(1 - p, 1e-6, None) * len(s) / len(t), *clip)
    return w / w.mean()
