import numpy as np
from sklearn.metrics import roc_auc_score

from sepsis.evaluate import PatientBootstrap, paired_difference, wilson


def _data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, 250, n)
    y = (rng.random(n) < 0.1).astype(int)
    weak = np.round(rng.random(n) + 0.2 * y, 2)        # rounded, so there are ties
    strong = rng.random(n) + 0.8 * y
    return y, groups, weak, strong


def test_weighted_auroc_matches_sklearn_with_ties():
    y, g, weak, _ = _data()
    b = PatientBootstrap(y, g, n_boot=1)
    b.counts = [np.ones(g.max() + 1, dtype=int)]       # a "draw" that keeps every patient once
    assert abs(b.aurocs(weak)[0] - roc_auc_score(y, weak)) < 1e-12


def test_interval_contains_estimate_and_paired_difference_detects_better_model():
    y, g, weak, strong = _data()
    b = PatientBootstrap(y, g, n_boot=200)
    auc, ci, _ = b.mean_auroc([weak])
    assert ci[0] < auc < ci[1]
    d = paired_difference(b, [weak], [strong])
    assert d["diff"] > 0 and d["diff_ci"][0] > 0
    same = paired_difference(b, [weak], [weak])
    assert same["diff"] == 0 and same["diff_ci"] == [0.0, 0.0]


def test_wilson_interval():
    lo, hi = wilson(8, 10)
    assert 0 <= lo < 80 < hi <= 100
    assert wilson(0, 10)[0] == 0
