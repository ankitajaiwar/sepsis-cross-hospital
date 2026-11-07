import numpy as np
import pandas as pd

from sepsis.active import STRATEGIES, k_center, run, select


def test_k_center_spreads_points():
    pts = np.array([[0, 0], [0.1, 0], [10, 10], [10.1, 10], [-10, 5]], dtype=float)
    picks = k_center(pts, 3, seed=0)
    assert len(set(picks)) == 3
    assert {2, 3} & set(picks) and 4 in picks          # covers the far clusters


def test_select_returns_unlabelled_patients():
    pool = pd.Index([1, 2, 3, 4, 5, 6])
    rows = pd.Series(np.repeat(pool.to_numpy(), 3))
    p = pd.Series(np.linspace(0.01, 0.9, len(rows)))
    emb = pd.DataFrame(np.random.default_rng(0).normal(size=(6, 4)), index=pool)
    dom = pd.Series(np.linspace(0, 1, 6), index=pool)
    for s in STRATEGIES:
        got = select(s, 3, pool, p, rows, emb, dom, [], seed=1)
        assert len(got) == 3 and set(got) <= set(pool) and len(set(got)) == 3
    assert set(select("high_risk", 2, pool, p, rows, emb, dom, [], 0)) == {5, 6}


def test_active_run_end_to_end(tmp_path):
    res = run(tmp_path, synthetic=True, limit=200, budgets=[5, 15], seeds=(0,),
              gbm_iter=30, log=lambda s: None)
    assert set(res.direction) == {"A->B", "B->A"}
    assert set(STRATEGIES) <= set(res.strategy)
    assert (res.groupby(["direction", "strategy"]).budget.max().drop("all_target_labels", level=1) == 15).all()
    for f in ["ACTIVE_REPORT.md", "active_auroc.png", "active_caught.png", "active_summary.csv"]:
        assert (tmp_path / f).exists()
