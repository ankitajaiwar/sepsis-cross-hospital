import numpy as np
import pandas as pd

from sepsis.active import BASELINES, STRATEGIES, caught_at_fixed_alerts, k_center, run, select


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
    assert set(STRATEGIES + BASELINES) <= set(res.strategy)
    for col in ["caught_at_fixed_alerts_pct", "tgt_septic_caught_pct", "hours_labelled"]:
        assert col in res.columns
    # threshold reset only never retrains, so its ranking equals the model as-is
    for d, g in res.groupby("direction"):
        as_is = g[(g.budget == 0) & (g.strategy == "random")].AUROC.iloc[0]
        assert np.allclose(g[(g.strategy == "recalibrate") & (g.budget > 0)].AUROC, as_is)
    # recalibrate and random label the same patients
    a = res[(res.strategy == "random") & (res.budget > 0)].sort_values(["direction", "budget"])
    b = res[(res.strategy == "recalibrate") & (res.budget > 0)].sort_values(["direction", "budget"])
    assert (a.hours_labelled.to_numpy() == b.hours_labelled.to_numpy()).all()
    preds = list((tmp_path / "checkpoints").glob("*/preds/A-B_seed0_hybrid.parquet"))
    assert preds and set(pd.read_parquet(preds[0]).columns) == {"b5", "b15"}
    assert (res.groupby(["direction", "strategy"]).budget.max().drop("all_target_labels", level=1) == 15).all()
    for f in ["ACTIVE_REPORT.md", "active_auroc.png", "active_caught.png", "active_caught_fixed.png",
              "active_caught_tgt.png", "active_summary.csv"]:
        assert (tmp_path / f).exists()


def test_caught_at_fixed_alerts():
    # two patients without sepsis (peak 0.2 and 0.8) and two septic (peak before onset 0.9 and 0.5)
    m = pd.DataFrame({"person_id": [1, 1, 2, 2, 3, 3, 4, 4], "hour": [0, 1] * 4,
                      "onset_hour": [np.nan] * 4 + [1, 1, 0, 0]})
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.4, 0.9, 0.5, 0.99])
    # alert half of the non-septic patients -> threshold at their median peak (0.5)
    assert caught_at_fixed_alerts(m, p, 50) == 100.0
    assert caught_at_fixed_alerts(m, p, 0) == 50.0      # threshold 0.8: only patient 3 flagged in time


def test_shared_features_drops_measurements_missing_at_one_hospital(tmp_path):
    # synthetic hospital B never records EtCO2, like the real data
    run(tmp_path, synthetic=True, limit=120, budgets=[5], seeds=(0,), gbm_iter=10,
        strategies=("random",), shared_features=True, log=lambda s: None)
    import json
    cfg = json.loads((tmp_path / "active_config.json").read_text())
    assert cfg["shared_features"] and any(f.startswith("EtCO2") for f in cfg["dropped_features"])
    assert "left out" in (tmp_path / "ACTIVE_REPORT.md").read_text()
