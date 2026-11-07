import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from sepsis.evaluate import alert_metrics, threshold_for_catch_rate
from sepsis.pipeline import run


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    out = tmp_path_factory.mktemp("art")
    res = run("", out, synthetic=True, epochs=4, n_boot=10, limit=400, gbm_iter=80, log=lambda s: None)
    return out, res


def test_all_settings_ran(artifacts):
    _, res = artifacts
    assert len(res) == 20
    assert set(res.setting) == {"within", "pooled", "cross"}
    assert {"importance_wt", "dann", "site_normalized"} <= set(res.variant)


def test_models_beat_chance(artifacts):
    _, res = artifacts
    assert (res[res.model == "gbm"].AUROC > 0.75).all()
    assert res[res.model == "gru"].AUROC.median() > 0.65   # small data: some runs are noisy


def test_outputs_written(artifacts):
    out, _ = artifacts
    for f in ["REPORT.md", "results.csv", "cross_hospital.png", "calibration.png",
              "alert_tradeoff.png", "data_shift.png", "model/gbm.joblib", "model/gru.pt",
              "omop/measurement.parquet"]:
        assert (out / f).exists(), f


def test_threshold_hits_target_catch_rate():
    rng = np.random.default_rng(0)
    rows = []
    for pid in range(200):
        septic = pid < 40
        T = 30
        onset = 20.0 if septic else np.nan
        for h in range(T):
            rows.append({"person_id": pid, "hour": h, "onset_hour": onset})
    meta = pd.DataFrame(rows)
    p = rng.random(len(meta)) * 0.5 + np.where(meta.onset_hour.notna(), 0.3, 0)
    thr = threshold_for_catch_rate(meta, p, 80)
    assert alert_metrics(meta, p, thr)["septic_caught_pct"] >= 80
    assert alert_metrics(meta, p, thr + 1e-6)["septic_caught_pct"] < 80


def test_api_scores_each_hour(artifacts, monkeypatch):
    out, _ = artifacts
    monkeypatch.setenv("MODEL_DIR", str(out / "model"))
    from sepsis import api
    api.load.cache_clear()
    c = TestClient(api.app)
    assert c.get("/health").json()["status"] == "ok"
    body = {"age": 67, "sex": "F", "icu_unit": "MICU",
            "hours": [{"HR": 88, "Temp": 37.0}, {"HR": 110, "Resp": 26, "Lactate": 3.5}]}
    r = c.post("/score", json=body)
    assert r.status_code == 200
    hrs = r.json()["hours"]
    assert len(hrs) == 2 and all(0 <= h["risk"] <= 1 for h in hrs)
    bad = c.post("/score", json={**body, "hours": [{"Blood": 1}]})
    assert bad.status_code == 422
    api.load.cache_clear()


def test_calibration_never_reverses_ranking():
    from sepsis.calibration import Platt
    s = np.linspace(-3, 3, 200)
    y = (s < 0).astype(float)          # validation set where the score looks backwards
    cal = Platt().fit(s, y)
    out = cal(s)
    assert np.all(np.diff(out) >= 0)
    assert abs(out.mean() - y.mean()) < 0.01
