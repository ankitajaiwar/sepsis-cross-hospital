import numpy as np
import pandas as pd

from sepsis import active, active_compare
from sepsis.evaluate import PatientBootstrap


def test_fixed_alert_bootstrap_matches_report_metric():
    rng = np.random.default_rng(0)
    rows = []
    for pid in range(60):
        n = int(rng.integers(5, 15))
        onset = float(rng.integers(2, n)) if pid % 5 == 0 else np.nan
        rows += [{"person_id": pid, "hour": h, "onset_hour": onset,
                  "SepsisLabel": int(not np.isnan(onset) and h >= onset - 6)} for h in range(n)]
    m = pd.DataFrame(rows)
    p = rng.random(len(m)) + 0.3 * m["SepsisLabel"].to_numpy()
    boot = PatientBootstrap(m["SepsisLabel"], m["person_id"], n_boot=20, seed=1)
    fab = active_compare.FixedAlertBootstrap(m, boot)
    assert np.isclose(fab.point(p), active.caught_at_fixed_alerts(m, p))
    assert fab.draws(p).shape == (20,)


def test_compare_runs_end_to_end(tmp_path):
    active.run(tmp_path, synthetic=True, limit=150, budgets=[10, 30], seeds=(0, 1), gbm_iter=10,
               strategies=("random", "hybrid", "recalibrate"), crossfit_folds=0, log=lambda s: None)
    res = active_compare.run(tmp_path, n_boot=20, log=lambda s: None)
    assert {"A->B", "B->A"} == set(res.direction)
    assert {"Random vs model as-is", "Hybrid vs Random"} <= set(res.comparison)
    assert res.p_holm.between(0, 1).all() and (res.diff_lo <= res.diff_hi).all()
    assert (tmp_path / "ACTIVE_COMPARISONS.md").exists() and (tmp_path / "active_comparisons.csv").exists()
    # a method compared with itself shows no difference
    m, preds = active_compare.load_preds(active_compare.find_pred_dir(tmp_path), "A-B")
    preds[("copy", 30)] = preds[("random", 30)]
    r = active_compare.compare_direction(m, preds, n_boot=20)
    same = r[r.method == "copy"]
    assert np.allclose(same["diff"], 0) and np.allclose(same[["diff_lo", "diff_hi"]], 0)
