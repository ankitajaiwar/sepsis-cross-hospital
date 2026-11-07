"""Run every train/test setting for both model types and collect results.

Settings
- within: train and test at the same hospital (different patients)
- pooled: train on both hospitals, test on each
- cross:  train at one hospital, test at the other, with these variants
    source_only     : no adjustment
    site_normalized : each hospital's values expressed relative to its own norms
    importance_wt   : (trees) source hours weighted to resemble the target hospital
    dann            : (neural net) domain-adversarial training on unlabeled target data
Thresholds are always chosen on the training hospital's validation patients, never
on the test hospital, so cross-hospital results show what would really happen.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .evaluate import alert_metrics, hourly_metrics, threshold_for_catch_rate
from .gbm import GBMModel, importance_weights
from .seq import GRUModel

META_COLS = ["person_id", "hour", "onset_hour", "SepsisLabel", "site", "split",
             "Age", "Gender", "Unit1", "Unit2"]


def _sel(meta, sites, split):
    return meta.index[meta["site"].isin(list(sites)) & (meta["split"] == split)]


def _score(name, model_type, setting, variant, train_sites, test_site, meta, p_test,
           test_idx, thr, n_boot, extra=None):
    m = meta.loc[test_idx]
    res = hourly_metrics(m["SepsisLabel"].to_numpy(), p_test, m["person_id"].to_numpy(), n_boot)
    res.update(alert_metrics(m, p_test, thr))
    res.update({"run": name, "model": model_type, "setting": setting, "variant": variant,
                "train": "+".join(train_sites), "test": test_site})
    if extra:
        res.update(extra)
    return res


def run_all(df: pd.DataFrame, X_raw: pd.DataFrame, X_norm: pd.DataFrame, df_norm: pd.DataFrame,
            models=("gbm", "gru"), epochs: int = 8, n_boot: int = 200, catch_target: float = 80.0,
            gbm_iter: int = 300, log=print):
    meta = df[META_COLS]
    results, keep = [], {}
    plans = [("within", ("A",), ["A"]), ("within", ("B",), ["B"]), ("pooled", ("A", "B"), ["A", "B"]),
             ("cross", ("A",), ["B"]), ("cross", ("B",), ["A"])]

    for mt in models:
        for setting, src, tests in plans:
            if setting == "cross":
                variants = ["source_only", "site_normalized"] + (["importance_wt"] if mt == "gbm" else ["dann"])
            else:
                variants = ["standard"]
            for var in variants:
                t0 = time.time()
                tr, va = _sel(meta, src, "train"), _sel(meta, src, "valid")
                X = X_norm if var == "site_normalized" else X_raw
                D = df_norm if var == "site_normalized" else df
                tgt_train = _sel(meta, tests, "train") if setting == "cross" else None
                if mt == "gbm":
                    w = importance_weights(X.loc[tr], X.loc[tgt_train]) if var == "importance_wt" else None
                    model = GBMModel(max_iter=gbm_iter).fit(X.loc[tr], meta.loc[tr], w)
                    model.calibrate(X.loc[va], meta.loc[va, "SepsisLabel"].to_numpy())
                    pred = lambda idx: model.predict(X.loc[idx])
                else:
                    model = GRUModel(epochs=epochs, dann=0.3 if var == "dann" else 0.0)
                    model.fit(D.loc[tr], D.loc[va], D.loc[tgt_train] if var == "dann" else None)
                    model.calibrate(D.loc[va])
                    pred = lambda idx: model.predict(D.loc[idx]).reindex(idx).to_numpy()
                thr = threshold_for_catch_rate(meta.loc[va], pred(va), catch_target)
                for ts in tests:
                    te = _sel(meta, [ts], "test")
                    p = pred(te)
                    name = f"{mt}:{setting}:{'+'.join(src)}->{ts}:{var}"
                    results.append(_score(name, mt, setting, var, src, ts, meta, p, te, thr, n_boot))
                    keep[name] = pd.Series(p, index=te)
                    r = results[-1]
                    log(f"{name:45s} AUROC {r['AUROC']:.3f}  caught {r['septic_caught_pct']:.0f}%  "
                        f"({time.time() - t0:.0f}s)")
                if setting == "pooled":
                    keep[f"{mt}:model"] = model
                    keep[f"{mt}:threshold"] = thr
    return pd.DataFrame(results), keep
