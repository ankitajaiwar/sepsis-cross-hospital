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

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import (PatientBootstrap, alert_metrics, hourly_metrics, paired_difference,
                       threshold_for_catch_rate)
from .gbm import GBMModel, importance_weights
from .seq import GRUModel

META_COLS = ["person_id", "hour", "onset_hour", "SepsisLabel", "site", "split",
             "Age", "Gender", "Unit1", "Unit2"]
PLANS = [("within", ("A",), ["A"]), ("within", ("B",), ["B"]), ("pooled", ("A", "B"), ["A", "B"]),
         ("cross", ("A",), ["B"]), ("cross", ("B",), ["A"])]
ID_COLS = ["run", "model", "setting", "variant", "train", "test"]


def _sel(meta, sites, split):
    return meta.index[meta["site"].isin(list(sites)) & (meta["split"] == split)]


def _variants(mt, setting):
    if setting != "cross":
        return ["standard"]
    return ["source_only", "site_normalized"] + (["importance_wt"] if mt == "gbm" else ["dann"])


def _safe(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _cache_paths(cache, key, seed):
    stem = Path(cache) / f"{_safe(key)}__seed{seed}"
    return stem.with_suffix(".pkl"), stem.with_suffix(".json")


def run_all(df: pd.DataFrame, X_raw: pd.DataFrame, X_norm: pd.DataFrame, df_norm: pd.DataFrame,
            models=("gbm", "gru"), epochs: int = 8, catch_target: float = 80.0,
            gbm_iter: int = 300, seeds=(0,), cache_dir: str | Path | None = None, log=print):
    """Train and test every setting once per seed. Returns one row per (run, seed) and the
    test predictions, keyed by (run name, seed). With cache_dir, each finished training
    is saved, so an interrupted run, or a later run with more seeds, picks up where it
    stopped."""
    meta = df[META_COLS]
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    results, keep = [], {}
    for mt in models:
        for setting, src, tests in PLANS:
            for var in _variants(mt, setting):
                cfg = f"epochs{epochs}" if mt == "gru" else f"iter{gbm_iter}"
                key = f"{mt}:{setting}:{'+'.join(src)}:{var}:{cfg}"
                for seed in seeds:
                    t0 = time.time()
                    pkl, js = _cache_paths(cache_dir, key, seed) if cache_dir else (None, None)
                    if pkl is not None and pkl.exists() and js.exists():
                        preds = pd.read_pickle(pkl)
                        thr = json.loads(js.read_text())["threshold"]
                        model, how = None, "cached"
                    else:
                        tr, va = _sel(meta, src, "train"), _sel(meta, src, "valid")
                        X = X_norm if var == "site_normalized" else X_raw
                        D = df_norm if var == "site_normalized" else df
                        tgt_train = _sel(meta, tests, "train") if setting == "cross" else None
                        if mt == "gbm":
                            w = (importance_weights(X.loc[tr], X.loc[tgt_train], seed=seed)
                                 if var == "importance_wt" else None)
                            model = GBMModel(max_iter=gbm_iter, seed=seed).fit(X.loc[tr], meta.loc[tr], w)
                            model.calibrate(X.loc[va], meta.loc[va, "SepsisLabel"].to_numpy())
                            pred = lambda idx: model.predict(X.loc[idx])
                        else:
                            model = GRUModel(epochs=epochs, dann=0.3 if var == "dann" else 0.0, seed=seed)
                            model.fit(D.loc[tr], D.loc[va], D.loc[tgt_train] if var == "dann" else None)
                            model.calibrate(D.loc[va])
                            pred = lambda idx: model.predict(D.loc[idx]).reindex(idx).to_numpy()
                        thr = threshold_for_catch_rate(meta.loc[va], pred(va), catch_target)
                        te_all = _sel(meta, tests, "test")
                        preds = pd.Series(np.asarray(pred(te_all), dtype="float32"), index=te_all)
                        if pkl is not None:
                            preds.to_pickle(pkl)
                            js.write_text(json.dumps({"threshold": float(thr)}))
                        how = f"{time.time() - t0:.0f}s"
                    for ts in tests:
                        te = _sel(meta, [ts], "test")
                        p = preds.loc[te].to_numpy()
                        m = meta.loc[te]
                        name = f"{mt}:{setting}:{'+'.join(src)}->{ts}:{var}"
                        r = hourly_metrics(m["SepsisLabel"].to_numpy(), p, n_boot=0)
                        r.update(alert_metrics(m, p, thr))
                        r.update({"run": name, "model": mt, "setting": setting, "variant": var,
                                  "train": "+".join(src), "test": ts, "seed": seed})
                        results.append(r)
                        keep[(name, seed)] = pd.Series(p, index=te)
                        log(f"{name:45s} seed {seed}  AUROC {r['AUROC']:.3f}  "
                            f"caught {r['septic_caught_pct']:.0f}%  ({how})")
                    if setting == "pooled" and seed == seeds[0]:
                        keep[f"{mt}:model"] = model
                        keep[f"{mt}:threshold"] = thr
    return pd.DataFrame(results), keep


def summarize(res: pd.DataFrame, keep: dict, meta: pd.DataFrame, n_boot: int = 200) -> pd.DataFrame:
    """One row per run: metrics averaged over seeds, an AUROC interval from resampling
    patients (for the seed-averaged AUROC), and the spread between seeds."""
    rows, boots = [], {}
    num = [c for c in res.columns if c not in ID_COLS + ["seed"] and pd.api.types.is_numeric_dtype(res[c])]
    for name, g in res.groupby("run", sort=False):
        seeds = sorted(g["seed"])
        te = keep[(name, seeds[0])].index
        m = meta.loc[te]
        bkey = tuple(sorted(set(m["site"])))
        if bkey not in boots:
            boots[bkey] = PatientBootstrap(m["SepsisLabel"].to_numpy(), m["person_id"].to_numpy(), n_boot)
        auc, ci, _ = boots[bkey].mean_auroc([keep[(name, s)].loc[te].to_numpy() for s in seeds])
        row = g.iloc[0][ID_COLS].to_dict()
        row.update(g[num].mean().to_dict())
        row.update({"AUROC": auc, "AUROC_ci": ci, "n_seeds": len(seeds),
                    "AUROC_seed_sd": float(g["AUROC"].std(ddof=1)) if len(seeds) > 1 else np.nan,
                    "AUROC_seed_min": float(g["AUROC"].min()), "AUROC_seed_max": float(g["AUROC"].max())})
        rows.append(row)
    return pd.DataFrame(rows)


def compare_to_source_only(res: pd.DataFrame, keep: dict, meta: pd.DataFrame,
                           n_boot: int = 200) -> pd.DataFrame:
    """For each model moved to the other hospital: how much each adaptation method changes
    AUROC versus no adjustment, and how much a model trained at the test hospital itself
    would gain. Paired: both are scored on the same resampled test patients."""
    rows = []
    for (mt, test), g in res[res.setting == "cross"].groupby(["model", "test"]):
        train = g["train"].iloc[0]
        base = f"{mt}:cross:{train}->{test}:source_only"
        seeds_of = lambda run: sorted(k[1] for k in keep if isinstance(k, tuple) and k[0] == run)
        te = keep[(base, seeds_of(base)[0])].index
        m = meta.loc[te]
        boot = PatientBootstrap(m["SepsisLabel"].to_numpy(), m["person_id"].to_numpy(), n_boot, seed=1)
        P = lambda run: [keep[(run, s)].loc[te].to_numpy() for s in seeds_of(run)]
        others = [(f"{mt}:cross:{train}->{test}:{v}", v) for v in _variants(mt, "cross") if v != "source_only"]
        others.append((f"{mt}:within:{test}->{test}:standard", "trained_at_test_hospital"))
        for run, label in others:
            if not seeds_of(run):
                continue
            d = paired_difference(boot, P(base), P(run))
            rows.append({"model": mt, "train": train, "test": test, "compared": label, **d})
    return pd.DataFrame(rows)
