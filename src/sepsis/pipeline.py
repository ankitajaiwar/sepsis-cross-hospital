"""Run everything: raw files -> OMOP -> cohort -> models -> evaluation -> report.

    python -m sepsis.pipeline --raw data/raw --out artifacts
    python -m sepsis.pipeline --synthetic --out artifacts_synthetic --fast
    python -m sepsis.pipeline --raw data/raw --out artifacts --models gbm   # trees only, faster
    python -m sepsis.pipeline --from-omop artifacts/omop --out artifacts --seeds 0,1,2
        # reuse the OMOP tables from an earlier run and train every model with 3 seeds;
        # finished trainings are saved in OUT/runs, so reruns only train what's new
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .evaluate import subgroup_table, tradeoff_curve
from .experiments import META_COLS, compare_to_source_only, run_all, summarize
from .features import assign_splits, build_features, normalize_by_site, site_stats
from .omop import OMOP, cohort_from_omop, to_omop
from .quality import clean, quality_report
from .raw import load_raw
from .report import (fig_calibration, fig_cross_hospital, fig_data_shift, fig_tradeoff,
                     write_report, NAMES)
from .synthetic import write_synthetic


def run(raw_dir: str, out: Path, synthetic: bool = False, models=("gbm", "gru"), epochs: int = 8,
        n_boot: int = 200, limit: int = 0, gbm_iter: int = 300, omop_dir: str | None = None,
        from_omop: str | None = None, seeds=(0,), log=print) -> pd.DataFrame:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    if from_omop:
        log("1-2/6 Reading the OMOP tables from an earlier run")
        df = cohort_from_omop(OMOP.load(Path(from_omop)))
    else:
        if synthetic:
            raw_dir = str(out / "synthetic_raw")
            write_synthetic(raw_dir, n_per_site=limit or 400)
        log("1/6 Reading raw files")
        raw = load_raw(raw_dir, limit=0 if synthetic else limit)
        log(f"    {raw.patient_id.nunique():,} patients, {len(raw):,} patient-hours")
        log("2/6 Converting to OMOP and building the cohort from it")
        db = to_omop(raw)
        del raw
        omop_path = Path(omop_dir) if omop_dir else out / "omop"
        db.save(omop_path)
        del db
        gc.collect()
        df = cohort_from_omop(OMOP.load(omop_path))
    gc.collect()
    log("3/6 Data-quality checks")
    qrep = quality_report(df)
    df, n_bad = clean(df)
    (out / "quality_report.json").write_text(json.dumps(qrep, indent=2), encoding="utf-8")
    df = assign_splits(df)
    log("4/6 Building features")
    df_norm = normalize_by_site(df, site_stats(df))
    X_raw, X_norm = build_features(df), build_features(df_norm)
    log("5/6 Training and testing models")
    runs, keep = run_all(df, X_raw, X_norm, df_norm, models=models, epochs=epochs,
                         gbm_iter=gbm_iter, seeds=seeds, cache_dir=out / "runs",
                         log=lambda s: log("    " + s))
    runs.to_csv(out / "results_by_seed.csv", index=False)
    meta = df[META_COLS]
    res = summarize(runs, keep, meta, n_boot)
    res.to_csv(out / "results.csv", index=False)
    comp = compare_to_source_only(runs, keep, meta, n_boot)
    comp.to_csv(out / "comparisons.csv", index=False)

    log("6/6 Report")
    test = meta[meta.split == "test"]
    curves, preds = {}, {}
    for m in models:
        p = pd.concat([keep[(f"{m}:pooled:A+B->A:standard", seeds[0])],
                       keep[(f"{m}:pooled:A+B->B:standard", seeds[0])]])
        preds[m] = p
        curves[m] = tradeoff_curve(test.loc[p.index], p.to_numpy())
        curves[m].to_csv(out / f"tradeoff_{m}.csv", index=False)
    main = "gbm" if "gbm" in models else models[0]
    sub = subgroup_table(test.loc[preds[main].index], preds[main].to_numpy(), keep[f"{main}:threshold"], n_boot)
    sub.to_csv(out / "subgroups.csv", index=False)
    fig_cross_hospital(res, out / "cross_hospital.png")
    fig_calibration(test, preds[main], out / "calibration.png", NAMES[main])
    fig_tradeoff(curves, out / "alert_tradeoff.png")
    fig_data_shift(qrep, out / "data_shift.png")
    write_report(out, res, qrep, n_bad, sub, curves, main, synthetic, comp, len(seeds))

    mdir = out / "model"
    mdir.mkdir(exist_ok=True)
    if "gbm" in models and keep["gbm:model"] is not None:
        joblib.dump(keep["gbm:model"], mdir / "gbm.joblib")
    if "gru" in models and keep["gru:model"] is not None:
        g = keep["gru:model"]
        torch.save({"state": g.net.state_dict(), "mu": g.mu, "sd": g.sd, "platt": g.platt,
                    "hidden": g.hidden}, mdir / "gru.pt")
    card = {"name": "sepsis-early-warning", "trained_on": "PhysioNet 2019 hospitals A + B (training split)",
            "served_model": main, "alert_threshold": float(keep[f"{main}:threshold"]),
            "intended_use": "Research demonstration of cross-hospital evaluation. Not a clinical tool.",
            "not_features": ["race / ethnicity (not in data)"]}
    (mdir / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    log(f"Done in {(time.time() - t0) / 60:.1f} min -> {out / 'REPORT.md'}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--omop", default=None, help="where to write OMOP tables (default: OUT/omop)")
    ap.add_argument("--from-omop", default=None,
                    help="skip reading raw files: build the cohort from OMOP tables saved by an earlier run")
    ap.add_argument("--seeds", default="0", help="comma-separated training seeds, e.g. 0,1,2")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--models", default="gbm,gru")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="patients per hospital (0 = all)")
    ap.add_argument("--fast", action="store_true", help="quick trial: fewer epochs and resamples")
    a = ap.parse_args()
    if a.fast:
        a.epochs, a.n_boot = min(a.epochs, 3), min(a.n_boot, 30)
    run(a.raw, Path(a.out), a.synthetic, tuple(a.models.split(",")), a.epochs, a.n_boot,
        a.limit, 100 if a.fast else 300, a.omop, a.from_omop,
        tuple(int(x) for x in a.seeds.split(",")))


if __name__ == "__main__":
    main()
