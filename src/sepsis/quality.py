"""Data-quality checks: counts, missingness and implausible values, per hospital.

Implausible values (outside the ranges in schema.MEASUREMENTS) are counted and then
set to missing. The challenge organisers deliberately left errors in the data, so
this step matters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import LABEL, MEASUREMENTS, PHYSIO


def quality_report(df: pd.DataFrame) -> dict:
    out = {"sites": {}}
    for site, g in df.groupby("site"):
        pts = g.groupby("person_id")
        septic = pts[LABEL].max()
        rows = []
        for v in PHYSIO:
            _, _, lo, hi = MEASUREMENTS[v]
            x = g[v]
            n_obs = int(x.notna().sum())
            bad = int(((x < lo) | (x > hi)).sum())
            rows.append({"variable": v, "missing_pct": float(x.isna().mean() * 100),
                         "observed": n_obs, "implausible": bad,
                         "implausible_pct": float(bad / n_obs * 100) if n_obs else 0.0})
        out["sites"][site] = {
            "patients": int(len(septic)), "septic_patients": int(septic.sum()),
            "septic_pct": float(septic.mean() * 100), "patient_hours": int(len(g)),
            "median_hours": float(pts.size().median()),
            "age_missing": int(g["Age"].isna().sum()),
            "variables": rows,
        }
    return out


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    total = 0
    for v in PHYSIO:
        _, _, lo, hi = MEASUREMENTS[v]
        bad = (df[v] < lo) | (df[v] > hi)
        total += int(bad.sum())
        df.loc[bad, v] = np.nan
    return df, total


def missingness_table(rep: dict) -> pd.DataFrame:
    rows = []
    for site, s in rep["sites"].items():
        for r in s["variables"]:
            rows.append({"site": site, **r})
    return (pd.DataFrame(rows)
            .pivot_table(index="variable", columns="site", values="missing_pct")
            .reindex(PHYSIO))
