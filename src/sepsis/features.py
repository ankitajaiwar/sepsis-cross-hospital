"""Splits, per-hospital normalisation and hourly features.

Every feature at hour t uses only data from hours <= t, so the model could run live.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import LABS, PHYSIO, VITALS

STATIC = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]


def assign_splits(df: pd.DataFrame, seed: int = 42, frac=(0.7, 0.15)) -> pd.DataFrame:
    """Random 70/15/15 split of patients within each hospital (never splits a patient)."""
    rng = np.random.default_rng(seed)
    split = {}
    for site, g in df.groupby("site"):
        ids = g["person_id"].unique()
        rng.shuffle(ids)
        a, b = int(len(ids) * frac[0]), int(len(ids) * (frac[0] + frac[1]))
        split.update({i: "train" for i in ids[:a]})
        split.update({i: "valid" for i in ids[a:b]})
        split.update({i: "test" for i in ids[b:]})
    df = df.copy()
    df["split"] = df["person_id"].map(split)
    return df


def site_stats(df: pd.DataFrame) -> dict:
    """Mean and SD of each measurement per hospital. Uses no outcome labels."""
    return {s: (g[PHYSIO].mean(), g[PHYSIO].std().replace(0, 1).fillna(1))
            for s, g in df.groupby("site")}


def normalize_by_site(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Express each hospital's measurements relative to that hospital's own typical
    values. A simple, label-free way to remove systematic differences between sites."""
    df = df.copy()
    for s, (mu, sd) in stats.items():
        m = df["site"] == s
        df.loc[m, PHYSIO] = ((df.loc[m, PHYSIO] - mu) / sd).astype("float32")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["person_id", "hour"])
    g = df.groupby("person_id", sort=False)
    filled = g[PHYSIO].ffill()
    out = {}
    fg = filled[VITALS].groupby(df["person_id"], sort=False)
    roll = fg.rolling(6, min_periods=1)
    mean6 = roll.mean().reset_index(level=0, drop=True)
    min6 = roll.min().reset_index(level=0, drop=True)
    max6 = roll.max().reset_index(level=0, drop=True)
    lag6 = fg.shift(6)
    for v in VITALS:
        out[f"{v}"] = filled[v]
        out[f"{v}_mean6"] = mean6[v]
        out[f"{v}_min6"] = min6[v]
        out[f"{v}_max6"] = max6[v]
        out[f"{v}_chg6"] = filled[v] - lag6[v]
    hour = df["hour"].astype("float32")
    for v in LABS:
        out[v] = filled[v]
        last = hour.where(df[v].notna()).groupby(df["person_id"], sort=False).ffill()
        out[f"{v}_hrs_since"] = hour - last
    for s in STATIC:
        out[s] = df[s]
    X = pd.DataFrame(out, index=df.index).astype("float32")
    return X


def subsample_negatives(df: pd.DataFrame, rate: float, seed: int = 0) -> pd.Index:
    """Keep every sepsis-labelled hour and a random share of the other hours."""
    rng = np.random.default_rng(seed)
    keep = (df["SepsisLabel"] == 1).to_numpy() | (rng.random(len(df)) < rate)
    return df.index[keep]
