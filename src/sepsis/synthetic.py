"""Write fake patient files in the exact PhysioNet 2019 format.

Used only for tests and CI. Hospital B is given deliberately different blood
pressure levels and recording habits, so the cross-hospital code has a real shift
to work on. Numbers produced from this data mean nothing clinically.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import ALL_COLUMNS, LABS, SITES

BASE = {"HR": 85, "O2Sat": 97, "Temp": 36.9, "SBP": 122, "MAP": 82, "DBP": 63,
        "Resp": 18, "EtCO2": 33}
LAB_BASE = {"BaseExcess": 0, "HCO3": 24, "FiO2": 0.4, "pH": 7.39, "PaCO2": 40,
            "SaO2": 95, "AST": 40, "BUN": 20, "Alkalinephos": 90, "Calcium": 8.5,
            "Chloride": 105, "Creatinine": 1.1, "Bilirubin_direct": 0.4,
            "Glucose": 130, "Lactate": 1.6, "Magnesium": 2.0, "Phosphate": 3.4,
            "Potassium": 4.1, "Bilirubin_total": 0.9, "TroponinI": 0.1, "Hct": 32,
            "Hgb": 10.5, "PTT": 32, "WBC": 10, "Fibrinogen": 300, "Platelets": 200}


def make_patient(rng, site: str, septic: bool) -> pd.DataFrame:
    T = int(rng.integers(12, 60))
    onset = int(rng.integers(10, T + 4)) if septic else None
    hours = np.arange(T)
    risk = np.zeros(T) if not septic else np.clip((hours - (onset - 10)) / 10, 0, 1.5)
    df = pd.DataFrame(index=hours, columns=ALL_COLUMNS, dtype=float)
    bp_shift = 12 if site == "B" else 0
    df["HR"] = BASE["HR"] + 18 * risk + rng.normal(0, 8, T)
    df["Temp"] = BASE["Temp"] + 0.9 * risk + rng.normal(0, 0.4, T)
    df["Resp"] = BASE["Resp"] + 5 * risk + rng.normal(0, 3, T)
    df["O2Sat"] = np.clip(BASE["O2Sat"] - 2 * risk + rng.normal(0, 1.5, T), 70, 100)
    df["SBP"] = BASE["SBP"] + bp_shift - 14 * risk + rng.normal(0, 12, T)
    df["MAP"] = BASE["MAP"] + bp_shift * 0.7 - 9 * risk + rng.normal(0, 8, T)
    df["DBP"] = BASE["DBP"] + bp_shift * 0.5 - 6 * risk + rng.normal(0, 7, T)
    df["EtCO2"] = (BASE["EtCO2"] + rng.normal(0, 4, T)) if site == "A" else np.nan
    for lab in LABS:
        base = LAB_BASE[lab]
        vals = base * (1 + rng.normal(0, 0.15, T))
        if lab == "Lactate":
            vals = vals + 1.8 * risk
        if lab == "WBC":
            vals = vals + 5 * risk
        if lab == "Creatinine":
            vals = vals + 0.4 * risk
        measured = rng.random(T) < (0.12 if site == "A" else 0.08)
        df[lab] = np.where(measured, vals, np.nan)
    # Vitals are mostly recorded, with gaps.
    for v in ["HR", "O2Sat", "SBP", "MAP", "DBP", "Resp"]:
        df.loc[rng.random(T) < 0.1, v] = np.nan
    df.loc[rng.random(T) < 0.65, "Temp"] = np.nan
    age = float(np.clip(rng.normal(62, 16), 18, 100))
    df["Age"] = round(age, 2)
    df["Gender"] = float(rng.integers(0, 2))
    unit = rng.integers(0, 3)
    df["Unit1"] = 1.0 if unit == 0 else (0.0 if unit == 1 else np.nan)
    df["Unit2"] = 0.0 if unit == 0 else (1.0 if unit == 1 else np.nan)
    df["HospAdmTime"] = round(-abs(rng.normal(10, 20)), 2)
    df["ICULOS"] = hours + 1.0
    lab = np.zeros(T)
    if septic:
        lab[max(0, onset - 6):] = 1
    df["SepsisLabel"] = lab
    return df


def write_synthetic(root: str | Path, n_per_site: int = 200, septic_rate: float = 0.12,
                    seed: int = 0) -> Path:
    root = Path(root)
    rng = np.random.default_rng(seed)
    for site, folder in SITES.items():
        d = root / folder
        d.mkdir(parents=True, exist_ok=True)
        start = 1 if site == "A" else 100001
        for i in range(n_per_site):
            df = make_patient(rng, site, rng.random() < septic_rate)
            df.to_csv(d / f"p{start + i:06d}.psv", sep="|", index=False, na_rep="NaN")
    return root
