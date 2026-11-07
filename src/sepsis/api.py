"""Scoring service: send one patient's hourly ICU data so far, get back the sepsis
risk for each hour and whether it would have triggered an alert.

    MODEL_DIR=artifacts/model uvicorn sepsis.api:app --port 8000
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .features import build_features
from .schema import LABS, PHYSIO, VITALS

app = FastAPI(title="Sepsis early warning (research demo)", version="0.1.0")


class Patient(BaseModel):
    age: float = Field(..., ge=14, le=110)
    sex: str = Field(..., pattern="^(F|M)$")
    icu_unit: str = Field("Unknown", pattern="^(MICU|SICU|Unknown)$")
    hours_since_hospital_admission: float = Field(0, ge=0)
    hours: list[dict[str, float | None]] = Field(
        ..., min_length=1, max_length=336,
        examples=[[{"HR": 92, "Temp": 37.4, "Resp": 20}, {"HR": 104, "Lactate": 2.8}]])


@lru_cache
def load():
    d = Path(os.environ.get("MODEL_DIR", "artifacts/model"))
    if not (d / "gbm.joblib").exists():
        raise RuntimeError(f"No model in {d}. Run the pipeline first.")
    return joblib.load(d / "gbm.joblib"), json.loads((d / "model_card.json").read_text(encoding="utf-8"))


@app.get("/health")
def health():
    try:
        load()
        return {"status": "ok"}
    except RuntimeError as e:
        raise HTTPException(503, str(e))


@app.get("/model-card")
def model_card():
    return load()[1]


@app.post("/score")
def score(p: Patient):
    model, card = load()
    unknown = {k for h in p.hours for k in h} - set(PHYSIO)
    if unknown:
        raise HTTPException(422, f"Unknown measurements: {sorted(unknown)}. Allowed: {PHYSIO}")
    T = len(p.hours)
    df = pd.DataFrame(p.hours, columns=PHYSIO, dtype="float32").reindex(range(T))
    df["person_id"], df["site"], df["hour"] = 1, "X", np.arange(T)
    df["Age"], df["Gender"] = p.age, 1.0 if p.sex == "M" else 0.0
    df["Unit1"] = {"MICU": 1.0, "SICU": 0.0}.get(p.icu_unit, np.nan)
    df["Unit2"] = {"MICU": 0.0, "SICU": 1.0}.get(p.icu_unit, np.nan)
    df["HospAdmTime"] = -p.hours_since_hospital_admission
    df["ICULOS"] = np.arange(1, T + 1, dtype="float32")
    risk = model.predict(build_features(df))
    thr = card["alert_threshold"]
    return {"alert_threshold": round(thr, 4),
            "hours": [{"hour": i, "risk": round(float(r), 4), "alert": bool(r >= thr)}
                      for i, r in enumerate(risk)]}
