"""Convert the challenge files to OMOP CDM tables, then build the modelling cohort
back out of those tables.

Going through OMOP on purpose: hospital data networks (OHDSI, multi-site studies)
share models by agreeing on these tables, so the cohort code below would run on
any OMOP database that has the same LOINC measurements.

Notes on this ETL
- The challenge files have no real dates. Each ICU stay is anchored at
  2100-01-01 00:00 (an obviously shifted date, as MIMIC does), and every hourly row
  is placed at anchor + hour.
- `measurement_concept_id` and `condition_concept_id` are 0 ("not yet mapped");
  the LOINC code is in `measurement_source_value`. Mapping to standard concept ids
  is a join to the OHDSI Athena vocabulary, which is licence-gated and not bundled.
- ICU unit (MICU / SICU) goes in `visit_detail`; time from hospital admission to
  ICU admission sets `visit_occurrence.visit_start_datetime`.
- Sepsis onset (Sepsis-3, as labelled by the challenge) is a `condition_occurrence`
  row at the onset hour.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import (DEMOGRAPHICS, GENDER_CONCEPT, LABEL, LABEL_LEAD_HOURS,
                     MEASUREMENTS, PHYSIO)

ANCHOR = pd.Timestamp("2100-01-01")
INPATIENT_VISIT = 9201
SEPSIS_SOURCE = "Sepsis-3 onset (PhysioNet Challenge 2019 label)"
LOINC_TO_VAR = {code: var for var, (code, *_r) in MEASUREMENTS.items()}


@dataclass
class OMOP:
    care_site: pd.DataFrame
    person: pd.DataFrame
    visit_occurrence: pd.DataFrame
    visit_detail: pd.DataFrame
    measurement: pd.DataFrame
    condition_occurrence: pd.DataFrame

    TABLES = ("care_site", "person", "visit_occurrence", "visit_detail",
              "measurement", "condition_occurrence")

    def save(self, d: str | Path) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        for t in self.TABLES:
            getattr(self, t).to_parquet(d / f"{t}.parquet", index=False)

    @classmethod
    def load(cls, d: str | Path) -> "OMOP":
        d = Path(d)
        return cls(**{t: pd.read_parquet(d / f"{t}.parquet") for t in cls.TABLES})


def to_omop(raw: pd.DataFrame) -> OMOP:
    raw = raw.sort_values(["site", "patient_id", "hour"])
    first = raw.groupby(["site", "patient_id"], sort=False).first().reset_index()
    first["person_id"] = np.arange(1, len(first) + 1)
    pid = first.set_index(["site", "patient_id"])["person_id"]

    care_site = pd.DataFrame({"care_site_id": [1, 2],
                              "care_site_name": ["Hospital system A", "Hospital system B"],
                              "care_site_source_value": ["A", "B"]})
    site_id = {"A": 1, "B": 2}

    person = pd.DataFrame({
        "person_id": first["person_id"],
        "gender_concept_id": first["Gender"].map(GENDER_CONCEPT).fillna(0).astype(int),
        "year_of_birth": (ANCHOR.year - np.floor(first["Age"])).astype(int),
        "race_concept_id": 0, "ethnicity_concept_id": 0,
        "care_site_id": first["site"].map(site_id),
        "person_source_value": first["site"] + ":" + first["patient_id"],
        "gender_source_value": first["Gender"].map({1.0: "M", 0.0: "F"}),
    })

    los = raw.groupby(["site", "patient_id"], sort=False)["hour"].max().reindex(
        pd.MultiIndex.from_frame(first[["site", "patient_id"]])).values + 1
    icu_end = ANCHOR + pd.to_timedelta(los, unit="h")
    hosp_start = ANCHOR + pd.to_timedelta(first["HospAdmTime"].fillna(0).values, unit="h")
    visit_occurrence = pd.DataFrame({
        "visit_occurrence_id": first["person_id"], "person_id": first["person_id"],
        "visit_concept_id": INPATIENT_VISIT,
        "visit_start_datetime": hosp_start, "visit_end_datetime": icu_end,
        "visit_type_concept_id": 32817,  # EHR
        "care_site_id": first["site"].map(site_id),
        "visit_source_value": first["site"] + ":" + first["patient_id"],
    })
    unit = np.select([first["Unit1"] == 1, first["Unit2"] == 1], ["MICU", "SICU"], "Unknown")
    visit_detail = pd.DataFrame({
        "visit_detail_id": first["person_id"], "person_id": first["person_id"],
        "visit_occurrence_id": first["person_id"], "visit_detail_concept_id": 32037,
        "visit_detail_start_datetime": ANCHOR, "visit_detail_end_datetime": icu_end,
        "visit_detail_type_concept_id": 32817, "care_site_id": first["site"].map(site_id),
        "visit_detail_source_value": unit,
    })

    person_ids = pid.reindex(pd.MultiIndex.from_frame(raw[["site", "patient_id"]])).values
    hours = raw["hour"].to_numpy()
    parts = []
    for v in PHYSIO:                       # one variable at a time keeps memory low
        x = raw[v].to_numpy()
        keep = ~np.isnan(x)
        code, unit, *_ = MEASUREMENTS[v]
        parts.append(pd.DataFrame({
            "person_id": person_ids[keep], "hour": hours[keep],
            "value_as_number": x[keep].astype("float32"),
            "unit_source_value": pd.Categorical([unit] * int(keep.sum())),
            "measurement_source_value": pd.Categorical([code] * int(keep.sum())),
        }))
    long = pd.concat(parts, ignore_index=True)
    measurement = pd.DataFrame({
        "measurement_id": np.arange(1, len(long) + 1),
        "person_id": long["person_id"].values,
        "measurement_concept_id": 0,
        "measurement_datetime": ANCHOR + pd.to_timedelta(long["hour"].values, unit="h"),
        "measurement_type_concept_id": 32817,
        "value_as_number": long["value_as_number"].values,
        "unit_source_value": long["unit_source_value"].astype(str).values,
        "visit_occurrence_id": long["person_id"].values,
        "measurement_source_value": long["measurement_source_value"].astype(str).values,
    })

    lab = raw.assign(person_id=person_ids)
    pos = lab[lab[LABEL] == 1].groupby("person_id")["hour"].min()
    onset = pos + LABEL_LEAD_HOURS
    condition_occurrence = pd.DataFrame({
        "condition_occurrence_id": np.arange(1, len(onset) + 1),
        "person_id": onset.index.values, "condition_concept_id": 0,
        "condition_start_datetime": ANCHOR + pd.to_timedelta(onset.values, unit="h"),
        "condition_type_concept_id": 32817, "visit_occurrence_id": onset.index.values,
        "condition_source_value": SEPSIS_SOURCE,
    })
    return OMOP(care_site, person, visit_occurrence, visit_detail, measurement,
                condition_occurrence)


def cohort_from_omop(db: OMOP) -> pd.DataFrame:
    """Rebuild one row per patient-hour from OMOP tables, with the sepsis label."""
    m = db.measurement
    var = m["measurement_source_value"].map(LOINC_TO_VAR)
    if var.isna().any():
        bad = m.loc[var.isna(), "measurement_source_value"].unique()[:5]
        raise ValueError(f"Unmapped measurement source codes: {bad}")

    vd = db.visit_detail.set_index("person_id").sort_index()
    vo = db.visit_occurrence.set_index("person_id")
    los = ((vd["visit_detail_end_datetime"] - ANCHOR) / pd.Timedelta(hours=1)).round().astype(int)
    offset = pd.Series(np.concatenate([[0], np.cumsum(los.values)[:-1]]), index=los.index)
    n = int(los.sum())
    df = pd.DataFrame({"person_id": np.repeat(los.index.values, los.values),
                       "hour": np.concatenate([np.arange(k) for k in los.values])})
    hour = ((m["measurement_datetime"] - ANCHOR) / pd.Timedelta(hours=1)).round().astype(int).to_numpy()
    row = m["person_id"].map(offset).to_numpy() + hour
    vals = m["value_as_number"].to_numpy()
    for v in PHYSIO:                       # scatter each variable into its hourly column
        col = np.full(n, np.nan, dtype="float32")
        sel = (var == v).to_numpy()
        col[row[sel]] = vals[sel]
        df[v] = col

    p = db.person.set_index("person_id")
    site = p["care_site_id"].map(db.care_site.set_index("care_site_id")["care_site_source_value"])
    birth_age = ANCHOR.year - p["year_of_birth"]
    df["site"] = df["person_id"].map(site).values
    df["patient_id"] = df["person_id"].map(p["person_source_value"].str.split(":").str[1]).values
    df["Age"] = df["person_id"].map(birth_age).astype("float32")
    df["Gender"] = df["person_id"].map(p["gender_source_value"].map({"M": 1.0, "F": 0.0}))
    unit = df["person_id"].map(vd["visit_detail_source_value"])
    df["Unit1"] = np.where(unit == "MICU", 1.0, np.where(unit == "SICU", 0.0, np.nan))
    df["Unit2"] = np.where(unit == "SICU", 1.0, np.where(unit == "MICU", 0.0, np.nan))
    df["HospAdmTime"] = df["person_id"].map(
        (vo["visit_start_datetime"] - ANCHOR) / pd.Timedelta(hours=1)).astype("float32")
    df["ICULOS"] = (df["hour"] + 1).astype("float32")

    c = db.condition_occurrence
    c = c[c["condition_source_value"] == SEPSIS_SOURCE]
    onset = ((c.set_index("person_id")["condition_start_datetime"] - ANCHOR)
             / pd.Timedelta(hours=1)).round()
    on = df["person_id"].map(onset)
    df[LABEL] = (df["hour"] >= on - LABEL_LEAD_HOURS).astype("float32")
    df["onset_hour"] = on.astype("float32")
    cols = ["person_id", "patient_id", "site", "hour"] + PHYSIO + DEMOGRAPHICS + [LABEL, "onset_hour"]
    out = df[cols].sort_values(["person_id", "hour"]).reset_index(drop=True)
    out[PHYSIO] = out[PHYSIO].astype("float32")
    return out
