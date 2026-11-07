import numpy as np
import pandas as pd
import pytest

from sepsis.features import assign_splits, build_features
from sepsis.omop import OMOP, cohort_from_omop, to_omop
from sepsis.quality import clean, quality_report
from sepsis.raw import load_raw
from sepsis.schema import LABEL, MEASUREMENTS, PHYSIO
from sepsis.synthetic import write_synthetic


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    write_synthetic(d, n_per_site=60, seed=3)
    return load_raw(d)


@pytest.fixture(scope="module")
def cohort(raw, tmp_path_factory):
    d = tmp_path_factory.mktemp("omop")
    to_omop(raw).save(d)
    return cohort_from_omop(OMOP.load(d))


def test_raw_has_both_hospitals(raw):
    assert set(raw.site) == {"A", "B"}
    assert raw.groupby("patient_id").hour.min().eq(0).all()


def test_omop_tables_are_well_formed(raw):
    db = to_omop(raw)
    assert db.person.person_id.is_unique
    assert set(db.person.gender_concept_id) <= {8507, 8532}
    loincs = {c for c, *_ in MEASUREMENTS.values()}
    assert set(db.measurement.measurement_source_value) <= loincs
    assert db.measurement.value_as_number.notna().all()
    septic = raw.groupby(["site", "patient_id"])[LABEL].max().sum()
    assert len(db.condition_occurrence) == septic


def test_omop_round_trip_rebuilds_the_data(raw, cohort):
    r = raw.sort_values(["site", "patient_id", "hour"]).reset_index(drop=True)
    c = cohort.sort_values(["site", "patient_id", "hour"]).reset_index(drop=True)
    assert len(r) == len(c)
    for v in PHYSIO + [LABEL, "Gender", "ICULOS"]:
        np.testing.assert_allclose(np.nan_to_num(r[v].to_numpy(float), nan=-1e9),
                                   np.nan_to_num(c[v].to_numpy(float), nan=-1e9), atol=1e-3)
    assert (r.Age - c.Age).abs().max() < 1.0   # OMOP keeps birth year only


def test_clean_removes_implausible_values(cohort):
    bad = cohort.copy()
    bad.loc[bad.index[0], "HR"] = 900
    cleaned, n = clean(bad)
    assert n >= 1 and np.isnan(cleaned.loc[bad.index[0], "HR"])


def test_quality_report_counts(cohort):
    rep = quality_report(cohort)
    assert rep["sites"]["A"]["patients"] == 60
    etco2_b = [v for v in rep["sites"]["B"]["variables"] if v["variable"] == "EtCO2"][0]
    assert etco2_b["missing_pct"] == 100.0


def test_splits_never_share_patients(cohort):
    df = assign_splits(cohort)
    per = df.groupby("person_id").split.nunique()
    assert per.max() == 1
    assert set(df.split) == {"train", "valid", "test"}


def test_features_use_only_the_past(cohort):
    """Changing a later hour must not change features for earlier hours."""
    pid = cohort.person_id.iloc[0]
    one = cohort[cohort.person_id == pid].copy()
    X1 = build_features(one)
    later = one.copy()
    later.loc[later.index[-1], ["HR", "Lactate"]] = [180, 9.0]
    X2 = build_features(later)
    pd.testing.assert_frame_equal(X1.iloc[:-1], X2.iloc[:-1])
