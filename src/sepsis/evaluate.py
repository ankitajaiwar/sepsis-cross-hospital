"""How well predictions work, hour by hour and as bedside alerts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def hourly_metrics(y, p, groups=None, n_boot: int = 200, seed: int = 0) -> dict:
    """AUROC, AUPRC and Brier, with 95% intervals from resampling whole patients."""
    y, p = np.asarray(y), np.asarray(p)
    out = {"AUROC": float(roc_auc_score(y, p)), "AUPRC": float(average_precision_score(y, p)),
           "Brier": float(brier_score_loss(y, p)), "prevalence": float(y.mean()),
           "mean_pred": float(p.mean())}
    if groups is not None and n_boot:
        rng = np.random.default_rng(seed)
        groups = np.asarray(groups)
        uniq, inv = np.unique(groups, return_inverse=True)
        rows = [np.flatnonzero(inv == i) for i in range(len(uniq))]
        a, b = [], []
        for _ in range(n_boot):
            pick = np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])
            if 0 < y[pick].sum() < len(pick):
                a.append(roc_auc_score(y[pick], p[pick]))
                b.append(average_precision_score(y[pick], p[pick]))
        out["AUROC_ci"] = [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
        out["AUPRC_ci"] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    return out


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    obs, pred = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": pred, "observed": obs})


def _patient_view(meta: pd.DataFrame, p: np.ndarray, thr: float) -> pd.DataFrame:
    d = meta[["person_id", "hour", "onset_hour"]].copy()
    d["alert"] = p >= thr
    first_alert = d[d["alert"]].groupby("person_id")["hour"].min()
    pts = d.groupby("person_id").agg(onset=("onset_hour", "first"), hours=("hour", "size"),
                                     alert_hours=("alert", "sum"))
    pts["first_alert"] = first_alert
    pts["septic"] = pts["onset"].notna()
    pts["alerted"] = pts["first_alert"].notna()
    pts["caught"] = pts["septic"] & (pts["first_alert"] <= pts["onset"])
    pts["hours_early"] = pts["onset"] - pts["first_alert"]
    return pts


def alert_metrics(meta: pd.DataFrame, p: np.ndarray, thr: float) -> dict:
    """Treat each hour above the threshold as an alert shown to the care team."""
    pts = _patient_view(meta, p, thr)
    sep, non = pts[pts["septic"]], pts[~pts["septic"]]
    alerted = pts[pts["alerted"]]
    return {
        "threshold": float(thr),
        "septic_caught_pct": float(sep["caught"].mean() * 100) if len(sep) else np.nan,
        "median_hours_early": float(sep.loc[sep["caught"], "hours_early"].median()) if sep["caught"].any() else np.nan,
        "nonseptic_alerted_pct": float(non["alerted"].mean() * 100) if len(non) else np.nan,
        "alerted_who_were_septic_pct": float(alerted["septic"].mean() * 100) if len(alerted) else np.nan,
        "alert_hours_per_100_patient_days": float(pts["alert_hours"].sum() / pts["hours"].sum() * 2400),
        "patients": int(len(pts)), "septic_patients": int(len(sep)),
    }


def threshold_for_catch_rate(meta: pd.DataFrame, p: np.ndarray, target_pct: float = 80.0) -> float:
    """Lowest-burden threshold that still flags `target_pct`% of septic patients by onset."""
    d = meta[["person_id", "hour", "onset_hour"]].assign(p=p)
    pre = d[d["onset_hour"].notna() & (d["hour"] <= d["onset_hour"])]
    best = pre.groupby("person_id")["p"].max()           # highest score before onset
    n_sep = d.loc[d["onset_hour"].notna(), "person_id"].nunique()
    if n_sep == 0 or best.empty:
        return 0.5
    k = int(np.ceil(target_pct / 100 * n_sep))
    vals = np.sort(best.to_numpy())[::-1]
    return float(vals[min(k, len(vals)) - 1])


def tradeoff_curve(meta: pd.DataFrame, p: np.ndarray, targets=range(30, 96, 5)) -> pd.DataFrame:
    rows = []
    for t in targets:
        thr = threshold_for_catch_rate(meta, p, t)
        rows.append(alert_metrics(meta, p, thr))
    return pd.DataFrame(rows)


def subgroup_table(meta: pd.DataFrame, p: np.ndarray, thr: float) -> pd.DataFrame:
    m = meta.copy()
    m["p"] = p
    m["Age group"] = pd.cut(m["Age"], [0, 45, 65, 80, 200], right=False,
                            labels=["Under 45", "45-64", "65-79", "80 and over"])
    m["Sex"] = m["Gender"].map({0.0: "Female", 1.0: "Male"})
    m["ICU unit"] = np.select([m["Unit1"] == 1, m["Unit2"] == 1], ["Medical ICU", "Surgical ICU"],
                              "Not recorded")
    rows = []
    for col in ["Hospital", "Age group", "Sex", "ICU unit"]:
        key = "site" if col == "Hospital" else col
        for g, d in m.groupby(key, observed=True):
            y = d["SepsisLabel"].to_numpy()
            am = alert_metrics(d, d["p"].to_numpy(), thr)
            rows.append({"attribute": col, "group": f"Hospital {g}" if col == "Hospital" else str(g),
                         "patients": am["patients"], "septic_patients": am["septic_patients"],
                         "AUROC": roc_auc_score(y, d["p"]) if 0 < y.sum() < len(y) else np.nan,
                         "septic_caught_pct": am["septic_caught_pct"],
                         "nonseptic_alerted_pct": am["nonseptic_alerted_pct"]})
    return pd.DataFrame(rows)
