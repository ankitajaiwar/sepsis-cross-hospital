"""How well predictions work, hour by hour and as bedside alerts."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


class PatientBootstrap:
    """Resample whole patients with replacement, the same draws for every model.

    Each draw is stored as a count per patient, so AUROC on a draw is a weighted AUROC
    over the original rows. That makes thousands of draws cheap, and using the same
    draws for two models gives a paired comparison."""

    def __init__(self, y, groups, n_boot: int = 200, seed: int = 0):
        self.y = np.asarray(y, dtype=float)
        _, self.inv = np.unique(np.asarray(groups), return_inverse=True)
        n_pat = self.inv.max() + 1
        rng = np.random.default_rng(seed)
        self.counts = [np.bincount(rng.integers(0, n_pat, n_pat), minlength=n_pat)
                       for _ in range(n_boot)]

    def _prep(self, p):
        o = np.argsort(np.asarray(p), kind="mergesort")
        ps = np.asarray(p)[o]
        starts = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1]])   # groups of tied scores
        return o, starts

    def _auc(self, prep, w):
        o, starts = prep
        ys, ws = self.y[o], w[o]
        pos = np.add.reduceat(ws * ys, starts)
        neg = np.add.reduceat(ws * (1 - ys), starts)
        P, N = pos.sum(), neg.sum()
        if P == 0 or N == 0:
            return np.nan
        below = np.cumsum(neg) - neg
        return float((pos * (below + 0.5 * neg)).sum() / (P * N))

    def aurocs(self, p) -> np.ndarray:
        prep = self._prep(p)
        return np.array([self._auc(prep, c[self.inv].astype(float)) for c in self.counts])

    def mean_auroc(self, preds) -> tuple[float, list[float], np.ndarray]:
        """AUROC averaged over several runs (seeds) of a model, with a 95% interval."""
        point = float(np.mean([roc_auc_score(self.y, p) for p in preds]))
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            draws = np.nanmean([self.aurocs(p) for p in preds], axis=0)
        return point, _ci(draws), draws


def _ci(draws) -> list[float]:
    d = np.asarray(draws)[~np.isnan(draws)]
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if len(d) else [np.nan, np.nan]


def paired_difference(boot: PatientBootstrap, preds_a, preds_b) -> dict:
    """AUROC of B minus AUROC of A on the same test patients, with a 95% interval."""
    a, _, da = boot.mean_auroc(preds_a)
    b, _, db = boot.mean_auroc(preds_b)
    d = db - da
    d = d[~np.isnan(d)]
    return {"diff": b - a, "diff_ci": _ci(d), "share_of_draws_better": float((d > 0).mean())}


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """95% interval for a proportion, in percent."""
    if n == 0:
        return [np.nan, np.nan]
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [float(max(0.0, 100 * (c - h))), float(min(100.0, 100 * (c + h)))]


def hourly_metrics(y, p, groups=None, n_boot: int = 200, seed: int = 0) -> dict:
    """AUROC, AUPRC and Brier, with a 95% AUROC interval from resampling whole patients."""
    y, p = np.asarray(y), np.asarray(p)
    out = {"AUROC": float(roc_auc_score(y, p)), "AUPRC": float(average_precision_score(y, p)),
           "Brier": float(brier_score_loss(y, p)), "prevalence": float(y.mean()),
           "mean_pred": float(p.mean())}
    if groups is not None and n_boot:
        out["AUROC_ci"] = PatientBootstrap(y, groups, n_boot, seed).mean_auroc([p])[1]
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


def subgroup_table(meta: pd.DataFrame, p: np.ndarray, thr: float, n_boot: int = 200) -> pd.DataFrame:
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
            pts = _patient_view(d, d["p"].to_numpy(), thr)
            sep, non = pts[pts["septic"]], pts[~pts["septic"]]
            ok = 0 < y.sum() < len(y)
            auc, ci = (PatientBootstrap(y, d["person_id"], n_boot).mean_auroc([d["p"].to_numpy()])[:2]
                       if ok and n_boot else (roc_auc_score(y, d["p"]) if ok else np.nan, [np.nan, np.nan]))
            rows.append({"attribute": col, "group": f"Hospital {g}" if col == "Hospital" else str(g),
                         "patients": am["patients"], "septic_patients": am["septic_patients"],
                         "AUROC": auc, "AUROC_ci": ci,
                         "septic_caught_pct": am["septic_caught_pct"],
                         "septic_caught_ci": wilson(int(sep["caught"].sum()), len(sep)),
                         "nonseptic_alerted_pct": am["nonseptic_alerted_pct"],
                         "nonseptic_alerted_ci": wilson(int(non["alerted"].sum()), len(non))})
    return pd.DataFrame(rows)
