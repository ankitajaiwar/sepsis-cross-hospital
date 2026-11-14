"""Paired comparisons for the active learning experiment, from saved test predictions.

For each direction and budget, each method is compared with Random (and Random with the
model as-is) on the same resampled test patients:
  - AUROC (hourly)
  - share of septic patients flagged by onset when 20% of patients without sepsis are alerted

Both are averaged over seeds, exactly as in ACTIVE_REPORT.md. Whole patients are resampled
(the same draws for every method), so each difference comes with a paired 95% interval.
Two-sided bootstrap p-values are Holm-adjusted within each direction and measure.

A point-estimate "equal effort" column gives Random's AUROC at the same number of labelled
hours, interpolated between Random's budgets (blank where a method labelled more hours than
Random did at its largest budget).

Run after `python -m sepsis.active`; nothing is retrained:
    python -m sepsis.active_compare --out artifacts_active_full
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import PatientBootstrap, _ci
from .active import FIXED_ALERT_PCT, LABELS

BASELINE = "random"


def find_pred_dir(out: Path) -> Path:
    """The checkpoint folder with saved predictions (the most recently written one)."""
    dirs = [d for d in (out / "checkpoints").glob("*/preds") if any(d.glob("*_test_patients.parquet"))]
    if not dirs:
        raise SystemExit(f"No saved predictions under {out / 'checkpoints'}. "
                         "Run sepsis.active with the current code first.")
    return max(dirs, key=lambda d: max(f.stat().st_mtime for f in d.iterdir()))


def load_preds(pred_dir: Path, tag: str):
    """Test patients for one direction, and {(method, budget): [pred array per seed]}."""
    m = pd.read_parquet(pred_dir / f"{tag}_test_patients.parquet")
    preds: dict[tuple[str, int], list[np.ndarray]] = {}
    pat = re.compile(rf"^{re.escape(tag)}_seed(\d+)_(.+)\.parquet$")
    for f in sorted(pred_dir.glob(f"{tag}_seed*_*.parquet")):
        mt = pat.match(f.name)
        if not mt:
            continue
        name, df = mt.group(2), pd.read_parquet(f)
        if name == "start":
            preds.setdefault(("as_is", 0), []).append(df["as_is"].to_numpy())
            preds.setdefault(("all_target_labels", -1), []).append(df["all_target_labels"].to_numpy())
            continue
        for col in df.columns:
            p = df[col].to_numpy()
            if not np.isnan(p).any():
                preds.setdefault((name, int(col[1:])), []).append(p)
    return m, preds


class FixedAlertBootstrap:
    """Share of septic patients flagged by onset at a fixed alert rate among patients
    without sepsis, on the same patient draws as a PatientBootstrap."""

    def __init__(self, m: pd.DataFrame, boot: PatientBootstrap, alert_pct: float = FIXED_ALERT_PCT):
        self.inv = boot.inv
        n_pat = self.inv.max() + 1
        septic_rows = m["onset_hour"].notna().to_numpy()
        self.septic = np.zeros(n_pat, bool)
        self.septic[self.inv[septic_rows]] = True
        self.pre_rows = septic_rows & (m["hour"] <= m["onset_hour"]).to_numpy()
        self.non_rows = ~septic_rows
        self.counts = np.array(boot.counts)            # draws x patients
        self.q = 1 - alert_pct / 100

    def _patient_max(self, p, rows):
        out = np.full(self.inv.max() + 1, -np.inf)
        np.maximum.at(out, self.inv[rows], p[rows])
        return out

    def point(self, p) -> float:
        return self._one(self._patient_max(p, self.non_rows), self._patient_max(p, self.pre_rows),
                         np.ones(len(self.septic), int))

    def _one(self, non_max, pre_max, w) -> float:
        non = ~self.septic & (w > 0)
        sep = self.septic & (w > 0)
        if not non.any() or not sep.any():
            return np.nan
        thr = np.quantile(np.repeat(non_max[non], w[non]), self.q)
        return float((w[sep] * (pre_max[sep] >= thr)).sum() / w[sep].sum() * 100)

    def draws(self, p) -> np.ndarray:
        non_max, pre_max = self._patient_max(p, self.non_rows), self._patient_max(p, self.pre_rows)
        return np.array([self._one(non_max, pre_max, w) for w in self.counts])


def _holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    m = len(p)
    adj, running = pd.Series(np.nan, index=p.index), 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[idx]))
        adj[idx] = running
    return adj


def compare_direction(m, preds, n_boot: int, seed: int = 1) -> pd.DataFrame:
    boot = PatientBootstrap(m["SepsisLabel"].to_numpy(), m["person_id"].to_numpy(), n_boot, seed)
    fab = FixedAlertBootstrap(m, boot)
    cache: dict = {}

    def stats(key):
        if key not in cache:
            ps = preds[key]
            auc_pt, _, auc_d = boot.mean_auroc(ps)
            fa_pt = float(np.mean([fab.point(p) for p in ps]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fa_d = np.nanmean([fab.draws(p) for p in ps], axis=0)
            cache[key] = (auc_pt, auc_d, fa_pt, fa_d, len(ps))
        return cache[key]

    budgets = sorted({b for (s, b) in preds if s == BASELINE})
    pairs = [((BASELINE, b), ("as_is", 0), "Random vs model as-is") for b in budgets]
    for (s, b) in sorted(preds):
        if s in (BASELINE, "as_is", "all_target_labels", "recalibrate") or (BASELINE, b) not in preds:
            continue
        pairs.append(((s, b), (BASELINE, b), f"{LABELS.get(s, s)} vs Random"))
    rows = []
    for (a, ref, label) in pairs:
        if ref not in preds:
            continue
        A, R = stats(a), stats(ref)
        for metric, i_pt, i_d in (("AUROC", 0, 1), ("flagged_at_20pct_alerts", 2, 3)):
            d = A[i_d] - R[i_d]
            d = d[~np.isnan(d)]
            share = float((d > 0).mean()) if len(d) else np.nan
            rows.append({"comparison": label, "method": a[0], "reference": ref[0], "budget": a[1],
                         "metric": metric, "method_value": A[i_pt], "reference_value": R[i_pt],
                         "diff": A[i_pt] - R[i_pt], "diff_lo": _ci(d)[0], "diff_hi": _ci(d)[1],
                         "share_of_draws_better": share,
                         "p_two_sided": min(1.0, 2 * min(share, 1 - share) + 1 / max(len(d), 1)),
                         "seeds": min(A[4], R[4])})
    return pd.DataFrame(rows)


def equal_hours(results_csv: Path) -> pd.DataFrame:
    """Random's AUROC at each method's number of labelled hours (linear interpolation)."""
    if not results_csv.exists():
        return pd.DataFrame()
    r = pd.read_csv(results_csv)
    g = r[r.budget > 0].groupby(["direction", "strategy", "budget"])[["hours_labelled", "AUROC"]].mean()
    g = g.reset_index()
    rows = []
    for d, gd in g.groupby("direction"):
        rnd = gd[gd.strategy == BASELINE].sort_values("hours_labelled")
        for _, x in gd[~gd.strategy.isin([BASELINE, "recalibrate", "target_only"])].iterrows():
            h = x.hours_labelled
            inside = rnd.hours_labelled.min() <= h <= rnd.hours_labelled.max()
            rows.append({"direction": d, "method": x.strategy, "budget": int(x.budget),
                         "hours_labelled": h,
                         "random_auroc_at_same_hours": float(np.interp(h, rnd.hours_labelled, rnd.AUROC))
                         if inside else np.nan})
    return pd.DataFrame(rows)


def write_markdown(out: Path, res: pd.DataFrame, eq: pd.DataFrame, n_boot: int):
    L = ["# Paired comparisons: active learning\n",
         f"Each method is compared with Random at the same budget, on the same test patients "
         f"resampled {n_boot} times (whole patients). Values are averaged over seeds. "
         "**Bold** = 95% interval excludes 0 *and* Holm-adjusted p < 0.05 (within each direction "
         "and measure). Random is also compared with the model as-is.\n"]
    eqd = {(r.direction, r.method, r.budget): r for r in eq.itertuples()} if len(eq) else {}
    for d in sorted(res.direction.unique()):
        a, b = d.split("->")
        L.append(f"## Hospital {a} model moved to hospital {b}\n")
        for metric, title, fmt in (("AUROC", "AUROC", "{:+.3f}"),
                                   ("flagged_at_20pct_alerts",
                                    "Septic patients flagged at 20% alerts (percentage points)", "{:+.1f}")):
            s = res[(res.direction == d) & (res.metric == metric)]
            L.append(f"### {title}\n")
            extra = metric == "AUROC" and bool(eqd)
            L.append("| Comparison | Patients labelled | Method | Reference | Difference (95% CI) | Holm p |"
                     + (" Random at same hours |" if extra else ""))
            L.append("|---|---|---|---|---|---|" + ("---|" if extra else ""))
            for _, r in s.sort_values(["comparison", "budget"]).iterrows():
                v = (lambda x: f"{x:.3f}") if metric == "AUROC" else (lambda x: f"{x:.0f}%")
                diff = f"{fmt.format(r['diff'])} ({fmt.format(r.diff_lo)} to {fmt.format(r.diff_hi)})"
                if r.p_holm < 0.05 and (r.diff_lo > 0 or r.diff_hi < 0):
                    diff = f"**{diff}**"
                line = (f"| {r.comparison} | {r.budget} | {v(r.method_value)} | {v(r.reference_value)} | "
                        f"{diff} | {r.p_holm:.3f} |")
                if extra:
                    e = eqd.get((d, r.method, r.budget))
                    line += f" {e.random_auroc_at_same_hours:.3f} |" if e is not None and not np.isnan(
                        e.random_auroc_at_same_hours) else " – |"
                L.append(line)
            L.append("")
    L.append("**Random at same hours:** Random's AUROC interpolated at the number of hours the method "
             "labelled (point estimate, from active_results.csv). Blank where the method labelled more "
             "hours than Random at its largest budget. Compare with the method's AUROC to see whether a "
             "gain survives when effort is measured in hours rather than patients.\n")
    L.append("Bootstrap p-values are approximate (resolution 1 / number of draws).\n")
    (out / "ACTIVE_COMPARISONS.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def run(out: Path, n_boot: int = 500, log=print) -> pd.DataFrame:
    pred_dir = find_pred_dir(out)
    log(f"Predictions from {pred_dir}")
    tags = sorted({f.name.split("_test_patients")[0] for f in pred_dir.glob("*_test_patients.parquet")})
    frames = []
    for tag in tags:
        m, preds = load_preds(pred_dir, tag)
        log(f"{tag}: {len(preds)} method x budget combinations, {m['person_id'].nunique():,} test patients")
        r = compare_direction(m, preds, n_boot)
        r.insert(0, "direction", tag.replace("-", "->"))
        frames.append(r)
    res = pd.concat(frames, ignore_index=True)
    res["p_holm"] = np.nan
    for _, idx in res.groupby(["direction", "metric"]).groups.items():
        res.loc[idx, "p_holm"] = _holm(res.loc[idx, "p_two_sided"])
    eq = equal_hours(out / "active_results.csv")
    res.to_csv(out / "active_comparisons.csv", index=False)
    if len(eq):
        eq.to_csv(out / "active_equal_hours.csv", index=False)
    write_markdown(out, res, eq, n_boot)
    log(f"Wrote {out / 'ACTIVE_COMPARISONS.md'}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="artifacts_active_full", help="folder written by sepsis.active")
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args()
    run(Path(a.out), a.n_boot)


if __name__ == "__main__":
    main()
