"""Which patients should a new hospital label? Active domain adaptation experiments.

A model trained at the source hospital is moved to the target hospital. The target
hospital can afford to review the charts of only a few of its own patients (the label
budget). Each strategy below picks which target patients to label, in rounds; after each
round the model is retrained on source data plus the labelled target patients, then
tested on held-out target patients.

Strategies (each scores whole patients, because chart review labels a whole stay)
  random       : pick at random (the usual baseline)
  uncertainty  : patients whose hourly risk scores are most uncertain on average
  high_risk    : patients with the highest peak risk (finds sepsis cases, which are rare)
  diversity    : spread the picks across the target population (k-center greedy)
  domain       : patients least like the source hospital, per a hospital classifier
  hybrid       : rank-average of uncertainty and domain scores, then diversity among the
                 top candidates (proposed in this project)

Comparison methods (both label the same random patients as `random`)
  recalibrate  : keep the source model unchanged and only reset the alert threshold on the
                 labelled patients
  target_only  : train only on the labelled new-hospital patients

Reference points: budget 0 (source model as-is) and "all target labels" (upper bound).

Every round is scored three ways: with the alert threshold set at the original hospital,
with the threshold reset on the labelled new-hospital patients, and as the share of septic
patients flagged when 20% of patients without sepsis are alerted (a fixed operating point,
so strategies are compared on ranking without threshold effects). Per-patient test
predictions are saved for every round so paired bootstrap comparisons can be run later.

    python -m sepsis.active --raw data/raw --out artifacts_active
    python -m sepsis.active --omop artifacts/omop --out artifacts_active  # reuse OMOP tables
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from .evaluate import alert_metrics, threshold_for_catch_rate
from .experiments import META_COLS
from .features import assign_splits, build_features
from .gbm import GBMModel
from .omop import OMOP, cohort_from_omop, to_omop
from .quality import clean
from .raw import load_raw

STRATEGIES = ["random", "uncertainty", "high_risk", "diversity", "domain", "hybrid"]
BASELINES = ["recalibrate", "target_only"]
DEFAULT_BUDGETS = [25, 50, 100, 200, 400]
MIN_SEPTIC_FOR_THRESHOLD = 3     # fewer labelled septic patients -> keep the original threshold
FIXED_ALERT_PCT = 20.0           # operating point for comparing ranking at the bedside
RESULTS_VERSION = 2              # bump when result columns change, so old checkpoints are not reused


# ---------- patient-level summaries ----------

def patient_embedding(X: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """One row per patient: mean and last value of each feature, standardised."""
    g = X.groupby(meta["person_id"].to_numpy(), sort=True)
    emb = pd.concat([g.mean().add_suffix("_mean"), g.last().add_suffix("_last")], axis=1)
    emb = (emb - emb.mean()) / emb.std().replace(0, 1)
    return emb.fillna(0).astype("float32")


def _rank01(s: pd.Series) -> pd.Series:
    return s.rank(pct=True)


def k_center(emb: np.ndarray, k: int, seed: int = 0) -> list[int]:
    """Greedy k-center: repeatedly take the point farthest from everything picked so far."""
    n = len(emb)
    if k >= n:
        return list(range(n))
    first = int(np.random.default_rng(seed).integers(n))
    picks = [first]
    d = ((emb - emb[first]) ** 2).sum(1)
    for _ in range(k - 1):
        i = int(np.argmax(d))
        picks.append(i)
        d = np.minimum(d, ((emb - emb[i]) ** 2).sum(1))
    return picks


def domain_scores(emb_src: pd.DataFrame, emb_tgt: pd.DataFrame, seed: int = 0) -> pd.Series:
    """P(target hospital | patient summary) for target patients. Uses no labels."""
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=seed)
    Xd = pd.concat([emb_src, emb_tgt])
    yd = np.r_[np.zeros(len(emb_src)), np.ones(len(emb_tgt))]
    # out-of-fold scores for the target patients, so the classifier isn't scoring its own training data
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, 3, len(Xd))
    out = np.zeros(len(Xd))
    for f in range(3):
        tr, te = fold != f, fold == f
        clf.fit(Xd[tr], yd[tr])
        out[te] = clf.predict_proba(Xd[te])[:, 1]
    return pd.Series(out[len(emb_src):], index=emb_tgt.index)


def select(strategy: str, k: int, pool: pd.Index, hourly_p: pd.Series, pid_of_row: pd.Series,
           emb_tgt: pd.DataFrame, dom: pd.Series, labelled: list, seed: int) -> list:
    """Return k patient ids from `pool` (patients not yet labelled)."""
    rng = np.random.default_rng(seed)
    k = min(k, len(pool))
    if strategy == "random":
        return list(rng.choice(pool.to_numpy(), k, replace=False))
    peak = hourly_p.groupby(pid_of_row).max()
    unc = (hourly_p * (1 - hourly_p)).groupby(pid_of_row).mean().reindex(pool)
    if strategy == "uncertainty":
        return list(unc.nlargest(k).index)
    if strategy == "high_risk":
        return list(peak.reindex(pool).nlargest(k).index)
    if strategy == "domain":
        return list(dom.reindex(pool).nlargest(k).index)
    if strategy == "diversity":
        E = emb_tgt.reindex(pool).to_numpy()
        picks = k_center(E, k, seed=seed)
        return list(pool[picks])
    if strategy == "hybrid":
        score = (_rank01(unc) + _rank01(dom.reindex(pool))) / 2
        cand = score.nlargest(min(len(pool), 3 * k)).index
        E = emb_tgt.reindex(cand).to_numpy()
        picks = k_center(E, k, seed=seed)
        return list(cand[picks])
    raise ValueError(strategy)


# ---------- one experiment ----------

def _worst_group(m: pd.DataFrame, p: np.ndarray, thr: float) -> float:
    """Lowest share of septic patients flagged in any age group or sex with >= 5 septic patients."""
    worst = []
    ps = pd.Series(p, index=m.index)
    age = pd.cut(m["Age"], [0, 45, 65, 80, 200], right=False)
    for key in (age, m["Gender"]):
        for _, idx in m.groupby(key, observed=True).groups.items():
            sub = m.loc[idx]
            if sub["onset_hour"].notna().groupby(sub["person_id"]).first().sum() >= 5:
                worst.append(alert_metrics(sub, ps.loc[idx].to_numpy(), thr)["septic_caught_pct"])
    return float(min(worst)) if worst else np.nan


def caught_at_fixed_alerts(m: pd.DataFrame, p: np.ndarray, alert_pct: float = FIXED_ALERT_PCT) -> float:
    """Share of septic patients flagged by onset when the threshold is chosen so that
    `alert_pct`% of patients without sepsis get at least one alert."""
    d = m[["person_id", "hour", "onset_hour"]].assign(p=p)
    septic = d["onset_hour"].notna()
    non_max = d[~septic].groupby("person_id")["p"].max()
    pre = d[septic & (d["hour"] <= d["onset_hour"])].groupby("person_id")["p"].max()
    n_sep = d.loc[septic, "person_id"].nunique()
    if non_max.empty or n_sep == 0:
        return np.nan
    thr = float(np.quantile(non_max.to_numpy(), 1 - alert_pct / 100))
    return float((pre >= thr).sum() / n_sep * 100)


def _evaluate(p_te: np.ndarray, m_te: pd.DataFrame, thr_src: float, thr_tgt: float | None) -> dict:
    """Scores for one model on the new hospital's test patients.

    thr_src: threshold set on the original hospital's validation patients.
    thr_tgt: threshold reset on the labelled new-hospital patients (None = not enough of
             them were septic, so the original threshold is kept)."""
    y = m_te["SepsisLabel"].to_numpy()
    if np.isnan(p_te).any():
        return {}
    out = {"AUROC": roc_auc_score(y, p_te), "AUPRC": average_precision_score(y, p_te),
           "mean_pred": float(p_te.mean()), "prevalence": float(y.mean()),
           "caught_at_fixed_alerts_pct": caught_at_fixed_alerts(m_te, p_te)}
    out.update(alert_metrics(m_te, p_te, thr_src))
    out["worst_group_caught_pct"] = _worst_group(m_te, p_te, thr_src)
    t = thr_src if thr_tgt is None else thr_tgt
    am = alert_metrics(m_te, p_te, t)
    out.update({"tgt_threshold": float(t), "tgt_threshold_from_target": int(thr_tgt is not None),
                "tgt_septic_caught_pct": am["septic_caught_pct"],
                "tgt_nonseptic_alerted_pct": am["nonseptic_alerted_pct"],
                "tgt_worst_group_caught_pct": _worst_group(m_te, p_te, t)})
    return out


def _target_threshold(m_lab: pd.DataFrame, p_lab: np.ndarray, catch_target: float) -> float | None:
    n_sep = m_lab.loc[m_lab["onset_hour"].notna(), "person_id"].nunique()
    if n_sep < MIN_SEPTIC_FOR_THRESHOLD:
        return None
    return threshold_for_catch_rate(m_lab, p_lab, catch_target)


def run_direction(X, meta, src: str, tgt: str, budgets, strategies, seeds, target_weight: float,
                  catch_target: float, gbm_iter: int, log=print, ckpt: Path | None = None,
                  weight_by: str = "hour") -> pd.DataFrame:
    """With ckpt, results are saved after every (seed, strategy), so a stopped run can be
    restarted with the same command and only the unfinished parts are run again.

    weight_by="hour": every labelled new-hospital hour counts `target_weight` times.
    weight_by="patient": each labelled patient carries the same total weight
    (`target_weight` x the average stay length), however long their stay."""
    sel = lambda site, split: meta.index[(meta.site == site) & (meta.split == split)]
    s_tr, s_va = sel(src, "train"), sel(src, "valid")
    t_pool_rows, t_te = sel(tgt, "train"), sel(tgt, "test")
    pid = meta["person_id"]
    pid_pool = pid.loc[t_pool_rows]
    pool_pids = pd.Index(np.sort(pid_pool.unique()))
    hours_per_pid = pid_pool.value_counts()
    mean_hours = float(hours_per_pid.mean())
    emb_src = patient_embedding(X.loc[s_tr], meta.loc[s_tr])
    emb_tgt = patient_embedding(X.loc[t_pool_rows], meta.loc[t_pool_rows])
    m_te = meta.loc[t_te]
    rows = []
    pred_dir = None
    if ckpt:
        pred_dir = ckpt / "preds"
        pred_dir.mkdir(exist_ok=True)
        f = pred_dir / f"{src}-{tgt}_test_patients.parquet"
        if not f.exists():
            m_te[["person_id", "hour", "onset_hour", "SepsisLabel", "Age", "Gender"]].reset_index(
                drop=True).to_parquet(f)

    def lab_rows(lab_pids):
        return t_pool_rows[pid_pool.isin(lab_pids).to_numpy()]

    def fit(seed, lab_pids, target_only=False):
        t_idx = lab_rows(lab_pids) if len(lab_pids) else t_pool_rows[:0]
        if weight_by == "patient":
            w_t = target_weight * mean_hours / hours_per_pid.reindex(pid.loc[t_idx]).to_numpy()
        else:
            w_t = np.full(len(t_idx), target_weight)
        if target_only:
            if meta.loc[t_idx, "SepsisLabel"].nunique() < 2:
                return None                       # no septic hours labelled yet: cannot train
            m = GBMModel(max_iter=gbm_iter, seed=seed)
            m.clf.set_params(min_samples_leaf=int(min(200, max(10, len(t_idx) // 50))))
            return m.fit(X.loc[t_idx], meta.loc[t_idx])
        idx = s_tr.append(t_idx)
        w = np.r_[np.ones(len(s_tr)), w_t]
        m = GBMModel(max_iter=gbm_iter, seed=seed).fit(X.loc[idx], meta.loc[idx], w)
        m.calibrate(X.loc[s_va], meta.loc[s_va, "SepsisLabel"].to_numpy())
        return m

    def score(model, lab_pids, threshold_model=None):
        """Evaluate `model` on the test patients. The original-hospital threshold comes from
        `threshold_model` (defaults to `model`); the new-hospital threshold from the labelled
        patients' scores under `model`."""
        if model is None:
            return {}, np.full(len(t_te), np.nan, dtype="float32")
        p_te = model.predict(X.loc[t_te])
        tm = threshold_model or model
        thr_src = threshold_for_catch_rate(meta.loc[s_va], tm.predict(X.loc[s_va]), catch_target)
        thr_tgt = None
        if len(lab_pids):
            li = lab_rows(lab_pids)
            thr_tgt = _target_threshold(meta.loc[li], model.predict(X.loc[li]), catch_target)
        return _evaluate(p_te, m_te, thr_src, thr_tgt), p_te.astype("float32")

    tag = f"{src}-{tgt}"
    methods = list(strategies)

    for seed in seeds:
        files = {name: (ckpt / f"{tag}_seed{seed}_{name}.csv") if ckpt else None
                 for name in ["start"] + methods}
        if ckpt and all(f.exists() for f in files.values()):
            for f in files.values():
                rows.extend(pd.read_csv(f).to_dict("records"))
            log(f"{src}->{tgt} seed {seed}: already finished, loaded from checkpoint")
            continue
        base = fit(seed, [])
        dom = domain_scores(emb_src, emb_tgt, seed)
        if ckpt and files["start"].exists():
            rows.extend(pd.read_csv(files["start"]).to_dict("records"))
        else:
            r0, p0 = score(base, [])
            start = [{"direction": f"{src}->{tgt}", "strategy": strat, "seed": seed, "budget": 0,
                      "septic_labelled": 0, "hours_labelled": 0, **r0} for strat in methods]
            # upper bound: all target training labels
            full = fit(seed, list(pool_pids))
            r_full, p_full = score(full, [])
            start.append({"direction": f"{src}->{tgt}", "strategy": "all_target_labels", "seed": seed,
                          "budget": len(pool_pids), "septic_labelled": int(
                              meta.loc[t_pool_rows].groupby("person_id")["SepsisLabel"].max().sum()),
                          "hours_labelled": len(t_pool_rows), **r_full})
            rows.extend(start)
            if ckpt:
                pd.DataFrame(start).to_csv(files["start"], index=False)
                pd.DataFrame({"as_is": p0, "all_target_labels": p_full}).to_parquet(
                    pred_dir / f"{tag}_seed{seed}_start.parquet")
        for strat in methods:
            if ckpt and files[strat].exists():
                rows.extend(pd.read_csv(files[strat]).to_dict("records"))
                log(f"{src}->{tgt} seed {seed} {strat:11s} loaded from checkpoint")
                continue
            t0 = time.time()
            picker = "random" if strat in BASELINES else strat
            model, labelled, done, preds = base, [], [], {}
            for b in budgets:
                p_pool = None if picker == "random" else pd.Series(
                    model.predict(X.loc[t_pool_rows]), index=t_pool_rows)
                remaining = pool_pids.difference(labelled)
                new = select(picker, b - len(labelled), remaining, p_pool, pid_pool,
                             emb_tgt, dom, labelled, seed + b)
                labelled = labelled + list(new)
                if strat == "recalibrate":
                    res, p_te = score(base, labelled)
                else:
                    model = fit(seed, labelled, target_only=(strat == "target_only"))
                    res, p_te = score(model, labelled)
                    if model is None:
                        model = base
                lab_meta = meta.loc[lab_rows(labelled)]
                n_sep = int(lab_meta.groupby("person_id")["SepsisLabel"].max().sum())
                done.append({"direction": f"{src}->{tgt}", "strategy": strat, "seed": seed,
                             "budget": b, "septic_labelled": n_sep, "hours_labelled": len(lab_meta),
                             **res})
                preds[f"b{b}"] = p_te
            rows.extend(done)
            if ckpt:
                pd.DataFrame(preds).to_parquet(pred_dir / f"{tag}_seed{seed}_{strat}.parquet")
                pd.DataFrame(done).to_csv(files[strat], index=False)
            log(f"{src}->{tgt} seed {seed} {strat:11s} AUROC@{budgets[-1]} "
                f"{done[-1].get('AUROC', np.nan):.3f} ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    keep = ["AUROC", "AUPRC", "septic_caught_pct", "nonseptic_alerted_pct",
            "worst_group_caught_pct", "septic_labelled", "hours_labelled",
            "caught_at_fixed_alerts_pct", "tgt_septic_caught_pct", "tgt_nonseptic_alerted_pct",
            "tgt_worst_group_caught_pct", "tgt_threshold_from_target"]
    keep = [k for k in keep if k in res.columns]
    g = res.groupby(["direction", "strategy", "budget"])[keep]
    out = g.mean().join(g.std().add_suffix("_sd")).reset_index()
    return out


# ---------- figures + report ----------

COLORS = {"random": "#6b6b66", "uncertainty": "#2a78d6", "high_risk": "#eb6834",
          "diversity": "#1baf7a", "domain": "#eda100", "hybrid": "#e87ba4",
          "recalibrate": "#2b2b29", "target_only": "#8a5cc2"}
MARKERS = {"random": "o", "uncertainty": "s", "high_risk": "^", "diversity": "D",
           "domain": "v", "hybrid": "P", "recalibrate": "x", "target_only": "*"}
LABELS = {"random": "Random", "uncertainty": "Uncertainty", "high_risk": "Highest risk",
          "diversity": "Diversity", "domain": "Least like source", "hybrid": "Hybrid",
          "recalibrate": "Threshold reset only", "target_only": "New hospital only"}


def fig_curves(summ: pd.DataFrame, metric: str, ylabel: str, title: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dirs = sorted(summ.direction.unique())
    fig, axes = plt.subplots(1, len(dirs), figsize=(5.2 * len(dirs), 3.6), squeeze=False)
    for ax, d in zip(axes[0], dirs):
        s = summ[summ.direction == d]
        ub = s[s.strategy == "all_target_labels"]
        if len(ub):
            ax.axhline(ub[metric].iloc[0], color="#2b2b29", ls="--", lw=1)
            ax.text(0.99, ub[metric].iloc[0], " all target labels", va="bottom", ha="right",
                    transform=ax.get_yaxis_transform(), fontsize=7.5, color="#2b2b29")
        for strat in [x for x in COLORS if x in set(s.strategy)]:
            c = s[s.strategy == strat].sort_values("budget")
            x = c.budget.replace(0, c.budget[c.budget > 0].min() / 2 if (c.budget > 0).any() else 1)
            ax.plot(x, c[metric], color=COLORS[strat], marker=MARKERS[strat], ms=6, lw=1.6,
                    ls="--" if strat in BASELINES else "-", mec="white", mew=1, label=LABELS[strat])
        ax.set_xscale("log")
        bud = sorted(s.loc[(s.strategy != "all_target_labels") & (s.budget > 0), "budget"].unique())
        if bud:
            ticks = [bud[0] / 2] + bud
            ax.set_xticks(ticks)
            ax.set_xticklabels(["0"] + [str(int(b)) for b in bud])
            ax.minorticks_off()
        ax.set_xlabel("Target-hospital patients labelled")
        ax.set_title(f"Hospital {d.split('->')[0]} model moved to hospital {d.split('->')[1]}",
                     fontsize=10, fontweight="bold", loc="left")
        ax.grid(color="#e4e3df", lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0][0].set_ylabel(ylabel)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=8, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, x=0.01, y=1.08, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(out: Path, summ: pd.DataFrame, budgets, seeds, target_weight, synthetic,
                 weight_by: str = "hour"):
    L = ["# Which patients should a new hospital label?\n"]
    if synthetic:
        L.append("> **Made-up test data.** These numbers only show the code runs.\n")
    per = "each hour" if weight_by == "hour" else "each patient, regardless of stay length,"
    L.append(f"A model trained at one hospital is moved to the other. The new hospital reviews the "
             f"charts of a small number of its own patients, chosen by each strategy, in rounds of "
             f"{', '.join(map(str, budgets))} patients in total. After each round the model is "
             f"retrained on the original hospital's data plus the newly labelled patients ({per} "
             f"weighted {target_weight:g}x), and tested on held-out patients at the new hospital. "
             f"Results are averaged over {len(seeds)} random seeds.\n")
    L.append("Two comparison methods label the same random patients as *Random*: *Threshold reset "
             "only* keeps the original model and only resets its alert threshold on the labelled "
             "patients; *New hospital only* trains on the labelled patients alone.\n")
    L.append("![ranking accuracy](active_auroc.png)\n")
    L.append("![septic patients flagged at a fixed alert rate](active_caught_fixed.png)\n")
    L.append("![septic patients flagged, threshold reset at the new hospital](active_caught_tgt.png)\n")
    has_tgt = "tgt_septic_caught_pct" in summ.columns
    for d in sorted(summ.direction.unique()):
        s = summ[summ.direction == d]
        L.append(f"## Hospital {d.split('->')[0]} model moved to hospital {d.split('->')[1]}\n")
        L.append("| Strategy | Patients labelled | Of them septic | Hours labelled | AUROC (± SD) | "
                 "Flagged at 20% alerts | Original threshold: flagged / alerted without sepsis | "
                 "Reset threshold: flagged / alerted without sepsis | Worst group flagged (original / reset) |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in s.sort_values(["budget", "strategy"]).iterrows():
            if r.budget == 0 and r.strategy != "random":
                continue
            name = "None (model as-is)" if r.budget == 0 else (
                "All target labels" if r.strategy == "all_target_labels" else LABELS.get(r.strategy, r.strategy))
            fmt = lambda v, suf="%": "–" if pd.isna(v) else f"{v:.0f}{suf}"
            if has_tgt and r.budget > 0 and r.strategy != "all_target_labels":
                reset = f"{fmt(r.tgt_septic_caught_pct)} / {fmt(r.tgt_nonseptic_alerted_pct)}"
                if r.tgt_threshold_from_target < 1:
                    reset += f" ({r.tgt_threshold_from_target * 100:.0f}% of seeds)"
                worst = f"{fmt(r.worst_group_caught_pct)} / {fmt(r.tgt_worst_group_caught_pct)}"
            else:
                reset, worst = "–", f"{fmt(r.worst_group_caught_pct)} / –"
            auroc = "–" if pd.isna(r.AUROC) else f"{r.AUROC:.3f} (± {r.AUROC_sd:.3f})"
            L.append(f"| {name} | {int(r.budget):,} | {r.septic_labelled:.1f} | "
                     f"{r.get('hours_labelled', np.nan):,.0f} | {auroc} | "
                     f"{fmt(r.get('caught_at_fixed_alerts_pct', np.nan))} | "
                     f"{fmt(r.septic_caught_pct)} / {fmt(r.nonseptic_alerted_pct)} | {reset} | {worst} |")
        L.append("")
    L.append("**Flagged at 20% alerts:** share of septic patients flagged by onset when the threshold "
             "is chosen so that 20% of patients without sepsis get an alert. This compares ranking "
             "at the bedside without any threshold effect.\n")
    L.append("**Original threshold:** set on the original hospital's validation patients to flag about "
             "80% of its septic patients.\n")
    L.append(f"**Reset threshold:** set on the labelled new-hospital patients to flag about 80% of "
             f"them. It needs at least {MIN_SEPTIC_FOR_THRESHOLD} labelled septic patients; otherwise "
             "the original threshold is kept, and the share of seeds where it could be reset is shown "
             "in brackets. For retrained models the labelled patients were also used for training, so "
             "this threshold is set on patients the model has seen and may flag fewer test patients "
             "than intended. *Threshold reset only* does not have this problem.\n")
    L.append("**Worst group:** the lowest share of septic patients flagged in any age group or sex "
             "with at least 5 septic patients.\n")
    (out / "ACTIVE_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------- CLI ----------

def prepare(raw: str | None, omop: str | None, synthetic_dir: Path | None, limit: int):
    if omop:
        df = cohort_from_omop(OMOP.load(omop))
    else:
        raw_df = load_raw(synthetic_dir or raw, limit=limit)
        df = cohort_from_omop(to_omop(raw_df))
    df, _ = clean(df)
    df = assign_splits(df)              # same seed as the main pipeline -> same patient splits
    X = build_features(df)
    return X, df[META_COLS]


def run(out: Path, raw="data/raw", omop=None, synthetic=False, budgets=DEFAULT_BUDGETS, seeds=(0, 1, 2),
        strategies=tuple(STRATEGIES + BASELINES), target_weight=10.0, catch_target=80.0, gbm_iter=200,
        limit=0, weight_by="hour", log=print):
    out.mkdir(parents=True, exist_ok=True)
    syn = None
    if synthetic:
        from .synthetic import write_synthetic
        syn = out / "synthetic_raw"
        write_synthetic(syn, n_per_site=limit or 400)
    X, meta = prepare(raw, omop, syn, 0 if synthetic else limit)
    # checkpoints are kept per settings, so changing budgets etc. never mixes old and new results
    cfg = {"budgets": list(budgets), "target_weight": target_weight, "catch_target": catch_target,
           "gbm_iter": gbm_iter, "limit": limit, "synthetic": synthetic, "weight_by": weight_by,
           "version": RESULTS_VERSION}
    key = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
    ckpt = out / "checkpoints" / key
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "settings.json").write_text(json.dumps(cfg, indent=2))
    res = pd.concat([run_direction(X, meta, s, t, budgets, strategies, seeds, target_weight,
                                   catch_target, gbm_iter, log, ckpt, weight_by)
                     for s, t in [("A", "B"), ("B", "A")]])
    res.to_csv(out / "active_results.csv", index=False)
    summ = summarize(res)
    summ.to_csv(out / "active_summary.csv", index=False)
    fig_curves(summ, "AUROC", "Ranking accuracy (AUROC)", "Ranking accuracy as more patients are labelled",
               out / "active_auroc.png")
    fig_curves(summ, "septic_caught_pct", "Septic patients flagged by onset (%)",
               "Septic patients flagged at the new hospital (original threshold)", out / "active_caught.png")
    fig_curves(summ, "caught_at_fixed_alerts_pct", "Septic patients flagged (%)",
               "Septic patients flagged when 20% of patients without sepsis are alerted",
               out / "active_caught_fixed.png")
    fig_curves(summ, "tgt_septic_caught_pct", "Septic patients flagged by onset (%)",
               "Septic patients flagged, threshold reset on the labelled new-hospital patients",
               out / "active_caught_tgt.png")
    write_report(out, summ, budgets, seeds, target_weight, synthetic, weight_by)
    (out / "active_config.json").write_text(json.dumps(
        {"budgets": list(budgets), "seeds": list(seeds), "strategies": list(strategies),
         "target_weight": target_weight, "weight_by": weight_by, "catch_target": catch_target,
         "gbm_iter": gbm_iter}, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--omop", default=None, help="reuse OMOP tables written by the main pipeline")
    ap.add_argument("--out", default="artifacts_active")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--strategies", default=",".join(STRATEGIES + BASELINES),
                    help="comma-separated; also accepts the comparison methods recalibrate, target_only")
    ap.add_argument("--target-weight", type=float, default=10.0)
    ap.add_argument("--weight-by", choices=["hour", "patient"], default="hour",
                    help="apply --target-weight to each labelled hour, or give each labelled patient "
                         "the same total weight regardless of stay length")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    run(Path(a.out), a.raw, a.omop, a.synthetic, [int(b) for b in a.budgets.split(",")],
        tuple(range(1 if a.fast else a.seeds)), a.strategies.split(","), a.target_weight,
        gbm_iter=80 if a.fast else 200, limit=a.limit, weight_by=a.weight_by)
    print(f"Done in {(time.time() - t0) / 60:.1f} min -> {Path(a.out) / 'ACTIVE_REPORT.md'}")


if __name__ == "__main__":
    main()
