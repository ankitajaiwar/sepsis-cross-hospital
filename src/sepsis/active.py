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

Reference points: budget 0 (source model as-is) and "all target labels" (upper bound).

    python -m sepsis.active --raw data/raw --out artifacts_active
    python -m sepsis.active --omop artifacts/omop --out artifacts_active  # reuse OMOP tables
"""
from __future__ import annotations

import argparse
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
DEFAULT_BUDGETS = [25, 50, 100, 200, 400]


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

def _evaluate(model, X, meta, test_idx, src_valid_idx, catch_target):
    p_te = model.predict(X.loc[test_idx])
    thr = threshold_for_catch_rate(meta.loc[src_valid_idx], model.predict(X.loc[src_valid_idx]), catch_target)
    m = meta.loc[test_idx]
    y = m["SepsisLabel"].to_numpy()
    out = {"AUROC": roc_auc_score(y, p_te), "AUPRC": average_precision_score(y, p_te),
           "mean_pred": float(p_te.mean()), "prevalence": float(y.mean())}
    out.update(alert_metrics(m, p_te, thr))
    # fairness: worst catch rate across age groups and sexes at the same threshold
    worst = []
    age = pd.cut(m["Age"], [0, 45, 65, 80, 200], right=False)
    for key in (age, m["Gender"]):
        for _, idx in m.groupby(key, observed=True).groups.items():
            sub = m.loc[idx]
            if sub["onset_hour"].notna().groupby(sub["person_id"]).first().sum() >= 5:
                worst.append(alert_metrics(sub, pd.Series(p_te, index=m.index).loc[idx].to_numpy(),
                                           thr)["septic_caught_pct"])
    out["worst_group_caught_pct"] = float(min(worst)) if worst else np.nan
    return out


def run_direction(X, meta, src: str, tgt: str, budgets, strategies, seeds, target_weight: float,
                  catch_target: float, gbm_iter: int, log=print) -> pd.DataFrame:
    sel = lambda site, split: meta.index[(meta.site == site) & (meta.split == split)]
    s_tr, s_va = sel(src, "train"), sel(src, "valid")
    t_pool_rows, t_te = sel(tgt, "train"), sel(tgt, "test")
    pid = meta["person_id"]
    pool_pids = pd.Index(np.sort(pid.loc[t_pool_rows].unique()))
    emb_src = patient_embedding(X.loc[s_tr], meta.loc[s_tr])
    emb_tgt = patient_embedding(X.loc[t_pool_rows], meta.loc[t_pool_rows])
    rows = []

    def fit(seed, lab_pids):
        idx = s_tr
        w = np.ones(len(s_tr))
        if len(lab_pids):
            t_idx = t_pool_rows[pid.loc[t_pool_rows].isin(lab_pids).to_numpy()]
            idx = s_tr.append(t_idx)
            w = np.r_[w, np.full(len(t_idx), target_weight)]
        m = GBMModel(max_iter=gbm_iter, seed=seed).fit(X.loc[idx], meta.loc[idx], w)
        m.calibrate(X.loc[s_va], meta.loc[s_va, "SepsisLabel"].to_numpy())
        return m

    for seed in seeds:
        base = fit(seed, [])
        r0 = _evaluate(base, X, meta, t_te, s_va, catch_target)
        dom = domain_scores(emb_src, emb_tgt, seed)
        for strat in strategies:
            rows.append({"direction": f"{src}->{tgt}", "strategy": strat, "seed": seed,
                         "budget": 0, "septic_labelled": 0, **r0})
        # upper bound: all target training labels
        full = fit(seed, list(pool_pids))
        rows.append({"direction": f"{src}->{tgt}", "strategy": "all_target_labels", "seed": seed,
                     "budget": len(pool_pids), "septic_labelled": int(
                         meta.loc[t_pool_rows].groupby("person_id")["SepsisLabel"].max().sum()),
                     **_evaluate(full, X, meta, t_te, s_va, catch_target)})
        for strat in strategies:
            t0 = time.time()
            model, labelled = base, []
            for b in budgets:
                p_pool = pd.Series(model.predict(X.loc[t_pool_rows]), index=t_pool_rows)
                remaining = pool_pids.difference(labelled)
                new = select(strat, b - len(labelled), remaining, p_pool, pid.loc[t_pool_rows],
                             emb_tgt, dom, labelled, seed + b)
                labelled = labelled + list(new)
                model = fit(seed, labelled)
                n_sep = int(meta.loc[t_pool_rows][pid.loc[t_pool_rows].isin(labelled).to_numpy()]
                            .groupby("person_id")["SepsisLabel"].max().sum())
                res = _evaluate(model, X, meta, t_te, s_va, catch_target)
                rows.append({"direction": f"{src}->{tgt}", "strategy": strat, "seed": seed,
                             "budget": b, "septic_labelled": n_sep, **res})
            log(f"{src}->{tgt} seed {seed} {strat:11s} AUROC@{budgets[-1]} "
                f"{rows[-1]['AUROC']:.3f} ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    keep = ["AUROC", "AUPRC", "septic_caught_pct", "nonseptic_alerted_pct",
            "worst_group_caught_pct", "septic_labelled"]
    g = res.groupby(["direction", "strategy", "budget"])[keep]
    out = g.mean().join(g.std().add_suffix("_sd")).reset_index()
    return out


# ---------- figures + report ----------

COLORS = {"random": "#6b6b66", "uncertainty": "#2a78d6", "high_risk": "#eb6834",
          "diversity": "#1baf7a", "domain": "#eda100", "hybrid": "#e87ba4"}
MARKERS = {"random": "o", "uncertainty": "s", "high_risk": "^", "diversity": "D",
           "domain": "v", "hybrid": "P"}
LABELS = {"random": "Random", "uncertainty": "Uncertainty", "high_risk": "Highest risk",
          "diversity": "Diversity", "domain": "Least like source", "hybrid": "Hybrid (proposed)"}


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
            ax.plot(x, c[metric], color=COLORS[strat], marker=MARKERS[strat], ms=6, lw=2 if strat == "hybrid" else 1.6,
                    mec="white", mew=1, label=LABELS[strat])
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
    fig.legend(h, l, loc="upper center", ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, x=0.01, y=1.08, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(out: Path, summ: pd.DataFrame, budgets, seeds, target_weight, synthetic):
    L = ["# Which patients should a new hospital label?\n"]
    if synthetic:
        L.append("> **Made-up test data.** These numbers only show the code runs.\n")
    L.append(f"A model trained at one hospital is moved to the other. The new hospital reviews the "
             f"charts of a small number of its own patients, chosen by each strategy, in rounds of "
             f"{', '.join(map(str, budgets))} patients in total. After each round the model is "
             f"retrained on the original hospital's data plus the newly labelled patients (each "
             f"weighted {target_weight:g}x), and tested on held-out patients at the new hospital. "
             f"Results are averaged over {len(seeds)} random seeds.\n")
    L.append("![ranking accuracy](active_auroc.png)\n")
    L.append("![septic patients flagged](active_caught.png)\n")
    for d in sorted(summ.direction.unique()):
        s = summ[summ.direction == d]
        L.append(f"## Hospital {d.split('->')[0]} model moved to hospital {d.split('->')[1]}\n")
        L.append("| Strategy | Patients labelled | Of them septic | AUROC (± SD) | Septic patients flagged | "
                 "Patients without sepsis alerted | Worst group flagged |")
        L.append("|---|---|---|---|---|---|---|")
        for _, r in s.sort_values(["budget", "strategy"]).iterrows():
            if r.budget == 0 and r.strategy != "random":
                continue
            name = "None (model as-is)" if r.budget == 0 else (
                "All target labels" if r.strategy == "all_target_labels" else LABELS.get(r.strategy, r.strategy))
            L.append(f"| {name} | {int(r.budget):,} | {r.septic_labelled:.1f} | "
                     f"{r.AUROC:.3f} (± {r.AUROC_sd:.3f}) | {r.septic_caught_pct:.0f}% | "
                     f"{r.nonseptic_alerted_pct:.0f}% | {r.worst_group_caught_pct:.0f}% |")
        L.append("")
    L.append("Alert threshold: set on the original hospital's validation patients to flag about 80% "
             "of its septic patients. \"Worst group\" is the lowest share of septic patients flagged "
             "in any age group or sex with at least 5 septic patients.\n")
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
        strategies=STRATEGIES, target_weight=10.0, catch_target=80.0, gbm_iter=200, limit=0, log=print):
    out.mkdir(parents=True, exist_ok=True)
    syn = None
    if synthetic:
        from .synthetic import write_synthetic
        syn = out / "synthetic_raw"
        write_synthetic(syn, n_per_site=limit or 400)
    X, meta = prepare(raw, omop, syn, 0 if synthetic else limit)
    res = pd.concat([run_direction(X, meta, s, t, budgets, strategies, seeds, target_weight,
                                   catch_target, gbm_iter, log) for s, t in [("A", "B"), ("B", "A")]])
    res.to_csv(out / "active_results.csv", index=False)
    summ = summarize(res)
    summ.to_csv(out / "active_summary.csv", index=False)
    fig_curves(summ, "AUROC", "Ranking accuracy (AUROC)", "Ranking accuracy as more patients are labelled",
               out / "active_auroc.png")
    fig_curves(summ, "septic_caught_pct", "Septic patients flagged by onset (%)",
               "Septic patients flagged at the new hospital", out / "active_caught.png")
    write_report(out, summ, budgets, seeds, target_weight, synthetic)
    (out / "active_config.json").write_text(json.dumps(
        {"budgets": list(budgets), "seeds": list(seeds), "strategies": list(strategies),
         "target_weight": target_weight, "catch_target": catch_target, "gbm_iter": gbm_iter}, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--omop", default=None, help="reuse OMOP tables written by the main pipeline")
    ap.add_argument("--out", default="artifacts_active")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--target-weight", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    run(Path(a.out), a.raw, a.omop, a.synthetic, [int(b) for b in a.budgets.split(",")],
        tuple(range(1 if a.fast else a.seeds)), a.strategies.split(","), a.target_weight,
        gbm_iter=80 if a.fast else 200, limit=a.limit)
    print(f"Done in {(time.time() - t0) / 60:.1f} min -> {Path(a.out) / 'ACTIVE_REPORT.md'}")


if __name__ == "__main__":
    main()
