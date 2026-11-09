"""Charts and a plain-language report for one run."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluate import calibration_table, tradeoff_curve
from .quality import missingness_table

TREE, NET = "#2a78d6", "#eb6834"           # validated categorical slots 1 and 2
INK, MUTED, GRID = "#2b2b29", "#6b6b66", "#e4e3df"
NAMES = {"gbm": "Gradient-boosted trees", "gru": "Neural network (GRU)"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "axes.titlelocation": "left"})


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


ROWS = [("within", "standard", "Trained at this hospital"),
        ("pooled", "standard", "Trained on both hospitals"),
        ("cross", "source_only", "Trained at the other hospital"),
        ("cross", "site_normalized", "  + adjusted to local norms"),
        ("cross", "adapt", "  + domain adaptation")]


def _row_value(res, model, test, setting, variant):
    r = res[(res.model == model) & (res.test == test) & (res.setting == setting)]
    if variant == "adapt":
        r = r[r.variant.isin(["importance_wt", "dann"])]
    else:
        r = r[r.variant == variant]
    return r.iloc[0] if len(r) else None


def fig_cross_hospital(res: pd.DataFrame, path: Path):
    models = [m for m in ("gbm", "gru") if m in set(res.model)]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), sharey=True)
    for ax, test in zip(axes, ["A", "B"]):
        for k, m in enumerate(models):
            ys, xs = [], []
            for i, (s, v, _) in enumerate(ROWS):
                r = _row_value(res, m, test, s, v)
                if r is not None:
                    ys.append(len(ROWS) - 1 - i + (0.12 if k == 0 else -0.12))
                    xs.append(r.AUROC)
            ax.scatter(xs, ys, s=70, color=TREE if m == "gbm" else NET,
                       marker="o" if m == "gbm" else "s", edgecolor="white", lw=1.5,
                       zorder=3, label=NAMES[m])
            for x, y in zip(xs, ys):
                ax.text(x + 0.004, y, f"{x:.3f}", va="center", fontsize=7.5, color=INK)
        ax.set_yticks(range(len(ROWS)))
        ax.set_yticklabels([r[2] for r in ROWS][::-1])
        ax.axhline(2.5, color=GRID, lw=0.8)
        ax.set_title(f"Tested on hospital {test} patients")
        ax.set_xlabel("Ranking accuracy (AUROC)")
        _clean(ax)
    lo = max(0.5, res.AUROC.min() - 0.03)
    for ax in axes:
        ax.set_xlim(lo, min(1.0, res.AUROC.max() + 0.04))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.6, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_calibration(meta_test: pd.DataFrame, p: pd.Series, path: Path, model_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))
    for ax, site in zip(axes, ["A", "B"]):
        m = meta_test[meta_test.site == site]
        cal = calibration_table(m.SepsisLabel.to_numpy(), p.loc[m.index].to_numpy())
        top = max(cal.max().max(), 0.05) * 1.1
        ax.plot([0, top], [0, top], ls="--", color=MUTED, lw=1)
        ax.plot(cal.mean_predicted, cal.observed, color=TREE, lw=2, marker="o", ms=6,
                mec="white", mew=1.2)
        ax.set_title(f"Hospital {site}")
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_xlabel("Predicted risk")
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Actual share of sepsis-labelled hours")
    fig.suptitle(f"Predicted vs. actual risk ({model_name}, trained on both hospitals)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_tradeoff(curves: dict, path: Path):
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    for m, c in curves.items():
        col = TREE if m == "gbm" else NET
        ax.plot(c.alert_hours_per_100_patient_days, c.septic_caught_pct, color=col, lw=2,
                marker="o" if m == "gbm" else "s", ms=5, mec="white", mew=1, label=NAMES[m])
    ax.set_xlabel("Alert-hours per 100 patient-days")
    ax.set_ylabel("Septic patients flagged by onset (%)")
    ax.set_title("More alerts catch more cases: the trade-off")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_data_shift(qrep: dict, path: Path, top: int = 12):
    t = missingness_table(qrep)
    t["gap"] = (t["A"] - t["B"]).abs()
    t = t.sort_values("gap", ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(6, 0.28 * len(t) + 1))
    y = np.arange(len(t))
    ax.hlines(y, t[["A", "B"]].min(axis=1), t[["A", "B"]].max(axis=1), color=GRID, lw=3)
    ax.scatter(t["A"], y, s=55, color=INK, zorder=3, label="Hospital A")
    ax.scatter(t["B"], y, s=55, color="white", edgecolor=INK, lw=1.5, zorder=3, label="Hospital B")
    ax.set_yticks(y)
    ax.set_yticklabels(t.index)
    ax.set_xlabel("Hours with the value missing (%)")
    ax.set_title("The two hospitals record data differently")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _clean(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _f(x, d=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def _ci(ci, d=3, pct=False):
    if not isinstance(ci, (list, tuple)) or any(np.isnan(c) for c in ci):
        return ""
    if pct:
        return f" ({max(ci[0], 0):.0f}–{min(ci[1], 100):.0f})"
    if ci[0] < 0:   # intervals for differences: avoid "-0.005–-0.001"
        return f" ({ci[0]:+.{d}f} to {ci[1]:+.{d}f})"
    return f" ({ci[0]:.{d}f}–{ci[1]:.{d}f})"


COMPARED = {"site_normalized": "adjusted to local norms", "importance_wt": "importance weighting",
            "dann": "domain-adversarial training",
            "trained_at_test_hospital": "model trained at the test hospital itself"}


def write_report(out: Path, res: pd.DataFrame, qrep: dict, n_bad: int, sub: pd.DataFrame,
                 curves: dict, main_model: str, synthetic: bool, comp: pd.DataFrame | None = None,
                 n_seeds: int = 1):
    L = ["# Predicting sepsis early, and checking whether it works at another hospital\n"]
    if synthetic:
        L.append("> **Made-up test data.** These numbers only show the code runs.\n")
    a, b = qrep["sites"]["A"], qrep["sites"]["B"]
    L.append("## The data\n")
    L.append(f"ICU records from two US hospital systems (PhysioNet/CinC Challenge 2019): "
             f"{a['patients'] + b['patients']:,} patients and "
             f"{a['patient_hours'] + b['patient_hours']:,} patient-hours of vital signs and lab results.\n")
    L.append("| | Hospital A | Hospital B |\n|---|---|---|")
    L.append(f"| Patients | {a['patients']:,} | {b['patients']:,} |")
    L.append(f"| Developed sepsis | {a['septic_patients']:,} ({a['septic_pct']:.1f}%) | "
             f"{b['septic_patients']:,} ({b['septic_pct']:.1f}%) |")
    L.append(f"| Typical ICU stay (hours, median) | {a['median_hours']:.0f} | {b['median_hours']:.0f} |\n")
    L.append(f"The files were first converted to the OMOP common data model, and the analysis "
             f"cohort was built from those tables. {n_bad:,} impossible values (for example a heart "
             f"rate outside 20–250) were set to missing.\n")
    L.append("![data shift](data_shift.png)\n")

    L.append("## Does a model trained at one hospital work at the other?\n")
    L.append("Each model was tested on patients it never saw. AUROC is the chance the model gives "
             "a higher score to an hour shortly before sepsis than to an hour that isn't (0.5 = coin flip).\n")
    L.append("![cross hospital](cross_hospital.png)\n")
    seed_col = n_seeds > 1
    L.append("| Model | Trained on | Tested on | Method | AUROC (95% CI) |"
             + (" Range across runs |" if seed_col else "")
             + " Septic patients flagged by onset | Patients without sepsis who got an alert |"
             " Alert-hours per 100 patient-days |")
    L.append("|---|---|---|---|---|" + ("---|" if seed_col else "") + "---|---|---|")
    label = {"standard": "–", "source_only": "no adjustment", "site_normalized": "adjusted to local norms",
             "importance_wt": "importance weighting", "dann": "domain-adversarial training"}
    for _, r in res.sort_values(["model", "test", "setting"]).iterrows():
        rng = f" {r.AUROC_seed_min:.3f}–{r.AUROC_seed_max:.3f} |" if seed_col else ""
        L.append(f"| {NAMES[r.model]} | {r.train.replace('+', ' + ')} | {r.test} | {label[r.variant]} | "
                 f"{r.AUROC:.3f}{_ci(r.get('AUROC_ci'))} |{rng} {_f(r.septic_caught_pct, 0)}% | "
                 f"{_f(r.nonseptic_alerted_pct, 0)}% | {_f(r.alert_hours_per_100_patient_days, 0)} |")
    L.append("\nThe alert threshold was always set on the training hospital's own patients "
             "(to flag about 80% of their septic patients by onset), as it would be before a real rollout. "
             "\"Patients without sepsis who got an alert\" counts anyone alerted at least once during "
             "their stay; alert-hours per 100 patient-days shows how much of the time alerts were on.\n")
    if seed_col:
        L.append(f"Every model was trained {n_seeds} times with different random seeds. The table shows "
                 "the average; the interval comes from resampling test patients, and the range shows "
                 "the lowest and highest AUROC across the separate trainings.\n")
    else:
        L.append("Each model was trained once. Neural networks, and domain-adversarial training in "
                 "particular, can vary between trainings, so differences of a few hundredths should be "
                 "confirmed with more seeds (`--seeds 0,1,2`).\n")

    if comp is not None and len(comp):
        L.append("### Does adapting to the new hospital help?\n")
        L.append("Each line compares a method with using the other hospital's model unchanged, on the "
                 "same test patients. An interval that includes 0 means no clear difference.\n")
        L.append("| Model | Trained on → tested on | Compared with no adjustment | Change in AUROC (95% CI) |")
        L.append("|---|---|---|---|")
        for _, r in comp.iterrows():
            L.append(f"| {NAMES[r.model]} | {r.train} → {r.test} | {COMPARED.get(r.compared, r.compared)} | "
                     f"{r['diff']:+.3f}{_ci(r.diff_ci)} |")
        L.append("")

    L.append("## What the care team would see\n")
    L.append("![trade-off](alert_tradeoff.png)\n")
    L.append("Each point is a different alert threshold. Moving right means more alerts for nurses "
             "to handle; moving up means more septic patients flagged before sepsis onset.\n")

    L.append(f"## Does it work equally well for everyone? ({NAMES[main_model]}, trained on both hospitals)\n")
    L.append("| Group | Patients | Developed sepsis | AUROC (95% CI) | Septic patients flagged (95% CI) | "
             "Patients without sepsis alerted (95% CI) |")
    L.append("|---|---|---|---|---|---|")
    for _, r in sub.iterrows():
        L.append(f"| {r.group} | {r.patients:,} | {r.septic_patients:,} | {_f(r.AUROC)}{_ci(r.get('AUROC_ci'))} | "
                 f"{_f(r.septic_caught_pct, 0)}%{_ci(r.get('septic_caught_ci'), pct=True)} | "
                 f"{_f(r.nonseptic_alerted_pct, 0)}%{_ci(r.get('nonseptic_alerted_ci'), pct=True)} |")
    n_unk = sub.loc[sub.group == "Not recorded", "patients"]
    if len(n_unk):
        L.append(f"\nICU type is missing in the source data for {int(n_unk.iloc[0]):,} test patients, so "
                 "they appear as \"Not recorded\". Small groups have wide intervals; differences "
                 "between groups whose intervals overlap may be chance.")
    L.append("\n![calibration](calibration.png)\n")
    L.append("Points on the dashed line mean the predicted risk matched what actually happened.\n")

    L.append("## Limits\n")
    L.append("- ICU data from two US hospital systems; a third was held back by the challenge "
             "organisers and isn't public.")
    L.append("- ICU type is missing for many patients in the source data.")
    L.append("- Sepsis labels follow the challenge's Sepsis-3 definition, which marks hours from 6 hours "
             "before onset.")
    L.append("- No real dates, so the split between training and test patients is random, not by time.")
    L.append("- This is a research demonstration, not a clinical tool.")
    (out / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
