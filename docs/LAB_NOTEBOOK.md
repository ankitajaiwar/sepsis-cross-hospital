# Lab notebook: active domain adaptation experiments

A dated record of what was run, what came out, and why each decision was made. Code details are in [TECHNICAL_REFERENCE.md](TECHNICAL_REFERENCE.md). Statements marked **hypothesis** are untested.

All results use the PhysioNet/CinC 2019 training data (hospitals A and B), tree models, and the new hospital's held-out test patients. AUROC is hourly. "A→B" means a model trained at A is moved to B.

---

## Nov 9: first active learning run (10x weight, 3 seeds)

**Setup.** Code at `f619358`. Six strategies, budgets 25 / 50 / 100 / 200 / 400 patients, labelled new-hospital hours weighted 10x, 200 trees, alert threshold set on the original hospital's validation patients.

**Results**

| Observation | Evidence | Reading |
|---|---|---|
| Hybrid did not rank patients better | A→B at 400: least like source 0.801, random 0.780, hybrid 0.773. B→A: hybrid below the model as-is (0.749) at every budget | Can't be presented as better as it stands |
| Hybrid's higher catch rate came with many more false alerts | A→B at 200: hybrid 84% flagged / 58% non-septic alerted; all target labels 76% / 22% | Mostly a shift in score level against a fixed threshold, not better ranking |
| Strategies picking many septic patients lowered AUROC; highest risk alerted less | Highest risk: 27–39% flagged A→B | Hypothesis: high-scoring hours before the label window, labelled negative, weighted 10x, pull scores down |
| In B→A, adding labelled patients mostly hurt | Random flat at 0.745–0.758; others lower; all target labels 0.817 | Labels are useful; how they are combined with source data is the problem |
| Diversity at 25 patients dropped AUROC by ~0.07–0.08 | Both directions | Too large for 25 patients; needs checking |
| Many differences within noise | SDs 0.01–0.02 | Random vs least like source not a real difference |

**Reinterpretation.** Labelled new-hospital patients can do two jobs: retraining the model, and resetting the alert threshold. The first run only used them for retraining. Strategies that find septic patients may matter more for the threshold, since random picks only ~3–15 septic patients at small budgets.

**Code check.** (1) The weight is per hour, so long stays count more. (2) Calibration and threshold both come from the original hospital. (3) Only summary metrics were saved, so no paired tests or matched operating points were possible.

**Code changes (`ec9a82f`)**

| Change | Why |
|---|---|
| Each round scored with the original threshold, a threshold reset on the labelled patients (≥ 3 septic needed), and the share flagged at a fixed 20% alert rate | Separates ranking from threshold effects |
| `recalibrate`: original model, threshold reset only, same patients as random | The cheap obvious alternative |
| `target_only`: train on labelled new-hospital patients alone | Does source data help at small budgets? |
| `--weight-by patient` | Tests the long-stay effect |
| Per-patient test predictions saved each round | Paired bootstrap tests |
| Hours labelled reported | Patients alone hide how much data a strategy adds |
| "Hybrid (proposed)" → "Hybrid" | No overclaiming while it doesn't win |

**Run plan changes.** Weight sensitivity at 1x and 3x instead of 50x (10x already looked too heavy). The 5-seed active learning run held until the new code was in, so it would not need repeating. Wider budgets (10–1,600) deferred.

**Checks noted from documenting the code** (details in the technical reference, section 9): hourly AUROC includes post-onset hours; onset is reconstructed as first label + 6 hours; local-norm statistics use all patients including test; "uncertainty" behaves like average risk because hourly risks are below 0.5; greedy k-centre picks outliers first; "all target labels" means source plus all target labels; tree models used scikit-learn's implicit early stopping.

---

## Nov 10: main results with 5 seeds, and weight sensitivity

### Main pipeline, 5 seeds (trees with implicit early stopping)
- Averages moved ≤ 0.007 from the single-seed run. Spread across seeds ~0.01 (trees), up to 0.03 (GRU).
- Moving a model costs 0.065–0.109 AUROC (paired gap to the test hospital's own model).
- Pooled matches within (trees 0.818 vs 0.811 at A, 0.846 vs 0.848 at B).
- Thresholds don't transfer: B's trees at A flag 80% (A's own 82%) but alert 57% of non-septic patients (A's own 36%); A's trees at B flag 53% (B's own 80%).
- Domain-adversarial training (GRU): **A→B +0.037 (95% CI 0.020–0.057), B→A −0.017 (−0.030 to −0.007).** The direction dependence is real, not single-seed noise. Even where it helps, it closes under half of the gap (0.037 of 0.079).
- Subgroup and calibration sections use the first seed's pooled model only.

### Weight sensitivity: 1x vs 3x vs 10x (3 seeds; 200 trees, early stopping)
At 25 patients each strategy labels exactly the same patients at every weight, so differences there come from the weight alone.

| AUROC | A→B 25 (10x / 3x / 1x) | A→B 400 | B→A 25 | B→A 400 |
|---|---|---|---|---|
| Random | 0.760 / 0.759 / 0.758 | 0.780 / 0.783 / 0.784 | 0.758 / 0.755 / 0.747 | 0.745 / 0.771 / 0.775 |
| Hybrid | 0.748 / 0.757 / 0.764 | 0.773 / 0.793 / 0.802 | 0.726 / 0.745 / 0.752 | 0.699 / 0.740 / 0.758 |
| Diversity | 0.677 / 0.712 / 0.736 | 0.754 / 0.773 / 0.794 | 0.678 / 0.720 / 0.740 | 0.681 / 0.718 / 0.737 |
| Highest risk | 0.699 / 0.728 / 0.740 | 0.749 / 0.761 / 0.776 | 0.718 / 0.735 / 0.744 | 0.717 / 0.741 / 0.755 |

- Heavier weight hurts targeted strategies monotonically (1x > 3x > 10x); random is almost unaffected. The large small-budget drops of the first run came mainly from the 10x weight on unusual or septic-heavy patients.
- The weight on labelled patients changes which strategy comes out ahead, so it is worth reporting as a finding in its own right.
- **Decision: 1x by default.** It is the no-tuning choice (labelled patients simply join the training data). It was chosen after seeing test results, so either justify it a priori or confirm it on the new hospital's validation patients; report 3x and 10x as sensitivity.

### Code changes (`41a271f`)
- Early stopping turned off for every tree model. scikit-learn's default held out a random 10% of training *hours*, so hours of one patient sat on both sides and the number of trees varied.
- Active learning uses 300 trees, the same as the main pipeline, so the model-as-is baseline matches.
- `--target-weight` defaults to 1.
- `--shared-features`: use only features recorded at both hospitals (drops the 5 EtCO2 features; B never records EtCO2).

**Hypothesis (EtCO2).** B never records EtCO2, so B's model ignores it; labelled A patients bring EtCO2 in, learned only from a handful of them. Least like source and hybrid may pick patients with EtCO2 recorded. This would explain why targeted strategies fail only in B→A. Tested Nov 11.

---

## Nov 11: no early stopping, base vs shared features

### Main pipeline rerun without early stopping (5 seeds)
- Tree AUROCs moved ≤ 0.004; conclusions unchanged. GRU rows identical (cached).
- Paired vs no adjustment: trees local norms B→A −0.027 (−0.038 to −0.017); importance weighting B→A +0.007 (0.001–0.013), A→B +0.011 (n.s.); GRU domain-adversarial unchanged.
- Importance weighting was the only label-free method that never hurt, but its gains were ≤ 0.011.

### Active learning base run (1x, 300 trees, no early stopping, 3 seeds)
- **Baselines aligned:** model as-is A→B 0.741, B→A 0.748 (pipeline 0.739 / 0.747).

| AUROC | A→B 25 / 100 / 400 | B→A 25 / 100 / 400 |
|---|---|---|
| Random | 0.746 / 0.763 / 0.783 | 0.750 / 0.756 / 0.771 |
| Least like source | 0.740 / 0.774 / **0.803** | 0.746 / 0.752 / 0.744 |
| Hybrid | 0.749 / 0.773 / 0.790 | 0.736 / 0.736 / 0.744 |
| Diversity | 0.712 / 0.752 / 0.792 | 0.731 / 0.718 / 0.728 |
| All target labels | 0.846 | 0.817 |

- A→B: least like source best (+0.020 vs random at 400), and it also gave 0.801 at 400 in the first run, so it is the most consistent A→B winner. Hybrid's 0.802 in the earlier 1x run did not reproduce (0.790).
- B→A: random best; every targeted strategy ends below the model as-is.
- **Hours labelled:** for 25 patients, diversity and hybrid label ~2,300–2,750 hours vs ~1,000 for random and ~900–950 for least like source. They pick long stays, so they get 1.5–2.5x more labelled data per patient. Report hours with patients, or compare at equal hours. Least like source's A→B win comes at about the same number of hours as random.
- **Flagged at a fixed 20% alert rate:** A→B 51% as-is → 59–65% at 400 (all labels 72%); B→A 53% → 52–58% (all labels 65%). Diversity is best A→B on this measure (65%) despite lower AUROC.
- **Reset threshold for retrained models doesn't work:** it flags only 2–30% of septic patients. The threshold is set on patients the model has trained on, so their scores are too high. Needs cross-fitting (or labelled patients held back for threshold setting) before it can be used.

### Shared-features run (EtCO2 dropped)

| AUROC at 400 | A→B all → shared | B→A all → shared |
|---|---|---|
| Random | 0.783 → 0.777 | 0.771 → 0.768 |
| Least like source | 0.803 → 0.794 | 0.744 → 0.749 |
| Hybrid | 0.790 → 0.777 | 0.744 → 0.749 |
| Diversity | 0.792 → 0.784 | 0.728 → 0.729 |

- **EtCO2 does not explain B→A.** Targeted strategies gain at most 0.003–0.007 (noise) and stay below random and below the model as-is. Least like source picks essentially the same patients, so the hospital classifier was not relying on EtCO2.
- A→B: dropping EtCO2 slightly hurts at 400 (−0.006 to −0.013) but removes diversity's dip at 25 patients (0.712 → 0.751). Unexplained.
- **Decision:** all features in the full run; shared features reported as a sensitivity analysis.

### Where the story stands
- Random labels help reliably: +0.023 to +0.042 AUROC at 400 patients in both directions.
- Choosing patients helps A→B (least like source) but every targeted strategy hurts B→A.
- No strategy wins in both directions; domain-adversarial training shows the same asymmetry. The paper should explain the asymmetry, not hide it.
- **Hypotheses for B→A (untested):** (1) the hospitals may differ partly in outcomes given the same measurements (labelling or care practice), so a biased sample of the new hospital misleads while a random one represents it; (2) targeted picks are septic-heavy (hybrid ~40% septic vs ~10% random) and long-stay, which may distort the base rate the model learns.
- Possible checks: `--weight-by patient`; compare `target_only` across directions; hospital classifier AUROC per direction.

### Code change: cross-fitted reset threshold
- **Decision:** 3-fold cross-fitting rather than holding back labelled patients for threshold setting. Holding patients back would change the AUROC results; cross-fitting keeps every labelled patient in training and only costs run time.
- For retrained models, the labelled patients are split into 3 groups; each group is scored by a model trained without it, and the reset threshold is set on these out-of-fold scores. About 3 extra trainings per round, only when ≥ 3 septic patients are labelled.

### Next
1. ~~Fix the reset threshold for retrained models.~~ Done (cross-fitting).
2. Full 5-seed run: all features, all strategies plus `recalibrate` and `target_only`.
3. Paired bootstrap comparisons vs random at each budget, from saved predictions.
4. Later: wider budgets, ICULOS/HospAdmTime ablation, qSOFA/NEWS2 baselines, challenge utility score, eICU after credentialing.

---

## Commands used

```
# Nov 9 night (code f619358 → runs used pre-patch code)
python -m sepsis.active --omop artifacts/omop --out artifacts_active_w1 --seeds 3 --target-weight 1 --strategies random,high_risk,diversity,hybrid --budgets 25,100,400
python -m sepsis.active --omop artifacts/omop --out artifacts_active_w3 --seeds 3 --target-weight 3 --strategies random,high_risk,diversity,hybrid --budgets 25,100,400
python -m sepsis.pipeline --from-omop artifacts/omop --out artifacts --seeds 0,1,2,3,4

# Nov 10 (code 41a271f)
python -m sepsis.active --omop artifacts/omop --out artifacts_active_base --seeds 3 --strategies random,diversity,domain,hybrid --budgets 25,100,400
python -m sepsis.active --omop artifacts/omop --out artifacts_active_shared --seeds 3 --strategies random,diversity,domain,hybrid --budgets 25,100,400 --shared-features
python -m sepsis.pipeline --from-omop artifacts/omop --out artifacts --seeds 0,1,2,3,4
```

Note: in `sepsis.active`, `--seeds` is a count (`--seeds 3` runs seeds 0–2); in `sepsis.pipeline` it is a list.
