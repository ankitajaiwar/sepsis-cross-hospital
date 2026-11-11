# Sepsis cross-hospital project: full technical reference

Describes the code as of Nov 11, 2026 (cross-fitted reset threshold added). Experiment log and results: [LAB_NOTEBOOK.md](LAB_NOTEBOOK.md).

## How to read this document

Reasons are marked by where they come from:

- **[code]** the reason is written in the code, its comments or the README.
- **[design]** the rationale for a design choice that is not written in the code.
- **⚠ Check** marks something to verify before it affects results or paper wording. These are collected again in section 9.

Contents
1. What the project does, end to end
2. Data and labels
3. Data preparation code (`schema`, `download`, `raw`, `omop`, `quality`, `features`)
4. Models (`gbm`, `calibration`, `seq`)
5. Evaluation (`evaluate`)
6. Main experiments (`experiments`, `pipeline`, `report`)
7. Active learning experiments (`active`)
8. Serving, synthetic data, tests, build files
9. Things to check before writing the paper
10. Every parameter in one place
11. Commit history

---

## 1. What the project does, end to end

```
PhysioNet 2019 .psv files (one per patient, hospitals A and B)
  → raw.py        one hourly table
  → omop.py       OMOP tables (person, visit_occurrence, visit_detail, measurement, condition_occurrence)
  → omop.py       cohort rebuilt from OMOP: one row per patient-hour, with label and onset hour
  → quality.py    per-hospital quality report; impossible values set to missing
  → features.py   70/15/15 patient split per hospital; hourly features; optional per-hospital normalisation
  → gbm.py / seq.py   gradient-boosted trees and GRU
  → experiments.py    within / pooled / cross-hospital runs, label-free adaptation, seeds, caching
  → evaluate.py       AUROC with patient bootstrap, alert metrics, subgroups, calibration
  → report.py         figures and REPORT.md
  → active.py         which new-hospital patients to label (separate entry point)
  → api.py            web service serving the pooled tree model
```

Two entry points produce results:
- `python -m sepsis.pipeline`: main cross-hospital study (sections 6.1–6.6).
- `python -m sepsis.active`: active domain adaptation study (section 7).

The research question, as written in the README **[code]**: if a model is trained at one hospital, how well does it work at another, and can that drop be reduced? The active learning part asks which patients a new hospital should label when it can afford only a few chart reviews.

---

## 2. Data and labels

**Source.** PhysioNet/CinC Challenge 2019, open access, training sets A and B: 40,336 ICU patients (A 20,336; B 20,000), 1,552,210 patient-hours. A third hospital was kept private by the organisers. Cite Reyna et al., *Crit Care Med* 2020.

**Variables (40 columns plus label).**
- 8 vitals: HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2.
- 26 labs: BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN, Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Phosphate, Potassium, Bilirubin_total, TroponinI, Hct, Hgb, PTT, WBC, Fibrinogen, Platelets.
- 6 demographic and admin fields: Age, Gender, Unit1 (MICU), Unit2 (SICU), HospAdmTime (hours from hospital admission to ICU admission, negative), ICULOS (hours in ICU).
- Label: `SepsisLabel`.

**Label definition [code].** The challenge marks every hour from 6 hours before Sepsis-3 onset onward as 1 (`LABEL_LEAD_HOURS = 6`). Hours after onset stay 1 until the end of the stay.

**Onset hour.** Not given directly in the files. The ETL reconstructs it as *first hour labelled 1 + 6* (`omop.py`).
- ⚠ Check: if a patient's first labelled hour is hour 0, the true onset could be earlier than hour 6, but the code still sets onset = 6. "Caught by onset" for such patients is measured against this reconstructed onset. Worth counting how many septic patients have their first label at hour 0, and reporting it or excluding them in a sensitivity analysis.

**Hospital differences found [code, REPORT.md].** Sepsis rate A 8.8%, B 5.7%; median stay 39 vs 38 hours; B never records EtCO2; missingness differs for many labs; 11,403 impossible values were removed in total.

---

## 3. Data preparation code

### 3.1 `schema.py`: variable definitions
- Lists of vitals, labs, demographics, the label column and the two sites (`training_setA`, `training_setB`).
- `MEASUREMENTS`: each variable → (LOINC code, unit, plausible low, plausible high). Examples: HR 8867-4, 20–250 beats/min; Temp 8310-5, 25–45 °C; Lactate 2524-7, 0.1–40 mmol/L; FiO2 3150-0, 0.15–1.0.
- **Why LOINC codes [code]:** they go into OMOP `measurement_source_value`, so the cohort query would run on any OMOP database with the same codes. The docstring notes the codes should be verified against OHDSI Athena before production use.
- **Why these plausible ranges [design]:** wide physiological limits meant to catch recording errors (wrong units, typos), not to clip extreme but real values. They are judgement calls; the paper's methods should state them or cite a source.
- OMOP gender concepts: male 8507, female 8532.

### 3.2 `download.py`: fetch the open data
- Lists files from the PhysioNet directory page, downloads in parallel (16 threads), 3 tries per file, 60-second timeout, and skips files already on disk.
- **Why [code]:** 40,336 small files; resumable so a dropped connection doesn't mean starting over.
- `--limit N` downloads N patients per hospital for a quick try.

### 3.3 `raw.py`: read patient files
- Reads each `.psv` (pipe-separated) as float32, checks that all 41 expected columns exist, and adds `patient_id` (file name), `site` and `hour` (row number, 0-based).
- Reads with 8 threads. `limit` keeps the first N files per hospital (sorted by name, so not random).
- **Why float32 [design]:** halves memory; 1.5 million rows × 41 columns.

### 3.4 `omop.py`: convert to OMOP and rebuild the cohort
**Why go through OMOP at all [code]:** hospital data networks (OHDSI, multi-site studies) share models by agreeing on these tables, so the cohort code would run on any OMOP database. It also makes the project look like a real multi-site study.

`to_omop(raw)` builds six tables:

| Table | What goes in it | Choices |
|---|---|---|
| `care_site` | Hospital A = 1, B = 2 | |
| `person` | one row per patient; `person_id` 1..N; gender concept; `year_of_birth = 2100 − floor(Age)` | race/ethnicity set to 0 (not in the data) |
| `visit_occurrence` | hospital stay; start = anchor + HospAdmTime hours; end = ICU end | inpatient visit concept 9201; type 32817 (EHR) |
| `visit_detail` | ICU stay; start = anchor; end = anchor + stay length; source value MICU / SICU / Unknown | ICU concept 32037 |
| `measurement` | one row per non-missing value; datetime = anchor + hour; LOINC in `measurement_source_value`; unit | `measurement_concept_id = 0` (not mapped) |
| `condition_occurrence` | one row per septic patient at the reconstructed onset hour | source value "Sepsis-3 onset (PhysioNet Challenge 2019 label)"; concept 0 |

- **Why the anchor date 2100-01-01 [code]:** the data has no real dates. An obviously shifted date, as MIMIC uses, keeps the tables valid OMOP without implying real dates.
- **Why concept ids are 0 [code]:** mapping LOINC to standard concepts needs the OHDSI Athena vocabulary, which is licence-gated and not bundled.
- **Why build measurements one variable at a time [code]:** keeps memory low.

`cohort_from_omop(db)` goes back the other way: one row per patient-hour for hours 0 to stay length − 1, each measurement placed at its hour, Age = 2100 − year of birth, Unit1/Unit2 rebuilt from the ICU source value, HospAdmTime from the visit start, ICULOS = hour + 1, label = 1 from onset − 6 onwards, `onset_hour` kept as a column. It raises an error if any measurement code is unknown.
- ⚠ Check (minor): Age is rebuilt as a whole number because the person table stores year of birth. Ages with decimals in the raw files are floored, about a year of precision at most. Harmless, but say "age in whole years" if you describe it.
- `test_omop_round_trip_rebuilds_the_data` checks that the round trip reproduces the raw data.

### 3.5 `quality.py`: data-quality checks and cleaning
- `quality_report`: per hospital, the number of patients, septic patients and %, patient-hours, median stay, missing Age, and per variable the % missing, number observed, and number and % outside the plausible range. Saved as `quality_report.json`.
- `clean`: sets values outside the plausible range to missing and returns the count.
- **Why [code]:** the challenge organisers deliberately left errors in the data.
- **Why set to missing rather than clip [design]:** an impossible value is most likely an error, so carrying the last good value forward is safer than pretending it was at the limit.
- `missingness_table`: % missing per variable per hospital, used in the data-shift figure.

### 3.6 `features.py`: splits, normalisation and features

**Patient split: `assign_splits(seed=42, frac=(0.7, 0.15))`.** Within each hospital, patients are shuffled and split 70% train, 15% validation, 15% test. A patient is never split across sets.
- **Why random rather than by time [code, README limits]:** the data has no real dates.
- **Why split within each hospital [design]:** every hospital has its own train, validation and test patients, so within, pooled and cross-hospital models are all tested on the same test patients.
- **Why a fixed seed of 42 for the split [design]:** the split never changes between runs or between the main pipeline and `active.py`, which calls the same function. Seeds 0–4 vary the model training, not the split.

**Hourly features: `build_features` (98 features).**
- **Rule [code]:** every feature at hour t uses only data from hours ≤ t, so the model could run live. `test_features_use_only_the_past` checks this.
- First, each measurement is forward-filled within the patient, so the last known value carries forward.

| Group | Features per variable | Count |
|---|---|---|
| 8 vitals | current (filled) value; mean, min and max over the last 6 hours (6-hour rolling window, at least 1 value); change vs 6 hours earlier | 8 × 5 = 40 |
| 26 labs | current (filled) value; hours since it was last measured | 26 × 2 = 52 |
| Static | Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS | 6 |

- Missing values stay missing (NaN). The trees handle them natively.
- **Why 6 hours [design]:** it matches the label's 6-hour lead time and is a common window for early warning summaries.
- **Why "hours since last measured" for labs [code, seq.py docstring]:** missingness is informative in ICU data. A lactate ordered an hour ago means something different from one measured two days ago.
- **Why vitals get trends and labs don't [design]:** vitals are recorded almost hourly, so trends are meaningful; labs are sparse, so the time since measurement carries more information than a 6-hour change.
- ⚠ Check: ICULOS and HospAdmTime are features. They can encode hospital practice (how long patients stay, how admissions flow) rather than physiology, which may help within a hospital and hurt across hospitals. The planned ablation (rerun without them) tests this.

**Per-hospital normalisation: `site_stats` and `normalize_by_site`.** For each hospital, the 34 measurements are expressed as (value − hospital mean) / hospital SD. Features are then rebuilt from the normalised values (`X_norm`). Static fields are not normalised.
- **Why [code]:** a simple, label-free way to remove systematic differences between hospitals, for example a different calibration of blood pressure devices.
- ⚠ Check: the means and SDs are computed over *all* patients of each hospital, including its test patients. No labels are used, so this is a standard transductive setup, but the methods section should say so. The fully clean alternative computes them on the training split only.

**`subsample_negatives(rate)`.** Keeps every hour labelled 1 and a random `rate` share of hours labelled 0. Used by the trees with rate 0.3.

---

## 4. Models

### 4.1 `gbm.py`: gradient-boosted trees

`GBMModel` wraps scikit-learn's `HistGradientBoostingClassifier`.

| Parameter | Value | Why |
|---|---|---|
| learning_rate | 0.05 | [design] slower learning with more trees; a standard robust setting |
| max_iter (trees) | 300 in pipeline and `active.py` (active used 200 before Nov 10); 100 with pipeline `--fast`; 80 for active `--fast` | same in both scripts so the model-as-is baseline matches |
| max_leaf_nodes | 31 | scikit-learn default |
| min_samples_leaf | 200 | [design] large leaves stop trees fitting single patients, given strong correlation between hours of the same patient |
| l2_regularization | 1.0 | [design] mild shrinkage of leaf values |
| random_state | the run's seed | varies across seeds 0–4 |
| negative-hour sampling | keep 30% of hours labelled 0 (all hours labelled 1) | [design] about 2% of hours are positive; subsampling speeds training and rebalances classes |
| prior correction | predicted odds × 0.3 | [code] undoes the effect of dropping 70% of negatives, so probabilities refer to the real class balance |
| columns used | only columns with at least one value in the training data | [code] e.g. EtCO2 is never recorded at B, so a model trained at B ignores it |
| calibration | Platt scaling on the training hospital's validation hours | see 4.2 |

- No hyperparameter tuning is done in the code; the values are fixed. ⚠ Check: reviewers may ask. The usual answer: fixed, sensible defaults chosen before looking at cross-hospital results, the same for every setting, so comparisons between settings are fair. Say so in the methods.
- **Early stopping: off since Nov 10 (`41a271f`).** Before that, the code didn't set `early_stopping`, so scikit-learn's default (`"auto"`, on above 10,000 rows) held out a random 10% of training *hours* and stopped after 10 trees without improvement. Hours of the same patient sat on both sides of that holdout, and the number of trees varied from run to run (a toy example stopped at 48 of 300). All results up to Nov 10 (the main pipeline's 1-seed and 5-seed runs, and the first active learning runs) were produced that way. Now every tree model, including the two hospital classifiers, uses exactly `max_iter` trees. The pipeline cache key for trees became `iter300noES`, so old tree runs are retrained while cached GRU runs are reused.

**Importance weighting: `importance_weights` (label-free adaptation for trees).**
- A second classifier learns to tell source-hospital hours from target-hospital hours: HistGB, 150 trees, learning rate 0.1, min_samples_leaf 500, trained on up to 200,000 randomly sampled hours from each hospital (the target hours come from its training split), using only columns recorded at both hospitals.
- Weight of each source hour = its odds of looking like the target × (number of source hours / number of target hours), clipped to [0.1, 10], then divided by the mean so weights average 1.
- **Why [code]:** source hours that resemble the new hospital count more, so the model leans towards the new hospital's population without using its outcomes.
- **Why clip to 0.1–10 [design]:** a few extreme weights would otherwise dominate training (the usual variance problem with importance weighting).

### 4.2 `calibration.py`: Platt scaling that never reverses ranking
- Fits a logistic regression from score to outcome on validation data (C = 10⁶, i.e. essentially no regularisation).
- If the fitted slope is ≤ 0, it keeps slope 1 and only shifts the intercept (by bisection) so the average prediction matches the observed rate.
- **Why [code]:** a flat or negative slope would scramble the ranking; this happens with a weak model on a small validation set. Calibration should change probabilities, never AUROC. `test_calibration_never_reverses_ranking` checks this.
- **Why calibrate at all [design]:** alert thresholds and the calibration plot need probabilities on a meaningful scale; Platt is monotone, so AUROC is unaffected.

### 4.3 `seq.py`: GRU recurrent network

**Inputs per hour (109 values) [code]:**

| Block | Values | How |
|---|---|---|
| Last known value | 34 | standardised with the *training data's* mean and SD, forward-filled within the patient, 0 if never measured, clipped to ±6 |
| Measured this hour | 34 | 1 if recorded at this hour, else 0 |
| Time since last measured | 34 | log(1 + hours) / log(337), scaled to 0–1; 1 if never measured |
| Static | 7 | (Age − 62)/16 (missing → 62); Gender (missing → 0.5); Unit1; Unit2 (missing → 0); a "unit missing" flag; −HospAdmTime/100 clipped to ±5; ICULOS/100 |

- **Why measured flags and time since measurement [code]:** missingness is informative in ICU data.
- **Why 336 hours maximum (`MAX_LEN`) [code]:** 14 days, the challenge maximum. Hours beyond it get the score of the last kept hour.
- **Why log-scaled time [design]:** the difference between 1 and 5 hours matters more than between 100 and 104.

**Network (`GRUNet`):** Linear(109 → 64) + ReLU → one GRU layer (64 hidden units) → Linear(64 → 1) gives one logit per hour. Domain head for adversarial training: Linear(64 → 32) + ReLU + Linear(32 → 1), connected through a gradient-reversal layer.

**Training (`GRUModel.fit`):**

| Parameter | Value | Why |
|---|---|---|
| epochs | 8 (3 with `--fast`) | [design] enough to converge on 28k training patients on a laptop CPU |
| batch | 64 patients, padded to the longest stay, padding masked | |
| optimiser | Adam, learning rate 1e-3 | standard |
| loss | binary cross-entropy per hour, padding masked | |
| positive-class weight | min(10, √(negatives / positives)) | [design] square root is a softer rebalance than full inverse frequency; capped at 10 for stability |
| gradient clipping | norm 5 | [design] recurrent nets can have exploding gradients |
| model selection | keep the epoch with the best validation AUROC (training hospital's validation patients) | |
| threads | CPU count − 1 | leaves one core free |
| seed | run seed for both PyTorch and NumPy | |
| calibration | Platt on validation logits | |

**Domain-adversarial training (DANN) [code]:** a label-free way to adapt, by teaching the network to ignore which hospital the data came from.
- Each step adds a batch of 64 random target-hospital training patients (no labels used). The domain head tries to tell source hours from target hours; the reversed gradient pushes the GRU's hidden states to make that impossible.
- Weight λ ramps up during training: λ = 0.3 × (2 / (1 + e^(−10·progress)) − 1), from 0 to about 0.3. The ramp is the schedule from the original DANN paper (Ganin et al. 2016); 0.3 is the maximum weight.
- ⚠ Check: the best epoch is still chosen by AUROC on the *source* hospital's validation patients. That is the standard label-free choice, but it can pick an epoch that adapted less. Worth one sentence in the paper, and a likely reason DANN results vary between seeds.
- ⚠ Check: the DANN weight (0.3) and schedule were not tuned. Fair for a label-free comparison, but state it.

---

## 5. Evaluation: `evaluate.py`

All metrics are computed on the test patients of the hospital being tested.

**Hourly metrics (`hourly_metrics`):** AUROC, AUPRC, Brier score, prevalence and mean prediction over all patient-hours.
- ⚠ Check: AUROC counts every hour, including hours after onset, which stay labelled 1 and are usually easy. This matches the challenge setup but is more optimistic than "predicting before onset". Consider also reporting AUROC on hours up to onset, or say clearly that it is hourly AUROC under the challenge labels.

**Patient bootstrap (`PatientBootstrap`).** Resamples whole patients with replacement (200 draws by default), stored as a count per patient, so AUROC on a draw is a weighted AUROC over the original rows.
- **Why patients, not hours [code]:** hours from one patient are correlated; resampling hours would make intervals far too narrow.
- **Why counts [code]:** makes thousands of draws cheap, and the same draws can be used for two models, giving a *paired* comparison.
- `mean_auroc`: AUROC averaged over seeds, with a 95% interval from the seed-averaged draws.
- `paired_difference`: AUROC of B minus A on the same draws, with a 95% interval and the share of draws where B is better.
- `wilson`: 95% Wilson interval for a proportion, used for flagged and alerted percentages in subgroups.

**Alerts (`_patient_view`, `alert_metrics`).** Every hour with risk ≥ threshold counts as an alert shown to the care team.

| Metric | Definition |
|---|---|
| Septic patients flagged by onset | septic patients whose first alert came at or before the onset hour |
| Median hours early | onset hour − first alert hour, among flagged septic patients |
| Patients without sepsis alerted | non-septic patients with at least one alert during the stay |
| Alerted patients who were septic | positive predictive value at patient level |
| Alert-hours per 100 patient-days | total alert hours / total hours × 2,400; how much of the time alerts are on |

**Threshold (`threshold_for_catch_rate`).** For each septic patient, take the highest score up to and including onset. Choose the threshold as the k-th highest of these, where k = ceil(80% × number of septic patients). This is the highest threshold that still flags 80% of septic patients, so it gives the fewest alerts for that catch rate. Returns 0.5 if there are no septic patients.
- **Why the training hospital's validation patients [code]:** so cross-hospital results show what would really happen at rollout. A hospital adopting a model does not get to tune the threshold on its future patients' outcomes.
- **Why 80% [design]:** a typical sensitivity target for sepsis screening; one fixed value makes all settings comparable. `tradeoff_curve` shows other targets (30% to 95% in steps of 5).

**Subgroups (`subgroup_table`).** By hospital, age group (under 45, 45–64, 65–79, 80 and over), sex, and ICU unit (medical, surgical, not recorded): AUROC with a bootstrap interval, flagged % and alerted % with Wilson intervals, all at the same threshold.

**Calibration (`calibration_table`).** 10 bins by quantile of predicted risk; mean predicted vs observed share of hours labelled 1.

---

## 6. Main experiments: `experiments.py`, `pipeline.py`, `report.py`

### 6.1 The settings (`PLANS`)

| Setting | Trained on | Tested on | Question it answers |
|---|---|---|---|
| within | A | A | how good is a hospital's own model (target for adaptation) |
| within | B | B | same for B |
| pooled | A + B | A and B | does combining hospitals cost anything at either one |
| cross | A | B | how much is lost when A's model is moved to B |
| cross | B | A | the other direction |

Every cross setting is run in variants:

| Variant | Trees | GRU | What it does |
|---|---|---|---|
| source_only | ✓ | ✓ | model used as-is (baseline for adaptation) |
| site_normalized | ✓ | ✓ | each hospital's values expressed relative to its own norms (3.6) |
| importance_wt | ✓ | | source hours weighted to look like the new hospital (4.1) |
| dann | | ✓ | domain-adversarial training (4.3) |

- **Why these three label-free methods [design]:** one each from the three main families of unsupervised domain adaptation: input normalisation, instance re-weighting, and representation alignment. Each is matched to the model type it naturally fits.
- **Why "within" and "pooled" are there [design]:** within gives the realistic ceiling for each hospital; pooled checks whether one model for both hospitals is a reasonable alternative to adaptation.
- Training uses the training split of the source hospital(s); validation patients of the source hospital(s) are used for model selection (GRU), calibration and the threshold; the target hospital's training patients are used *without labels* by importance weighting and DANN; testing is on the target hospital's test patients.

### 6.2 Seeds and caching (`run_all`)
- Every setting and variant is trained once per seed (`--seeds 0,1,2,3,4`). The seed changes tree randomness, negative-hour sampling, GRU initialisation and batch order, and the importance-weight sample. It does *not* change the patient split (fixed at seed 42).
- **Why several seeds [code, report text]:** neural networks, and DANN in particular, vary between trainings; differences of a few hundredths need confirming. Today's plan uses 5 seeds.
- Cache: each finished training saves its test predictions and threshold to `artifacts/runs/<run>__seed<k>.pkl/.json`. The cache key includes the model type, setting, source, variant and the epochs or tree count, so changing those trains new models instead of reusing old ones. A rerun only trains what's missing.
- ⚠ Check: the cache key does *not* include the code version. If you change features or model code, delete `artifacts/runs` before rerunning, or old predictions will be mixed in.

### 6.3 Summaries and comparisons
- `summarize`: one row per run with metrics averaged over seeds, the seed-averaged AUROC with a 95% bootstrap interval (200 patient draws), and the SD, minimum and maximum of AUROC across seeds.
- `compare_to_source_only`: for each model moved to the other hospital, the paired AUROC difference vs using it unchanged, for each adaptation method and for a model trained at the test hospital itself (the gap still to close). Same bootstrap draws for both (seed 1), so the comparison is paired.

### 6.4 Pipeline steps (`pipeline.run`)
1–2. Read raw files and convert to OMOP, saved to `artifacts/omop`; or with `--from-omop`, read existing OMOP tables.
3. Quality report, cleaning, splits.
4. Per-hospital normalisation; build `X_raw` and `X_norm`.
5. Train and test every setting (6.1) for each seed.
6. Report, figures, saved model.

**Command-line options:** `--raw`, `--out`, `--omop` (where to write OMOP), `--from-omop` (reuse), `--seeds` (comma-separated list, e.g. `0,1,2,3,4`), `--synthetic`, `--models gbm,gru`, `--epochs 8`, `--n-boot 200`, `--limit` (patients per hospital), `--fast` (3 epochs, 30 bootstrap draws, 100 trees).

**Outputs in `artifacts/`:** `quality_report.json`, `results_by_seed.csv`, `results.csv`, `comparisons.csv`, `tradeoff_gbm.csv`, `tradeoff_gru.csv`, `subgroups.csv`, the figures `cross_hospital.png`, `calibration.png`, `alert_tradeoff.png` and `data_shift.png`, `REPORT.md`, and `model/` (`gbm.joblib`, `gru.pt`, `model_card.json`).

### 6.5 What the report shows (`report.py`)
- Data summary, data-shift figure (the 12 variables with the biggest missingness gap between hospitals).
- Cross-hospital AUROC figure and full table, with the seed range when more than one seed is run.
- Paired comparisons of each adaptation method vs no adjustment.
- Alert trade-off curve, subgroup table and calibration plot. These use the **pooled tree model with the first seed only** (the model the API serves).
- Limits section.

### 6.6 First-run results (one seed; `docs/REPORT.md`)
- Within-hospital AUROC 0.798–0.846; pooled matches within.
- Moving a model drops AUROC by 0.07–0.11.
- Thresholds don't transfer: B's tree model at A alerts 60% of non-septic patients (A's own: 35%); A's model at B flags 53% of septic patients by onset (B's own: 82%).
- Label-free adaptation is inconsistent: importance weighting changes little; local norms hurt B→A for trees; DANN A→B 0.746 → 0.790 but B→A 0.732 → 0.715.

---

## 7. Active learning experiments: `active.py`

### 7.1 The question and the design
A model trained at the original hospital is moved to a new one. The new hospital can afford chart review for only a few of its own patients. **Which patients should it label, and what does that buy?** [code]

One run, for each direction (A→B, B→A) and each seed:
1. Train the base model on the source hospital's training patients, calibrated on its validation patients.
2. **Pool** = the new hospital's training patients (labels hidden). **Test** = the new hospital's test patients. The new hospital's validation patients are not used.
3. For each strategy, in rounds with cumulative totals of 25, 50, 100, 200 and 400 patients:
   - the strategy picks the next patients from the pool, using the current model's predictions;
   - the model is **retrained from scratch** on all source training patients plus every labelled patient so far, with labelled patients weighted;
   - it is recalibrated on the source validation patients and scored on the test patients.
4. Reference points: budget 0 (model as-is) and "all target labels".

- **Why whole patients, not hours [code]:** chart review labels a whole stay.
- **Why retrain from scratch each round rather than fine-tune [design]:** gradient-boosted trees can't be updated with new data, and retraining keeps every round comparable.
- **Why rounds [design]:** the standard active learning protocol, where each round's choice uses the model improved by earlier labels. It also gives a learning curve.
- **Why the weight is an option [design]:** 25 patients are about 1,000 hours against roughly 700,000 source hours, so extra weight was the first choice (10x). The Nov 10 comparison showed that 1x works best, and it is now the default.
- **Why tree models only [design]:** hundreds of retrainings per run; the GRU would take far longer.

### 7.2 Strategies

| Strategy (code name) | How it picks | Notes |
|---|---|---|
| Random (`random`) | uniformly at random | the usual baseline |
| Uncertainty (`uncertainty`) | highest mean of p × (1 − p) over the patient's hours | ⚠ Check: almost all calibrated hourly risks are below 0.5, where p(1 − p) rises with p, so this behaves like "highest average risk". That explains why it picks many septic patients. Say so, or define uncertainty differently (e.g. closest to the alert threshold) |
| Highest risk (`high_risk`) | highest peak hourly risk | finds sepsis cases, which are rare [code] |
| Diversity (`diversity`) | greedy k-centre on patient summaries: start from a random patient, then repeatedly take the one farthest from all picked so far | ⚠ Check: greedy k-centre picks the most extreme patients first; at small budgets this means outliers. A likely reason diversity is poor at 25 patients |
| Least like source (`domain`) | highest out-of-fold probability of belonging to the new hospital, from a hospital classifier on patient summaries | uses no labels |
| Hybrid (`hybrid`) | average the percentile ranks of uncertainty and least-like-source, take the top 3 × k candidates, then k-centre among them | the strategy this project introduced; renamed from "Hybrid (proposed)" on Nov 9 |
| Threshold reset only (`recalibrate`) | same patients as random; model not retrained; only the alert threshold is reset on the labelled patients | added Nov 9 as the cheap obvious alternative |
| New hospital only (`target_only`) | same patients as random; trained on the labelled patients alone (no source data); smaller minimum leaf (between 10 and 200, by data size); skipped until a septic hour is labelled | added Nov 9 |

**Patient summaries (for diversity, least-like-source and hybrid):** for each patient, the mean and last value of each of the 98 features, standardised, with missing values set to 0.

**Hospital classifier (least-like-source):** HistGB, 150 trees, learning rate 0.1, trained to tell source from target patient summaries; target scores come from 3-fold out-of-fold predictions so the classifier never scores its own training data [code].
- ⚠ Check: summaries include ICULOS and HospAdmTime, so "least like source" may partly mean "different stay length or admission pattern".

**Randomness:** the base model uses the run seed; selection randomness uses seed + budget; the hospital classifier uses the run seed. Random, recalibrate and target_only pick identical patients for the same seed.

### 7.3 Weighting
- `--target-weight` (default 1 since Nov 10; 10 before): weight of labelled new-hospital data. In the 1x/3x/10x comparison (Nov 10), 1x was best for every targeted strategy in both directions.
- `--weight-by hour` (default, the only option before Nov 9): every labelled hour counts 10x. ⚠ Check: long stays carry more weight.
- `--weight-by patient` (added Nov 9): each labelled patient carries the same total, 10 × the average stay length in hours, however long their stay.
- Negative-hour subsampling (30%) still applies to labelled patients, as to source patients.

### 7.4 How each round is scored
Each round gets several columns (`active_results.csv`). The first run had only the original-threshold columns.

| Measure | Definition | Why |
|---|---|---|
| AUROC, AUPRC | hourly, on the test patients | ranking quality |
| Original threshold: flagged / alerted without sepsis | threshold set on the source hospital's validation patients for 80% catch | what happens if the hospital keeps the vendor's threshold |
| Worst group flagged | lowest flagged % in any age group or sex with ≥ 5 septic patients | does the worst-served group benefit |
| Reset threshold: flagged / alerted without sepsis (added Nov 9) | threshold set on the labelled new-hospital patients for 80% catch; needs ≥ 3 labelled septic patients, otherwise the original threshold is kept and the report shows the share of seeds where it could be reset | the labels' second job: setting the alert threshold |
| Flagged at 20% alerts (added Nov 9) | share of septic patients flagged by onset when the threshold is chosen *on the test patients* so that 20% of non-septic patients are alerted | compares ranking at the bedside with threshold effects removed; like AUROC, it is an evaluation metric, not a deployable threshold |
| Patients labelled, of them septic, hours labelled | counts | hours added on Nov 9 because patients alone hide how much weight a strategy adds |

- **Cross-fitting (since Nov 11).** For retrained models, the reset threshold used to be set on the retrained model's own scores for the labelled patients. Those patients were in its training data, so their scores were too high, and the threshold flagged only 2–30% of septic test patients (Nov 11 base run). Now the labelled patients are split into 3 groups (`--crossfit-folds 3`); each group is scored by a model trained without it, and the threshold is set on these out-of-fold scores. It costs 3 extra trainings per round, and only runs when ≥ 3 septic patients are labelled. `--crossfit-folds 0` restores the old in-sample behaviour. Threshold reset only never trains on the labelled patients, so it uses its own scores.
- ⚠ Check: the fold models are calibrated on the same source validation patients as the final model, so their score scale is close but not identical. This is the usual cross-fitting approximation; state it in the methods.
- ⚠ Check: the worst group uses age and sex only; the main pipeline's subgroup table also includes ICU unit. Fine, but say which in each table.

### 7.5 Reference points
- **Budget 0:** the base model as-is, identical for every strategy in a seed.
- **All target labels:** source training patients plus *all* new-hospital training patients at the same weight. ⚠ Check: it is "source + all target labels", not "target only". Describe it that way in the paper.

### 7.6 Command-line options and outputs
`--raw`, `--omop` (reuse OMOP tables from the pipeline), `--out`, `--synthetic`, `--budgets 25,50,100,200,400`, `--seeds 3`, `--strategies` (default: all six plus the two comparison methods), `--target-weight 1`, `--weight-by hour|patient`, `--shared-features` (use only features recorded at both hospitals; drops the 5 EtCO2 features, added Nov 10), `--crossfit-folds 3` (added Nov 11), `--limit`, `--fast` (1 seed, 80 trees).
- ⚠ Note: `--seeds` here is a **count** (`--seeds 5` runs seeds 0–4), while in the pipeline it is a **list** (`--seeds 0,1,2,3,4`).
- Fixed in code: catch target 80%, 300 trees (`GBM_TREES`), no early stopping, minimum 3 septic patients to reset the threshold, fixed operating point 20%.

Outputs in the `--out` folder: `active_results.csv` (one row per direction × strategy × seed × budget), `active_summary.csv` (means and SDs over seeds), `active_auroc.png`, `active_caught.png` (original threshold), `active_caught_fixed.png` and `active_caught_tgt.png` (added Nov 9), `ACTIVE_REPORT.md`, `active_config.json`, and `checkpoints/<settings>/`: one CSV per direction, seed and strategy, a `settings.json`, and (since Nov 9) `preds/`, with each round's test predictions in parquet plus the test patients' IDs, hours, onset and age/sex once per direction.
- **Why checkpoints are keyed by settings [code]:** changing budgets, weights or the results version never mixes old and new results; a stopped run resumes with the same command.

### 7.7 First results (3 seeds, per-hour 10x weight)
Summarised in the decisions log. In short: random and least-like-source were best on AUROC; hybrid did not rank better and its higher catch rate came with many more false alerts; strategies that pick many septic patients lowered AUROC; in B→A most strategies made AUROC worse than using the model as-is. The tests queued for tonight (1x and 3x weights) check whether the weighting causes this.

---

## 8. Serving, synthetic data, tests, build files

**`api.py`: web service (FastAPI).**
- Serves the pooled tree model (first seed) from `artifacts/model`, with its alert threshold from `model_card.json`.
- `GET /health`, `GET /model-card`, `POST /score`. Input: age 14–110, sex F/M, ICU unit MICU/SICU/Unknown, hours since hospital admission, and 1–336 hourly dictionaries of measurements. It rejects unknown measurement names, builds the same features, and returns the risk and whether it alerts for each hour.
- The model card says "Research demonstration of cross-hospital evaluation. Not a clinical tool."

**`synthetic.py`: made-up data in the exact PhysioNet format, for tests only.** Stays of 12–60 hours, a 12% sepsis rate, vitals and labs that drift with a risk ramp before onset, hospital B with higher blood pressure and no EtCO2, labs measured less often at B. **Why [code]:** the cross-hospital code has a real shift to work on; the numbers mean nothing clinically.

**Tests (`python -m pytest -q`, 20 tests):**
- *Data:* both hospitals present; OMOP tables well formed; the OMOP round trip rebuilds the data; cleaning removes impossible values; quality counts; splits never share patients; features use only the past.
- *Evaluation:* weighted AUROC matches scikit-learn with ties; the interval contains the estimate and the paired difference detects a better model; Wilson interval.
- *Models:* all settings run; models beat chance; outputs are written; the threshold hits the target catch rate; the API scores each hour; calibration never reverses ranking.
- *Active learning:* k-centre spreads points; selection returns unlabelled patients; end-to-end run with all strategies; since Nov 9, the new columns, recalibrate equal to as-is on AUROC, recalibrate and random labelling the same patients, saved predictions, and the fixed-alert metric on a hand-worked example.

**Build files:** `Makefile` targets `install`, `data`, `run`, `run-seeds`, `run-fast`, `active`, `test`, `serve`, `docker`. `Dockerfile`: Python 3.11, CPU-only PyTorch, copies `src/` and `artifacts/model/`, runs uvicorn as a non-root user. `requirements.txt`: pandas, numpy, scikit-learn, pyarrow, torch, matplotlib, joblib, fastapi, uvicorn, pydantic, httpx, pytest. There is no `setup.py` or `pyproject.toml`; run with `PYTHONPATH=src`.

---

## 9. Things to check before writing the paper

Most are about clear wording, not errors.

| # | Item | Why it matters | Suggested action |
|---|---|---|---|
| 1 | ~~Trees in `active.py` used 200 trees, the pipeline 300.~~ **Fixed Nov 10**: both use 300, no early stopping | As-is AUROC differed (A→B 0.756 active vs 0.739 pipeline) | Rerun both with the new code |
| 2 | ~~HistGB early stopping on by default.~~ **Fixed Nov 10**: off everywhere | Results before Nov 10 used it | Methods: "300 trees, no early stopping"; rerun tree results |
| 3 | Hourly AUROC includes post-onset hours | More optimistic than early prediction | Report pre-onset AUROC too, or word it as "hourly AUROC under the challenge labels" |
| 4 | Onset = first label + 6, wrong when the first label is at hour 0 | "Flagged by onset" is measured against an estimated onset for these patients | Count them; sensitivity analysis excluding them |
| 5 | Local-norm statistics use all patients, including test | Transductive, label-free | State it, or compute on training patients only |
| 6 | ICULOS and HospAdmTime are features, and in active learning patient summaries | May encode hospital practice | Planned ablation |
| 7 | "Uncertainty" ≈ average risk because p < 0.5 | Its name overstates what it does | Explain, or redefine relative to the alert threshold |
| 8 | Greedy k-centre picks outliers first | Diversity is poor at small budgets for a known reason | Mention; consider a k-means++ or density-weighted variant |
| 9 | Per-hour weighting favours long stays; 10x weight distorted targeted strategies | Confirmed Nov 10: 1x > 3x > 10x | Default now 1x; justify 1x a priori or check on validation patients; report 3x/10x as sensitivity |
| 10 | "All target labels" = source + target at 10x | Not a pure target-only upper bound | Describe accurately; `target_only` with a full budget gives the other version |
| 11 | ~~Reset threshold for retrained models was in-sample.~~ **Fixed Nov 11**: 3-fold cross-fitting | In-sample version flagged only 2–30% | Describe cross-fitting in the methods |
| 12 | DANN model selection uses source validation; λ = 0.3 not tuned | Explains seed variability; reviewers ask | One sentence in methods |
| 13 | No hyperparameter tuning anywhere | Reviewers ask | State fixed defaults, chosen before cross-hospital testing, the same for all settings |
| 14 | Pipeline cache key ignores code version | Old predictions could be mixed in after code changes | Delete `artifacts/runs` after changing model or feature code |
| 15 | Subgroups and calibration use the first seed's pooled tree model only | Not seed-averaged | Fine for a description; say so |
| 16 | Age is stored as whole years after the OMOP round trip | Minor | "Age in years" |

---

## 10. Every parameter in one place

| Where | Parameter | Value |
|---|---|---|
| Labels | lead time before onset | 6 h |
| Split | seed; fractions | 42; 70/15/15 per hospital |
| Features | window for vital summaries | 6 h |
| Features | count | 98 (40 vital, 52 lab, 6 static) |
| Trees | learning rate; leaves; min leaf; L2 | 0.05; 31; 200; 1.0 |
| Trees | number of trees | 300 in pipeline and active (fast 100 / 80); early stopping off (since Nov 10) |
| Trees | negative-hour share kept | 0.3, with odds correction |
| Importance weights | domain classifier | HistGB 150 trees, lr 0.1, min leaf 500 |
| Importance weights | sample; clip | 200,000 hours per hospital; [0.1, 10], mean 1 |
| Calibration | method | Platt, C = 10⁶, slope forced positive |
| GRU | inputs; hidden size | 109; 64 |
| GRU | epochs; batch; optimiser | 8 (fast 3); 64 patients; Adam 1e-3 |
| GRU | positive weight; gradient clip | min(10, √(neg/pos)); 5 |
| GRU | max stay | 336 h |
| GRU | selection | best epoch by source validation AUROC |
| DANN | max weight; schedule | 0.3; 2/(1 + e^(−10p)) − 1 |
| DANN | target batch | 64 random target training patients per step |
| Threshold | target | flag 80% of septic patients by onset, on the training hospital's validation patients |
| Trade-off curve | targets | 30% to 95%, step 5 |
| Bootstrap | draws; unit | 200 (fast 30); whole patients; paired across models |
| Subgroups | age bands | < 45, 45–64, 65–79, ≥ 80 |
| Active | budgets | 25, 50, 100, 200, 400 (cumulative) |
| Active | seeds | 3 in first run; 5 planned |
| Active | target weight; weighting | 1 (10 before Nov 10); per hour (per patient optional) |
| Active | hybrid candidate pool | top 3 × k by averaged rank |
| Active | hospital classifier | HistGB 150 trees, lr 0.1, 3-fold out-of-fold |
| Active | worst group | age band or sex with ≥ 5 septic test patients |
| Active | reset threshold | needs ≥ 3 labelled septic patients; out-of-fold scores, 3 folds, for retrained models |
| Active | fixed operating point | 20% of non-septic test patients alerted |

---

## 11. Commit history

| Commit | Date | What |
|---|---|---|
| `bb060b6` | Nov 7 | OMOP ETL, models, cross-site evaluation, API |
| `f619358` | Nov 7 | Active domain adaptation experiments (the version that produced the first active learning results) |
| `02b9d74` | Nov 9 | Repeat trainings over seeds, paired comparisons, subgroup intervals |
| `e78c32c` | Nov 9 | Active learning saves progress after each strategy |
| `da455e6` | Nov 9 | Results from the full run |
| `ec9a82f` | Nov 9 (pushed Nov 10) | Active learning: reset thresholds, fixed alert-rate metric, comparison methods, saved predictions |
| `41a271f` | Nov 10 | Trees without early stopping; active learning 300 trees, 1x weight, `--shared-features` |

The changes between `f619358` and `da455e6` don't touch the training, selection, threshold or alert metric code used by `active.py`, so the first active learning results stand.
