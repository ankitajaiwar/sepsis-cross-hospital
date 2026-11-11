# Predicting sepsis early, and checking whether it works at another hospital

Results from the full run on the PhysioNet/CinC 2019 data: every model trained 5 times with different seeds; trees with 300 trees and no early stopping. Charts are written to `artifacts/` when you run the pipeline. Subgroup and calibration results use the first seed's model trained on both hospitals.

## The data

ICU records from two US hospital systems (PhysioNet/CinC Challenge 2019): 40,336 patients and 1,552,210 patient-hours of vital signs and lab results.

| | Hospital A | Hospital B |
|---|---|---|
| Patients | 20,336 | 20,000 |
| Developed sepsis | 1,790 (8.8%) | 1,142 (5.7%) |
| Typical ICU stay (hours, median) | 39 | 38 |

The files were first converted to the OMOP common data model, and the analysis cohort was built from those tables. 11,403 impossible values (for example a heart rate outside 20–250) were set to missing.

## Does a model trained at one hospital work at the other?

Each model was tested on patients it never saw. AUROC is the chance the model gives a higher score to an hour shortly before sepsis than to an hour that isn't (0.5 = coin flip).

| Model | Trained on | Tested on | Method | AUROC (95% CI) | Range across runs | Septic patients flagged by onset | Patients without sepsis who got an alert | Alert-hours per 100 patient-days |
|---|---|---|---|---|---|---|---|---|
| Gradient-boosted trees | B | A | no adjustment | 0.747 (0.725–0.769) | 0.742–0.751 | 80% | 58% | 449 |
| Gradient-boosted trees | B | A | adjusted to local norms | 0.720 (0.700–0.745) | 0.714–0.724 | 75% | 53% | 325 |
| Gradient-boosted trees | B | A | importance weighting | 0.755 (0.734–0.779) | 0.746–0.760 | 80% | 51% | 429 |
| Gradient-boosted trees | A + B | A | – | 0.818 (0.798–0.840) | 0.815–0.823 | 83% | 35% | 395 |
| Gradient-boosted trees | A | A | – | 0.811 (0.788–0.836) | 0.810–0.813 | 83% | 37% | 398 |
| Gradient-boosted trees | A | B | no adjustment | 0.739 (0.710–0.780) | 0.735–0.746 | 53% | 22% | 193 |
| Gradient-boosted trees | A | B | adjusted to local norms | 0.746 (0.716–0.785) | 0.741–0.748 | 57% | 24% | 213 |
| Gradient-boosted trees | A | B | importance weighting | 0.750 (0.720–0.785) | 0.747–0.755 | 52% | 19% | 186 |
| Gradient-boosted trees | A + B | B | – | 0.846 (0.819–0.872) | 0.841–0.848 | 75% | 22% | 239 |
| Gradient-boosted trees | B | B | – | 0.847 (0.821–0.873) | 0.846–0.850 | 80% | 25% | 204 |
| Neural network (GRU) | B | A | no adjustment | 0.731 (0.706–0.758) | 0.727–0.740 | 78% | 57% | 563 |
| Neural network (GRU) | B | A | adjusted to local norms | 0.733 (0.709–0.757) | 0.720–0.743 | 71% | 48% | 401 |
| Neural network (GRU) | B | A | domain-adversarial training | 0.714 (0.689–0.740) | 0.707–0.725 | 59% | 31% | 326 |
| Neural network (GRU) | A + B | A | – | 0.800 (0.782–0.820) | 0.793–0.814 | 85% | 42% | 477 |
| Neural network (GRU) | A | A | – | 0.797 (0.779–0.818) | 0.787–0.803 | 82% | 38% | 441 |
| Neural network (GRU) | A | B | no adjustment | 0.752 (0.721–0.781) | 0.736–0.766 | 61% | 23% | 240 |
| Neural network (GRU) | A | B | adjusted to local norms | 0.754 (0.723–0.784) | 0.732–0.767 | 65% | 29% | 281 |
| Neural network (GRU) | A | B | domain-adversarial training | 0.789 (0.759–0.816) | 0.776–0.802 | 75% | 35% | 432 |
| Neural network (GRU) | A + B | B | – | 0.835 (0.812–0.858) | 0.833–0.838 | 80% | 29% | 285 |
| Neural network (GRU) | B | B | – | 0.831 (0.806–0.853) | 0.828–0.837 | 81% | 33% | 308 |

The alert threshold was always set on the training hospital's own patients (to flag about 80% of their septic patients by onset), as it would be before a real rollout. "Patients without sepsis who got an alert" counts anyone alerted at least once during their stay; alert-hours per 100 patient-days shows how much of the time alerts were on.

Every model was trained 5 times with different random seeds. The table shows the average; the interval comes from resampling test patients, and the range shows the lowest and highest AUROC across the separate trainings.

### Does adapting to the new hospital help?

Each line compares a method with using the other hospital's model unchanged, on the same test patients. An interval that includes 0 means no clear difference.

| Model | Trained on → tested on | Compared with no adjustment | Change in AUROC (95% CI) |
|---|---|---|---|
| Gradient-boosted trees | B → A | adjusted to local norms | -0.027 (-0.038 to -0.017) |
| Gradient-boosted trees | B → A | importance weighting | +0.007 (0.001–0.013) |
| Gradient-boosted trees | B → A | model trained at the test hospital itself | +0.064 (0.045–0.083) |
| Gradient-boosted trees | A → B | adjusted to local norms | +0.007 (-0.003 to +0.017) |
| Gradient-boosted trees | A → B | importance weighting | +0.011 (-0.003 to +0.024) |
| Gradient-boosted trees | A → B | model trained at the test hospital itself | +0.108 (0.076–0.141) |
| Neural network (GRU) | B → A | adjusted to local norms | +0.002 (-0.003 to +0.005) |
| Neural network (GRU) | B → A | domain-adversarial training | -0.017 (-0.030 to -0.007) |
| Neural network (GRU) | B → A | model trained at the test hospital itself | +0.066 (0.045–0.089) |
| Neural network (GRU) | A → B | adjusted to local norms | +0.002 (-0.003 to +0.008) |
| Neural network (GRU) | A → B | domain-adversarial training | +0.037 (0.020–0.057) |
| Neural network (GRU) | A → B | model trained at the test hospital itself | +0.079 (0.061–0.104) |

## What the care team would see

Each point is a different alert threshold. Moving right means more alerts for nurses to handle; moving up means more septic patients flagged before sepsis onset.

## Does it work equally well for everyone? (Gradient-boosted trees, trained on both hospitals)

| Group | Patients | Developed sepsis | AUROC (95% CI) | Septic patients flagged (95% CI) | Patients without sepsis alerted (95% CI) |
|---|---|---|---|---|---|
| Hospital A | 3,051 | 285 | 0.823 (0.803–0.844) | 84% (80–88) | 36% (34–38) |
| Hospital B | 3,000 | 174 | 0.848 (0.823–0.874) | 76% (69–82) | 23% (22–25) |
| Under 45 | 964 | 70 | 0.858 (0.821–0.893) | 86% (76–92) | 28% (25–31) |
| 45-64 | 2,199 | 158 | 0.840 (0.814–0.864) | 83% (76–88) | 29% (27–31) |
| 65-79 | 2,063 | 165 | 0.835 (0.808–0.859) | 77% (70–83) | 29% (27–31) |
| 80 and over | 825 | 66 | 0.833 (0.797–0.869) | 82% (71–89) | 36% (32–39) |
| Female | 2,665 | 188 | 0.848 (0.826–0.871) | 81% (75–86) | 27% (26–29) |
| Male | 3,386 | 271 | 0.833 (0.812–0.851) | 81% (76–85) | 31% (30–33) |
| Medical ICU | 1,840 | 163 | 0.841 (0.820–0.863) | 87% (80–91) | 37% (34–39) |
| Not recorded | 2,359 | 196 | 0.817 (0.791–0.840) | 82% (76–86) | 32% (30–34) |
| Surgical ICU | 1,852 | 100 | 0.866 (0.837–0.895) | 71% (61–79) | 20% (18–22) |

ICU type is missing in the source data for 2,359 test patients, so they appear as "Not recorded". Small groups have wide intervals; differences between groups whose intervals overlap may be chance.

Points on the dashed line mean the predicted risk matched what actually happened.

## Limits

- ICU data from two US hospital systems; a third was held back by the challenge organisers and isn't public.
- ICU type is missing for many patients in the source data.
- Sepsis labels follow the challenge's Sepsis-3 definition, which marks hours from 6 hours before onset.
- No real dates, so the split between training and test patients is random, not by time.
- This is a research demonstration, not a clinical tool.
