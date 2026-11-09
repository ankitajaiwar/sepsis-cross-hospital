# Predicting sepsis early, and checking whether it works at another hospital

Results from the full run on the PhysioNet/CinC 2019 data (one training run per model). Charts are written to `artifacts/` when you run the pipeline.

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

| Model | Trained on | Tested on | Method | AUROC (95% CI) | Septic patients flagged by onset | Patients without sepsis who got an alert |
|---|---|---|---|---|---|---|
| Gradient-boosted trees | B | A | no adjustment | 0.743 (0.723–0.766) | 80% | 60% |
| Gradient-boosted trees | B | A | adjusted to local norms | 0.721 (0.699–0.746) | 72% | 52% |
| Gradient-boosted trees | B | A | importance weighting | 0.748 (0.728–0.773) | 80% | 53% |
| Gradient-boosted trees | A + B | A | – | 0.820 (0.802–0.841) | 84% | 35% |
| Gradient-boosted trees | A | A | – | 0.813 (0.791–0.838) | 82% | 35% |
| Gradient-boosted trees | A | B | no adjustment | 0.734 (0.704–0.776) | 53% | 21% |
| Gradient-boosted trees | A | B | adjusted to local norms | 0.739 (0.709–0.778) | 57% | 25% |
| Gradient-boosted trees | A | B | importance weighting | 0.739 (0.711–0.775) | 52% | 20% |
| Gradient-boosted trees | A + B | B | – | 0.845 (0.817–0.872) | 75% | 23% |
| Gradient-boosted trees | B | B | – | 0.846 (0.819–0.872) | 82% | 25% |
| Neural network (GRU) | B | A | no adjustment | 0.732 (0.704–0.757) | 76% | 55% |
| Neural network (GRU) | B | A | adjusted to local norms | 0.736 (0.710–0.763) | 71% | 49% |
| Neural network (GRU) | B | A | domain-adversarial training | 0.715 (0.688–0.740) | 54% | 21% |
| Neural network (GRU) | A + B | A | – | 0.814 (0.795–0.832) | 87% | 42% |
| Neural network (GRU) | A | A | – | 0.798 (0.777–0.822) | 81% | 38% |
| Neural network (GRU) | A | B | no adjustment | 0.746 (0.710–0.781) | 63% | 27% |
| Neural network (GRU) | A | B | adjusted to local norms | 0.754 (0.721–0.788) | 66% | 33% |
| Neural network (GRU) | A | B | domain-adversarial training | 0.790 (0.758–0.819) | 70% | 26% |
| Neural network (GRU) | A + B | B | – | 0.838 (0.814–0.863) | 82% | 31% |
| Neural network (GRU) | B | B | – | 0.830 (0.806–0.855) | 79% | 32% |

The alert threshold was always set on the training hospital's own patients (to flag about 80% of their septic patients by onset), as it would be before a real rollout.

## What the care team would see

Each point is a different alert threshold. Moving right means more alerts for nurses to handle; moving up means more septic patients flagged before sepsis onset.

## Does it work equally well for everyone? (Gradient-boosted trees, trained on both hospitals)

| Group | Patients | Developed sepsis | AUROC | Septic patients flagged | Patients without sepsis alerted |
|---|---|---|---|---|---|
| Hospital A | 3,051 | 285 | 0.820 | 84% | 35% |
| Hospital B | 3,000 | 174 | 0.845 | 75% | 23% |
| Under 45 | 964 | 70 | 0.854 | 87% | 28% |
| 45-64 | 2,199 | 158 | 0.841 | 81% | 28% |
| 65-79 | 2,063 | 165 | 0.832 | 76% | 28% |
| 80 and over | 825 | 66 | 0.825 | 85% | 35% |
| Female | 2,665 | 188 | 0.845 | 79% | 27% |
| Male | 3,386 | 271 | 0.831 | 82% | 31% |
| Medical ICU | 1,840 | 163 | 0.839 | 86% | 37% |
| Not recorded | 2,359 | 196 | 0.815 | 81% | 31% |
| Surgical ICU | 1,852 | 100 | 0.858 | 72% | 18% |

Points on the dashed line mean the predicted risk matched what actually happened.

## Limits

- ICU data from two US hospital systems; a third was held back by the challenge organisers and isn't public.
- ICU type is missing for many patients in the source data.
- Sepsis labels follow the challenge's Sepsis-3 definition, which marks hours from 6 hours before onset.
- No real dates, so the split between training and test patients is random, not by time.
- This is a research demonstration, not a clinical tool.
