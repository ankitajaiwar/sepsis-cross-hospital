# Predicting sepsis early, and testing whether it works at a different hospital

Sepsis is the body's extreme reaction to an infection, and every hour of delay in
treatment makes it more dangerous. Hospitals use early-warning models to flag
patients before sepsis is obvious. The catch: a model built at one hospital often
works worse at the next, because hospitals measure and record things differently.

This project builds sepsis early-warning models from real ICU records and asks:
**if you train at one hospital, how well does it work at another, and can that drop
be reduced?**

## The data

The open [PhysioNet/CinC Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/)
dataset: hourly vital signs, lab results and demographics for 40,336 ICU patients
from two US hospital systems (A and B). It's real electronic health record data, with
the gaps and errors left in on purpose.

> Reyna MA, et al. Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing
> in Cardiology Challenge 2019. *Critical Care Medicine* 48(2):210–217, 2020.

## What the project does

1. **Converts the data to OMOP.** OMOP is the standard format many hospital networks
   use to share research data. The raw files go into OMOP tables (person, visit,
   measurement with LOINC lab codes, condition), and the analysis cohort is built back
   out of those tables, the same way a multi-hospital study would.
2. **Checks data quality.** Missing data and impossible values are counted at each
   hospital, and impossible values are removed. The two hospitals turn out to record
   data quite differently (for example, hospital B never records exhaled CO2).
3. **Trains two kinds of model:**
   - gradient-boosted trees on summaries of the last few hours, and
   - a recurrent neural network (GRU, in PyTorch) that reads each patient's full
     hourly history, including *when* each value was last measured.
4. **Tests across hospitals.** Each model is trained:
   - at the same hospital it's tested on,
   - on both hospitals together, and
   - at the *other* hospital, then tested as-is and with three ways of adapting
     to the new hospital without using any of its outcomes:
     - adjusting each hospital's values to its own norms,
     - re-weighting training patients to look like the new hospital (trees), and
     - domain-adversarial training, which teaches the network to ignore which
       hospital the data came from.
5. **Evaluates the way a hospital would.** Besides ranking accuracy, it reports
   calibration, how many septic patients get flagged before onset, how early, how
   many patients *without* sepsis get alerts, and whether performance holds across
   age groups, sex and ICU type. Alert thresholds are always set at the training
   hospital, never tuned on the test hospital.
6. **Asks which patients a new hospital should label.** A hospital adopting a model
   built elsewhere can usually afford chart review for only a few of its own
   patients. `active.py` compares ways of choosing them, in rounds of 25 to 400
   patients: at random, by model uncertainty, by highest risk, by spreading picks
   across the population, by picking patients least like the original hospital, and a
   hybrid of uncertainty, hospital difference and spread. After each round the model is
   retrained and judged on accuracy, alerts and the worst-served patient group.
7. **Serves predictions.** A small web service takes a patient's hourly data and
   returns the risk for each hour and whether it would trigger an alert.

## Results

_To be filled in after the full run. See [docs/REPORT.md](docs/REPORT.md)._

## Running it

You need Python 3.10 or newer and about 8 GB of free memory for the full dataset.

**Mac / Linux**
```bash
pip install -r requirements.txt
make test        # 19 tests on made-up data, a few minutes
make data        # downloads the 40,336 patient files (resumable)
make run         # full analysis, roughly 30-60 minutes on a laptop
make active      # which-patients-to-label experiments, a few hours (run after `make run`)
make serve       # web service at http://localhost:8000/docs
```

**Windows (PowerShell)**
```powershell
pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m pytest -q
python -m sepsis.download
python -m sepsis.pipeline --raw data/raw --out artifacts
python -m sepsis.active --omop artifacts/omop --out artifacts_active
```

To train every model several times with different random seeds (recommended before
comparing methods), reusing the OMOP tables from the first run:
`python -m sepsis.pipeline --from-omop artifacts/omop --out artifacts --seeds 0,1,2`.
Finished trainings are saved in `artifacts/runs`, so an interrupted run picks up where it
stopped and only new seeds are trained.

For a quick trial on part of the data: `python -m sepsis.pipeline --raw data/raw --out artifacts_fast --limit 2000 --fast`.

## Code map

| File | What it does |
|---|---|
| `download.py` | Fetches the open data |
| `raw.py` | Reads the hourly patient files |
| `omop.py` | Converts to OMOP tables and builds the cohort from them |
| `quality.py` | Data-quality report and cleaning |
| `features.py` | Patient splits, per-hospital normalisation, hourly features |
| `gbm.py` | Gradient-boosted trees and importance weighting |
| `seq.py` | GRU network and domain-adversarial training |
| `calibration.py` | Turns scores into probabilities without reordering them |
| `evaluate.py` | Accuracy, calibration, alert and subgroup metrics |
| `experiments.py` | Runs every training and testing setup |
| `report.py` | Charts and the written report |
| `active.py` | Which-patients-to-label experiments |
| `api.py` | Scoring web service |

## Limits

Two US hospital systems; a third was kept private by the challenge
organisers. The data has no real dates, so patients are split at random rather than by
time. Sepsis labels follow the challenge's definition. This is a research
demonstration, not a clinical tool.

---
Ankita Singh, Ph.D. · [LinkedIn](https://www.linkedin.com/in/ankitasingh1292) · [Google Scholar](https://scholar.google.com/citations?user=xRtUSq0AAAAJ)
