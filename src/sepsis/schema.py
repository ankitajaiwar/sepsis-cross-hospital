"""Variables in the PhysioNet/CinC Challenge 2019 files and how they map to OMOP.

LOINC codes are the source codes we put in OMOP `measurement_source_value`.
`measurement_concept_id` is left as 0 in this ETL; in a production OMOP build it is
filled by joining these LOINC codes to the OHDSI Athena vocabulary.
Verify codes against Athena before production use.
"""

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]

LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
        "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
        "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
        "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen",
        "Platelets"]

PHYSIO = VITALS + LABS
DEMOGRAPHICS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
LABEL = "SepsisLabel"
ALL_COLUMNS = PHYSIO + DEMOGRAPHICS + [LABEL]

# variable -> (LOINC code, unit, plausible low, plausible high)
MEASUREMENTS = {
    "HR": ("8867-4", "beats/min", 20, 250),
    "O2Sat": ("59408-5", "%", 40, 100),
    "Temp": ("8310-5", "Cel", 25, 45),
    "SBP": ("8480-6", "mm[Hg]", 40, 300),
    "MAP": ("8478-0", "mm[Hg]", 20, 250),
    "DBP": ("8462-4", "mm[Hg]", 10, 200),
    "Resp": ("9279-1", "/min", 2, 80),
    "EtCO2": ("19889-5", "mm[Hg]", 5, 100),
    "BaseExcess": ("1925-7", "mmol/L", -40, 40),
    "HCO3": ("1959-6", "mmol/L", 2, 60),
    "FiO2": ("3150-0", "1", 0.15, 1.0),
    "pH": ("2744-1", "[pH]", 6.5, 8.0),
    "PaCO2": ("2019-8", "mm[Hg]", 5, 150),
    "SaO2": ("2708-6", "%", 20, 100),
    "AST": ("1920-8", "U/L", 1, 20000),
    "BUN": ("3094-0", "mg/dL", 1, 300),
    "Alkalinephos": ("6768-6", "U/L", 5, 5000),
    "Calcium": ("17861-6", "mg/dL", 2, 20),
    "Chloride": ("2075-0", "mmol/L", 50, 160),
    "Creatinine": ("2160-0", "mg/dL", 0.1, 30),
    "Bilirubin_direct": ("1968-7", "mg/dL", 0, 50),
    "Glucose": ("2345-7", "mg/dL", 10, 1500),
    "Lactate": ("2524-7", "mmol/L", 0.1, 40),
    "Magnesium": ("19123-9", "mg/dL", 0.2, 10),
    "Phosphate": ("2777-1", "mg/dL", 0.2, 20),
    "Potassium": ("2823-3", "mmol/L", 1, 12),
    "Bilirubin_total": ("1975-2", "mg/dL", 0, 60),
    "TroponinI": ("10839-9", "ng/mL", 0, 500),
    "Hct": ("4544-3", "%", 5, 75),
    "Hgb": ("718-7", "g/dL", 1, 25),
    "PTT": ("3173-2", "s", 10, 250),
    "WBC": ("6690-2", "10*3/uL", 0, 500),
    "Fibrinogen": ("3255-7", "mg/dL", 20, 1500),
    "Platelets": ("777-3", "10*3/uL", 1, 2500),
}

SITES = {"A": "training_setA", "B": "training_setB"}

# The challenge marks every hour from 6 hours before sepsis onset onward as 1.
LABEL_LEAD_HOURS = 6

# Standard OMOP gender concepts
GENDER_CONCEPT = {1: 8507, 0: 8532}   # male, female
