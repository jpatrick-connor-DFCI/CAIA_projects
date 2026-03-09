# CAIA

PySpark preprocessing pipelines for extracting longitudinal clinical lab data from OMOP CDM (Observational Medical Outcomes Partnership Common Data Model) for time-to-event modeling with SurvLatentODE.

Data source: DFCI deidentified OMOP CDM via Snowflake (`dfci_ia_aistudio.omop_caia_denorm` for clinical data, `snowflake_aistudio_full_catalog.omop_cdm_deid` for vocabulary).

## Repository Structure

```text
CAIA/
├── COMPASS/                              # Prostate cancer workflow
│   ├── rhino_preprocessing/
│   │   ├── compass_preprocessing.py      # Main COMPASS preprocessing pipeline
│   │   ├── compass_cohort_sensitivity.py # Cohort parameter sweep analysis
│   │   └── compass_preprocessing_prostate_only_no_parp.py
│   ├── ERIS_testing/
│   │   ├── compile_prostate_data.py      # Data extraction utilities
│   │   ├── compile_MRNs_for_manual_review.py
│   │   └── extract_units.ipynb           # Unit analysis notebook
│   └── COMPASS_Rhino Schema_v3.csv       # Output schema documentation
├── IPIO/                                 # Immunotherapy workflow
│   ├── rhino_preprocessing/
│   │   └── ipio_preprocessing.py         # Main IPIO preprocessing pipeline
│   └── IPIO_Rhino Schema.xlsx            # Output schema documentation
├── common_OMOP/                          # Shared lookup tables
│   └── all_common_OMOP_units.csv         # Allowed measurement/unit combinations
└── README.md
```

## Tech Stack

- **Language**: Python (PySpark)
- **Compute**: Apache Spark (Databricks)
- **Database**: Snowflake
- **Data Standard**: OMOP CDM v5
- **Downstream**: SurvLatentODE and other time-to-event models

---

## COMPASS (`compass_preprocessing.py`)

### Problem Formulation

Predict survival and/or transformation to neuroendocrine prostate cancer (NEPC) in prostate cancer patients using routine clinical laboratory measurements as irregular time series input to SurvLatentODE or other time-to-event models.

**Proxy for NEPC transformation**: Initiation of platinum-based chemotherapy (cisplatin, carboplatin, oxaliplatin) is used as a proxy event for NEPC transformation. This proxy is currently being stress-tested on internal DFCI data.

### Cohort Definition

- **Cancer**: Prostate cancer (OMOP condition concepts for malignant neoplasm of prostate and variants)
- **Eligibility**: Patients with >= 10 PSA measurements after (or before) their first prostate cancer diagnosis
- **Event**: First platinum-based chemotherapy exposure (one row per patient via ROW_NUMBER)

### Data Extraction

1. **Concept sets**: Prostate cancer diagnoses, PSA measurements (27 concept variants), routine clinical labs (46 measurements + 32 variant codings), platinum-based drugs (cisplatin, carboplatin, oxaliplatin)
2. **First prostate diagnosis**: Earliest condition_occurrence per patient (ROW_NUMBER, one row per patient)
3. **PSA eligibility filter**: Patients with >= 10 PSA measurements
4. **First platinum exposure**: Earliest platinum drug exposure per patient (ROW_NUMBER, one row per patient)
5. **Lab extraction**: All routine clinical labs for eligible patients (including pre-diagnosis labs)
6. **Follow-up**: Last observation date from the latest of: measurement, condition, drug exposure, visit, or death dates
7. **Concept remapping**: 38 variant LOINC concept IDs remapped to canonical IDs (e.g., PSA variants → 3013603, basophils generic → automated)
8. **Unit filtering**: Inner join on allowed measurement/unit combinations to exclude clearly erroneous unit pairings
9. **Unit conversion**: Conversion rules standardizing lab values to canonical units (e.g., g/L to g/dL, umol/L to mg/dL, pmol/L to ng/dL, cells/uL to 10^3/uL)
10. **Physiologic range filtering**: Left join with concept-specific physiologic ranges to remove implausible values
11. **Time-to-event columns**: `event_platinum` (0/1), `time_to_platinum_or_censor`, `time_to_death_or_censor`

### Output Schema (33 columns)

| Category | Columns |
|---|---|
| Patient demographics | `person_id`, `gender`, `race`, `ethnicity`, `age_at_diagnosis`, `date_of_birth` |
| Cancer | `prostate_cancer_diagnosis_date` |
| Treatment | `drug_concept_id`, `drug_name`, `drug_type`, `drug_initiation_date` |
| Follow-up & death | `last_followup_date`, `death_date`, `is_deceased` |
| Time-to-event | `event_platinum`, `time_to_platinum_or_censor`, `time_to_death_or_censor` |
| Lab info | `measurement_concept_id`, `lab_name`, `measurement_date`, `lab_value`, `unit_concept_id`, `lab_unit_name`, `lab_values_converted`, `unit_converted_id`, `unit_converted_name`, `lab_value_final` |
| Relative timing | `days_relative_to_diagnosis`, `days_relative_to_platinum_chemo_start`, `days_relative_to_last_followup` |

### Time-to-Event Variables

- **`event_platinum`**: Binary indicator (1 = patient received platinum chemotherapy, 0 = censored)
- **`time_to_platinum_or_censor`**: Days from prostate cancer diagnosis to platinum initiation (if event) or last follow-up (if censored)
- **`time_to_death_or_censor`**: Days from prostate cancer diagnosis to death (if deceased) or last follow-up (if alive)

---

## IPIO (`ipio_preprocessing.py`)

### Problem Formulation

Predict survival and/or immunotherapy toxicities in multi-cancer patients receiving immune checkpoint inhibitor (ICI) therapy, using routine clinical laboratory measurements as irregular time series input to SurvLatentODE or other time-to-event models.

**Proxy for immunotherapy toxicity**: ICI discontinuation and time on ICI are used as proxies for toxicity or patient tolerance, due to lack of structured toxicity documentation. A multi-signal discontinuation classification system infers the likely reason for ICI discontinuation.

### Cohort Definition

- **Cancer types**: NSCLC, Bladder, Kidney, Melanoma, and other solid tumors
- **Treatment**: Patients who received at least one ICI (PD-1, PD-L1, CTLA-4, LAG-3 inhibitors: nivolumab, pembrolizumab, atezolizumab, durvalumab, avelumab, ipilimumab, tremelimumab, relatlimab)
- **Event**: ICI discontinuation with inferred cause classification

### ICI Discontinuation Classification

Treatment blocks are identified using a gap-and-island technique (>60-day gap between ICI exposures = new treatment block). The **first** treatment block per patient is selected. Discontinuation cause is classified with the following priority:

| Priority | Cause | Definition |
|---|---|---|
| 1 | `DEATH` | Patient died within 60 days of last ICI cycle |
| 2 | `OBS_END` | Last ICI cycle is within 60 days of observation period end (right-censored) |
| 3 | `PROGRESSION` | Non-ICI antineoplastic therapy (ATC L01, excluding ICI) started within 90 days after ICI block end |
| 4 | `TOXICITY_LIKELY` | Systemic corticosteroid (ATC H02AB) administered within +/-14 days of ICI block end |
| 5 | `COMPLETED` | ICI treatment duration >= 330 days (~1 year, typical course length) |
| 6 | `UNDETERMINED` | None of the above signals detected |

### Data Extraction

1. **Concept sets**: Cancer diagnoses (by type), ICI drugs, non-ICI antineoplastics (ATC L01 hierarchy via concept_ancestor), systemic corticosteroids (ATC H02AB hierarchy via concept_ancestor), routine clinical labs (46 measurements + 32 variant codings)
2. **Cancer diagnosis**: Earliest diagnosis per patient per cancer type (ROW_NUMBER)
3. **ICI treatment blocks**: Gap-and-island detection of treatment blocks, first block selected
4. **Discontinuation inference**: Multi-signal classification (see above)
5. **Lab extraction**: All routine clinical labs for eligible patients (including pre-diagnosis labs)
6. **Follow-up**: Last observation date excluding ICI drug exposures (prevents self-referential censoring)
7. **Concept remapping**: 38 variant LOINC concept IDs remapped to canonical IDs (e.g., PSA variants → 3013603, basophils generic → automated)
8. **Unit filtering**: Inner join on allowed measurement/unit combinations
9. **Unit conversion**: Conversion rules standardizing to canonical units (e.g., g/L to g/dL, cells/uL to 10^3/uL, pmol/L to ng/dL)
10. **Physiologic range filtering**: Concept-specific physiologic ranges to remove implausible values
11. **Time-to-event columns**: `event_ici_discontinued` (0/1), `time_on_ici`, `time_to_death_or_censor`

### Output Schema (37 columns)

| Category | Columns |
|---|---|
| Patient demographics | `person_id`, `gender`, `race`, `ethnicity`, `age_at_diagnosis`, `date_of_birth` |
| Cancer | `cancer_type`, `cancer_subtype`, `diagnosis_date` |
| ICI treatment | `ici_start_date`, `last_ici_cycle_start_date`, `ici_type` |
| Follow-up & death | `last_followup_date`, `death_date`, `is_deceased`, `ici_discontinuation_date`, `ici_discontinuation_cause` |
| Time-to-event | `time_on_ici`, `event_ici_discontinued`, `time_to_death_or_censor` |
| Lab info | `measurement_concept_id`, `lab_name`, `measurement_date`, `lab_value`, `unit_concept_id`, `lab_unit_name`, `lab_values_converted`, `unit_converted_id`, `unit_converted_name`, `lab_value_final` |
| Relative timing | `days_relative_to_diagnosis`, `days_relative_to_ici_start`, `days_relative_to_ici_discontinuation`, `days_relative_to_last_followup` |

### Time-to-Event Variables

- **`event_ici_discontinued`**: Binary indicator (1 = ICI was actively discontinued [PROGRESSION, TOXICITY_LIKELY, COMPLETED, UNDETERMINED], 0 = censored [DEATH, OBS_END])
- **`time_on_ici`**: Days from ICI start date to ICI discontinuation date
- **`time_to_death_or_censor`**: Days from cancer diagnosis to death (if deceased) or last follow-up (if alive)

---

## Shared Processing Steps

Both pipelines share identical logic for:

- **Concept remapping**: 38 variant LOINC concept IDs are remapped to canonical IDs post-extraction (e.g., multiple PSA codings → 3013603, basophils/eosinophils generic → automated counts). This consolidates equivalent measurements before unit filtering.
- **Allowed unit combinations** (`all_common_OMOP_units.csv`): Validated measurement/unit pairings used to filter out erroneous unit assignments
- **Unit conversions**: Rules converting non-standard units to canonical forms (e.g., g/L -> g/dL for albumin, umol/L -> mg/dL for creatinine/bilirubin, cells/uL -> 10^3/uL for differentials, pmol/L -> ng/dL for Free T4, various TSH units -> mIU/L)
- **Physiologic range filtering**: Concept-specific plausible ranges to remove data entry errors and instrument artifacts
- **Lab value finalization**: `lab_value_final = COALESCE(lab_values_converted, lab_value)` -- uses converted value when available, otherwise the original

## Output

Each script produces a single long-format DataFrame (one row per patient per lab measurement per date) that is uploaded to Snowflake for downstream modeling. Summary statistics are printed at the end of each script covering cohort demographics, treatment distributions, lab measurement distributions, unit conversion audits, follow-up/survival, time-to-event variables, pre-diagnosis labs, and null value audits.

---

## Supporting Scripts

- **`compass_cohort_sensitivity.py`**: Parameter sweep over PSA concept sets and count thresholds (1–30) to show patient inclusion/exclusion at each filter step.
- **`ERIS_testing/compile_prostate_data.py`**: Data extraction utilities for the DFCI prostate cohort (text notes, health history, medications, labs, somatic data).
- **`ERIS_testing/extract_units.ipynb`**: Jupyter notebook for unit analysis and validation.
