# CAIA

Research workflow for assembling a prostate cancer cohort from local DFCI / Profile / OncDRS exports, labeling each patient via a v3 single-call LLM classifier, and running landmark survival analysis (Cox / XGBoost / Dynamic-DeepHit) on the resulting cohort.

The code is pandas-based and the entry points are command-line scripts plus a local-runs notebook.

## Repository Structure

```text
CAIA/
├── COMPASS/
│   ├── data_preprocessing/        # prostate cohort + source-data compilation
│   │   ├── compile_prostate_data.py
│   │   ├── compile_MRNs_for_manual_review.py
│   │   └── compile_text_for_LLM_review.py
│   ├── v3/                        # current LLM labeling pipeline
│   │   ├── compile_prostate_note_bundle.py
│   │   ├── helpers.py
│   │   └── run_v3_pipeline.py
│   ├── survival_analysis/         # cohort consolidation + survival models
│   │   ├── consolidate_dfci_labs.py
│   │   ├── longitudinal_data_processing.py
│   │   ├── build_prediction_inputs.py
│   │   ├── build_genomic_inputs.py
│   │   ├── cox_aggregated.py
│   │   ├── cox_genomic_univariate.py
│   │   ├── cox_pgs_adjusted.py
│   │   ├── landmark_xgboost.py
│   │   ├── dynamic_deephit.py
│   │   ├── helper.py
│   │   ├── run_locally.ipynb
│   │   └── generate_figures.ipynb
│   ├── utils.py                   # shared note-cleaning rules + clean_note()
│   ├── OMOP_to_DFCI_lab_ids.csv
│   └── unique_lab_ids_w_units.csv
├── common_OMOP/
└── README.md
```

## Pipeline

### 1. Compile the prostate cohort source data

`COMPASS/data_preprocessing/compile_prostate_data.py`

Pulls prostate MRNs from the DFCI first-treatments table and filters the related raw exports (ICDs, health history, medications, labs, somatic, total PSA, platinum-chemo records) down to the prostate cohort. Outputs:

- `prostate_text_data.csv`, `prostate_icd_data.csv`, `prostate_health_history_data.csv`, `prostate_medications_data.csv`, `prostate_labs_data.csv`, `prostate_somatic_data.csv`
- `total_psa_records.csv`, `platinum_chemo_records.csv`

These CSVs are the upstream inputs for both the v3 LLM pipeline and the survival pipeline.

### 2. v3 LLM patient labeling

`COMPASS/v3/run_v3_pipeline.py`

One LLM call per patient against an Azure OpenAI `gpt-4o` deployment. Each patient is classified as **NEPC**, **AVPC**, **biomarker-driven**, or **conventional**, with structured fields covering NE features, AVPC criteria, biomarker genes, visceral metastasis patterns, supporting quotes, and confidence.

Two steps:

1. `compile_prostate_note_bundle.py` — gather all OncDRS notes for a list of DFCI MRNs into a gzipped JSON bundle.
2. `run_v3_pipeline.py` — read the bundle, build per-patient snippets, call the LLM with `CLASSIFY_SYSTEM_PROMPT`, parse the JSON response, and write `LLM_v3_labels.tsv`.

Helpers (`compile_prostate_note_bundle.py`, `helpers.py`) import `clean_note` from `COMPASS/utils.py`.

### 3. Survival analysis

`COMPASS/survival_analysis/`

A four-stage chain. All stages are driven from `run_locally.ipynb`, which invokes each script with `!{PYTHON} ...` so a single notebook kernel runs the whole pipeline end-to-end.

1. **Lab consolidation** — `longitudinal_data_processing.py` (which internally calls `consolidate_dfci_labs.consolidate_dfci_labs`) folds the raw `prostate_labs_data.csv` and health-history rows into a per-patient longitudinal prediction frame. Applies the sentinel-value filter and per-measurement physiologic-range filter. Output: `longitudinal_prediction_data.csv`.

   Cohort filters at this stage (in `apply_cohort_filters`):
   - C61 prostate ICD required; non-prostate primary excluded
   - Must have a recorded first treatment (`FIRST_TREATMENT == 1`)
   - ≥ 5 PSA rows
   - PARPi-exposed patients excluded

2. **Prediction inputs** — `build_prediction_inputs.py` (plus `build_genomic_inputs.py` for the genomic-anchored arm). Derives landmark cohorts (default landmarks: 0 and 90 days post first-treatment), 3-way train/valid/test split, canonical lab feature sets, AUC horizon grids, and writes per-landmark aggregated + longitudinal CSVs into `prediction_inputs/`.

3. **Models** — fit and evaluate at each landmark on the **PLATINUM** (time-to-first-platinum) and **DEATH** endpoints:
   - `cox_aggregated.py` — Cox univariate + multivariable (elastic net) on aggregated features
   - `landmark_xgboost.py` — XGBoost survival on the same aggregated features
   - `dynamic_deephit.py` — Dynamic-DeepHit on the longitudinal feature CSVs (PLATINUM and competing-risk configs)
   - `cox_pgs_adjusted.py` — PGS-adjusted Cox sweep for selected labs (e.g. Testosterone, PSA)
   - `cox_genomic_univariate.py` — genomic-feature univariate Cox on the t_sample-anchored cohort

4. **Inspection** — `run_locally.ipynb` (sections 4–6) collects headline metrics and runs cohort/feature diagnostics. `generate_figures.ipynb` builds the three platinum-arm figures (paired univariate volcano, grouped AUC barplot, 2×2 importance grid) from the same outputs and writes them to `figures/`.

## `utils.py`

`COMPASS/utils.py` contains note-cleaning regex rules (universal, clinician, imaging, pathology) and the `clean_note(text, note_type=None)` helper used by `COMPASS/v3/helpers.py`.

## Dependencies and Runtime Assumptions

No packaged environment definition is checked in. The scripts assume:

- Python with `pandas`, `numpy`, `tqdm`, `scikit-learn`, `scikit-survival`, `xgboost`, `lifelines`, `pycox`/`torch` (DeepHit)
- `openai`, `azure-identity` (LLM labeling)
- valid Azure credentials for `DefaultAzureCredential`
- access to the Azure OpenAI endpoint hard-coded in v3 / configured via env vars
- CSV and JSON data files under `/data/gusev/...` (raw exports and outputs)

Paths are mostly hard-coded to the cluster filesystem; the survival notebook exposes them as variables for local overrides.

## `IPIO`

`IPIO/` currently contains only `IPIO_Cohort_Report.docx`. The older IPIO preprocessing code referenced by prior README versions is not in this checkout.

## Notes

- `.ipynb_checkpoints/` and `__pycache__/` artifacts exist in the tree but are not part of the workflow.
- Several scripts read and write data outside the repository root.
- Earlier LLM-labeling code (`COMPASS/v2/`, `COMPASS/generate_LLM_labels.py`, `COMPASS/regex_generation/`) has been removed; v3 is the only supported labeling pipeline.
