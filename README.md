# COMPASS / IPIO Survival Analysis

Research workflow for assembling a prostate-cancer cohort from local DFCI / Profile / OncDRS
exports and running landmark survival analysis (Cox / XGBoost) on the resulting cohort.

The code is pandas-based. Entry points are command-line scripts orchestrated from notebooks.
COMPASS uses one `COMPASS_run_locally.ipynb` notebook with separate sections for the first-treatment
(`t_first_treatment`) and treatment-anchored (`t_treatment_anchor`) analyses; figures are produced
by the paired `COMPASS_generate_figures.ipynb` notebook.

> This README is the canonical reference for editing the pipeline. It documents the directory
> layout, the data flow, every script's inputs/outputs, the **conventions and invariants that
> must be preserved** (split discipline, train-only fitting, landmark base, ID-column handling),
> and a list of **known issues / footguns**. Read the "Conventions & invariants" and "Known
> issues" sections before changing modeling or data-prep code.

---

## Repository structure

Only `COMPASS/`, `IPIO/`, `survival_common/`, and `data_preprocessing_common/` are tracked — see
`.gitignore` (`/*` is ignored except an allowlist). `common_OMOP/` exists on disk but is **not**
tracked.

```text
survival_common/                         # shared survival-analysis library used by COMPASS PROFILE + IPIO
├── config.py                            # project config hooks for shared runners
├── cox_runners.py                       # shared univariate/multivariable Cox CLI orchestration
├── cox_models.py                        # shared Cox feature selection, CV, final-fit, and manifest helpers
├── cohort.py                            # landmark/outcome/feature-matrix builders
├── cox_engine.py                        # shared Cox / Coxnet / IPCW AUC(t) primitives
├── xgboost_engine.py                    # shared XGBoost survival:cox primitives
├── xgboost_runners.py                   # shared XGBoost CLI orchestration
├── helper.py                            # canonical labs, horizons, Brier, fold/leakage guards
├── loaders.py                           # PROFILE longitudinal loader helpers
├── plotting.py                          # lab taxonomy, IRNT, overlay hist/KM, Wilson CI
└── projects/                            # COMPASS PROFILE / IPIO study-specific configs

data_preprocessing_common/               # shared data-preprocessing utilities/resources
├── dfci_labs.py                         # unit standardization + sentinel/physiologic filtering
├── resources/lab_mappings/
│   └── OMOP_to_DFCI_lab_ids.csv          # canonical lab-id -> OMOP mapping
└── projects/                            # per-project preprocessing defaults

COMPASS/
├── data_preprocessing/                   # raw exports + survival model input builders
│   ├── compile_prostate_data.py          # ENTRY: build prostate_* source tables
│   ├── compile_MRNs_for_manual_review.py # build review sheet (platinum + ICD + PARPi + BRCA2)
│   ├── longitudinal_data_processing.py   # ENTRY: raw exports -> longitudinal_prediction_data.csv
│   ├── build_prediction_inputs.py        # ENTRY: landmark cohorts, split, canonical labs, horizons
│   └── build_genomic_inputs.py           # optional: genomic arm inputs
│
└── survival_analysis/
    ├── cox_aggregated.py                 # PROFILE adapter/config for shared survival code
    ├── univariate_analysis.py            # ENTRY: univariate Cox associations
    ├── multivariate_analysis.py          # ENTRY: elastic-net Cox or XGBoost survival:cox
    └── COMPASS_run_locally.ipynb / COMPASS_generate_figures.ipynb
```

---

## Data flow

```text
 raw DFCI / OncDRS / Profile exports  (/data/gusev/USERS/jpconnor/data/...)
        │
        ▼  data_preprocessing/compile_prostate_data.py
 prostate_text_data.csv, prostate_icd_data.csv, prostate_health_history_data.csv,
 prostate_medications_data.csv, prostate_labs_data.csv, prostate_somatic_data.csv,
 total_psa_records.csv, platinum_chemo_records.csv
        │
        ▼  data_preprocessing/longitudinal_data_processing.py
 longitudinal_prediction_data.csv
        │
        ▼  data_preprocessing/build_prediction_inputs.py     (+ build_genomic_inputs.py)
 prediction_inputs/  (aggregated + pre-treatment long labs + split + horizons)
        │
        ▼  survival_analysis/univariate_analysis.py
           survival_analysis/multivariate_analysis.py
        │
        ▼  COMPASS_generate_figures.ipynb (first-treatment and treatment-anchor sections)
 figures/
```

---

## Stage 1 — Compile prostate cohort source data

`data_preprocessing/compile_prostate_data.py` (module-level script, no CLI). Derives the broad
prostate MRN set from the inferred-cancer table, removes patients with a clear non-prostate-primary
ICD, loads any available batched note JSONs into `prostate_text_data.csv`, then filters the raw ICD /
health-history / medication / lab / somatic exports down to that broad prostate set and derives
`total_psa_records.csv` and `platinum_chemo_records.csv`.

- **Inputs (hard-coded under `DATA_PATH = /data/gusev/USERS/jpconnor/data/`):**
  `first_treatments_dfci_w_inferred_cancers.csv`, `full_VTE_embeddings_metadata.csv`,
  `VTE_notes_with_full_metadata_batch_*.json`, `timestamped_icd_info.csv.gz`, `HEALTH_HISTORY.csv`,
  `MEDICATIONS.csv`, `OUTPT_LAB_RESULTS_LABS.csv`, `complete_somatic_data_df.csv`.
- **Outputs (under `NEPC_PROJ_PATH = DATA_PATH/CAIA/COMPASS/`):** the eight `prostate_*` /
  `*_records.csv` tables listed in the data-flow diagram.
- **Cohort definition:** inferred-cancer prostate patients after ICD-based non-prostate-primary
  exclusion, not only patients with first-treatment anchors or batched note text. Notes remain a
  useful subset, but the structured longitudinal lab pipeline is built from the full prostate set and
  cohort-specific selection happens downstream.

`compile_MRNs_for_manual_review.py` builds an auxiliary manual-review MRN sheet from the
stage-1 outputs (platinum mentions, non-prostate-primary ICDs, PARPi exposure, BRCA2 status).

---

## Stage 2 — Survival analysis

COMPASS survival analysis is PROFILE-only. Shared implementation details that are reused by
COMPASS PROFILE and IPIO live in `survival_common/`.

- **COMPASS PROFILE** — reads `longitudinal_prediction_data.csv` and runs the prostate-cancer
  landmark analyses.
- **IPIO** — has its own cohort/outcome assembly but reuses the generic survival mechanics in
  `survival_common/`.

### 2.1 — `data_preprocessing/longitudinal_data_processing.py` → `longitudinal_prediction_data.csv`

Consolidates/standardizes labs (via `data_preprocessing_common/dfci_labs.py`), attaches the first
prostate (`C61`) diagnosis date when available and outcomes, rebases all timing to
`FIRST_RECORD_DATE = min(first lab, diagnosis, first treatment)`, and writes the broad row-level
prostate lab frame. Patient-level cohort filters are deferred to `build_prediction_inputs.py` so
different anchors can select their own cohorts.

- **Broad processing:** inferred-cancer prostate required and non-prostate-primary ICDs excluded at
  stage 1; a C61 diagnosis date is attached when present but is not required. No first-treatment,
  PSA-count, or PARPi filter is applied to `longitudinal_prediction_data.csv`. `PARPI_EXPOSED` is
  carried as a patient-level flag for downstream filtering.
- **Lab QC (`consolidate_dfci_labs`):** unit standardization to canonical units, sentinel nulling
  (e.g. `9999999`), physiologic-range nulling, combined-BP splitting. Out-of-range values are **nulled,
  not row-dropped** — downstream must filter on `conversion_status` (or pass `--successful-only`).
- **Shared lab resources:** the canonical mapping lives at
  `data_preprocessing_common/resources/lab_mappings/OMOP_to_DFCI_lab_ids.csv`. The
  `unique_lab_ids_w_units.csv` inventory is generated per project under the project data root for
  diagnostics / optional mapping refreshes; it is not a repo source of truth.
- **Timing semantics:** `t_lab`, `t_diagnosis`, `t_first_treatment`, `t_platinum`, `t_last_contact`.
  `DEATH` / `t_death` may remain in source tables as follow-up metadata, but COMPASS models do not use
  death as an endpoint.

### 2.2 — `data_preprocessing/build_prediction_inputs.py` → `prediction_inputs/`

The **single source of truth for model inputs**. For each landmark it cohort-filters, intersects MRNs
across landmarks, derives the train/valid/test split **once on the base landmark**, and writes per-landmark
aggregated tables plus pre-treatment long labs and shared split / canonical-lab / horizon artifacts.

- **Key CLI:** `--data`, `--landmark-days 0 90` (default from `cox_aggregated.DEFAULT_LANDMARK_DAYS`),
  `--seed`, `--test-frac`, `--val-frac`, `--time-unit-days 7`, `--min-patient-coverage`,
  `--auc-quantiles`, `--id-col`, `--age-col`, `--anchor-col`, `--stage-file`,
  `--restrict-to-mrns`, `--require-first-treatment` / `--no-require-first-treatment`,
  `--min-psa-count`, `--exclude-parpi` / `--include-parpi`.
- **Default downstream cohort filters:** `FIRST_TREATMENT == 1`, ≥5 PSA rows, and PARPi exclusion
  (when `PARPI_EXPOSED` is present). These defaults preserve the original first-treatment cohort, but
  alternate anchors can relax them explicitly.
- **Outputs:** `aggregated_landmark{D}.csv`, `pre_treatment_lab_long_landmark{D}.csv`,
  `split_assignments.csv`, `landmark_mrn_availability.csv`, `canonical_labs_train_val.csv`,
  `landmark_attrition.json`, `build_manifest.json`.
- `build_genomic_inputs.py` builds the parallel `prediction_inputs/genomic/` arm (index time =
  `SAMPLE_COLLECTION_DT`, plus 12 binary genomic indicators `{TP53,RB1,PTEN}×{SV,DEL,AMP,SNV}`). It
  **reuses** the main `split_assignments.csv` so test stays test.

### 2.3 — Models

All read prebuilt inputs and the `split` column; none re-derive the split. COMPASS models use the
`platinum` endpoint only (time to first platinum). Landmarks default `[0, 90]`. Metrics: Harrell
C-index, IPCW mean AUC(t), integrated IPCW Brier — horizons come from `build_manifest.json` so all
models share a grid.

| Script | Model | CLI notes |
|---|---|---|
| `univariate_analysis.py` | Cox: univariate n_obs-adjusted associations | `--landmark-days`, `--endpoints` |
| `multivariate_analysis.py --model elastic-net` | Elastic-net Cox multivariable model (sksurv `CoxnetSurvivalAnalysis`, 5-fold CV, AGE unpenalized) | `--landmark-days`, `--endpoints`, `--n-folds` |
| `multivariate_analysis.py --model xgboost` | XGBoost `survival:cox`, 5-fold CV grid (`max_depth × eta × min_child_weight`) | `--landmark-days`, `--endpoints`, `--max-features` |

`cox_aggregated.py` is now a project adapter: endpoint constants, cohort-specific covariates/restrictions,
and per-landmark context. The univariate/elastic-net CLI orchestration lives in
`survival_common/cox_runners.py`; reusable Cox feature selection, CV, final-fit, and manifest helpers live in
`survival_common/cox_models.py`; low-level Cox fitting/evaluation primitives live in
`survival_common/cox_engine.py`; XGBoost orchestration lives in `survival_common/xgboost_runners.py`;
low-level XGBoost mechanics live in `survival_common/xgboost_engine.py`.

### 2.4 — Notebooks

COMPASS PROFILE has one run notebook and one figure notebook:

- `COMPASS_run_locally.ipynb` — drives preprocessing, then has one section for the default
  first-treatment arm (`t_first_treatment`) and one section for the treatment-anchored arm
  (`t_treatment_anchor`). Each section builds its own prediction inputs and runs univariate,
  elastic-net, and XGBoost models at landmarks 0/90.
- `COMPASS_generate_figures.ipynb` — has separate sections for the first-treatment and
  treatment-anchored arms. Each section builds Figure 1 (cohort overview), Figure 3 (paired
  univariate volcano), and Figure 4a/4b/compiled (discrimination + importance grid).
---

## Conventions & invariants (preserve these when editing)

1. **The split is derived once, in `build_prediction_inputs.py`, on the base landmark.** The base landmark
   is `--landmark-days[0]` — **order matters**. Every model reads `split_assignments.csv`; never re-split
   inside a model script. `build_genomic_inputs.py` reuses the same file.
2. **Fit on the training block; never touch test for fitting.** Imputers, `StandardScaler`,
   canonical-lab selection, and Breslow baselines are all fit on train+valid (or fold-train inside CV) and
   applied to eval. Per-fold canonical labs are recomputed inside CV. The leakage guards
   `assert_no_test_leakage` / `assert_disjoint_folds` live in `survival_common/helper.py`.
3. **Canonical labs and horizon grids are training-block artifacts.** The main canonical lab set and
   AUC/Brier horizons are derived on **train+valid**; CV recomputes canonical labs on each fold-train.
   Do not derive these from held-out test patients.
4. **Endpoint and duration:** COMPASS uses `(t_platinum, PLATINUM)` only. For non-platinum patients,
   the anchor time is filled with `t_last_contact` (censoring). After landmark rebasing, the validity
   filter requires duration `> 0`, which silently drops patients with platinum before/at the landmark —
   add count logging if you depend on it.
5. **ID/age columns are injected at runtime.** PROFILE defaults to `DFCI_MRN` / `AGE_AT_TREATMENTSTART`;
   IPIO defaults to `DFCI_MRN` with its own baseline covariates. `build_*` and model `main()` functions
   mutate module globals **and monkey-patch `cox_aggregated.ID_COL/AGE_COL`**. If you add a function that captures
   `ID_COL` at import time (default arg, module constant), it will not see the patch — thread the column
   through as a parameter instead.
6. **Horizon grid is shared via `build_manifest.json`** so Cox/XGBoost AUC & Brier are comparable.
   Don't compute horizons ad hoc in a model script.

---

## Configuration & paths

- **Hard-coded cluster roots** (all overridable by CLI except module-level constants):
  - Data: `/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/`
  - Survival results: `/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis`
  - Figures: `/data/gusev/USERS/jpconnor/figures/CAIA/COMPASS/`
- `data_preprocessing_common/dfci_labs.py` uses the checked-in shared
  `resources/lab_mappings/OMOP_to_DFCI_lab_ids.csv` by default. Per-project lab inventory outputs
  default to `/data/gusev/USERS/jpconnor/data/CAIA/<project>/unique_lab_ids_w_units.csv`.

## Recommended run order

```bash
# Stage 1 (cluster paths hard-coded)
python COMPASS/data_preprocessing/compile_prostate_data.py

# Stage 2 — or just run COMPASS/survival_analysis/COMPASS_run_locally.ipynb top to bottom
python COMPASS/data_preprocessing/longitudinal_data_processing.py
python COMPASS/data_preprocessing/build_prediction_inputs.py --landmark-days 0 90 --time-unit-days 7
python COMPASS/survival_analysis/univariate_analysis.py --inputs-dir <...>/prediction_inputs --landmark-days 0
python COMPASS/survival_analysis/multivariate_analysis.py --model elastic-net --inputs-dir <...>/prediction_inputs --landmark-days 0
python COMPASS/survival_analysis/multivariate_analysis.py --model xgboost --inputs-dir <...>/prediction_inputs --landmark-days 0
```

## Dependencies

No packaged environment is checked in. Assumed: `pandas`, `numpy`, `scipy`, `tqdm`, `scikit-learn`,
`scikit-survival` (`sksurv`), `xgboost`, `lifelines`, `matplotlib`. Python **3.10+** is recommended
for the modern type-hint syntax used by the shared modules.

---

## Known issues / footguns

These are real, verified items found in code review. Fix opportunistically; at minimum, don't be
surprised by them.

### High impact

- **No true date of death in source data** (`longitudinal_data_processing.py`). `DEATH` / `t_death`
  remain metadata only; COMPASS modeling intentionally ignores death as an endpoint.

### Medium impact

- **`auc_max_time_units = 260` is the default** in `cox_aggregated.py` and `multivariate_analysis.py` and
  admin-censors AUC/Brier unless `--auc-max-time-units` is overridden.
- **Silent patient drops** at several inner-joins and `valid`-mask filters (diagnosis/death inner joins,
  duration `> 0` filter). Downstream cohort filters now log attrition in `build_prediction_inputs.py`;
  keep that pattern for any new cohort-selection rule.

### Low impact / cleanliness

- `iterrows`/`apply`-based row loops in `data_preprocessing_common/dfci_labs.py` and `longitudinal_data_processing.py`
  are slow on full DFCI-scale pulls — vectorize if performance bites.

---

## Notes

- `.ipynb_checkpoints/` and `__pycache__/` artifacts are git-ignored and not part of the workflow.
- Several scripts read and write data **outside** the repository root (the `/data/gusev/...` cluster paths).
