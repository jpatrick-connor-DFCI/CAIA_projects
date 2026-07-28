# COMPASS / IPIO Survival Analysis

Research workflow for assembling a prostate-cancer cohort from local DFCI / Profile / OncDRS
exports and running landmark survival analysis (Cox / XGBoost) on the resulting cohort.

The preprocessing and modeling code is pandas-based; publication figures are generated only in R.
Entry points are command-line scripts orchestrated from notebooks.
COMPASS uses one `COMPASS_run_locally.ipynb` notebook for four treatment-anchored cohorts:
`without_other_primaries` and `with_other_primaries` (anchored at first ARPI/chemo exposure), plus
`adt_without_other_primaries` and `adt_with_other_primaries` (anchored at first ADT exposure,
years earlier in the treatment sequence). The paired, R-only `COMPASS_generate_figures.ipynb`
emits manuscript figures for all four.

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
│   ├── compile_COMPASS_cohort_data.py    # ENTRY: build prostate_* source tables + survival cohort
│   ├── compile_MRNs_for_manual_review.py # build review sheet (platinum + ICD + PARPi + BRCA2)
│   ├── longitudinal_data_processing.py   # ENTRY: raw exports -> longitudinal_prediction_data.csv
│   ├── build_prediction_inputs.py        # ENTRY: landmark cohorts, split, canonical labs, horizons
│   └── build_genomic_inputs.py           # optional: genomic arm inputs
│
└── survival_analysis/
    ├── cox_aggregated.py                 # PROFILE adapter/config for shared survival code
    ├── univariate_analysis.py            # ENTRY: univariate Cox associations
    ├── multivariate_analysis.py          # ENTRY: elastic-net Cox or XGBoost survival:cox
    ├── COMPASS_generate_figures_pipeline.R # sole figure-generation implementation
    ├── COMPASS_nominally_significant_univariate.ipynb # review nominal univariate hits
    └── COMPASS_run_locally.ipynb / COMPASS_generate_figures.ipynb # Python models / R figures
```

---

## Data flow

```text
 raw DFCI / OncDRS / Profile exports  (/data/gusev/USERS/jpconnor/data/...)
        │
        ▼  data_preprocessing/compile_COMPASS_cohort_data.py
 prostate_icd_data.csv,
 prostate_arpi_survival_cohort_{without,with}_other_primaries.csv,
 prostate_arpi_survival_cohort_{without,with}_other_primaries_arpi.csv,
 prostate_adt_survival_cohort_{without,with}_other_primaries_adt.csv
        │
        ▼  data_preprocessing/longitudinal_data_processing.py  (--anchor-med-set {arpi,adt})
 longitudinal_prediction_data.csv, longitudinal_prediction_data_adt.csv
        │
        ▼  data_preprocessing/build_prediction_inputs.py     (+ build_genomic_inputs.py)
 prediction_inputs/  (aggregated + pre-treatment long labs + split + horizons)
        │
        ▼  survival_analysis/univariate_analysis.py
           survival_analysis/multivariate_analysis.py
        │
        ▼  COMPASS_generate_figures.ipynb (R; 4 cohorts: ARPI/ADT × with/without other primaries)
 figures/
```

---

## Stage 1 — Compile prostate cohort source data

`data_preprocessing/compile_COMPASS_cohort_data.py` (module-level script, argparse CLI for path
overrides). Derives two ICD-C61 cohorts, with and without patients who have another primary
malignancy. Each definition has a full and treatment-anchor-restricted output, restricted
separately per anchor. The same script builds both the ARPI/chemo-anchored and the
ADT-anchored survival data (age, treatment anchor, death, time-to-platinum) directly from the raw
OncDRS pull, computing `anchor_df`/`adt_anchor_df` from the same medications scan.

- **Inputs (hard-coded under `DATA_PATH = /data/gusev/USERS/jpconnor/data/`, plus the raw OncDRS pull
  at `ONCDRS_PATH`):** `EHR_DIAGNOSIS.csv`, `HEALTH_HISTORY.csv`, `MEDICATIONS.csv`,
  `OUTPT_LAB_RESULTS_LABS.csv`, `complete_somatic_data_df.csv.gz`, `PT_INFO_STATUS_REGISTRATION.csv`.
- **Outputs (under `NEPC_PROJ_PATH = DATA_PATH/CAIA/COMPASS/`):** four bare (anchor-independent)
  survival-cohort CSVs (with/without other primaries, full and unrestricted), four
  ARPI-treatment-anchor-restricted CSVs (`..._arpi.csv`), four ADT-treatment-anchor-restricted CSVs
  (`prostate_adt_survival_cohort_{cohort_key}_adt.csv`), corresponding bare-MRN lists for each
  (including `mrn_lists/{cohort_key}_adt_mrns.csv`), `prostate_icd_data.csv`, and
  `mrn_lists/icd_prostate_mrn_flags.csv`. The latter contains every ICD-C61 MRN plus binary
  indicators for a non-prostate primary, PARPi exposure, ARPI/docetaxel exposure, ADT exposure
  (`ADT_EXPOSED`), and at least five broad PSA tests.
- **Cohort definitions:** `without_other_primaries` applies the ICD-based non-prostate-primary
  exclusion; `with_other_primaries` retains every ICD-C61 patient. Each also emits an `_arpi`
  ARPI-treatment-anchor-restricted subset and an `_adt` ADT-treatment-anchor-restricted subset.
- **Anchor sets:** `ARPI_ANCHOR_MEDS` (7 drugs: ARPIs, taxanes, radium-223 — unchanged,
  `TREATMENT_ANCHOR_MEDS` alias retained) and `ADT_ANCHOR_MEDS` (10 drugs: GnRH agonists/antagonists
  + first-generation antiandrogens). `compute_treatment_anchor(meds, meds_set=...)` takes either set;
  `ANCHOR_MED_SETS = {"arpi": ..., "adt": ...}` maps the CLI-facing key to the drug set.

`compile_MRNs_for_manual_review.py` builds an auxiliary manual-review MRN sheet from the
stage-1 outputs (platinum records, non-prostate-primary ICDs, PARPi exposure, BRCA2 status).

---

## Stage 2 — Survival analysis

COMPASS survival analysis is PROFILE-only. Shared implementation details that are reused by
COMPASS PROFILE and IPIO live in `survival_common/`.

- **COMPASS PROFILE** — reads `longitudinal_prediction_data.csv` (ARPI anchor) or
  `longitudinal_prediction_data_adt.csv` (ADT anchor, via `--anchor-med-set adt`) and runs the
  prostate-cancer landmark analyses.
- **IPIO** — has its own cohort/outcome assembly but reuses the generic survival mechanics in
  `survival_common/`.

### 2.1 — `data_preprocessing/longitudinal_data_processing.py` → `longitudinal_prediction_data.csv`

Consolidates/standardizes labs (via `data_preprocessing_common/dfci_labs.py`), attaches the first
prostate (`C61`) diagnosis date when available and outcomes, rebases all timing to
`FIRST_RECORD_DATE = min(first lab, diagnosis, first treatment)`, and writes the row-level
prostate lab frame used by the current treatment-anchored analyses.

- **Anchor selection:** `--anchor-med-set {arpi,adt}` (default `arpi`) selects `ARPI_ANCHOR_MEDS` or
  `ADT_ANCHOR_MEDS` (mirrored from Stage 1) and switches the defaults for `--survival-cohort-csv`,
  `--output-csv`, and `--consolidated-cache-parquet` to the ADT-suffixed files/cache when set to
  `adt`. Both anchors reuse the same lab-standardization code; only the anchor date and the derived
  timing columns differ.
- **Fast cohort scope:** by default, raw scans start from the widest anchored
  `with_other_primaries` cohort (or its ADT counterpart, under `--anchor-med-set adt`). PARPi-exposed
  patients and patients with fewer than five broad PSA records are removed before expensive lab
  standardization. Use `--prefilter-include-parpi`, `--prefilter-min-psa-count 0`, and/or a broader
  `--survival-cohort-csv` to rebuild a less restricted source frame.
- **Lab QC (`consolidate_dfci_labs`):** unit standardization to canonical units, sentinel nulling
  (e.g. `9999999`), physiologic-range nulling, combined-BP splitting. Out-of-range values are **nulled,
  not row-dropped** — downstream must filter on `conversion_status` (or pass `--successful-only`).
- **Performance:** raw CSV scans project only required columns; lab consolidation is vectorized.
  Standardized rows are cached in `consolidated_longitudinal_data.parquet` (ARPI) or
  `consolidated_longitudinal_data_adt.parquet` (ADT) with a provenance manifest that includes
  `anchor_med_set`, so switching `--anchor-med-set` cannot silently serve a stale cache built under
  the other anchor. Use `--refresh-cache` to rebuild or `--no-cache` to bypass it. Large diagnostic
  CSVs are opt-in via `--write-unique-labs`, `--write-uncondensed`, and `--write-consolidated`.
- **Shared lab resources:** the canonical mapping lives at
  `data_preprocessing_common/resources/lab_mappings/OMOP_to_DFCI_lab_ids.csv`. The
  `unique_lab_ids_w_units.csv` inventory can be generated per project with
  `--write-unique-labs` for diagnostics or mapping refreshes; it is not a repo source of truth.
- **Timing semantics:** `t_lab`, `t_diagnosis`, `t_first_treatment`, `t_treatment_anchor`,
  `t_platinum`, `t_last_contact`, `t_death`. `t_death` is a real death-date-derived duration when the
  survival cohort's `death_date` is available (falls back to the last-contact proxy for dead patients
  with no recorded date); COMPASS models still use the `platinum` endpoint only.

### 2.2 — `data_preprocessing/build_prediction_inputs.py` → `prediction_inputs/`

The **single source of truth for model inputs**. For each landmark it cohort-filters, intersects MRNs
across landmarks, derives the train/valid/test split **once on the base landmark**, and writes per-landmark
aggregated tables plus pre-treatment long labs and shared split / canonical-lab / horizon artifacts.

- **Key CLI:** `--data`, `--landmark-days 0 90` (default from `cox_aggregated.DEFAULT_LANDMARK_DAYS`),
  `--seed`, `--test-frac`, `--val-frac`, `--time-unit-days 7`, `--min-patient-coverage`,
  `--auc-quantiles`, `--id-col`, `--age-col`, `--anchor-col`,
  `--restrict-to-mrns`, `--require-first-treatment` / `--no-require-first-treatment`,
  `--min-psa-count`, `--exclude-parpi` / `--include-parpi`.
- **Default downstream cohort filters:** `FIRST_TREATMENT == 1`, ≥5 PSA rows, and PARPi exclusion
  (when `PARPI_EXPOSED` is present). These defaults preserve the original first-treatment cohort, but
  alternate anchors can relax them explicitly.
- **Outputs:** `aggregated_landmark{D}.csv`, `pre_treatment_lab_long_landmark{D}.csv`,
  `split_assignments.csv`, `landmark_mrn_availability.csv`, `canonical_labs_train_val.csv`,
  `landmark_attrition.json`, `build_manifest.json`.
- `IPIO/data_preprocessing/build_genomic_inputs.py` builds the parallel
  `prediction_inputs/genomic/` landmark-0 arm anchored at IO start (`t_first_treatment`), restricts
  to patients with an actual somatic sample, attaches dynamic binary `<GENE>_<SV|SNV|AMP|DEL>`
  indicators, and **reuses** the main `split_assignments.csv` so test stays test. It writes both
  genomic provenance files and runner-compatible aliases (`aggregated_landmark0.csv`,
  `pre_treatment_lab_long_landmark0.csv`, `canonical_labs_train_val.csv`, `build_manifest.json`).

### 2.3 — Models

All read prebuilt inputs and the `split` column; none re-derive the split. COMPASS models use the
`platinum` endpoint only (time to first platinum). Landmarks default `[0, 90]` for the ARPI-anchored
arms; the ADT-anchored arms use `[0, 90, 365]` (the longer pre-platinum runway from anchoring at ADT
initiation instead of ARPI initiation justifies a third landmark). Metrics: Harrell C-index, IPCW
mean AUC(t), integrated IPCW Brier — horizons come from `build_manifest.json` so all models share a
grid.

| Script | Model | CLI notes |
|---|---|---|
| `univariate_analysis.py` | Cox: univariate n_obs-adjusted associations | `--landmark-days`, `--endpoints`; IPIO also supports `--feature-subset {labs,genomics,all}` |
| `multivariate_analysis.py --model elastic-net` | Elastic-net Cox multivariable model (sksurv `CoxnetSurvivalAnalysis`, 5-fold CV, AGE unpenalized) | `--landmark-days`, `--endpoints`, `--n-folds`; IPIO also supports `--feature-subset {labs,genomics,all}` |
| `multivariate_analysis.py --model xgboost` | XGBoost `survival:cox`, 5-fold CV grid (`max_depth × eta × min_child_weight`) | `--landmark-days`, `--endpoints`, `--max-features`; IPIO also supports `--feature-subset {labs,genomics,all}` |

`cox_aggregated.py` is now a project adapter: endpoint constants, cohort-specific covariates/restrictions,
and per-landmark context. The univariate/elastic-net CLI orchestration lives in
`survival_common/cox_runners.py`; reusable Cox feature selection, CV, final-fit, and manifest helpers live in
`survival_common/cox_models.py`; low-level Cox fitting/evaluation primitives live in
`survival_common/cox_engine.py`; XGBoost orchestration lives in `survival_common/xgboost_runners.py`;
low-level XGBoost mechanics live in `survival_common/xgboost_engine.py`.

### 2.4 — Notebooks

COMPASS PROFILE has one run notebook, one figure notebook, and a focused
univariate-results review notebook:

- `COMPASS_run_locally.ipynb` — drives preprocessing and runs all four `COHORT_SPECS` arms:
  `without_other_primaries`, `with_other_primaries` (ARPI anchor, landmarks 0/90), and
  `adt_without_other_primaries`, `adt_with_other_primaries` (ADT anchor, landmarks 0/90/365). Each
  cohort gets independent prediction inputs and univariate, elastic-net, and XGBoost models at its
  own landmark list (`tasks_for_run(run)` builds the per-run task grid from `run["landmarks"]`), and
  the Stage 2 cell runs `longitudinal_data_processing.py` once per anchor
  (`--anchor-med-set {arpi,adt}`).
- `COMPASS_generate_figures.ipynb` — the sole COMPASS figure notebook, using the R kernel and
  `COMPASS_generate_figures_pipeline.R`. It renders all four cohorts' overview, LLM-label,
  univariate, multivariate, KM, androgen-distribution, and androgen-trajectory figures — ADT arms
  get a third volcano/importance/KM/androgen panel (landmark 365) that ARPI arms don't. Outputs are
  grouped as `FIG_ROOT/<figure>/<plot-stem>/<cohort>_<plot-stem>.png` with no `R/` language
  subdirectory. Figure 1A reads `mrn_lists/icd_prostate_mrn_flags.csv` and displays cumulative
  ICD-C61 cohort selection through the non-prostate-primary, PARPi, and ≥5-PSA-test criteria, plus
  either the ARPI/docetaxel or the ADT exposure criterion depending on the cohort's anchor. Axis and
  table labels throughout name the cohort's anchor ("ARPI/chemo initiation" vs. "ADT initiation") via
  `ANCHOR_LABEL`.
- `COMPASS_nominally_significant_univariate.ipynb` loads all shared-landmark univariate results
  for `with_other_primaries`, filters to nominal `p_value < 0.05`, displays every
  hit, and exports the filtered table beside the Cox outputs.

IPIO has a paired run/figure notebook as well:

- `IPIO_run_locally.ipynb` — builds standard lab landmark inputs at 0/90 plus the genomic landmark-0
  inputs, then runs univariate Cox, elastic-net Cox, and XGBoost for the lab arm, genomics-only arm,
  and genomics+labs arm separately.
- `IPIO_generate_figures.ipynb` — writes a labs-only paired volcano and a separate genomics-only
  volcano, plus the lab-arm discrimination, genomic-arm discrimination, and lab-arm importance
  figures.
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
# Stage 1 (cluster paths hard-coded, override via CLI flags if needed)
python COMPASS/data_preprocessing/compile_COMPASS_cohort_data.py

# Stage 2 — or just run COMPASS/survival_analysis/COMPASS_run_locally.ipynb top to bottom.
# Run once per anchor; --anchor-med-set switches the default survival-cohort/output/cache paths.
python COMPASS/data_preprocessing/longitudinal_data_processing.py --anchor-med-set arpi
python COMPASS/data_preprocessing/longitudinal_data_processing.py --anchor-med-set adt
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
