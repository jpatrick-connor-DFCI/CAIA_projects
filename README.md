# COMPASS / IPIO Survival Analysis

Research workflow for assembling a prostate-cancer cohort from local DFCI / Profile / OncDRS
exports and running landmark survival analysis (Cox / XGBoost) on the resulting cohort.

The preprocessing and modeling code is pandas-based; publication figures are generated only in R.
Entry points are command-line scripts orchestrated from notebooks.
COMPASS uses one `COMPASS_run_locally.ipynb` notebook for two treatment-anchored cohort arms,
both drawn from an ADT-entry ICD-C61 cohort: `arpi` (anchored at first ARPI/chemo exposure) and
`adt` (anchored at first ADT exposure, years earlier in the treatment sequence). Patients with
bladder (C67), lung (C34), head-and-neck (C00-C14/C30-C32), or testicular (C62) cancer diagnosed
strictly after ADT start are excluded. The paired, R-only `COMPASS_generate_figures.ipynb` emits
manuscript figures for both.

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
 prostate_arpi_survival_cohort_arpi.csv,
 prostate_adt_survival_cohort_adt.csv
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
        ▼  COMPASS_generate_figures.ipynb (R; 2 cohort arms: arpi, adt)
 figures/
```

---

## Stage 1 — Compile prostate cohort source data

`data_preprocessing/compile_COMPASS_cohort_data.py` (module-level script, argparse CLI for path
overrides). Derives the ICD-C61 universe, requires a dated ADT exposure as the primary entry
criterion, and excludes bladder (C67), lung (C34), head-and-neck (C00-C14/C30-C32), and
testicular (C62) cancer diagnosed strictly after first ADT. Other non-prostate primaries remain
descriptive only. The same eligible base cohort feeds the ARPI/chemo-anchored and ADT-anchored
survival data (age, treatment anchor, death, time-to-platinum), with `anchor_df`/`adt_anchor_df`
computed from the same medication scan.

- **Inputs (hard-coded under `DATA_PATH = /data/gusev/USERS/jpconnor/data/`, plus the raw OncDRS pull
  at `ONCDRS_PATH`):** `EHR_DIAGNOSIS.csv`, `HEALTH_HISTORY.csv`, `MEDICATIONS.csv`,
  `OUTPT_LAB_RESULTS_LABS.csv`, `complete_somatic_data_df.csv.gz`, `PT_INFO_STATUS_REGISTRATION.csv`.
- **Outputs (under `NEPC_PROJ_PATH = DATA_PATH/CAIA/COMPASS/`):** `prostate_arpi_survival_cohort_arpi.csv`
  and `prostate_adt_survival_cohort_adt.csv` (ARPI-treatment-anchor-restricted and
  ADT-treatment-anchor-restricted respectively), corresponding bare-MRN lists
  (`mrn_lists/arpi_mrns.csv`, `mrn_lists/adt_mrns.csv`), the regenerated all-ICD-C61
  platinum-recipient list (`mrn_lists/platinum_MRN_list.csv`), `prostate_icd_data.csv`, and
  `mrn_lists/icd_prostate_mrn_flags.csv`. The latter contains every ICD-C61 MRN plus binary
  indicators for a non-prostate primary (`HAS_NON_PROSTATE_PRIMARY`, descriptive only), a
  requested post-ADT cancer exclusion (`HAS_POST_ADT_EXCLUSION_CANCER`), PARPi exposure,
  ARPI/docetaxel exposure, ADT exposure (`ADT_EXPOSED`), and at least five broad PSA tests.
- **Cohort definition:** ICD-C61 plus dated ADT exposure, excluding the four specified cancer
  groups only when diagnosis is strictly after ADT start. The `adt` arm uses this cohort directly;
  the `arpi` arm additionally requires an ARPI/chemo anchor.
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
- **Fast cohort scope:** by default, raw scans start from the anchored `arpi` cohort (or its ADT
  counterpart, under `--anchor-med-set adt`). PARPi-exposed patients and patients with fewer than
  five broad PSA records are removed before expensive lab standardization. Use
  `--prefilter-include-parpi`, `--prefilter-min-psa-count 0`, and/or a broader
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

- `COMPASS_run_locally.ipynb` — drives preprocessing and runs both `COHORT_SPECS` arms: `arpi`
  (landmarks 0/90) and `adt` (landmarks 0/90/365), both over the common ADT-entry eligible cohort.
  Each arm gets
  independent prediction inputs and univariate, elastic-net, and XGBoost models at its own landmark
  list (`tasks_for_run(run)` builds the per-run task grid from `run["landmarks"]`), and the Stage 2
  cell runs `longitudinal_data_processing.py` once per anchor (`--anchor-med-set {arpi,adt}`).
- `COMPASS_generate_figures.ipynb` — the sole COMPASS figure notebook, using the R kernel and
  `COMPASS_generate_figures_pipeline.R`. It renders both arms' overview, LLM-label, univariate,
  multivariate, KM, and per-lab distribution/trajectory figures — the ADT arm gets a third
  volcano/importance/KM/lab panel (landmark 365) that the ARPI arm doesn't. Figure 1A reads
  `mrn_lists/icd_prostate_mrn_flags.csv` and displays cumulative ICD-C61 cohort selection through
  ADT entry, the requested post-ADT cancer exclusion, PARPi, and ≥5-PSA-test criteria; the ARPI
  arm additionally displays its ARPI/docetaxel exposure criterion. Axis and table labels throughout
  name the arm's anchor ("ARPI/chemo initiation" vs. "ADT initiation") via `ANCHOR_LABEL`.
  - **LLM label strata:** alongside the existing `LLM_v3_labels.tsv`-driven Figure 2, the pipeline
    loads `LLM_NEPC_classifier_labels.tsv` from
    `/data/gusev/USERS/jpconnor/data/LLM_annotations/LLM_NEPC_labels/` (`DFCI_MRN`, `primary_label`
    [conventional/avpc/nepc/biomarker], `has_nepc`, `has_avpc`) via `load_llm_strata()`, re-deriving
    `is_platinum` from `mrn_lists/platinum_MRN_list.csv`. If the file is absent, all LLM-strata plots
    are skipped with a `message()` rather than failing. This file drives:
    - **Figure 2 v2** (`figure2v2_llm/`) — a parallel confusion matrix / metric bar / subtype
      landscape / platinum-enrichment panel set (`figure2v2_confusion_matrix`,
      `figure2v2_metric_bar`, `figure2v2_confusion_has_nepc`, `figure2v2_confusion_has_avpc`,
      `figure2v2_subtype_landscape`, `figure2v2_enrichment`, `figure2v2_llm_subtype_platinum`),
      reusing the original Figure 2's render helpers. Unlike the original Figure 2, it has **no
      hardcoded `stopifnot` count assertions** — captions are computed dynamically from the loaded
      labels, since the new file's counts differ from `LLM_v3_labels.tsv`.
    - **All-lab longitudinal dynamics** — Figure 7's group-mean-CI trajectory machinery
      (`bin_group_ci()`/`plot_group_ci_panel()`) generalized from PSA/Testosterone-only to every
      canonical lab in `CATEGORY_MAP` (~40 labs across CBC/CMP/LFT/Vitals/Androgen/Other), stratified
      by platinum status plus `primary_label`/`has_nepc`/`has_avpc`. Stems:
      `longitudinal_<stratum>_<lab>`.
    - **All-lab KM-by-quartile** — Figure 5's Q1-vs-Q4 platinum-free KM curves, generalized the same
      way via `resolve_mean_col()`. Stems: `km_quartile_<lab>_landmark<D>`.
    - **All-lab distribution panels** — Figure 6's log/raw platinum-split distribution plots,
      generalized the same way. Stems: `dist_by_platinum_{log,raw}_<lab>_landmark<D>`.
    - **LLM-stratified KM sanity check** — new time-to-platinum KM curves stratified by
      `primary_label`/`has_nepc`/`has_avpc`, using the same `platinum_km_inputs()` +
      `overlay_km()` path as the quartile curves. Stems: `km_llm_<scheme>_landmark<D>`.
  - **Output layout:** each arm has its own `FIG_ROOT/ARPI/` or `FIG_ROOT/ADT/` subtree. Non-lab
    figures use `<arm>/<figure>/<plot-stem>/<cohort>_<plot-stem>.png`; per-lab panels
    (longitudinal, km_quartile, distribution) use
    `<arm>/labs/<lab_category>/<lab_name>/<panel_type>/<cohort>_<stem>.png`. A lab-aware branch in
    `figure_group()` uses `assign_category()` for CBC/CMP/LFT/Vitals/Androgen axis/Other. Any
    plot stem `figure_group()` can't route still raises
    `stop("Unmapped figure output stem")`.
- `COMPASS_nominally_significant_univariate.ipynb` loads all shared-landmark univariate results
  for the `arpi` arm, filters to nominal `p_value < 0.05`, displays every hit, and exports the
  filtered table beside the Cox outputs.

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
