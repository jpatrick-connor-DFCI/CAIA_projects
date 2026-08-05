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
├── fast_io.py                           # lazy cohort-filtered scans (all raw reads funnel here)
├── oncdrs_sources.py                    # source-agnostic scan: raw OncDRS CSV *or* merged Parquet
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
    ├── COMPASS_run_locally.ipynb / COMPASS_generate_figures.ipynb # Python models / R figures
    └── COMPASS_run_locally_profile_data.ipynb / COMPASS_generate_figures_profile_data.ipynb
                                          # same pipeline, merged PROFILE_data_processing parquets
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

Every raw OncDRS read in Stage 1 and Stage 2 goes through
`data_preprocessing_common/oncdrs_sources.scan_source()`, which dispatches on file suffix, so
every path flag below accepts either a raw release CSV or a merged Parquet. See
[Two OncDRS source roots](#two-oncdrs-source-roots).

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
  Each of the four OncDRS tables has its own flag — `--icd-source`, `--medications-source`,
  `--labs-csv`, `--patient-status-source` — and every one accepts a raw CSV or a merged Parquet.
  The two `*-source` flags default to `<--oncdrs-path>/{MEDICATIONS,PT_INFO_STATUS_REGISTRATION}.csv`,
  so existing invocations are unchanged. `load_patient_status()` takes a **file** path, not a
  directory.
- **Outputs (under `NEPC_PROJ_PATH = DATA_PATH/CAIA/COMPASS/`, or `--out-dir`; all outputs including
  `prostate_icd_data.csv` honour it):** `prostate_arpi_survival_cohort_arpi.csv`
  and `prostate_adt_survival_cohort_adt.csv` (ARPI-treatment-anchor-restricted and
  ADT-treatment-anchor-restricted respectively), corresponding bare-MRN lists
  (`mrn_lists/arpi_mrns.csv`, `mrn_lists/adt_mrns.csv`), the ADT-entry-cohort
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

Consolidates/standardizes labs and vital signs (via `data_preprocessing_common/dfci_labs.py`), attaches the first
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
- **Below-detection imputation:** PSA and Testosterone only, and only where the `9999999` sentinel is
  paired with a `TEXT_RESULT` containing `<` (e.g. `"<0.1"`). Those become an exact `0.0` rather than
  being nulled with the other sentinels; every other sentinel/measurement combination is still nulled.
  The rule runs before sentinel nulling and before unit conversion, and the PSA/Testosterone
  physiologic ranges are inclusive at `0`, so the imputed zeros survive to the model inputs.
  Every run prints a `[below-detection]` table — rows and patients zeroed per measurement, plus two
  near-miss counts (`9999999` without a `<`, and `<` alongside a *different* sentinel). The same
  counts are written to `cohort_attrition.json` under `below_detection_imputation` and stored in the
  consolidated-cache manifest, so cache-hit runs replay the report instead of showing nothing.
  Read the near-miss columns, not just the zeroed count: if `TEXT_RESULT` is missing or unpopulated
  the rule silently cannot fire, and a zeroed count of `0` looks identical to a cohort that genuinely
  has no undetectable results.
- **Vital signs included:** COMPASS scans `HEALTH_HISTORY.csv` for `CODE_TYPE = "Vital Signs"`,
  combines those records with outpatient labs before standardization, and retains the canonical
  vital measurements in model inputs and figures. Cache version 5 forces older caches—including
  those built with the incorrect `999999` below-detection sentinel—to rebuild.
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

The **single source of truth for model inputs**. It builds an independent eligible risk set at each
landmark and derives a separate train/valid/test split within each risk set. Day-0 membership therefore
does not require surviving event-free to day 90, avoiding an immortal-time restriction on the earlier
cohort.

- **Key CLI:** `--data`, `--landmark-days 0 90 180` (default from
  `cox_aggregated.DEFAULT_LANDMARK_DAYS`),
  `--seed`, `--test-frac`, `--val-frac`, `--time-unit-days 7`, `--min-patient-coverage`,
  `--auc-quantiles`, `--id-col`, `--age-col`, `--anchor-col`,
  `--restrict-to-mrns`, `--require-first-treatment` / `--no-require-first-treatment`,
  `--min-psa-count`, `--exclude-parpi` / `--include-parpi`.
- **Default downstream cohort filters:** `FIRST_TREATMENT == 1`, ≥5 PSA rows, and PARPi exclusion
  (when `PARPI_EXPOSED` is present). These defaults preserve the original first-treatment cohort, but
  alternate anchors can relax them explicitly.
- **Outputs:** `aggregated_landmark{D}.csv`, `pre_treatment_lab_long_landmark{D}.csv`,
  `split_assignments_landmark{D}.csv`, the base-landmark compatibility copy
  `split_assignments.csv`, `landmark_mrn_availability.csv`, `canonical_labs_train_val.csv`,
  `landmark_attrition.json`, and `build_manifest.json`.
- **Optional GAM trajectory features** (produced by `gam_trajectory_features.R`, not by this
  script — see §2.3): `gam_trajectory_features_landmark{D}.csv` and
  `gam_fit_diagnostics_landmark{D}.csv`. When present in `prediction_inputs/`,
  `load_prebuilt_landmark` (`survival_common/cox_models.py`) left-joins the features onto
  `aggregated_landmark{D}.csv` before the split partition; when absent, behavior is unchanged
  (the merge is a no-op by construction, not by convention).
- `IPIO/data_preprocessing/build_genomic_inputs.py` builds the parallel
  `prediction_inputs/genomic/` landmark-0 arm anchored at IO start (`t_first_treatment`), restricts
  to patients with an actual somatic sample, attaches dynamic binary `<GENE>_<SV|SNV|AMP|DEL>`
  indicators, and **reuses** the main `split_assignments.csv` so test stays test. It writes both
  genomic provenance files and runner-compatible aliases (`aggregated_landmark0.csv`,
  `pre_treatment_lab_long_landmark0.csv`, `canonical_labs_train_val.csv`, `build_manifest.json`).

### 2.3 — Models

All read prebuilt inputs and the `split` column; none re-derive the split. COMPASS models use the
`platinum` endpoint only (time to first platinum). Both ARPI- and ADT-anchored arms use landmarks
`[0, 90, 180]`. Metrics: Harrell C-index, IPCW
mean AUC(t), integrated IPCW Brier — horizons come from `build_manifest.json` so all models share a
grid. The outer train+valid event-time quantiles define a bounded interval that is filled with up to
25 evenly spaced time points (interior quantiles are provenance only, not horizons). The builders clamp
that interval to stay strictly inside `--auc-max-time-units` and record the cap in the manifest, and the
runners evaluate with the manifest's cap by default, so no requested horizon is inestimable by
construction. Each evaluation
split records which requested points have cases, controls, and adequate follow-up; **AUC(t) and Brier
both pre-screen the timeline for their method-specific support requirements and use the same
administrative censoring cap**, since one horizon past a fold's follow-up can otherwise blank that
fold's entire metric. Mean AUC(t) and integrated Brier each require at least two valid points and 50%
timeline coverage; mean AUC(t) is always sksurv's censoring-weighted integral over the valid horizons,
never a substitute average. AUC first scores the eligible timeline in one batch and retries individual
horizons only if that batch fails. Metrics carry `train_val_*` / `test_*` prefixed AUC and Brier
requested/eligible/valid horizon counts, coverage, valid bounds, and status, in the same schema for Cox
and XGBoost.
Timeline schema version 3 is stored in the build manifest; model runners reject older manifests so
the input builder must be rerun after this change.

| Script | Model | CLI notes |
|---|---|---|
| `univariate_analysis.py` | Cox: univariate n_obs-adjusted associations | `--landmark-days`, `--endpoints`; IPIO also supports `--feature-subset {labs,genomics,all}` |
| `multivariate_analysis.py --model elastic-net` | Elastic-net Cox multivariable model (sksurv `CoxnetSurvivalAnalysis`, 5-fold CV, AGE unpenalized) | `--landmark-days`, `--endpoints`, `--n-folds`; IPIO also supports `--feature-subset {labs,genomics,all}` |
| `multivariate_analysis.py --model xgboost` | XGBoost `survival:cox`, 5-fold CV grid (`max_depth × eta × min_child_weight`) | `--landmark-days`, `--endpoints`, `--max-features`; IPIO also supports `--feature-subset {labs,genomics,all}` |
| `gam_trajectory_features.R` (COMPASS only) | Hierarchical GAM (`mgcv::bam`, `bs="fs"` factor-smooth per patient, shrinking sparse patients toward the population curve) per canonical lab, replacing the two-point `__delta` with `__gam_level` / `__gam_slope` / `__gam_curvature` / `__gam_auc` / `__gam_dev` evaluated at the landmark boundary | `--inputs-dir`, `--landmark-days`, `--k-pop`, `--k-pat`, `--trailing-window-days`, `--nthreads`, `--fit-split {all,train_val}` |
| `gam_cox_nonlinearity.R` (COMPASS only) | Penalized-spline Cox (`mgcv::gam(family=cox.ph())`) per selected feature: fits a smooth and a linear model of the same feature and reports `edf`/`p_lrt`/`q_lrt`/`delta_aic` — flags features whose hazard association is not actually linear | `--inputs-dir`, `--output-dir`, `--landmark-days` |

`cox_aggregated.py` is now a project adapter: endpoint constants, cohort-specific covariates/restrictions,
and per-landmark context. The univariate/elastic-net CLI orchestration lives in
`survival_common/cox_runners.py`; reusable Cox feature selection, CV, final-fit, and manifest helpers live in
`survival_common/cox_models.py`; low-level Cox fitting/evaluation primitives live in
`survival_common/cox_engine.py`; XGBoost orchestration lives in `survival_common/xgboost_runners.py`;
low-level XGBoost mechanics live in `survival_common/xgboost_engine.py`.

**GAM stages — run order and leakage stance.** Both R scripts are base R + `mgcv` + `data.table`
only (no `tidyverse`/`survminer`/`broom`, unlike `COMPASS_generate_figures_pipeline.R`) and live under
`COMPASS/survival_analysis/`, with self-checking synthetic smoke tests in
`COMPASS/survival_analysis/tests/`. `gam_trajectory_features.R` runs after
`build_prediction_inputs.py` and before `univariate_analysis.py`: it reads
`pre_treatment_lab_long_landmark{D}.csv` and `canonical_labs_train_val.csv` and writes
`gam_trajectory_features_landmark{D}.csv` (one row per `DFCI_MRN`) plus a
`gam_fit_diagnostics_landmark{D}.csv` sidecar recording which basis (`fs` vs. the random
intercept+slope fallback) was used per lab, EDF, fit seconds, and convergence. The per-patient
smooth is unsupervised — it never sees `t_platinum`/`PLATINUM` — and by default is fit on **all**
cohort patients, not just train_val; pass `--fit-split train_val` to refit population smooths on
train+valid only as a leakage sensitivity check. `univariate_analysis.py` then automatically tests
the new `__gam_*` columns alongside the existing stats (`load_prebuilt_landmark` left-joins
`gam_trajectory_features_landmark{D}.csv` onto `aggregated_landmark{D}.csv` when it exists; the merge
is a no-op if the file is absent). `gam_cox_nonlinearity.R` runs **after** `univariate_analysis.py`,
since it depends on the feature list in `cox_agg_feature_selection.csv`; it fits on the full merged
table (train+valid+test), mirroring the same row-fitting asymmetry as
`run_univariate_nobs_adjusted_associations`, and writes `gam_cox_nonlinearity_landmark{D}.csv`.

### 2.4 — Notebooks

COMPASS PROFILE has one run notebook, one figure notebook, a focused
univariate-results review notebook, and a parallel `*_profile_data` run/figure pair that executes
the identical pipeline against the merged PROFILE_data_processing parquets:

- `COMPASS_run_locally.ipynb` — drives preprocessing and runs both `COHORT_SPECS` arms: `arpi`
  and `adt` (both landmarks 0/90), over the common ADT-entry eligible cohort.
  Each arm gets
  independent prediction inputs and univariate, elastic-net, and XGBoost models at its own landmark
  list (`tasks_for_run(run)` builds the per-run task grid from `run["landmarks"]`), and the Stage 2
  cell runs `longitudinal_data_processing.py` once per anchor (`--anchor-med-set {arpi,adt}`).
- `COMPASS_generate_figures.ipynb` — the sole COMPASS figure notebook, using the R kernel and
  `COMPASS_generate_figures_pipeline.R`. It renders both arms' overview, LLM-label, univariate,
  multivariate, KM, and per-lab distribution/trajectory figures at landmarks 0 and 90. Figure 1A reads
  `mrn_lists/icd_prostate_mrn_flags.csv` and displays cumulative ICD-C61 cohort selection through
  ADT entry, the requested post-ADT cancer exclusion, PARPi, and ≥5-PSA-test criteria; the ARPI
  arm additionally displays its ARPI/docetaxel exposure criterion. Axis and table labels throughout
  name the arm's anchor ("ARPI/chemo initiation" vs. "ADT initiation") via `ANCHOR_LABEL`.
  - **LLM label strata:** alongside the existing `LLM_v3_labels.tsv`-driven Figure 2, the pipeline
    loads `LLM_NEPC_classifier_labels.tsv` from
    `/data/gusev/USERS/jpconnor/data/LLM_annotations/LLM_NEPC_labels/` (`DFCI_MRN`, `primary_label`
    [conventional/avpc/nepc/biomarker], `has_nepc`, `has_avpc`, and a reported-biomarker field) via
    `load_llm_strata()`, re-deriving
    `is_platinum` from `mrn_lists/platinum_MRN_list.csv`. If the file is absent, all LLM-strata plots
    are skipped with a `message()` rather than failing. **Primary-label normalization:**
    `primary_label = biomarker` is retained only when the reported biomarkers contain a
    token-matched BRCA1, BRCA2, PTEN, TP53, or RB1. Otherwise the label falls back in order to
    NEPC when `has_nepc = 1`, AVPC when `has_avpc = 1`, and conventional. Rows whose category is
    missing, blank, `NaN`, `NA`, `null`, or `none` are removed from all classifier-derived LLM
    plots. This file drives:
    - **Figure 2 v2** (`ADT/figure2v2_llm/`, ADT cohort only) — a parallel confusion matrix / metric bar / subtype
      landscape / platinum-enrichment panel set (`figure2v2_confusion_matrix`,
      `figure2v2_metric_bar`, `figure2v2_confusion_has_nepc`, `figure2v2_confusion_has_avpc`,
      `figure2v2_subtype_landscape`, `figure2v2_enrichment`, `figure2v2_llm_subtype_platinum`),
      reusing the original Figure 2's render helpers. Unlike the original Figure 2, it has **no
      hardcoded `stopifnot` count assertions** — captions are computed dynamically after classifier
      labels are intersected with the complete ADT time-0 MRN set. Both Figure 2 and Figure 2v2 are
      ADT/day-0-only; the ARPI figure pass skips them, and captions report labeled/evaluable coverage
      against the total ADT day-0 cohort.
    - **Figure 2 v3** (`ADT/figure2v3_llm/`) — the same label source and panel set as v2
      (`figure2v3_confusion_matrix`, `figure2v3_metric_bar`, `figure2v3_confusion_has_nepc`,
      `figure2v3_confusion_has_avpc`, `figure2v3_subtype_landscape`, `figure2v3_enrichment`,
      `figure2v3_llm_subtype_platinum`) over a **wider patient universe**: every MRN with
      `ADT_EXPOSED = 1` in `mrn_lists/icd_prostate_mrn_flags.csv` — the "ADT entry requirement"
      step of the Figure 1 CONSORT — rather than only the landmark-0 prediction cohort
      (`eligible_landmark_0` in `landmark_mrn_availability.csv`) that v2 uses. So v3 includes
      patients later dropped by the post-ADT exclusion-cancer, PARPi, and ≥5-PSA-test criteria.
      Like v2 it carries no `stopifnot` count assertions; it is emitted from the ADT pass only.
    - **All-lab longitudinal dynamics** — Figure 7's group-mean-CI trajectory machinery
      (`bin_group_ci()`/`plot_group_ci_panel()`) generalized from PSA/Testosterone-only to every
      canonical lab in `CATEGORY_MAP` (CBC/CMP/LFT/Vitals/Androgen/Other), stratified
      by platinum status plus `primary_label`/`has_nepc`/`has_avpc`. Stems:
      `longitudinal_<stratum>_<lab>`.
    - **All-lab KM-by-quartile** — Figure 5's Q1-vs-Q4 platinum-free KM curves, generalized the same
      way via `resolve_mean_col()`. Stems: `km_quartile_<lab>_landmark<D>`.
    - **All-lab distribution panels** — Figure 6's log/raw platinum-split distribution plots,
      generalized the same way. Stems: `dist_by_platinum_{log,raw}_<lab>_landmark<D>`.
      `PLOT_NON_ANDROGEN_DISTRIBUTIONS` controls whether distributions extend beyond PSA and
      Testosterone.
    - **Lab-panel toggles** — the figure notebook and pipeline defaults set both
      `PLOT_NON_ANDROGEN_DISTRIBUTIONS <- FALSE` and
      `PLOT_NON_ANDROGEN_LAB_FIGURES <- FALSE`. Lab-specific distribution, KM, and longitudinal
      panels are therefore limited to PSA and Testosterone. Non-androgen labs, including vitals,
      remain available to models and aggregate feature plots.
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
  for both `arpi` and `adt`, filters each to nominal `p_value < 0.05`, displays every hit, and exports
  a separate `cox/nominally_significant_univariate_results.csv` beneath each arm's run directory.
- `COMPASS_run_locally_profile_data.ipynb` / `COMPASS_generate_figures_profile_data.ipynb` — the
  same two notebooks pointed at the merged PROFILE_data_processing parquets and a separate output
  root. Every shared run-helper cell is byte-identical to the baseline notebooks and
  `COMPASS_generate_figures_pipeline.R` is unmodified, so only the data source differs. See
  [Two OncDRS source roots](#two-oncdrs-source-roots).

IPIO has a paired run/figure notebook as well:

- `IPIO_run_locally.ipynb` — builds standard lab landmark inputs at 0/90 plus the genomic landmark-0
  inputs, then runs univariate Cox, elastic-net Cox, and XGBoost for the lab arm, genomics-only arm,
  and genomics+labs arm separately.
- `IPIO_generate_figures.ipynb` — writes a labs-only paired volcano and a separate genomics-only
  volcano, plus the lab-arm discrimination, genomic-arm discrimination, and lab-arm importance
  figures.

---

## Two OncDRS source roots

COMPASS can be run against either of two raw-data roots. Both produce the same file layout; they
differ only in which OncDRS extract they read and where they write.

| | Baseline | PROFILE_data_processing |
|---|---|---|
| Source | `/data/gusev/PROFILE/CLINICAL/OncDRS/ALL_2025_03/*.csv` (one release) | `/data/gusev/USERS/jpconnor/data/PROFILE_DATA/*.parquet` (7 releases merged + deduplicated) |
| Run notebook | `COMPASS_run_locally.ipynb` | `COMPASS_run_locally_profile_data.ipynb` |
| Figure notebook | `COMPASS_generate_figures.ipynb` | `COMPASS_generate_figures_profile_data.ipynb` |
| Data root | `data/CAIA/COMPASS/` | `data/CAIA/COMPASS_PROFILE_DATA/` |
| Figure root | `figures/CAIA/COMPASS/` | `figures/CAIA/COMPASS_PROFILE_DATA/` |

The roots are disjoint, so the two runs can be compared side by side and neither clobbers the
other. The parquets are built by the sibling repo `PROFILE_data_processing`
(`compile_OncDRS_data.ipynb`), which folds releases `ALL_2021_11` … `ALL_2026_03` into one
Parquet per table.

**`oncdrs_sources.scan_source()`.** No entry point knows which root it is reading. Every raw scan
goes through `data_preprocessing_common/oncdrs_sources.py`:

```python
scan_source(path)  # .parquet/.pq -> pl.scan_parquet(path).select(pl.all().cast(pl.Utf8))
                   # anything else -> pl.scan_csv(path, infer_schema_length=0)
```

The `cast(pl.Utf8)` is not cosmetic — it reproduces `infer_schema_length=0`, which is what makes
every column a string. The whole pipeline relies on that (`.str.to_datetime()`,
`.str.to_uppercase()`, `fast_io.recover_numeric()`, then explicit casts back to `Int64`). Parquet
columns are typed, so reading one without the cast breaks every `.str.*` call. The only genuinely
numeric column on the COMPASS path is `LABS.NUMERIC_RESULT`, and `recover_numeric()` parses it
back to `Float64` losslessly. `fast_io.scan_filter()` calls `scan_source`, so CSV callers — all of
IPIO, and the baseline COMPASS notebooks — are byte-for-byte unaffected.

**Upstream column-name requirement.** `compile_OncDRS_data.ipynb` inserts any requested-but-absent
column as null with only a printed warning, so a rename between releases yields an **all-null
column rather than an error**. The COMPASS pipeline requires these canonical spellings in the
merged parquets:

| Table | Required column | Consequence if null-filled |
|---|---|---|
| `MEDICATIONS` | `NCI_PREFERRED_MED_NM` (not `..._MED_NAME`) | every anchor / platinum / PARPi computation silently returns empty |
| `PT_INFO_STATUS_REGISTRATION` | `DERIVED_LAST_ALIVE_DATE` (not `DERIVED_LAST_CONTACT_DT`) | `LAST_CONTACT_DATE` null → censoring and `TT_DEATH` broken |
| `EHR_DIAGNOSES` | `DIAGNOSIS_ICD10_NM` / `_NM2` / `_NM3` | only affects `compile_MRNs_for_manual_review.py` |

The upstream `COLUMN_MAP` requests both the old and new spellings and coalesces them via
`ALIAS_MAP`, since release schemas differ across the seven pulls.

The first code cell of `COMPASS_run_locally_profile_data.ipynb` is a **schema audit** that guards
this. It scans all five tables and checks columns in three tiers — REQUIRED (absent *or* all-null
raises), EXPECTED (absent raises, all-null warns — for legitimately sparse columns like
`HYBRID_DEATH_DT` and `LABS.TEXT_RESULT`), OPTIONAL (warns either way) — and raises a single
`RuntimeError` listing every problem. Run it before anything else; an all-null
`NCI_PREFERRED_MED_NM` means going back to the upstream compile, not debugging COMPASS.

The last cell of the same notebook prints a **cohort comparison** table: Stage 1 cohort/platinum
counts and Stage 2 per-landmark attrition for both roots, side by side with deltas. Expect the
merged run to be larger (more releases), but note that upstream full-row deduplication of `LABS`
can push a few patients *below* the ≥5-broad-PSA gate, so attrition can move in both directions.
Both arms write `cohort_attrition.json` to the same path, so that readout reflects whichever
anchor ran last — it is labelled as such in the cell.

---

## Conventions & invariants (preserve these when editing)

1. **Each landmark has its own risk set and split.** `build_prediction_inputs.py` writes the applicable
   split directly into each `aggregated_landmark{D}.csv` and also writes
   `split_assignments_landmark{D}.csv`. Models use the aggregated table's split and never re-split.
   `split_assignments.csv` remains a base-landmark compatibility copy for `build_genomic_inputs.py`.
2. **Fit on the training block; never touch test for fitting.** Imputers, `StandardScaler`,
   canonical-lab selection, and Breslow baselines are all fit on train+valid (or fold-train inside CV) and
   applied to eval. Per-fold canonical labs are recomputed inside CV. The leakage guards
   `assert_no_test_leakage` / `assert_disjoint_folds` live in `survival_common/helper.py`.
3. **Canonical labs and horizon grids are training-block artifacts.** The main canonical lab set and
   AUC/Brier horizons are derived on **train+valid**; CV recomputes canonical labs on each fold-train.
   Evaluation data may only mask requested horizons that are not estimable—it never supplies
   replacement times. Do not derive the requested timeline from held-out test patients.
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
- **Raw OncDRS roots** — `ALL_2025_03` release CSVs at
  `/data/gusev/PROFILE/CLINICAL/OncDRS/ALL_2025_03/`, or merged parquets at
  `/data/gusev/USERS/jpconnor/data/PROFILE_DATA/`. Both defaults live in
  `data_preprocessing_common/oncdrs_sources.py` (`DEFAULT_ONCDRS_RELEASE`,
  `DEFAULT_PROFILE_DATA_ROOT`) alongside the `TABLE_FILES` basename map. The parquet run writes to
  the `COMPASS_PROFILE_DATA` data/figure roots — see
  [Two OncDRS source roots](#two-oncdrs-source-roots).
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
python COMPASS/data_preprocessing/build_prediction_inputs.py --landmark-days 0 90 180 --time-unit-days 7
python COMPASS/survival_analysis/univariate_analysis.py --inputs-dir <...>/prediction_inputs --landmark-days 0
python COMPASS/survival_analysis/multivariate_analysis.py --model elastic-net --inputs-dir <...>/prediction_inputs --landmark-days 0
python COMPASS/survival_analysis/multivariate_analysis.py --model xgboost --inputs-dir <...>/prediction_inputs --landmark-days 0
```

To run against the merged PROFILE_data_processing parquets instead, use
`COMPASS_run_locally_profile_data.ipynb` → `COMPASS_generate_figures_profile_data.ipynb` (both
top to bottom). They pass the `PROFILE_DATA/*.parquet` paths and the `COMPASS_PROFILE_DATA` roots to the
same scripts; nothing under `data/CAIA/COMPASS/` or `figures/CAIA/COMPASS/` is written. The figure
notebook symlinks `LLM_NEPC_labels/` from the baseline root, since those hand-curated annotations
are an input to `generate_figures()` rather than something the pipeline produces.

## Dependencies

No packaged environment is checked in. Assumed: `pandas`, `numpy`, `scipy`, `tqdm`, `scikit-learn`,
`scikit-survival` (`sksurv`), `xgboost`, `lifelines`, `matplotlib`. Python **3.10+** is recommended
for the modern type-hint syntax used by the shared modules.

---

## Known issues / footguns

These are real, verified items found in code review. Fix opportunistically; at minimum, don't be
surprised by them.

### Medium impact

- **`auc_max_time_units = 260` is the default** (`DEFAULT_AUC_MAX_TIME_UNITS` in `survival_common/helper.py`)
  and admin-censors AUC/Brier unless `--auc-max-time-units` is overridden. The builders clamp the horizon
  grid to it and runners read it back from the manifest; overriding it at run time only (not at build
  time) makes the two diverge, which the runner warns about.
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
