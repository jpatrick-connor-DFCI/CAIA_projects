# COMPASS / IPIO Survival Analysis

Research workflow for assembling a prostate-cancer cohort from local DFCI / Profile / OncDRS
exports and running landmark survival analysis (Cox / XGBoost) on the resulting cohort.

The preprocessing and modeling code is pandas-based; publication figures are generated only in R.
Entry points are command-line scripts orchestrated from notebooks.
COMPASS uses numbered stage notebooks (`01_preprocessing.ipynb`, `02_univariate.ipynb`,
`03_multivariate.ipynb`) for two treatment-anchored cohort arms, both drawn from an ADT-entry
ICD-C61 cohort: `arpi` (anchored at first ARPI/chemo exposure) and `adt` (anchored at first ADT
exposure, years earlier in the treatment sequence). Entry requires a dated prostate diagnosis, male
sex, at least five PSA measurements, and ADT on/after diagnosis; PARPi exposure, platinum before
diagnosis, and bladder (C67), lung (C34), head-and-neck (C00-C14/C30-C32), or testicular (C62)
cancer diagnosed strictly after first ADT are excluded. The paired, R-only `05_figures.Rmd` emits
manuscript figures for both.

**Arm and endpoint are two orthogonal axes.** The *arm* sets time 0 (the index-date anchor); the
*endpoint* sets the event being modelled. They are selected independently, and the pair is what
defines a run — see [Arms and endpoints](#arms-and-endpoints).

| Axis | Registry | Declared in | Values |
| --- | --- | --- | --- |
| **Arm** (index date / time 0) | `_ARM_SPECS` | `COMPASS/survival_analysis/compass_pipeline.py` | `arpi`, `adt` |
| **Endpoint** (the event) | `ENDPOINTS` | `COMPASS/survival_analysis/cox_aggregated.py` | `platinum`, `nepc` |

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
├── cohort.py                            # landmark/outcome/feature-matrix/person-period builders
├── cox_engine.py                        # shared Cox / Coxnet / IPCW AUC(t) primitives
├── xgboost_engine.py                    # shared XGBoost survival:cox primitives
├── longitudinal_targets.py              # torch-free cause/competing target + horizon semantics
├── deephit_engine.py                    # torch-gated Dynamic-DeepHit model + training engine
├── longitudinal_runners.py              # Dynamic-DeepHit CLI orchestration
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
│   ├── build_genomic_inputs.py           # legacy sample-anchored genomic + lab arm
│   └── build_somatic_gleason_inputs.py   # sequencing, Gleason, and PRS inputs
│
└── survival_analysis/
    ├── compass_pipeline.py               # shared setup/helper logic behind the 4 Python notebooks
    ├── cox_aggregated.py                 # PROFILE adapter/config for shared survival code
    ├── univariate_analysis.py            # ENTRY: univariate Cox associations
    ├── multivariate_analysis.py          # ENTRY: elastic-net Cox or XGBoost survival:cox
    ├── COMPASS_generate_figures_pipeline.R # sole figure-generation implementation
    ├── 01_preprocessing.ipynb            # schema audit, cohort compile, preprocessing, diagnostics
    ├── 02_univariate.ipynb               # univariate arms + nominal-significance filter
    ├── 03_multivariate.ipynb             # elastic-net + XGBoost + summary tables
    ├── 03b_multivariate_longitudinal.ipynb # Dynamic-DeepHit (torch, optional; SurvLatent ODE off by default)
    ├── 05_figures.Rmd                    # R figures from merged profile_data outputs
    ├── 06_abstract_numbers.ipynb         # read-only: abstract/manuscript counts from built artifacts
    ├── 07_endpoint_comparison.ipynb      # read-only: platinum vs nepc endpoint comparison
    ├── multivariate_longitudinal/
    │   ├── dynamic_deephit.py            # ENTRY: thin CLI over survival_common/deephit_engine.py
    │   ├── survlatent_ode.py             # ENTRY: adapter around the bundled editable checkout
    │   └── README.md                     # bundled-checkout/conda-env prerequisites, chdir caveat
    └── GAM/
        ├── gam_trajectory_features.R     # hierarchical longitudinal GAM features
        ├── gam_cox_nonlinearity.R        # smooth-vs-linear Cox GAM tests
        └── 04_gam.ipynb                  # R-kernel GAM runner on merged profile_data
```

All eight numbered/lettered notebooks use the merged PROFILE_data_processing parquets and the
standard `COMPASS` data/figure roots. The Python notebooks share `compass_pipeline.py`; no
single-release baseline run is part of this workflow. `03b` is optional and torch-gated (see
invariant #7 below) — the other notebooks never require torch. `06` and `07` are read-only
reporting notebooks: they read already-generated artifacts and refit nothing.

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
        ▼  data_preprocessing/build_prediction_inputs.py     [forks per endpoint]
 prediction_inputs_<arm>/        (aggregated + pre-treatment long labs + split + horizons)
 prediction_inputs_<arm>_nepc/   (same, built with --require-nepc: incident-NEPC risk set)
        │
        ├──► build_somatic_gleason_inputs.py
        │    prediction_inputs_<arm>/somatic_gleason/
        │
        ▼  survival_analysis/univariate_analysis.py
           survival_analysis/multivariate_analysis.py
 local_runs_<arm>/ , local_runs_<arm>_nepc/
        │
        ├──► 05_figures.Rmd (R; 2 cohort arms: arpi, adt)
        │    figures/
        │
        ▼  07_endpoint_comparison.ipynb (read-only; platinum vs nepc)
```

Everything above `build_prediction_inputs.py` is endpoint-independent and shared: one survival
cohort carries every endpoint's columns. Only Stage 3 onward is duplicated per endpoint.

Every raw OncDRS read in Stage 1 and Stage 2 goes through
`data_preprocessing_common/oncdrs_sources.scan_source()`. The COMPASS notebooks pass the merged
`PROFILE_DATA/*.parquet` sources explicitly.

---

## Stage 1 — Compile prostate cohort source data

`data_preprocessing/compile_COMPASS_cohort_data.py` (module-level script, argparse CLI for path
overrides). It applies the same seven criteria as `caia-project-compass`: dated ICD-C61 prostate
diagnosis; male sex; at least five broad-PSA measurements at any time; ADT on/after diagnosis; no
PARPi exposure; no carboplatin/cisplatin exposure before diagnosis; and no bladder (C67), lung
(C34), head-and-neck (C00-C14/C30-C32), or testicular (C62) cancer diagnosed strictly after first
ADT anywhere in the record. Other non-prostate primaries remain descriptive only. The same
eligible base cohort feeds the ARPI/chemo-anchored and ADT-anchored survival data.

- **Inputs (hard-coded under `DATA_PATH = /data/gusev/USERS/jpconnor/data/`, plus the raw OncDRS pull
  at `ONCDRS_PATH`):** `EHR_DIAGNOSIS.csv`, `HEALTH_HISTORY.csv`, `MEDICATIONS.csv`,
  `OUTPT_LAB_RESULTS_LABS.csv`, `complete_somatic_data_df.csv.gz`, `PT_INFO_STATUS_REGISTRATION.csv`.
  Each of the four OncDRS tables has its own flag — `--icd-source`, `--medications-source`,
  `--labs-csv`, `--patient-status-source` — and every one accepts a raw CSV or a merged Parquet.
  The two `*-source` flags default to `<--oncdrs-path>/{MEDICATIONS,PT_INFO_STATUS_REGISTRATION}.csv`,
  so existing invocations are unchanged. `load_patient_status()` takes a **file** path, not a
  directory.
  `--nepc-labels` points at the strict LLM NEPC diagnosis labels
  (`LLM_annotations/LLM_nepc_diagnosis/nepc_dx_labels.parquet`, one row per patient, written by the
  sibling `LLM_clinical_annotations` repo). A **missing file is non-fatal**: the stage warns and
  emits no NEPC columns, leaving the platinum pipeline unaffected. Note `diagnosis_date` arrives as
  an ISO **string**, not a date type, with partial dates already normalized (year → Jan 1,
  year-month → the 1st).
- **Outputs (under `NEPC_PROJ_PATH = DATA_PATH/CAIA/COMPASS/`, or `--out-dir`; all outputs including
  `prostate_icd_data.csv` honour it):** `prostate_arpi_survival_cohort_arpi.csv`
  and `prostate_adt_survival_cohort_adt.csv` (ARPI-treatment-anchor-restricted and
  ADT-treatment-anchor-restricted respectively), corresponding bare-MRN lists
  (`mrn_lists/arpi_mrns.csv`, `mrn_lists/adt_mrns.csv`), the ADT-entry-cohort
  platinum-recipient list (`mrn_lists/platinum_MRN_list.csv`), `prostate_icd_data.csv`, and
  `mrn_lists/icd_prostate_mrn_flags.csv`. The latter contains every ICD-C61 MRN plus binary
  indicators for dated prostate diagnosis, male sex, a non-prostate primary
  (`HAS_NON_PROSTATE_PRIMARY`, descriptive only), the requested post-ADT cancer exclusion,
  PARPi exposure, pre-diagnosis platinum, ARPI/docetaxel exposure, ADT on/after diagnosis,
  at least five broad PSA tests, and final eligibility. When the NEPC labels are available, the two
  survival cohorts additionally carry `NEPC`, `NEPC_DATE`, `TT_NEPC`, and the three provenance
  columns (`NEPC_DATE_SOURCE`, `NEPC_DATE_PRECISION`, `NEPC_LABEL_SOURCE`); one cohort file serves
  **both** endpoints. The printed summary reports NEPC positives, the provenance breakdowns, and how
  many diagnoses are prevalent at the anchor — read it before committing to NEPC modelling.
- **Cohort definition:** the seven criteria above are enforced in Stage 1. The `adt` arm uses this
  cohort directly; the `arpi` arm additionally requires a post-diagnosis ARPI/chemo anchor.
- **Anchor sets:** `ARPI_ANCHOR_MEDS` (7 drugs: ARPIs, taxanes, radium-223 — unchanged,
  `TREATMENT_ANCHOR_MEDS` alias retained) and `ADT_ANCHOR_MEDS` (the eight CAIA-COMPASS ADT
  ingredients; PROFILE has separate preferred names for triptorelin and triptorelin pamoate).
  `compute_treatment_anchor(meds, meds_set=...)` takes either set;
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
  `t_platinum`, `t_nepc`, `t_last_contact`, `t_death`. `t_death` is a real death-date-derived duration
  when the survival cohort's `death_date` is available (falls back to the last-contact proxy for dead
  patients with no recorded date); it is not itself a registered model endpoint. `t_nepc` is built
  exactly like `t_platinum` (days from anchor to `NEPC_DATE` for positives, else `t_last_contact`)
  and is emitted only when the upstream NEPC labels were available at Stage 1.

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
  `--min-psa-count`, `--exclude-parpi` / `--include-parpi`, `--require-nepc`.
- **`--require-nepc`** restricts the risk set to patients with an incident NEPC diagnosis
  (`t_nepc` present and `> 0` after landmark rebasing). Off by default, and recorded in
  `build_manifest.json`. It changes cohort membership, so a build made with it must go to its own
  `prediction_inputs_<arm>_nepc/` directory — `compass_pipeline.make_runs(output_suffix=...)` does
  that automatically. The horizon-grid loop skips any registered endpoint whose duration/event
  columns are absent from the build, so a platinum-only cohort is unaffected by the `nepc`
  registration.
- **Default downstream cohort filters:** ≥5 PSA rows and PARPi exclusion are repeated as consistency
  guards after the Stage-1 restriction. They should produce no further criterion-related attrition
  for newly rebuilt cohorts; alternate inputs can relax them explicitly.
- **Outputs:** `aggregated_landmark{D}.csv`, `pre_treatment_lab_long_landmark{D}.csv`,
  `split_assignments_landmark{D}.csv`, the base-landmark compatibility copy
  `split_assignments.csv`, `landmark_mrn_availability.csv`, `canonical_labs_train_val.csv`,
  `landmark_attrition.json`, and `build_manifest.json`.
- **Optional GAM trajectory features** (produced by `gam_trajectory_features.R`, not by this
  script — see §2.3): `gam_trajectory_features_landmark{D}.csv` and
  `gam_fit_diagnostics_landmark{D}.csv`, plus
  `gam_trajectory_curves_landmark{D}.csv` containing the per-patient fitted grid used for figures.
  When present in `prediction_inputs/`,
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

All read prebuilt inputs and the `split` column; none re-derive the split. COMPASS models default to
the `platinum` endpoint (time to first platinum) and additionally support `nepc` (time to
LLM-adjudicated NEPC diagnosis) — see [Arms and endpoints](#arms-and-endpoints). `--endpoints` is
built from `cox.ENDPOINTS`, so every runner gets the choice for free. Both ARPI- and ADT-anchored
arms use landmarks `[0, 90, 180]`. Metrics: Harrell C-index, IPCW
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
| `univariate_analysis.py` | Cox: univariate n_obs-adjusted lab associations, or separate sequencing, Gleason, and PRS COMPASS associations | `--landmark-days`, `--endpoints`; sequencing/Gleason use the observation nearest ADT as origin, while PRS uses ADT start; IPIO supports `--feature-subset {labs,genomics,all}` |
| `multivariate_analysis.py --model elastic-net` | Elastic-net Cox multivariable model (sksurv `CoxnetSurvivalAnalysis`, 5-fold CV, AGE unpenalized) | `--landmark-days`, `--endpoints`, `--n-folds`; IPIO also supports `--feature-subset {labs,genomics,all}` |
| `multivariate_analysis.py --model xgboost` | XGBoost `survival:cox`, 5-fold CV grid (`max_depth × eta × min_child_weight`) | `--landmark-days`, `--endpoints`, `--max-features`; IPIO also supports `--feature-subset {labs,genomics,all}` |
| `gam_trajectory_features.R` (COMPASS only) | Hierarchical GAM (`mgcv::bam`, `bs="fs"` factor-smooth per patient, shrinking sparse patients toward the population curve) per canonical lab, replacing the two-point `__delta` with `__gam_level` / `__gam_slope` / `__gam_curvature` / `__gam_auc` / `__gam_dev` evaluated at the landmark boundary | `--inputs-dir`, `--landmark-days`, `--k-pop`, `--k-pat`, `--trailing-window-days`, `--nthreads`, `--fit-split {all,train_val}` |
| `gam_cox_nonlinearity.R` (COMPASS only) | Penalized-spline Cox (`mgcv::gam(family=cox.ph())`) per selected feature: fits a smooth and a linear model of the same feature and reports `edf`/`p_lrt`/`q_lrt`/`delta_aic` — flags features whose hazard association is not actually linear | `--inputs-dir`, `--output-dir`, `--landmark-days`, `--feature-selection-csv` |
| `multivariate_longitudinal/dynamic_deephit.py` (COMPASS only, torch, optional) | Discrete-time competing-risks GRU (Dynamic-DeepHit) fit directly on the person-period lab sequence, in a cause-only config (death censored) or a competing config (`0=censored,1=cause,2=death`) | `--inputs-dir`, `--output-dir`, `--landmark-day`, `--config {platinum,competing,nepc,nepc_competing}` |
| `multivariate_longitudinal/survlatent_ode.py` (COMPASS only, torch, optional) | Adapter around the bundled editable `survlatent_ode_repo/` source, same configs and person-period input as Dynamic-DeepHit. Off by default in `03b` (`cp.RUN_SURVLATENT = False`) | `--survlatent-repo` defaults to the bundled checkout; `--inputs-dir`, `--output-dir`, `--landmark-day`, `--config {platinum,competing,nepc,nepc_competing}` |

`cox_aggregated.py` is now a project adapter: endpoint constants, cohort-specific covariates/restrictions,
and per-landmark context. The univariate/elastic-net CLI orchestration lives in
`survival_common/cox_runners.py`; reusable Cox feature selection, CV, final-fit, and manifest helpers live in
`survival_common/cox_models.py`; low-level Cox fitting/evaluation primitives live in
`survival_common/cox_engine.py`; XGBoost orchestration lives in
`COMPASS/survival_analysis/multivariate_analysis.py`'s `run_xgboost()`, which calls directly into
low-level XGBoost mechanics in `survival_common/xgboost_engine.py`.

**GAM stages — run order and leakage stance.** Both R scripts are base R + `mgcv` + `data.table`
only (no `tidyverse`/`survminer`/`broom`, unlike `COMPASS_generate_figures_pipeline.R`) and live under
`COMPASS/survival_analysis/`, with self-checking synthetic smoke tests in
`COMPASS/survival_analysis/tests/`. `gam_trajectory_features.R` runs after
`build_prediction_inputs.py` and before `univariate_analysis.py`: it reads
`pre_treatment_lab_long_landmark{D}.csv` and `canonical_labs_train_val.csv` and writes
`gam_trajectory_features_landmark{D}.csv` (one row per `DFCI_MRN`) plus a
`gam_fit_diagnostics_landmark{D}.csv` sidecar recording which basis (`fs`, the random
intercept+slope fallback, or the scalable two-stage ridge path) was used per lab, EDF, fit seconds,
and convergence. It also writes
`gam_trajectory_curves_landmark{D}.csv`, a long patient×lab×time grid of fitted values. The figure
pipeline uses that grid for paired GAM-smoothed trajectory panels stratified by platinum exposure
and classifier `has_nepc`; ribbons are 95% confidence intervals across patient-specific fitted
curves, not `mgcv` coefficient intervals. To prevent out-of-memory kills from an all-patient
factor-smooth design, labs with more than 500 patients automatically use a scalable decomposition:
one population GAM plus independently ridge-shrunk patient intercept/slope deviations. Adjust the
switch with `--max-fs-patients` and the shrinkage with `--patient-ridge-lambda`. The per-patient
smooth is unsupervised — it never sees `t_platinum`/`PLATINUM` — and by default is fit on **all**
cohort patients, not just train_val; pass `--fit-split train_val` to refit population smooths on
train+valid only as a leakage sensitivity check. `univariate_analysis.py` then automatically tests
the new `__gam_*` columns alongside the existing stats (`load_prebuilt_landmark` left-joins
`gam_trajectory_features_landmark{D}.csv` onto `aggregated_landmark{D}.csv` when it exists; the merge
is a no-op if the file is absent). `gam_cox_nonlinearity.R` runs **after** `univariate_analysis.py`,
since it depends on the feature list in `cox_agg_feature_selection.csv`; it fits on the full merged
table (train+valid+test), mirroring the same row-fitting asymmetry as
`run_univariate_nobs_adjusted_associations`, and writes `gam_cox_nonlinearity_landmark{D}.csv`.
The R stages are isolated in the R-kernel `GAM/04_gam.ipynb` so the Python run notebooks and
the `mgcv` conda environment do not need to coexist. The enforced handoff is: build inputs in Python,
run trajectory GAMs in R, set `REBUILD_PREDICTION_INPUTS = False` and rerun Python Stage 3 so its
models consume the trajectory features, then return to R for nonlinear Cox GAMs. Stage B rejects a
feature-selection file older than its trajectory-feature file, preventing a stale pre-GAM selection
from being tested accidentally. The R notebook uses the merged-profile root exclusively.

### 2.4 — Notebooks

COMPASS PROFILE has four Python stage notebooks sharing `compass_pipeline.py` (three numbered plus
`03b`), two read-only Python reporting notebooks (`06`, `07`), one R figure notebook, and one R GAM
notebook. All operate on the merged `profile_data` run:

- `01_preprocessing.ipynb` — drives preprocessing (schema audit, cohort compile, longitudinal
  preprocessing, prediction-input build, diagnostics) for whichever arms are selected (`arpi`
  and/or `adt`, with landmarks 0/90/180), over the common ADT-entry eligible cohort. Each arm gets
  independent prediction inputs at its own landmark list, and Stage 2 runs
  `longitudinal_data_processing.py` once per anchor (`--anchor-med-set {arpi,adt}`). Stages 0-2 are
  endpoint-independent and only need running once; Stage 3 forks per endpoint.
- `02_univariate.ipynb` / `03_multivariate.ipynb` — read `01`'s prediction inputs and run
  univariate, elastic-net, and XGBoost models independently of preprocessing
  (`tasks_for_run(run)` builds the per-run task grid from `run["landmarks"]`); either can be
  re-run alone without touching Stage 1-3 outputs.
- `03b_multivariate_longitudinal.ipynb` — optional, torch-gated (README invariant #7).
  Follows `ENDPOINT` like `02`/`03`, via its own config registry in
  `survival_common/longitudinal_targets.py` (see
  [Longitudinal configs](#longitudinal-configs)). **SurvLatent ODE is off by default**
  (`cp.RUN_SURVLATENT = False`) so the notebook runs Dynamic-DeepHit alone without the bundled
  external checkout and its conda env; set it to `True` to re-enable. Reads `01`'s
  `longitudinal_landmark{D}.csv` person-period inputs and runs the enabled models via
  `compass_pipeline.run_multivariate_longitudinal`; kept out of `03_multivariate.ipynb` so that
  notebook stays runnable with no torch installed. When enabled, SurvLatent tasks default to the
  bundled editable `survlatent_ode_repo/` checkout through `cp.SURVLATENT_REPO` — see
  `multivariate_longitudinal/README.md`.
- `06_abstract_numbers.ipynb` — read-only. Collects the cohort/event/performance counts quoted in
  the abstract and manuscript from already-generated artifacts. Refits nothing.
- `07_endpoint_comparison.ipynb` — read-only. Compares the `platinum` and `nepc` endpoint runs
  side by side: cohort and event counts per landmark (including how many diagnoses the incident
  gate removed as prevalent, and a loud warning when an endpoint has too few events to interpret),
  NEPC date/label provenance breakdowns, held-out performance pivots with `nepc - platinum`
  deltas, univariate association overlap flagged shared / platinum-only / nepc-only, and overlaid
  KM curves. Requires `01`-`03` to have been run for **both** endpoints; missing artifacts are
  collected and reported rather than raising. Its header restates the two caveats that govern
  reading it — the cohorts are not the same patients, and the strict NEPC label is narrower than
  the Figure 2 classifier definition.
- `05_figures.Rmd` — the sole COMPASS figure document, using R Markdown and
  `COMPASS_generate_figures_pipeline.R`. It renders both arms' overview, LLM-label, univariate,
  multivariate, KM, and per-lab distribution/trajectory figures at landmarks 0 and 90. Figure 1A reads
  `mrn_lists/icd_prostate_mrn_flags.csv` and displays cumulative ICD-C61 cohort selection through
  dated diagnosis, male sex, ≥5 PSA tests, post-diagnosis ADT, PARPi exclusion, pre-diagnosis
  platinum exclusion, and the requested post-ADT cancer exclusion; the ARPI arm additionally
  displays its post-diagnosis ARPI/docetaxel exposure criterion. Axis and table labels throughout
  name the arm's anchor ("ARPI/chemo initiation" vs. "ADT initiation") via `ANCHOR_LABEL`.
  Figure 4 additionally emits `figure4s_multivariate_all_models`, a supplemental held-out comparison
  of elastic-net Cox, XGBoost, and death-censored Dynamic-DeepHit using mean AUC(t), C-index, and
  integrated Brier score. If optional `03b_multivariate_longitudinal.ipynb` results are absent, the
  figure and data CSV mark Dynamic-DeepHit as missing instead of suppressing the supplement.
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
      `ADT_EXPOSED = 1` in `mrn_lists/icd_prostate_mrn_flags.csv` — ADT on/after diagnosis —
      step of the Figure 1 CONSORT — rather than only the landmark-0 prediction cohort
      (`eligible_landmark_0` in `landmark_mrn_availability.csv`) that v2 uses. So v3 includes
      patients later dropped by the male-sex, post-ADT cancer, PARPi, pre-diagnosis-platinum,
      and ≥5-PSA-test criteria.
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
  `02_univariate.ipynb`'s final section loads all shared-landmark univariate results for both
  `arpi` and `adt`, filters each to nominal `p_value < 0.05`, displays every hit, and exports a
  separate `cox/nominally_significant_univariate_results.csv` beneath each arm's run directory.
- `compass_pipeline.py`, the figure notebook, and the GAM notebook all point directly to the
  `PROFILE_DATA` source and `COMPASS` output roots; there is no data-variant selector.

IPIO has a paired run/figure notebook as well:

- `IPIO_run_locally.ipynb` — builds standard lab landmark inputs at 0/90 plus the genomic landmark-0
  inputs, then runs univariate Cox, elastic-net Cox, and XGBoost for the lab arm, genomics-only arm,
  and genomics+labs arm separately.
- `IPIO_generate_figures.ipynb` — writes a labs-only paired volcano and a separate genomics-only
  volcano, plus the lab-arm discrimination, genomic-arm discrimination, and lab-arm importance
  figures.

---

## Merged OncDRS source

COMPASS uses only the merged, deduplicated parquets published by the sibling
`PROFILE_data_processing` repository. `compile_OncDRS_data.ipynb` folds releases
`ALL_2021_11` … `ALL_2026_03` into one Parquet per table.

| | Path |
|---|---|
| Source | `/data/gusev/USERS/jpconnor/data/PROFILE_DATA/*.parquet` |
| Data root | `/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/` |
| Figure root | `/data/gusev/USERS/jpconnor/figures/CAIA/COMPASS/` |

Every raw scan goes through `data_preprocessing_common/oncdrs_sources.py`:

```python
scan_source(path)  # .parquet/.pq -> pl.scan_parquet(path).select(pl.all().cast(pl.Utf8))
                   # anything else -> pl.scan_csv(path, infer_schema_length=0)
```

The `cast(pl.Utf8)` is not cosmetic — it reproduces `infer_schema_length=0`, which is what makes
every column a string. The whole pipeline relies on that (`.str.to_datetime()`,
`.str.to_uppercase()`, `fast_io.recover_numeric()`, then explicit casts back to `Int64`). Parquet
columns are typed, so reading one without the cast breaks every `.str.*` call. The only genuinely
numeric column on the COMPASS path is `LABS.NUMERIC_RESULT`, and `recover_numeric()` parses it
back to `Float64` losslessly. `fast_io.scan_filter()` calls `scan_source`, so other projects may
continue to pass CSV sources even though the COMPASS notebooks always pass merged Parquets.

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

The second code cell of `01_preprocessing.ipynb` runs a **schema audit**
(`compass_pipeline.audit_schema`) that guards this. It scans all five tables and checks columns in
three tiers — REQUIRED (absent *or* all-null raises), EXPECTED (absent raises, all-null warns — for legitimately sparse columns like
`HYBRID_DEATH_DT` and `LABS.TEXT_RESULT`), OPTIONAL (warns either way) — and raises a single
`RuntimeError` listing every problem. Run it before anything else; an all-null
`NCI_PREFERRED_MED_NM` means going back to the upstream compile, not debugging COMPASS.

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
4. **Endpoint and duration:** COMPASS registers `(t_platinum, PLATINUM)` and `(t_nepc, NEPC)`;
   `platinum` is the default and the one the manuscript figures use. For patients without the event,
   the anchor time is filled with `t_last_contact` (censoring). After landmark rebasing, the validity
   filter requires duration `> 0`, which silently drops patients whose event falls before/at the
   landmark — add count logging if you depend on it. The `t_nepc` conditions are applied **only**
   under `--require-nepc`, so registering the second endpoint cannot change the platinum cohort;
   `tests/test_nepc_endpoint.py` asserts that equivalence.
   **Any new endpoint's outcome columns must also be added to
   `cox_aggregated.OUTCOME_METADATA_COLUMNS`**, the set consumed by `outcome_columns()` to keep
   outcome data out of the feature matrix. Miss it and the event becomes a predictor of itself. The
   set lists every endpoint's columns unconditionally, including on runs that don't model them.
5. **ID/age columns are injected at runtime.** PROFILE defaults to `DFCI_MRN` / `AGE_AT_TREATMENTSTART`;
   IPIO defaults to `DFCI_MRN` with its own baseline covariates. `build_*` and model `main()` functions
   mutate module globals **and monkey-patch `cox_aggregated.ID_COL/AGE_COL`**. If you add a function that captures
   `ID_COL` at import time (default arg, module constant), it will not see the patch — thread the column
   through as a parameter instead. Concrete example already in the codebase: `survival_common/cox_models.py`
   imports `AGE_COL`/`ID_COL` from `cohort.py` at module load as `DEFAULT_AGE_COL`/`DEFAULT_ID_COL` — a
   third, unpatched copy that neither the module-global rebind nor the `cox_aggregated` monkey-patch touches.
6. **Horizon grid is shared via `build_manifest.json`** so Cox/XGBoost AUC & Brier are comparable.
   Don't compute horizons ad hoc in a model script.
7. **Three-layer torch gating.** `multivariate_longitudinal/` is the only torch consumer in this repo,
   and it must stay optional: (1) `survival_common/deephit_engine.py` does
   `try/except ModuleNotFoundError` at import time, falling back to `torch = None; Dataset = object` —
   the `Dataset = object` fallback matters because `class SequenceDataset(Dataset)` is evaluated at
   import time even when torch is absent; (2) `DynamicDeepHitGRU` is defined under `if nn is not None:`
   with an `else:` stub that calls `require_torch()` if ever instantiated; (3) `require_torch()` is
   called at the top of `run_deephit`, never at module import. `survlatent_ode.py` follows the same
   shape: `import_survlatent()` (which does the external repo's `sys.path`/`os.chdir` setup) is called
   only from `main()`, never at import. Net effect: `dynamic_deephit.py --help` and
   `survlatent_ode.py --help` both exit 0, and the full test suite imports cleanly, with no torch
   installed and without importing the bundled SurvLatent source.

---

## Configuration & paths

- **Hard-coded cluster roots** (all overridable by CLI except module-level constants):
  - Data: `/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/`
  - Survival results: `/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis`
  - Figures: `/data/gusev/USERS/jpconnor/figures/CAIA/COMPASS/`
- **Raw OncDRS root** — merged Parquets at
  `/data/gusev/USERS/jpconnor/data/PROFILE_DATA/`. The source resolver lives in
  `data_preprocessing_common/oncdrs_sources.py` alongside the `TABLE_FILES` basename map.
- `data_preprocessing_common/dfci_labs.py` uses the checked-in shared
  `resources/lab_mappings/OMOP_to_DFCI_lab_ids.csv` by default. Per-project lab inventory outputs
  default to `/data/gusev/USERS/jpconnor/data/CAIA/<project>/unique_lab_ids_w_units.csv`.

## Arms and endpoints

A run is an (arm, endpoint) pair. The arm sets time 0; the endpoint sets the event. Selecting a
different endpoint does **not** fork the cohort assembly — Stages 0-2 are endpoint-independent and
write one survival cohort carrying every endpoint's columns. Stage 3 is where the fork happens.

### Endpoints

| Endpoint | Duration / event | Source of the event date |
| --- | --- | --- |
| `platinum` | `t_platinum` / `PLATINUM` | First carboplatin/cisplatin exposure in `MEDICATIONS` |
| `nepc` | `t_nepc` / `NEPC` | `LLM_annotations/LLM_nepc_diagnosis/nepc_dx_labels.parquet` |

The `nepc` endpoint models **time from the treatment anchor to LLM-adjudicated NEPC diagnosis**.
Four properties of it differ from `platinum` and matter for interpretation:

- **It is an incident endpoint, gated by `--require-nepc`.** That flag adds `t_nepc notna` and
  `t_nepc > 0` to the `make_outcome_df` validity conditions, dropping patients whose NEPC diagnosis
  falls at or before the landmark (prevalent, not incident). The gate is **off by default** so it
  can never silently shrink the platinum cohort — `compass_pipeline` passes it only when
  `ENDPOINT == "nepc"`. The consequence is that the two endpoints' cohorts are **not the same
  patients**; per-landmark attrition is reported in the Stage 3 build log.
- **The label definition is strict, and narrower than the classifier used in Figure 2.** The
  `nepc_dx_labels` labels are veto-gated and precision-biased; the Figure 2 `has_nepc` strata come
  from the broader "any NE feature → NEPC" binary classifier. They are not interchangeable — name
  which definition is in play in any resulting text.
- **Event counts may be low.** Check the Stage 1 summary and the `07` notebook's count table
  *before* spending modelling time; `07` prints an explicit underpowered warning below 50 events.
- **Date provenance is carried, not filtered.** `NEPC_DATE_SOURCE`, `NEPC_DATE_PRECISION`, and
  `NEPC_LABEL_SOURCE` ride through to the prediction inputs for sensitivity analysis. Two are worth
  knowing: `date_source = note_date` means *earliest documentation*, not onset; and
  `label_source = auto_negative_no_evidence` patients were never seen by an LLM — legitimate
  censored observations, but distinct from adjudicated negatives.

If `nepc_dx_labels.parquet` is not mounted, Stage 1 warns and emits no NEPC columns. Every
downstream NEPC touchpoint is presence-guarded, so the platinum path runs unchanged.

### Longitudinal configs

`03b`'s Dynamic-DeepHit / SurvLatent arm uses a **second, related registry**:
`survival_common/longitudinal_targets.py`'s `LONGITUDINAL_CONFIGS`. It selects a cause of
interest plus an optional competing cause, which is a finer axis than `ENDPOINTS`' single
(duration, event) pair — hence `--config`, not `--endpoints`:

| `--config` | causes (label 1, label 2) | AUC(t) grid read |
| --- | --- | --- |
| `platinum` | platinum; death censored | `platinum` |
| `competing` | platinum, death | `platinum` |
| `nepc` | nepc; death censored | `nepc` |
| `nepc_competing` | nepc, death | `nepc` |

`CONFIG_ENDPOINTS` maps each config to the `ENDPOINTS` key whose horizon grid it scores on, so a
NEPC model is evaluated on the NEPC timeline and stays comparable to the NEPC Cox/XGBoost arms
(README invariant #6). `compass_pipeline.longitudinal_task_specs(endpoint)` picks the pair for the
selected endpoint, so `03b` needs only `ENDPOINT` set — the same knob as `02`/`03`.

Two ordering invariants are asserted at import: the cause of interest is always label 1 and death
always label 2 (a reorder would silently reinterpret every downstream risk column), and
`survlatent_ode.py`'s parallel `EVENT_CONFIGS` must agree with this registry on both names **and**
columns, so the two models cannot drift into meaning different things by the same config name.

The cause-only configs (`platinum`, `nepc`) censor at death and are the ones comparable to
Cox/XGBoost for that endpoint; `summarize_longitudinal_outputs` filters to that row. The competing
configs' death rows are written to disk but excluded from the summary.

### Running a second endpoint

Set the parameter cell of `01`/`02`/`03`, then run each top to bottom:

```python
ARMS = ["adt"]
ENDPOINT = "nepc"        # "platinum" reproduces the original run exactly
OUTPUT_SUFFIX = "_nepc"  # "" for the original tree
OVERWRITE = False

cp.ENDPOINT = ENDPOINT
cp.FORCE_RERUN = OVERWRITE
RUNS = cp.make_runs(ARMS, output_suffix=OUTPUT_SUFFIX)
```

`OUTPUT_SUFFIX` suffixes **both** `prediction_inputs_<arm>` and `local_runs_<arm>`. Both are
required: because `--require-nepc` changes which patients survive the landmark filter, and that
filter runs at *preprocessing* time inside `build_landmark_merged`, the NEPC build needs its own
inputs tree — not just its own output tree. With the suffix set, `prediction_inputs_adt/` and
`local_runs_adt/` are never touched.

Stages 0-2 of `01` (schema audit, cohort compile, lab preprocessing) only need running once; they
are shared. Re-run Stage 3 onward per endpoint.

`03b` takes the same `ENDPOINT` / `OUTPUT_SUFFIX` pair and derives its `--config` values from it
(see [Longitudinal configs](#longitudinal-configs)); it additionally has `RUN_SURVLATENT`, left off
by default.

## Recommended run order

Run each notebook top to bottom; select ARPI/ADT with each Python notebook's `ARMS` setting, and
the event with its `ENDPOINT` / `OUTPUT_SUFFIX` setting (see [Arms and endpoints](#arms-and-endpoints)):

1. `COMPASS/survival_analysis/01_preprocessing.ipynb`
2. `COMPASS/survival_analysis/02_univariate.ipynb`
3. `COMPASS/survival_analysis/03_multivariate.ipynb`
4. `COMPASS/survival_analysis/03b_multivariate_longitudinal.ipynb` (optional — requires torch;
   see [Dependencies](#dependencies) and `multivariate_longitudinal/README.md`)
5. `COMPASS/survival_analysis/GAM/04_gam.ipynb`
6. `COMPASS/survival_analysis/05_figures.Rmd`
7. `COMPASS/survival_analysis/06_abstract_numbers.ipynb` (read-only; abstract/manuscript counts)
8. `COMPASS/survival_analysis/07_endpoint_comparison.ipynb` (read-only; only after steps 1-3 have
   been run for **both** endpoints)

The notebooks pass `PROFILE_DATA/*.parquet` paths explicitly to the lower-level scripts. Existing
hand-curated `LLM_NEPC_labels/` inputs remain under the shared `COMPASS` data root.

## Dependencies

No packaged environment is checked in. Assumed: `pandas`, `numpy`, `scipy`, `tqdm`, `scikit-learn`,
`scikit-survival` (`sksurv`), `xgboost`, `lifelines`, `matplotlib`. Python **3.10+** is recommended
for the modern type-hint syntax used by the shared modules.

`torch` is an **optional** dependency, needed only for
`COMPASS/survival_analysis/multivariate_longitudinal/` (Dynamic-DeepHit directly; SurvLatent ODE via
its own external repo/conda env). It stays commented out in `requirements.txt` — see that file's
header and README invariant #7 — so the rest of the pipeline (data compilation, Cox, XGBoost) never
requires it.

---

## Known issues / footguns

These are real, verified items found in code review. Fix opportunistically; at minimum, don't be
surprised by them.

### High impact

- **`t_platinum > 0` drops patients at later landmarks — read +180d results with this in mind.**
  The validity filter runs after landmark rebasing, so at +180d every patient whose platinum event
  fell in the first 180 days is dropped — the later-landmark cohorts are both smaller and
  systematically depleted of early-progressing (highest-risk) patients. This bites
  `multivariate_longitudinal` hardest, since a GRU/ODE needs more data than Cox; treat +180d results
  there as possibly underpowered rather than as a negative result. This is a structural landmark-design
  consequence, not a coding bug, and it also shapes how Finding 5's full-cohort univariate screen and
  Finding 10's genomic-prevalence floor should be interpreted at later landmarks (a feature's measured
  prevalence, and hence whether it clears the floor, shifts as the cohort composition shifts).
- **The `nepc` endpoint is likely event-poor, and its cohort is not the platinum cohort.** The
  strict `nepc_dx_labels` definition is precision-biased by design, and `--require-nepc` further
  removes prevalent diagnoses, so NEPC event counts can be far smaller than platinum's at the same
  landmark. Two consequences: multivariate NEPC results may be underpowered rather than negative
  (`07` warns below 50 events), and **platinum-vs-NEPC metric differences are confounded by cohort
  composition** — they are not a like-for-like model comparison. If counts are too low to work
  with, the fallback is the broader `avpc_nepc_timeline.parquet` definition (more events, lower
  precision), which is a different label and would need its own endpoint.
- **Two different NEPC definitions are in play in this repo.** The `nepc` *endpoint* uses the
  strict, veto-gated, dated `nepc_dx_labels.parquet`. The Figure 2 / Figure 2v2 / Figure 2v3
  `has_nepc` *strata* use the broader "any NE feature → NEPC" binary classifier from
  `LLM_NEPC_labels/LLM_NEPC_classifier_labels.tsv`. They are **not interchangeable** and will not
  agree on patient counts. Always name which one a given number came from.
- **Competing-config Brier is the binary cause-of-interest Brier, not the cumulative-incidence
  Brier — and so is the competing-risks IPCW AUC(t).** Both the Brier score and the `cumulative_dynamic_auc`
  IPCW reference distribution binarize on `event = (label == cause)`, folding any competing event
  (a different cause, or true censoring) into "censored" for the KM censoring-weight estimator. This
  keeps the `competing` config's platinum row numerically comparable to the `platinum` config and to
  Cox/XGBoost, but it is *not* the methodologically correct CIF/Aalen-Johansen-based estimator for
  competing risks (a `cif_brier` metric was scoped but not implemented — see the plan's §7). The two
  metrics share one root convention (see `cox_engine.compute_ipcw_auc_t`'s and
  `deephit_engine.compute_metrics`'s docstrings) — a reader who internalizes the Brier caveat but not
  the AUC(t) one would still misread the AUC(t) numbers the same way. Document as a stated
  methodological choice in the paper's methods section; don't over-read either metric as a rigorous
  competing-risks score without checking which convention you're looking at.

### Medium impact

- **`auc_max_time_units = 260` is the default** (`DEFAULT_AUC_MAX_TIME_UNITS` in `survival_common/helper.py`)
  and admin-censors AUC/Brier unless `--auc-max-time-units` is overridden. The builders clamp the horizon
  grid to it and runners read it back from the manifest; overriding it at run time only (not at build
  time) makes the two diverge, which the runner warns about.
- **Silent patient drops** at several inner-joins and `valid`-mask filters (diagnosis/death inner joins,
  duration `> 0` filter). Downstream cohort filters now log attrition in `build_prediction_inputs.py`;
  keep that pattern for any new cohort-selection rule.
- **`multivariate_longitudinal` per-landmark cohorts are independent** (`cohort_mode:
  "independent_by_landmark"`) — the same MRN can be train at one landmark and test at another. Fine
  within a landmark, since each model only ever reads its own landmark's split, but results must never
  be pooled across landmarks.
- **`survlatent_ode.py` does `os.chdir` into the cloned external repo** (`import_survlatent()`,
  called only from `main()`). `--output-dir`/`--inputs-dir` are resolved to absolute paths before that
  chdir happens; a hypothetical caller that constructs those paths lazily after `main()` starts would
  silently write into the wrong place. See `multivariate_longitudinal/README.md`.
- **Dynamic-DeepHit's CV grid is expensive**: 27 hyperparameter combos × 5 folds, run serially per
  (landmark, config) by `compass_pipeline._run_tasks`. Reduce the grid for a first pass and rely on
  `FORCE_RERUN = False` to resume.

### Low impact / cleanliness

- `iterrows`/`apply`-based row loops in `data_preprocessing_common/dfci_labs.py` and `longitudinal_data_processing.py`
  are slow on full DFCI-scale pulls — vectorize if performance bites.

---

## Notes

- `.ipynb_checkpoints/` and `__pycache__/` artifacts are git-ignored and not part of the workflow.
- Several scripts read and write data **outside** the repository root (the `/data/gusev/...` cluster paths).
