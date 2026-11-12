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
cancer diagnosed strictly after first ADT are excluded. The R-only `COMPASS/notebooks/figures/run_figure_render.R` script emits
the manuscript figures for the ADT arm.

**Arm, cohort, and endpoint are three orthogonal axes.** The *arm* sets time 0 (the index-date
anchor); the *cohort* restricts which patients are modelled; the *endpoint* sets the event being
modelled. All three are selected independently, and the triple is what defines a run — see
[Arms, cohorts, and endpoints](#arms-cohorts-and-endpoints).

| Axis | Registry | Declared in | Values |
| --- | --- | --- | --- |
| **Arm** (index date / time 0) | `_ARM_SPECS` | `COMPASS/pipeline/compass_pipeline.py` | `arpi`, `adt` |
| **Cohort** (patient subset) | `COHORT_SPECS` | `COMPASS/pipeline/compass_pipeline.py` | `all`, `metastatic`, `localized` |
| **Endpoint** (the event) | `ENDPOINTS` | `COMPASS/pipeline/cox_aggregated.py` | `platinum`, `nepc`, `avpc` |
| **Landmark** (days after time 0) | `_ARM_SPECS[arm]["landmarks"]` | `COMPASS/pipeline/compass_pipeline.py` | `0`, `90`, `180` |

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
│   └── build_somatic_gleason_inputs.py   # sequencing and Gleason inputs
│
├── notebooks/
│   ├── manuscript/                       # 00-05, run in order from this directory
│   │   ├── 00_update_manual_annotations.ipynb
│   │   ├── 01_preprocessing.ipynb        # cohort compile, ADT-intent strata + label review, preprocessing, diagnostics
│   │   ├── 02_univariate.ipynb           # univariate arms + nominal-significance filter
│   │   ├── 02a_cohort_comparison.ipynb   # read-only: 6 cohorts x 2 endpoints, landmark +180d
│   │   ├── 03_multivariate.ipynb         # elastic-net + XGBoost + summary tables + predictive checklists
│   │   ├── 03b_multivariate_longitudinal.ipynb # Dynamic-DeepHit (torch, optional)
│   │   ├── 03c_multivariate_adjust_only.ipynb # adjust-only refits into *_adjust_only trees
│   │   ├── 03d_shallow_decision_tree.ipynb
│   │   ├── 04_federated_transfer.ipynb   # frozen rhino_data_v2 bundles scored on local DFCI endpoints
│   │   └── 05_prep_figure_data.ipynb     # Polars figure tables + cache manifest
│   ├── testing/                          # 06-08, reviewer/exploratory analyses
│   │   ├── 06_reviewer_suggestions.ipynb
│   │   ├── 07_checklist_test.ipynb
│   │   └── 08_metastatic_transfer_test.ipynb
│   └── figures/                          # R entry scripts (code they source stays in COMPASS/figures/)
│       ├── final_figure_data_prep.R
│       ├── run_figure_render.R
│       └── ESMO_AI_Figures.R
├── federated_models/                     # 04
│   └── federated_clinical_comparison.py  # clinical comparison and transfer tables
└── figures/                              # 05 prep + R figures
    ├── figure_run.R                      # toggles/paths shared by the two entry scripts in notebooks/figures/
    │                                     #   (final_figure_data_prep.R, run_figure_render.R)
    ├── figure_workflow.R                 # cached prepare/render, federated + manuscript figures
    ├── COMPASS_generate_figures_pipeline.R # per-cohort DFCI figures (Figs 1-2)
    ├── figure_supplements.R              # cohort forest, cohort overview, metastatic labels
    ├── checklist_supplement.R            # predictive-checklist cards
    ├── figure_typography.R               # shared typography (survival_common/figure_typography.json)
    ├── prepare_figure_data.py            # Python prep called by 05
    ├── retained_exports.txt              # the rendered PDF set
    └── README.md
```

All numbered/lettered notebooks use the merged PROFILE_data_processing parquets and the
standard `COMPASS` data/figure roots. The Python notebooks share `compass_pipeline.py`; no
single-release baseline run is part of this workflow. `03b` is optional and torch-gated (see
invariant #7 below) — the other notebooks never require torch. `02a` is a read-only
reporting notebook: it reads already-generated artifacts and refit nothing.

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
 prediction_inputs_<arm>_nepc/   (same, built with --endpoint nepc: incident-NEPC risk set)
 prediction_inputs_<arm>_avpc/   (same pattern, other endpoint)
        │
        ├──► build_somatic_gleason_inputs.py
        │    prediction_inputs_<arm>/somatic_gleason/
        │
        ▼  lab_associations/univariate_analysis.py
           multivariable_models/multivariate_analysis.py
 local_runs_<arm>/ , local_runs_<arm>_nepc/
        │
        ├──► 05_prep_figure_data.ipynb, then COMPASS/notebooks/figures/run_figure_render.R (R; ADT, platinum)
        │    <fig_root>/ (PDF exports)
        │
        ▼  02a_cohort_comparison.ipynb (read-only; 6 cohorts x 2 endpoints)
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
  `--subtype-labels` points at the per-patient LLM primary-subtype labels
  (`SUBTYPE_LABELS_PATH`, `LLM_annotations/LLM_primary_subtype/primary_subtype_labels.parquet`, one
  row per patient, written by the sibling `LLM_clinical_annotations` repo's
  `notebooks/primary_subtype.ipynb`, which full-joins the `tasks/nepc` and `tasks/avpc` runner
  outputs). `load_subtype_labels()` emits `NEPC`/`NEPC_DATE` (`nepc_status == "nepc"`,
  `nepc_date`) and `AVPC`/`AVPC_DATE` (the AVPC runner's `is_avpc`/`avpc_date`, taken as-is — no
  criteria-count threshold is applied here), plus the joint `AVPC_NEPC` audit columns. Patients the
  upstream runners left unresolved (null status, review failed/not run) or that are absent from
  the file are **censored as 0**; their review status rides along in
  `NEPC_LABEL_SOURCE`/`AVPC_LABEL_SOURCE` (`missing_from_labels` for join misses). Positives with
  no parseable date are demoted to 0 and counted in the log. `NEPC_STATED_DX`/`NEPC_STATED_DX_DATE`
  (the stricter stated-diagnosis tier), `AVPC_N_CRITERIA`/`AVPC_CRITERIA` and `PRIMARY_LABEL` are
  audit-only. A **missing file is non-fatal**: the stage warns and emits no NEPC/AVPC/AVPC_NEPC
  columns, leaving the platinum pipeline unaffected. Note the date columns arrive as ISO
  **strings**, not a date type, with partial dates already normalized upstream (precision is in
  `nepc_date_precision`).
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
  at least five broad PSA tests, and final eligibility. When the labels are available, the two
  survival cohorts additionally carry `NEPC`/`NEPC_DATE`/`TT_NEPC`, `AVPC`/`AVPC_DATE`/`TT_AVPC`,
  plus joint `AVPC_NEPC` audit fields. One cohort file serves the **three modeled endpoints**
  (`platinum`, `nepc`, `avpc`). The printed summary
  reports positives, the provenance breakdowns, and how many diagnoses are prevalent at the anchor
  for each — read it before committing to NEPC/AVPC modelling.
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
  `t_platinum`, `t_nepc`, `t_avpc`, `t_last_contact`, `t_death`. `t_death` is a real
  death-date-derived duration when the survival cohort's `death_date` is available (falls back to
  the last-contact proxy for dead patients with no recorded date); it is not itself a registered
  model endpoint. `t_nepc`/`t_avpc` are each built exactly like `t_platinum` (days
  from anchor to the endpoint's date column for positives, else `t_last_contact`) and are emitted
  only when the upstream LLM primary-subtype labels were available at Stage 1.

### 2.2 — `data_preprocessing/build_prediction_inputs.py` → `prediction_inputs/`

The **single source of truth for model inputs**. It builds an independent eligible risk set at each
landmark. Day-0 membership therefore does not require surviving event-free to day 90, avoiding an
immortal-time restriction on the earlier cohort. By default (`--split-mode shared`) the train/valid/test
split is derived once on the union of MRNs across all landmarks and reindexed onto each landmark's risk
set, so held-out results are comparable across landmarks; `--split-mode independent` derives a separate
split within each landmark's risk set instead.

- **Key CLI:** `--data`, `--landmark-days 0 90 180` (default from
  `cox_aggregated.DEFAULT_LANDMARK_DAYS`),
  `--split-mode {shared,independent}` (default `shared`),
  `--seed`, `--test-frac`, `--val-frac`, `--time-unit-days 7`, `--min-patient-coverage`,
  `--auc-quantiles`, `--id-col`, `--age-col`, `--anchor-col`,
  `--restrict-to-mrns`, `--require-first-treatment` / `--no-require-first-treatment`,
  `--min-psa-count`, `--exclude-parpi` / `--include-parpi`, `--endpoint` (`--require-nepc` is a
  deprecated alias for `--endpoint nepc`).
- **`--endpoint`** restricts the risk set to patients with an incident event for the selected
  endpoint (that endpoint's `t_*` duration present and `> 0` after landmark rebasing) — `platinum`
  by default, or `nepc`/`avpc`. Non-`platinum` values are recorded in
  `build_manifest.json`. It changes cohort membership, so a build made with a non-`platinum`
  endpoint must go to its own `prediction_inputs_<arm>_<endpoint>/` directory —
  `compass_pipeline.make_runs(output_suffix=...)` does that automatically. The horizon-grid loop
  skips any registered endpoint whose duration/event columns are absent from the build, so a
  platinum-only cohort is unaffected by any of these registrations.
- **Default downstream cohort filters:** ≥5 PSA rows and PARPi exclusion are repeated as consistency
  guards after the Stage-1 restriction. They should produce no further criterion-related attrition
  for newly rebuilt cohorts; alternate inputs can relax them explicitly.
- **Outputs:** `aggregated_landmark{D}.csv`, `pre_treatment_lab_long_landmark{D}.csv`,
  `split_assignments_landmark{D}.csv`, the base-landmark compatibility copy
  `split_assignments.csv`, `landmark_mrn_availability.csv`, `canonical_labs_train_val.csv`,
  `landmark_attrition.json`, and `build_manifest.json`.
- `IPIO/data_preprocessing/build_genomic_inputs.py` builds the parallel
  `prediction_inputs/genomic/` landmark-0 arm anchored at IO start (`t_first_treatment`), restricts
  to patients with an actual somatic sample, attaches dynamic binary `<GENE>_SNV`
  indicators, and **reuses** the main `split_assignments.csv` so test stays test. It writes both
  genomic provenance files and runner-compatible aliases (`aggregated_landmark0.csv`,
  `pre_treatment_lab_long_landmark0.csv`, `canonical_labs_train_val.csv`, `build_manifest.json`).

### 2.3 — Models

All read prebuilt inputs and the `split` column; none re-derive the split. COMPASS models default to
the `platinum` endpoint (time to first platinum) and additionally support `nepc` and `avpc`
(time to the LLM NEPC / AVPC date from `primary_subtype_labels.parquet`) — see
[Arms, cohorts, and endpoints](#arms-cohorts-and-endpoints). `--endpoints` is built from `cox.ENDPOINTS`, so every
runner gets the choice for free. Both ARPI- and ADT-anchored
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
| `univariate_analysis.py` | Cox: univariate n_obs-adjusted lab associations, or separate sequencing (SNV-only; any-alteration + altered-gene count) and Gleason COMPASS associations | `--landmark-days`, `--endpoints`; sequencing/Gleason use the observation nearest ADT as origin; IPIO supports `--feature-subset {labs,genomics,all}` |
| `multivariate_analysis.py --model elastic-net` | Elastic-net Cox multivariable model (sksurv `CoxnetSurvivalAnalysis`, 5-fold CV, AGE unpenalized) | `--landmark-days`, `--endpoints`, `--n-folds`; IPIO also supports `--feature-subset {labs,genomics,all}` |
| `multivariate_analysis.py --model xgboost` | XGBoost `survival:cox`, 5-fold CV grid (`max_depth × eta × min_child_weight`) | `--landmark-days`, `--endpoints`, `--max-features`; IPIO also supports `--feature-subset {labs,genomics,all}` |
| `multivariable_models/longitudinal/dynamic_deephit.py` (COMPASS only, torch, optional) | Discrete-time competing-risks GRU (Dynamic-DeepHit) fit directly on the person-period lab sequence, in a cause-only config (death censored) or a competing config (`0=censored,1=cause,2=death`) | `--inputs-dir`, `--output-dir`, `--landmark-day`, `--config {platinum,competing,nepc,nepc_competing}` |
| `multivariable_models/longitudinal/dynamic_deephit.py --dynamic` (COMPASS only, torch, optional) | The **dynamic arm**: the same GRU, but the unit of prediction becomes one `(patient, prediction time)` pair instead of one patient — a CIF at every observation time, from history up to that time only. Requires the full-follow-up frame. Off by default (`cp.RUN_DYNAMIC = False`) | adds `--dynamic` to the flags above; writes to `dynamic_deephit_dynamic/` with the `dynamic_deephit_dyn_` metrics prefix, so the two arms resume independently |
| `multivariable_models/longitudinal/incremental_risk.py` (COMPASS only, **no torch**) | Does accruing lab history sharpen the 6-month estimate? Pure post-processing of the dynamic arm's predictions — no refit. Emits sequential-landmark metrics (5a, confounded by risk-set thinning) and a held-back-window ablation (5b, the one to lead with) | `--output-dir` (the dynamic arm's results dir), `--endpoint`, `--landmark-day` (**singular**), `--config`, `--cohort`, `--horizon-days`, `--delta-days`, `--grid-days`, `--min-risk-set`, `--min-events` |

`cox_aggregated.py` is now a project adapter: endpoint constants, cohort-specific covariates/restrictions,
and per-landmark context. The univariate/elastic-net CLI orchestration lives in
`survival_common/cox_runners.py`; reusable Cox feature selection, CV, final-fit, and manifest helpers live in
`survival_common/cox_models.py`; low-level Cox fitting/evaluation primitives live in
`survival_common/cox_engine.py`; XGBoost orchestration lives in
`COMPASS/multivariable_models/multivariate_analysis.py`'s `run_xgboost()`, which calls directly into
low-level XGBoost mechanics in `survival_common/xgboost_engine.py`.

### 2.4 — Notebooks

The [predictive checklists](COMPASS/multivariable_models/predictive_checklists/README.md) section of
`03_multivariate.ipynb` fits the Zhang et al. M-of-N model with a local inverse-censoring-weight
extension on the full ADT cohort and on the Gleason + TP53/RB1/PTEN subset. Training is off by
default (`RUN_CHECKLISTS`) until preflight counts have been reviewed. The runner reads the separate
platinum/NEPC prebuilt input trees and existing splits; day 0 is primary, with day 90/180
secondary, and default horizons are 1-5 years after each landmark. It exports calibrated horizon
risks, matched age/Cox/XGBoost comparisons, solver diagnostics, and R figures. These are
independent horizon classifiers, not continuous survival curves or competing-risk cumulative
incidence. Its manifest checks require rebuilding inputs that carry the old endpoint-metadata
shadowing bug. Solver dependencies remain isolated from the existing Cox/XGBoost and torch
workflows.

COMPASS PROFILE has four Python stage notebooks sharing `compass_pipeline.py` (three numbered plus
`03b`), one read-only Python reporting notebook (`02a`), and one R figure notebook. All operate
on the merged `profile_data` run:

- `01_preprocessing.ipynb` — drives preprocessing (cohort compile, ADT-intent
  stratum build and label review, longitudinal preprocessing, prediction-input build, diagnostics) for whichever
  arms are selected (`arpi` and/or `adt`, with landmarks 0/90/180), over the common ADT-entry
  eligible cohort. Each arm gets independent prediction inputs at its own landmark list, and
  Stage 2 runs `longitudinal_data_processing.py` once per anchor
  (`--anchor-med-set {arpi,adt}`) via `cp.stage2_runs(RUNS)`. **Stage 1b** writes the
  medication-derived ADT-intent MRN lists and their preliminary endpoint counts; **Stage 2c**
  (after Stage 2, whose ADT-arm longitudinal file its event counts read) re-derives that label
  over every ADT-exposed patient with `export_adt_intent_outputs.build_labels()` and prints
  text-only checks on it; it draws no figures and writes nothing. Stages 0-2 are
  independent of both cohort and endpoint and only need running once; Stage 3 forks per
  (cohort, endpoint).
- `02_univariate.ipynb` / `03_multivariate.ipynb` — read `01`'s prediction inputs and run
  univariate, elastic-net, and XGBoost models independently of preprocessing.
  `02`'s sequencing and Gleason univariate screens (`gleason`, `sequencing_snv`,
  `sequencing_any`, landmark 0) run once per endpoint on the full cohort only (cohort `all`,
  no exclusion); metastatic and pre-ADT-castrate runs skip them.
  `03` also runs an available-case sensitivity at 0, +90, and +180 days:
  Gleason-versus-labs models among identical Gleason-available patients (the Figure 4C
  mean AUC(t) panel).
  (`tasks_for_run(run)` builds the per-run task grid from `run["landmarks"]`); either can be
  re-run alone without touching Stage 1-3 outputs.
  - **Clinical/somatic component combinations** (`03`, `run_multivariate_clinical_combinations` /
    `summarize_clinical_combinations`) — every non-empty combination of {labs, gleason, somatic}
    on the matched `gleason_somatic_available_case` cohort, written under
    `<output_dir>/clinical_combinations/<cohort>/<arm>/<cox|xgboost>/landmark_<D>/both/`.
    `summarize_clinical_combinations` adds a paired bootstrap `delta_c_index` against each
    cohort's `labs` arm on the shared test-block patients.
  - **Risk-score stratification** (`03`, `run_risk_stratification`) — stratifies the full-cohort
    labs risk score by Gleason/stage on the ADT clock (from `build_somatic_gleason_inputs.py`'s
    `somatic_gleason/clinical_stratifiers/clinical_stratifiers_landmark{D}.csv`), and separately
    compares the labs arm against every other matched arm within the
    `gleason_somatic_available_case` cohort via `risk_score_stratified_figures.py --comparison-risks`,
    stratified by Gleason/stage/TP53-PTEN-RB1 on the sequencing clock. Every somatic cohort is
    anchored on the sequencing report date (`REPORT_DT`) reported closest to ADT start; no somatic
    data is used on the ADT clock.
    Runs once per scoring scheme, written to `<output_dir>/risk_stratification/<test|cv_oof>/full_cohort/landmark_<D>/`
    and `<output_dir>/risk_stratification/<test|cv_oof>/matched/<cohort>/<cox|xgboost>/landmark_<D>/`
    — **the two schemes are never pooled into the same directory or KM panel.**
- `03b_multivariate_longitudinal.ipynb` — optional, torch-gated (README invariant #7).
  Follows `ENDPOINT` like `02`/`03`, via its own config registry in
  `survival_common/longitudinal_targets.py` (see
  [Longitudinal configs](#longitudinal-configs)). It runs Dynamic-DeepHit; SurvLatent ODE has
  been removed. **The dynamic arm is off by default** (`cp.RUN_DYNAMIC = False`): set it to `True` and the same
  `run_multivariate_longitudinal` call additionally runs per-timepoint Dynamic-DeepHit and then
  `incremental_risk.py` on its predictions, with the last two cells reading the tables back via
  `cp.load_incremental_risk_results(run, "ablation" | "by_landmark")`. That arm needs the
  full-follow-up inputs (`--longitudinal-full-followup` in `01`; see invariant #10). Reads `01`'s
  `longitudinal_landmark{D}.csv` person-period inputs and runs the enabled models via
  `compass_pipeline.run_multivariate_longitudinal`; kept out of `03_multivariate.ipynb` so that
  notebook stays runnable with no torch installed. See
  `COMPASS/multivariable_models/longitudinal/README.md`.
- `02a_cohort_comparison.ipynb` — read-only. Compares the 6 patient cohorts (3 cohorts `all` /
  `metastatic_adt` / `metastatic_llm` x 2 exclusions `none` / `pre_adt_castrate`) against both
  endpoints at the **+180d landmark only**: cohort and event counts (including how many diagnoses
  the incident gate removed as prevalent, and a loud warning when a cell has too few events to
  interpret), PSA and testosterone pre/post-treatment distributions stratified by event, univariate
  PSA/testosterone associations as a hazard-ratio forest plot across every cohort x endpoint cell,
  and the cascade effect of narrowing `all` -> metastatic -> +noprecastrate. Requires `01`-`02` to have been run for **both** endpoints; missing artifacts are
  collected and reported rather than raising. Its header restates the two caveats that govern
  reading it — the cohorts are not the same patients, and the strict NEPC label is narrower than
  the Figure 2 classifier definition.
- `COMPASS/figures/` — R manuscript figures for the ADT cohort, platinum endpoint. Run
  `Rscript COMPASS/notebooks/figures/run_figure_render.R` after `05_prep_figure_data.ipynb`; the `FIGURES`
  toggles at its top (or `COMPASS_FIGURE_GROUPS`) select the figure groups `cohort`, `llm`,
  `cohort_sensitivity`, `federated_univariable`, `federated_multivariable`,
  `transfer_survival` and `transfer_checklists`. Only the PDFs in
  `COMPASS/figures/retained_exports.txt` (the curated `Desktop/ADT` export) are rendered. See
  [`COMPASS/figures/README.md`](COMPASS/figures/README.md).
  `02_univariate.ipynb`'s final section loads all shared-landmark univariate results for both
  `arpi` and `adt`, filters each to nominal `p_value < 0.05`, displays every hit, and exports a
  separate `cox/nominally_significant_univariate_results.csv` beneath each arm's run directory.
- `compass_pipeline.py` and the figure notebook both point directly to the
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

There is no automated guard for this: a renamed or null-filled column surfaces as an empty
cohort or a silently-dropped feature further down the pipeline. An all-null
`NCI_PREFERRED_MED_NM` means going back to the upstream compile, not debugging COMPASS.

---

## Conventions & invariants (preserve these when editing)

1. **Each landmark has its own risk set; the split is shared across landmarks by default.**
   `build_prediction_inputs.py` builds an independent eligible risk set per landmark, but with the
   default `--split-mode shared` a single split is derived on the union of MRNs across all landmarks
   and reindexed onto each landmark's own risk set, so a patient eligible at multiple landmarks keeps
   the same train/valid/test label everywhere. `--split-mode independent` restores the legacy
   behavior of deriving a separate split within each landmark's risk set.
   `assert_split_consistent_across_landmarks` (`survival_common/helper.py`) enforces this in shared
   mode. The chosen split is written directly into each `aggregated_landmark{D}.csv` and also into
   `split_assignments_landmark{D}.csv`; models use the aggregated table's split and never re-split.
   `split_assignments.csv` remains a base-landmark compatibility copy for `build_genomic_inputs.py`,
   and in shared mode it is written from the shared split over the MRN union rather than from a
   single landmark. `split_mode` and per-landmark split sizes are recorded in `build_manifest.json`
   and `landmark_attrition.json`.
2. **Fit on the training block; never touch test for fitting.** Imputers, `StandardScaler`,
   canonical-lab selection, and Breslow baselines are all fit on train+valid (or fold-train inside CV) and
   applied to eval. Per-fold canonical labs are recomputed inside CV. The leakage guards
   `assert_no_test_leakage` / `assert_disjoint_folds` live in `survival_common/helper.py`.
3. **Canonical labs and horizon grids are training-block artifacts.** The main canonical lab set and
   AUC/Brier horizons are derived on **train+valid**; CV recomputes canonical labs on each fold-train.
   Evaluation data may only mask requested horizons that are not estimable—it never supplies
   replacement times. Do not derive the requested timeline from held-out test patients.
4. **Endpoint and duration:** COMPASS registers `(t_platinum, PLATINUM)`, `(t_nepc, NEPC)`,
   and `(t_avpc, AVPC)`; `platinum` is the default and the one the
   manuscript figures use. For patients without the event, the anchor time is filled with
   `t_last_contact` (censoring). After landmark rebasing, the validity filter requires duration
   `> 0`, which silently drops patients whose event falls before/at the landmark — add count
   logging if you depend on it. Each non-`platinum` endpoint's duration conditions are applied
   **only** when that endpoint is selected (`--endpoint nepc|avpc`, or the deprecated
   `--require-nepc` alias for `nepc`), so registering additional endpoints cannot change the
   platinum cohort; `tests/test_nepc_endpoint.py` asserts that equivalence.
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
7. **Three-layer torch gating.** `COMPASS/multivariable_models/longitudinal/` is the only torch consumer in this repo,
   and it must stay optional: (1) `survival_common/deephit_engine.py` does
   `try/except ModuleNotFoundError` at import time, falling back to `torch = None; Dataset = object` —
   the `Dataset = object` fallback matters because `class SequenceDataset(Dataset)` is evaluated at
   import time even when torch is absent; (2) `DynamicDeepHitGRU` is defined under `if nn is not None:`
   with an `else:` stub that calls `require_torch()` if ever instantiated; (3) `require_torch()` is
   called at the top of `run_deephit`, never at module import. Net effect: `dynamic_deephit.py --help`
   exits 0 and the full test suite imports cleanly with no torch installed.
8. **The genomic arm is SNV-only.** Both `build_genomic_inputs.py` files pin
   `--variant-types` to `SNV`, and `build_somatic_gleason_inputs.py` filters the
   `SOMATIC_FEATURE_MANIFEST.parquet` feature list through the `<GENE>_SNV` regex before use —
   that was the one path where CNV/SV features entered with no opt-in at all. **Exception:** in
   `build_somatic_gleason_inputs.py` (the Gleason + somatic available-case cohort, multivariate
   arms, clinical stratifiers, and their figures), TP53, PTEN, and RB1 are not SNV-only. Each is
   one `<GENE>_ANY` indicator, the max over every declared `SNV`/`DEL`/`AMP`/`SV` column for
   that gene. It is 1 if any class is called, 0 if every observed class is 0, and missing only if
   no class was observed. It replaces `<GENE>_SNV` for those three genes. PTEN and RB1 are
   canonically lost by deletion, so an SNV-only call misses most carriers. `GLEASON_SCORE`
   carries a different `feature_kind` value and passes through untouched; only the
   `somatic_binary` block is filtered. On the consumer side, `cox_aggregated.py` splits the two
   regexes it needs: `GENOMIC_FEATURE_RE` (`_SNV$`) selects the genomic feature set, and the
   broader `ANY_VARIANT_RE` (`SV|SNV|AMP|DEL`) excludes every variant column from the `labs`
   subset — narrowing only the first would have let CNV columns fall *into* the lab arm.
   The sequencing-indexed univariate screen does not use this hybrid: it runs twice, as
   `sequencing_snv` (`<GENE>_SNV` for every gene, unadjusted) and `sequencing_any`
   (`<GENE>_ANY` for every gene, with the per-patient altered-gene count
   `SOMATIC_N_GENES_ALTERED` added to every base Cox model).
   `ALL_VARIANT_TYPES` survives for the detection regex, and `genomic_variant_types` stays in the
   build manifest as provenance.

   > **Caveat.** No `INDEL`/`FRAMESHIFT` suffix exists anywhere in the schema, so whatever
   > produced `SOMATIC_WIDE_BY_SAMPLE.parquet` folded indels into `_SNV`. "SNV-only" is therefore
   > **small-variant-only**, not strictly point mutations. Confirm against the panel documentation
   > before making the stricter claim in a manuscript.
9. **One canonical metrics schema across model families.** Every `*_metrics_*.csv` — elastic-net
   Cox, XGBoost, and Dynamic-DeepHit alike — carries the identity block
   (`model`, `cohort`, `endpoint`, `landmark_days`, `config`), the counts
   (`n_train_val`, `n_test`, `n_events_train_val`, `n_events_test`), and the performance block
   (`train_val_c_index`, `test_c_index`, `train_val_mean_auc_t`, `test_mean_auc_t`,
   `test_integrated_brier`). `survival_common/metrics_schema.py` is the single definition;
   `order_canonical_first()` raises if a writer omits a column, so drift fails loudly at write
   time rather than surfacing as an all-NA figure panel. Families that score only the held-out
   block emit **NaN** train-side twins rather than omitting the columns, so readers can assume the
   columns exist. Hyperparameter columns (`selected_penalizer`, `xgb_params`,
   `selected_hidden_dim`, per-horizon `*_auc_h*`) are family-specific and deliberately not
   unified — "identical columns" means the shared metric block, not the tuning record.
   `tests/test_metrics_schema_consistency.py` is the regression guard.

   The per-family `cv_summary` files stay outside this schema on purpose: they have genuinely
   different shapes (DeepHit is competing-risks and legitimately per-event).
   `landmark_day` (singular) also remains correct in the per-landmark CV-fold, patient-risk,
   feature-importance, and run-manifest frames, which are separate files.

   The dynamic arm respects this rather than widening it. Its
   `dynamic_deephit_dyn_metrics_{config}.csv` carries the canonical block with **one row per
   cause**, at `prediction_time == landmark_time` — the row directly comparable to Cox/XGBoost.
   The per-prediction-time series goes to a *separate*
   `dynamic_deephit_dyn_metrics_by_time_{config}.csv`, precisely so the canonical schema never
   gains a prediction-time axis. `incremental_risk.py`'s outputs are named
   `incremental_risk_*.csv`, deliberately outside the `*_metrics_*.csv` namespace this invariant
   governs: they report a *paired* comparison indexed by prediction time, not one model's
   held-out performance. A prediction time whose risk set cannot support a metric gets NaN
   metrics and a `note` beginning `underpowered:` rather than being dropped — a gap in the series
   should say why it is missing.
10. **Two longitudinal input frames; on only one of them is the landmark `max(TIME)`.**
    `longitudinal_landmark{D}.csv` ends at the landmark, so `max(TIME) == landmark_time` per
    patient and code can infer the landmark that way. `longitudinal_full_landmark{D}.csv`
    (written by `--longitudinal-full-followup`, **alongside** the landmark frame, never in place
    of it) runs to end of follow-up, which breaks that identity: `max(TIME)` becomes the
    patient's last visit. Anything anchoring to it there shifts per patient by the length of that
    patient's own post-landmark history — silently, and with plausible-looking output. On the
    full frame, read the explicit `landmark_time` column (`build_person_period_wide` does). The full frame's manifest
    carries `include_post_landmark: true`, and consumers cross-check it against their own mode
    and refuse a mismatch **in either direction** — post-landmark labs are legitimate inputs only
    for predictions made *after* the landmark.


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

## Arms, cohorts, and endpoints

A run is a (arm, cohort, endpoint) triple. The arm sets time 0, the cohort restricts which
patients are modelled, and the endpoint sets the event. Neither the cohort nor the endpoint forks
cohort assembly — Stages 0-2 are independent of both and write one survival cohort carrying every
endpoint's columns. Stage 3 is where the fork happens, crossing
**3 cohorts x 4 endpoints x 3 landmarks** for the ADT arm.

### Cohorts

| Cohort | Restriction | Label / trees |
| --- | --- | --- |
| `all` | none beyond the arm's Stage-1 survival cohort | `adt` -> `prediction_inputs_adt`, `local_runs_adt` |
| `metastatic_adt` | `ADT_INTENT == METASTATIC` | `adt_metastatic_adt` -> `prediction_inputs_adt_metastatic_adt`, ... |
| `metastatic_llm` | LLM `has_metastatic_disease == true` | `adt_metastatic_llm` -> `prediction_inputs_adt_metastatic_llm`, ... |
| `localized` | `ADT_INTENT == LOCALIZED_ADJUVANT` (retired, not in `DEFAULT_COHORTS`) | `adt_localized` -> ... |
| `nonmetastatic_llm` | LLM `has_metastatic_disease == false` (retired, not in `DEFAULT_COHORTS`) | `adt_nonmetastatic_llm` -> ... |

The modelled stratified cohorts come from two independent label sources: `metastatic_adt` is a
**medication-derived ADT-intent stratum**, and `metastatic_llm` is the **LLM-adjudicated
metastatic status** from the `met_diagnosis` pipeline. Stage 1b
(`cp.build_adt_intent_mrn_lists()`, in notebook `01`) writes one MRN list per stratum plus a
preliminary incident-endpoint count table under `<data_root>/mrn_lists/`; Stage 3 consumes those
lists through `--restrict-to-mrns`, which is the single channel by which stratification enters
the pipeline. `COHORT_SPECS` in `compass_pipeline.py` is the registry.

> **`ADT_INTENT` is retrospective.** It is derived from the full observed ADT course — duration,
> cessation/restart, and later definitive escalation — so it is *not* a metastatic-status label
> available prospectively at the landmark. Platinum is excluded from the classifier, but
> cohort-stratified results still describe patients grouped by how their treatment eventually
> unfolded. Say so wherever these results are reported.

Localized event counts are thin for the incident endpoints. Read Stage
1b's count table before spending modelling time; the runners' insufficient-events guards handle
the rest.

### Endpoints

| Endpoint | Duration / event | Source of the event date |
| --- | --- | --- |
| `platinum` | `t_platinum` / `PLATINUM` | First carboplatin/cisplatin exposure in `MEDICATIONS` |
| `nepc` | `t_nepc` / `NEPC` | `LLM_annotations/LLM_primary_subtype/primary_subtype_labels.parquet`: `nepc_status == "nepc"` dated by `nepc_date` (any gated NE finding, independent of AVPC) |
| `avpc` | `t_avpc` / `AVPC` | Same file: the AVPC runner's `is_avpc` dated by `avpc_date` (no criteria-count threshold applied here; independent of NEPC) |

`nepc` and `avpc` read the **same** primary-subtype labels file, a full join of the
`LLM_clinical_annotations` `tasks/nepc` and `tasks/avpc` runner outputs, through a single loader
(`compile_COMPASS_cohort_data.py`'s `load_subtype_labels`). A patient who is NEPC and also
`is_avpc` counts as an event for both endpoints. Unresolved or absent patients are censored (0),
with the upstream review status kept in `NEPC_LABEL_SOURCE`/`AVPC_LABEL_SOURCE`.

Four properties of `nepc`/`avpc` differ from `platinum` and matter for interpretation:

- **They are incident endpoints, gated per-endpoint.** `--require-nepc` (legacy alias for
  `endpoint=nepc`) adds `t_nepc notna` and `t_nepc > 0` to the `make_outcome_df` validity
  conditions; `avpc` gets the analogous `t_avpc` gate. Each drops
  patients whose event falls at or before the landmark (prevalent, not incident). The gate is
  **off by default** so it can never silently shrink the platinum cohort — `compass_pipeline`
  passes it only when the endpoint matches. The consequence is that these endpoints' cohorts are
  **not the same patients** as `platinum`'s, nor as each other's; per-landmark attrition is
  reported in the Stage 3 build log.
- **The endpoints and the Figure 2 strata share one label snapshot.** Figure 2's `has_nepc` /
  `has_avpc` / `primary_label` strata are derived from the same `primary_subtype_labels.parquet`
  (`has_nepc = nepc_status == "nepc"`, `has_avpc = is_avpc`). The only differences are that the
  endpoints censor unresolved patients as 0 and demote undated positives, whereas the figures drop
  unresolved patients and do not need dates.
- **Event counts may be low**, especially for `nepc`. Check
  the Stage 1 summary and the `02a` notebook's count table *before* spending modelling time; `02a`
  prints an explicit underpowered warning below 50 events.
- **Date provenance is carried, not filtered.** Endpoint-specific NEPC provenance rides through
  to prediction inputs; joint AVPC_NEPC provenance remains in the Stage-1 audit cohort only. Two are
  worth knowing: `date_source = note_date` means *earliest documentation*, not onset; and
  censored patients whose `NEPC_LABEL_SOURCE`/`AVPC_LABEL_SOURCE` is `failed`, `not_run` or
  `missing_from_labels` were never resolved by an LLM — kept as censored observations, but
  distinct from adjudicated negatives.

If `primary_subtype_labels.parquet` is not mounted, Stage 1 warns and emits no NEPC/AVPC/AVPC_NEPC
columns. Every downstream touchpoint for the two modeled LLM-derived endpoints is presence-guarded, so the
platinum path runs unchanged.

### Longitudinal configs

`03b`'s Dynamic-DeepHit arm uses a **second, related registry**:
`survival_common/longitudinal_targets.py`'s `LONGITUDINAL_CONFIGS`. It selects a cause of
interest plus an optional competing cause, which is a finer axis than `ENDPOINTS`' single
(duration, event) pair — hence `--config`, not `--endpoints`:

| `--config` | causes (label 1, label 2) | AUC(t) grid read |
| --- | --- | --- |
| `platinum` | platinum; death censored | `platinum` |
| `competing` | platinum, death | `platinum` |
| `nepc` | nepc; death censored | `nepc` |
| `nepc_competing` | nepc, death | `nepc` |
| `avpc` | avpc; death censored | `avpc` |
| `avpc_competing` | avpc, death | `avpc` |

`CONFIG_ENDPOINTS` maps each config to the `ENDPOINTS` key whose horizon grid it scores on, so a
NEPC model is evaluated on the NEPC timeline and stays comparable to the NEPC Cox/XGBoost arms
(README invariant #6). `compass_pipeline.longitudinal_task_specs(endpoint)` picks the pair for the
selected endpoint, so `03b` needs only `ENDPOINT` set — the same knob as `02`/`03`.

Two ordering invariants are asserted at import: the cause of interest is always label 1 and death
always label 2 (a reorder would silently reinterpret every downstream risk column).

The cause-only configs (`platinum`, `nepc`, `avpc`) censor at death and are the ones
comparable to Cox/XGBoost for that endpoint; `summarize_longitudinal_outputs` filters to that row.
The competing configs' death rows are written to disk but excluded from the summary.

### Running additional endpoints

Set the parameter cell of `01`/`02`/`03`/`03b`, then run each top to bottom:

```python
ARMS = ["adt"]
ENDPOINTS = ("platinum", "nepc")   # any subset of cox_aggregated.ENDPOINTS
COHORTS = ("all", "metastatic_adt", "metastatic_llm")   # any subset of cp.DEFAULT_COHORTS
EXCLUSIONS = ("none", "pre_adt_castrate")
OVERWRITE = False

cp.FORCE_RERUN = OVERWRITE
RUNS = cp.make_endpoint_runs(ARMS, endpoints=ENDPOINTS, cohorts=COHORTS, exclusions=EXCLUSIONS)
```

This is the notebook default: 2 endpoints x 3 cohorts x 2 exclusions = a 12-run `RUNS` list per
arm. To iterate faster, narrow any tuple — the notebooks' `for run in RUNS:` loops are unchanged
either way.

`make_endpoint_runs(arms, *, endpoints=..., cohorts=DEFAULT_COHORTS, prediction_input_dirs_by_endpoint=None)`
builds one independent input/output tree per requested (cohort, endpoint) pair by calling
`make_runs` once per pair under the hood, with `output_suffix` set to `""` for `platinum` and
`f"_{endpoint}"` for everything else (e.g. `"_nepc"`, `"_avpc"`), and the cohort's
suffix composed onto the arm label (`""`, `"_metastatic_adt"`, `"_metastatic_llm"`). Each notebook's
`for run in RUNS:` loop iterates every (arm, cohort, endpoint) triple produced. Every notebook
calls `make_endpoint_runs` directly; the single-run `cp.make_runs(..., output_suffix=..., cohort=...)`
is the lower-level helper it wraps.

Three helpers keep the wider cross from leaking into the notebooks:

- `cp.endpoint_suffix(endpoint)` — the one definition of the suffix rule, which used to be
  inlined in four places.
- `cp.run_key(run)` — `(cohort, endpoint, label)`, for summary dicts that must stay unique now
  that a label alone no longer identifies a run.
- `cp.stage2_runs(runs)` — one run per treatment anchor. Lab preprocessing is independent of both
  cohort and endpoint, so notebook `01`'s Stage 2 iterates this rather than filtering `RUNS` by
  hand; the old `if run["endpoint"] != "platinum": continue` guard would have run Stage 2 once per
  cohort as soon as cohorts were added.

The suffix pattern suffixes **both** `prediction_inputs_<arm>` and `local_runs_<arm>`. Both are
required: because each endpoint's incident gate (`--require-nepc` and its `avpc`
analogues) changes which patients survive the landmark filter, and that filter runs at
*preprocessing* time inside `build_landmark_merged`, each optional endpoint needs its own inputs
tree — not just its own output tree. With the suffix set, `prediction_inputs_adt/` and
`local_runs_adt/` (the unsuffixed `platinum` tree) are never touched by an `nepc` or `avpc` run.

Stages 1-2 of `01` (cohort compile, lab preprocessing) only need running once; they
are shared across all endpoints. Re-run Stage 3 onward per endpoint — `RUNS` already contains one
entry per (arm, endpoint) pair, so the existing `for run in RUNS:` loops handle this.

`03b` takes the same `ENDPOINTS` tuple and derives each endpoint's `--config` values from it via
`cp.longitudinal_task_specs(endpoint)` (see [Longitudinal configs](#longitudinal-configs)); it
additionally has `RUN_DYNAMIC`, left off by default, and an optional
`PREDICTION_INPUT_DIRS_BY_ENDPOINT` override for cluster-mounted prediction-input paths, passed
through as `make_endpoint_runs`'s `prediction_input_dirs_by_endpoint`.

## Recommended run order

Run each notebook top to bottom; select ARPI/ADT with each Python notebook's `ARMS` setting, the
patient subsets with `COHORTS`, and the event(s) with `ENDPOINTS`
(see [Arms, cohorts, and endpoints](#arms-cohorts-and-endpoints)):

1. `COMPASS/notebooks/manuscript/01_preprocessing.ipynb`
2. `COMPASS/notebooks/manuscript/02_univariate.ipynb`
3. `COMPASS/notebooks/manuscript/02a_cohort_comparison.ipynb` (read-only; only after steps 1-2 have
   been run for **all** cohorts and endpoints being compared)
4. `COMPASS/notebooks/manuscript/03_multivariate.ipynb`
5. `COMPASS/notebooks/manuscript/03b_multivariate_longitudinal.ipynb` (optional — requires torch;
   see [Dependencies](#dependencies) and `COMPASS/multivariable_models/longitudinal/README.md`)
6. `COMPASS/notebooks/manuscript/04_federated_transfer.ipynb` (scores the frozen federated bundles
   on the local endpoints; needs the `rhino_data_v2/` delivery)
7. `COMPASS/notebooks/manuscript/05_prep_figure_data.ipynb` (Polars figure tables; reads step 6's
   transfer outputs)
8. `Rscript COMPASS/notebooks/figures/run_figure_render.R` (figure-group and manuscript-only toggles are in `COMPASS/figures/figure_run.R`; manuscript-only defaults to on for both R preparation and rendering)

The notebooks pass `PROFILE_DATA/*.parquet` paths explicitly to the lower-level scripts. Existing
hand-curated `LLM_NEPC_labels/` inputs remain under the shared `COMPASS` data root.

## Dependencies

No packaged environment is checked in. Assumed: `pandas`, `numpy`, `scipy`, `tqdm`, `scikit-learn`,
`scikit-survival` (`sksurv`), `xgboost`, `lifelines`, `matplotlib`. Python **3.10+** is recommended
for the modern type-hint syntax used by the shared modules.

`torch` is an **optional** dependency, needed only for
`COMPASS/multivariable_models/longitudinal/` (Dynamic-DeepHit). It stays commented out in `requirements.txt` — see that file's
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
- **`nepc` and `avpc` are likely event-poor, and neither's cohort is the platinum cohort.** Both
  come from the same `primary_subtype_labels.parquet` (`nepc` = `nepc_status == "nepc"`; `avpc` =
  the AVPC runner's `is_avpc`). Each endpoint's incident gate
  (`--endpoint nepc|avpc`, or the deprecated
  `--require-nepc` alias for `nepc`) further removes prevalent events, so event counts can be far
  smaller than platinum's at the same landmark. Two consequences: multivariate results may be
  underpowered rather than negative (`02a` warns below 50 events), and **platinum-vs-nepc/avpc
  metric differences are confounded by cohort composition** — they are not a
  like-for-like model comparison. Check each endpoint's own event count against the 50-event floor
  before treating it as adequately powered.
- **One LLM subtype source, two NEPC tiers.** The `nepc` and `avpc` endpoints and the Figure 2 /
  Figure 2v2 / Figure 2v3 strata all read `primary_subtype_labels.parquet` (built by
  `notebooks/primary_subtype.ipynb` in the sibling `LLM_clinical_annotations` repo). The AVPC_NEPC
  union remains audit metadata, not a model endpoint. The file also carries a stricter
  stated-diagnosis NEPC tier (`has_stated_nepc_dx`, surfaced as the audit column
  `NEPC_STATED_DX`); it is narrower and more precision-biased than `nepc_status == "nepc"` and is
  not used by any endpoint or figure. Endpoint and figure counts still differ (censored vs dropped
  unresolved patients, undated demotion, landmark gating), so always name which one a given
  number came from.
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

- **`auc_max_time_units = 522` is the default** (`DEFAULT_AUC_MAX_TIME_UNITS` in `survival_common/helper.py`; ceil(3650 / 7), matching the 10-year administrative censoring)
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
- **Dynamic-DeepHit's CV grid is expensive**: 27 hyperparameter combos × 5 folds, run serially per
  (landmark, config) by `compass_pipeline._run_tasks`. Reduce the grid for a first pass and rely on
  `FORCE_RERUN = False` to resume.

### Low impact / cleanliness

- `iterrows`/`apply`-based row loops in `data_preprocessing_common/dfci_labs.py` and `longitudinal_data_processing.py`
  are slow on full DFCI-scale pulls — vectorize if performance bites.

---

## Scoring federated models on local data

The federated runs in `caia-project-compass/rhino_scripts` fit XGBoost,
elastic-net Cox, DeepHit and predictive checklist models across sites and export each fitted model as a
`caia-federated-model-bundle` JSON. Two modules here consume those bundles:

| Module | Takes | Does |
| --- | --- | --- |
| `survival_common/federated_inference.py` | a wide frame for Cox, XGBoost and checklists; ordered long histories and a cohort frame for DeepHit | applies frozen preprocessing and returns risk or survival; can also restore the trained estimator |
| `survival_common/federated_scoring.py` | the **long-format** table a site actually holds | builds each landmark's cohort and wide frame by the federated rules, scores the selected models, reports coverage and local C-index |

Use `federated_scoring` on the cluster; it is the one with a CLI over raw data:

```bash
python -m survival_common.federated_scoring \
    --bundle xgboost_federated_model_adt.json \
           cox_federated_elasticnet_model_adt.json \
    --data  /data/gusev/.../compass_long.parquet \
    --output-dir risk_scores/ \
    --evaluate --horizons 365,730,1095
```

Outputs, per bundle plus two combined tables:

- `<bundle>_risk_scores.csv` — one row per patient per model: `risk_score`
  (log-hazard ratio, higher = higher risk), the outcome columns when the local
  rows carry one, and `survival_<h>d` when the bundle has a Breslow baseline
  (elastic-net only; the XGBoost export carries none, so it gives relative risk only).
- `federated_inference_metrics.csv` — per model: cohort counts, risk mean/SD,
  `n_covariates_absent_locally`, and `c_index` under `--evaluate`.
- `federated_inference_coverage.csv` — per covariate: whether it exists locally
  and how often it is observed.
- `federated_inference_run.json` — the exact invocation, for provenance.

To get at the trained object itself rather than a score:

```python
from survival_common.federated_inference import load_fitted_model

booster, bundle, model = load_fitted_model("xgboost_federated_model_adt.json", 90)
booster.trees_to_dataframe()            # a real xgboost.Booster

cox, bundle, model = load_fitted_model("cox_federated_elasticnet_model_adt.json", 90)
cox.params_                             # coefficients by covariate name
```

### Transporting the platinum models to local NEPC / AVPC

The federated bundles are trained on time to platinum only (NEPC/AVPC are not
annotated at the other sites). `survival_common/federated_endpoint_transfer.py`
scores the bundles refit without DFCI (`federated_*_no_DFCI/`) on the full DFCI
ADT cohort and evaluates the same risk score against the local LLM-derived NEPC
and AVPC timelines, with platinum as the reference:

```bash
RHINO=/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/rhino_data_v2
python -m survival_common.federated_endpoint_transfer \
    --bundle $RHINO/federated_elastic_net_cox_no_DFCI/cox_federated_elasticnet_model_adt.json \
    --data /data/gusev/USERS/jpconnor/data/CAIA/COMPASS/longitudinal_prediction_data_adt.csv \
    --output-dir $RHINO/federated_transfer_adt_no_DFCI/ \
    --n-bootstrap 1000
```

The CLI keys bundles by file name and rejects duplicates. The per-landmark
XGBoost, DeepHit and checklist deliveries all share one file name per family,
so score those through 04 (below), which merges them first.

All federated inputs live in the `rhino_data_v2/` delivery under the COMPASS
data root (override with `COMPASS_RHINO_ROOT`):

| Path | Content |
| --- | --- |
| `cox_federated_univariate_adt.csv` | pooled univariate Cox across the four sites |
| `within_site_cox_univariate/` | the same fits within each site |
| `federated_elastic_net_cox_{w,no}_DFCI/` | elastic-net Cox trained with and without DFCI; one bundle holds landmarks 0/90/180 × full/age-only |
| `feature_set_models/{w,no}_DFCI/predictive_checklists/` | the default checklist: adjust-only (feature-set) refit, one bundle and metrics file for all landmarks |
| `federated_predictive_checklists/lm_{0,90,180}/{w,no}_DFCI/` | the regular checklist, one bundle and its metrics per landmark; read only by Figure S1 |
| `federated_xgboost*/lm_<days>/{w,no}_DFCI/`, `federated_*deephit*/lm_<days>/{w,no}_DFCI/` | landmark XGBoost and DeepHit, delivered like the checklists (optionally under `adt/`) |
| `federated_transfer_adt_no_DFCI/` | 04's output |
| `figure_data/` | aggregate KM curves and trajectories; see below |

To read the earlier `rhino_results/` delivery instead, set `COMPASS_RHINO_ROOT`
to it, `COMPASS_FEDERATED_RESULTS` to its `federated_model_results/` CSV, and
`COMPASS_FEDERATED_FIGURE_DATA` to its `misc_figure_data/`.

`COMPASS/notebooks/manuscript/04_federated_transfer.ipynb` wraps the same call for
`rhino_data_v2`. It scores elastic-net Cox, XGBoost, DeepHit and the predictive
checklists from the bundles trained on Fred Hutch, Johns Hopkins and MSK only. The
bundles trained with DFCI are not scored (DFCI patients would be in-sample).

- **Discovery.** Bundles are found by file name anywhere under `RHINO_ROOT`, in a
  `no_DFCI` or `*_no_DFCI` directory. The exact XGBoost and DeepHit folder names
  therefore do not matter. A family that has not been delivered yet is skipped
  with a warning.
- **Merge.** `federated_inference.merge_bundles` combines each family's
  per-landmark bundles into one. Format version, model family, analysis label,
  endpoint, time unit and library version must agree, and a landmark exported
  twice is an error.
- **Age-only comparator for XGBoost.** The landmark XGBoost packages export no
  age-only model, so XGBoost's paired Δ is taken against the elastic-net Cox
  `baseline` (`baseline_fallback`; recorded in `baseline_bundle`). An age-only
  model's C and AUC(t) depend only on how it ranks age, so the comparator is the
  same.
- **First-delivery fallback.** If `rhino_data_v2` has no no-DFCI XGBoost bundle,
  04 scores `rhino_results/federated_model_results/federated_xgboost_no_DFCI/` instead.
  `COMPASS_RHINO_V1_ROOT` sets the earlier delivery's root, and
  `V1_FALLBACK_FAMILIES` sets which families may fall back; `()` turns this off.
  That bundle keeps its own age-only model if it has one. The bundle summary and
  `transfer_run.json` (`delivery`) record which delivery each family came from.
- **Comparison table.** The notebook shows each family's federated held-out test
  metrics next to the local platinum ones. Those are read from the metrics CSV
  beside each source bundle.

The transfer module maps local `LAB_NAME` to the OMOP names via `OMOP_to_DFCI_lab_ids.csv`, puts
each endpoint's date in the federated chain's event slot, and reuses
`federated_scoring.build_landmark_frame` unchanged, so events at/before the
landmark are dropped (incident risk set) and follow-up is censored at 3650 days.
NEPC/AVPC are censored at last contact (death is not a competing event).
Outputs: `transfer_metrics.csv` (mean AUC(t) + bootstrap CI and paired ΔAUC(t) vs the
age-only baseline, which the figures plot; Harrell C + CI, paired ΔC, Uno's C, HR per SD
of risk),
`transfer_auc_t.csv`, `transfer_scores.csv`, `transfer_coverage.csv`, and
`transfer_unit_check.csv`. The last one compares local feature means with the
bundle's pooled means; a flagged row (>3x off) almost always means a unit
mismatch between the DFCI and OMOP lab pipelines. Resolve those before you read
the discrimination metrics.

The no-DFCI bundles never saw a DFCI patient, so this is an external
validation on all three endpoints. 04 writes `transfer_metrics.csv` and
`transfer_auc_t.csv` to `COMPASS_FEDERATED_TRANSFER_NO_DFCI` (default
`$COMPASS_RHINO_ROOT/federated_transfer_adt_no_DFCI`). These primary tables hold
elastic-net Cox and XGBoost only, one row per endpoint, bundle, landmark and
config, as Figure 5 requires. The sequencing-clock run and the Gleason /
TP53-RB1-PTEN clinical comparison (Figure 5 b–h) also use those two families. Without
an age-only XGBoost model, XGBoost's `transferred_age` rows in the clinical
comparison stay empty, and its other comparisons are unaffected. Rerun
`05_prep_figure_data.ipynb` after 04, then
`COMPASS_FIGURE_GROUPS=transfer_survival Rscript COMPASS/notebooks/figures/run_figure_render.R`
to render `figures/ADT_figures/05_transfer_survival/transfer_discrimination_no_dfci__platinum.pdf` (table in `figures/ADT_figure_data/05_transfer_survival/`
(mean AUC(t) and paired ΔAUC(t) vs. age only, by endpoint) and federated manuscript
Figure 5.

The feature-set predictive checklists are scored in a second pass on the same
frames and bootstrap resamples, each at its exported horizon; their patient
flags feed Figure 6. DeepHit, the regular checklists and display-only summaries
are not run here because no wired figure reads them.
The `training_arm`, `model_horizon_days`, and `evaluation_horizon_days` columns
identify each prediction. Checklist `risk_score` is the raw item count;
`probability` and `flag` are the frozen calibrator and decision output.
The horizon-specific rows also report an IPCW Brier score for DeepHit and
checklist probabilities, with censoring estimated on the DFCI cohort.
For a CLI run, pass the additional JSON paths to the existing `--bundle` list,
one bundle per file name. Only downloaded bundles from runs that excluded DFCI
belong in this comparison.
After rerunning `05_prep_figure_data.ipynb` and `COMPASS/notebooks/figures/run_figure_render.R`, the dynamic arm's
held-out federated metrics appear in the multivariate main figures when the
corresponding DeepHit metrics CSV is available. The figures still render with
Cox and XGBoost alone when it is absent. Transferred predictive checklist
results are not added to the main figures.

The same federated run also renders the pooled federated models themselves,
from the exports trained with DFCI next to the univariate CSV. Elastic-net Cox
comes from `federated_elastic_net_cox_w_DFCI/`; XGBoost and DeepHit come from
their `lm_<days>/w_DFCI/` files, which are stacked into one table. The outputs are
`figures/ADT_figures/04_multivariate/federated/{elasticnet,xgboost}_performance__platinum`,
`elasticnet_coefficients__platinum`, and `xgboost_importance__platinum`.
Manuscript Figure 8 (`08_federated_multivariable`) puts them in the local
Figure 4 layout. The `no_DFCI` exports render as `*_no_dfci__platinum`.
XGBoost exports no age-only model, so its baseline series is left empty.

The default predictive checklists trained with DFCI
(the feature-set refits in `feature_set_models/w_DFCI/predictive_checklists/`) render as
`figures/ADT_figures/04_multivariate/federated/checklist_supplement__platinum`. It has
one column per landmark (0/90/180 days) and three rows:

- a: held-out test AUC(t) at 1 and 5 years, with 95% intervals, for the checklist and the age-only checklist;
- b: the 1-year checklist cards;
- c: the 5-year checklist cards.

Each card gives the M-of-N rule, its items, the held-out n and events, AUC and
Brier for both models, and the flag operating point. Held-out metrics are pooled
over the four sites' test patients; the export has no per-site breakdown. The
`_items` and `_metrics` CSV sidecars keep exact thresholds and every interval.
The no-DFCI checklists are shown only in the transfer supplement.

Three resolution rules apply:

- An export in the earlier layout, with the file directly in
  `federated_cox_multivariate{,_no_DFCI}/`, `federated_xgboost{,_no_DFCI}/` or
  `{fast_,}federated_deephit{,_no_DFCI}/`, wins over the per-landmark files.
- XGBoost metrics/importance missing from `rhino_data_v2` are read from the first
  delivery, `rhino_results/federated_model_results/federated_xgboost{,_no_DFCI}/`
  (`COMPASS_RHINO_V1_ROOT`), as 04 does for the transfer bundles. The panel captions
  say so. Without XGBoost, the federated Figure 4 and Figure 8 are skipped.
- A landmark found twice fails as a duplicate. 05 fingerprints every export it
  finds, so an export delivered after 05 last ran makes 07 stop and ask for a
  rerun of 05.

`figure_data/` holds aggregate KM curves (`cohort_km.csv`,
`lab_extreme_{km,summary}.csv`) and binned PSA/testosterone trajectories
(`lab_trajector{ies,y_cohort}.csv`), each pooled across the four sites and, in
the `within_site_*` twins, per site, plus a `manifest.json` with the binning
settings and per-site counts. There are no patient-level rows. The federated run
draws them with the local trajectory and KM plotters as
`federated_trajectory_<lab>`, `federated_km_quintile_<lab>_landmark<N>` and
per-site `_by_site` pages (`COMPASS_FEDERATED_FIGURE_DATA` overrides the folder).

The federated job also writes a separate manuscript set, each figure in the `manuscript/`
subfolder of its section (`03_univariate/federated/`, `04_multivariate/federated/`,
`05_transfer_survival/`, `06_transfer_checklists/`), with captions in `manuscript/captions/federated_figureN.{md,txt}`:

| File | Content |
|---|---|
| `federated_03_univariable` | Figure 3 layout: pooled univariable Cox volcanoes at days 0/90/180 (with a log-HR direction arrow under the key), pooled PSA/testosterone trajectories and day-180 extreme-decile KM curves (site `extreme_fraction`, 10% by default) |
| `federated_04_multivariable` | Pooled four-site models; excludes the XGBoost age baseline and ranks measured features after excluding observation counts and missingness indicators |
| `federated_04_multivariable__full_features` | Complete model comparison and all supplied coefficients and gains, including observation counts and missingness indicators |
| `federated_05_transfer_no_dfci__no_d` | Panels a–c only (no KM curves or probability panel); caption `federated_figure5_no_d` |
| `federated_06_checklist_no_dfci__lm{0,90,180}_h{365,1825}` | Figure 6 for each landmark × horizon, from `checklist_grid/` (notebook 04); copied with the main and supplement figures to `ADT_figures/manuscript_renders/` |
| `federated_06_checklist_guideline_no_dfci__lm{0,90,180}_h{365,1825}` | Generalized (guideline) version of the grid: same checklists with item thresholds rounded (<1 → nearest 0.25; single digits → whole number; tens → multiple of 5; hundreds → 10; thousands → 100, and so on), from `checklist_grid_guidelines/` (notebook 04); copied to `ADT_figures/manuscript_renders/supplement/generalized_federated_checklist_transfer/` |
| `S7_full_checklist_transfer` | Paired generalized and direct-transfer checklist bars for full Dana-Farber ADT-cohort horizon-specific IPCW AUC (balanced accuracy for the binary rule), with bootstrap confidence intervals, by endpoint, landmark and horizon. Requires full-cohort accuracy exports for both grids from notebook 04. |
| `federated_06_checklist_no_dfci__extension_lm{0,90,180}` (S8) | Reviewer extension: the generalized checklist is applied frozen, exactly as transferred, and its yes/no result becomes one item of internal checklists learned on Dana-Farber platinum by 5-fold cross-fitting, with a Gleason item (≥ 7/8/9) and/or TP53, RB1, PTEN (separate items, or any of the three); the lab-checklist item is forced into the "Labs + …" checklists and the federated model is not re-tuned (`federated_models/checklist_extension.py`, last cell of notebook 04 → `checklist_extension/`). Out-of-fold horizon AUC and paired ΔAUC vs the transferred lab checklist per endpoint × cohort (Gleason available, ADT clock; sequenced, sequencing clock; sequenced with Gleason) × horizon, one page per landmark; copied to `ADT_figures/manuscript_renders/supplement/S8_checklist_extension/` |
| `03_cohort_sensitivity_supplement`, `adt_vs_llm__platinum__all__incl` | Copied to `ADT_figures/manuscript_renders/supplement/` as `S3_cohort_sensitivity.pdf` and `S2_llm_metastatic_vs_adt_intent.pdf`, respectively. Figure 1 (`01_cohort_overview`) has panels a-e only and is copied as `S1_cohort_overview.pdf`. Checklist transfer figures gain a full-DFCI-cohort accuracy panel (b) from `transfer_checklist_full_accuracy*.csv` (rerun notebook 04); older exports fall back to the three-panel layout |

The local Figure 3 (`03_dfci_univariable`) still uses DFCI patient-level data.

### Invariants (these change predictions silently if broken)

- **The bundle's preprocessing is pooled across the federation and is reused
  verbatim.** Re-fitting an imputer or scaler on local data would centre the
  features on the local cohort and shift every prediction, while still returning
  a plausible-looking score. `federated_inference` never refits; don't add it.
- **Covariate order is load-bearing.** XGBoost scores by position, so a reordered
  design matrix returns wrong risks without raising.
- **Match the cohort arm.** A bundle whose `analysis_label` ends in
  `noprecastrate` was trained after the pre-anchor castrate exclusion, so score
  it with `--exclusion pre_anchor_castrate`. The CLI warns on an obvious mismatch.
- **`federated_scoring` is a second copy of the federated feature derivation**
  (`rhino_scripts/*/preprocessing.py`), because the bundle carries the pooled
  *transform* but not the *derivation*. Two copies drift. After pulling
  caia-project-compass, re-run the cross-repo diff:

  ```bash
  python tests/check_federated_preprocessing_parity.py
  ```

  It imports the federated modules and asserts the wide frames match exactly,
  over both cohort arms and all three landmarks; it exits non-zero on any
  difference. `tests/test_federated_scoring.py` pins the same rules from this
  side and runs in the normal suite.
- **A high `n_covariates_absent_locally` means the extract is wrong**, usually a
  lab dictionary that does not match the training site's. Absent covariates are
  imputed to the pooled train mean, which is legitimate for a few features and
  meaningless for most of them. Check the coverage table before trusting a score.

---

## Notes

- `.ipynb_checkpoints/` and `__pycache__/` artifacts are git-ignored and not part of the workflow.
- Several scripts read and write data **outside** the repository root (the `/data/gusev/...` cluster paths).
