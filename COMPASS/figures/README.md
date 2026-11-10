# COMPASS figures

Manuscript figures for the ADT cohort, platinum endpoint. The R stages prepare and render the manuscript figures as PDF and SVG under
`<COMPASS_FIG_ROOT>/ADT_figures/main/` and `supplement/`.

## Running

1. For the standard figure groups, run `COMPASS/notebooks/05_prep_figure_data.ipynb`.
   It calls `prepare_figure_data.py` to scan the cohort data and federated
   deliveries into Polars tables plus a manifest under `COMPASS_FIGURE_DATA_ROOT`.
2. Run the two R entry scripts (both in `COMPASS/notebooks/`; the code they source
   stays in this folder):

   ```sh
   Rscript COMPASS/notebooks/final_figure_data_prep.R   # R-side data prep / cache
   Rscript COMPASS/notebooks/run_figure_render.R        # draw PDF and SVG figures, write figure data
   ```

   Flip the `FIGURES` toggles at the top of `figure_run.R` to run a subset, or
   override them without editing the file:

   ```sh
   COMPASS_FIGURE_GROUPS=transfer_survival,transfer_checklists Rscript COMPASS/notebooks/run_figure_render.R
   ```

   Preparation computes the source panels and tables needed by composites, but
   saves and renders only manuscript scenes. After changing the figure code, rerun
   the preparation stage before rendering so the scene cache is current.

The R code never calls Python. Missing or changed prepared inputs stop with a
message to rerun notebook 05 (notebook 07 for `checklist_augmentation`).

## Figure groups

For the four Results blocks in `COMPASS_ESMO_AI_proposed` (9 November 2026),
use the separate poster entry point:

```sh
Rscript COMPASS/figures/ESMO_AI_Figures.R
```

In an R session, `source("COMPASS/figures/ESMO_AI_Figures.R")` or RStudio's
Source button also runs generation. The script prints the output directory,
preparation progress, and each saved figure set. For tests or function-only
loading, set `options(compass.esmo.autorun = FALSE)` before sourcing, then call
`esmo_main(character())` explicitly when ready to generate.

It reads the existing analysis outputs directly; run notebooks 01–04 and have
the federated Cox/KM/model deliveries available first. It does not require an
R scene cache or modify the manuscript exports. The default destination is
`<COMPASS_FIG_ROOT>/ESMO_AI/results_{1,2,3,4}/`, with a composite and individual
panels as PDF only, aggregate CSVs in `data/`, and `caption.txt`.
No patient IDs or patient-level tables are exported. Results are calculated
from the inputs, so updated analyses can differ from the proposal's old figures.

| Results block | Panels |
| --- | --- |
| 1 | Manual-review confusion matrix, subtype fractions by platinum status, AVPC/NEPC platinum enrichment with Wilson CIs and one-sided Fisher test |
| 2 | Pooled day-0/day-180 mean-lab volcanoes on common axes, day-180 PSA/testosterone extreme-group KM curves with confidence bands and patients/events |
| 3 | Age-only Cox / elastic-net / XGBoost held-out mean AUC(t) at all three landmarks; day-180 top-15 eligible XGBoost lab features |
| 4 | Three-site XGBoost transfer mean AUC(t) with bootstrap CIs at all three ADT landmarks; platinum/AVPC/NEPC five-year event probabilities at sequencing report +180 days |

The lower row of Results 4 uses, within each endpoint, the same sequenced
patients with available Gleason and known TP53/RB1/PTEN indicators for all three
stratifiers. Q1/Q5 cutpoints are computed in that paired subset, retaining ties;
Gleason groups are ≤7/≥8 and TSG groups are WT/any alteration. These are
`1 − KM survival`, with reversed confidence bounds, and death is censored.
Unavailable horizon risks are labeled. The row uses a common 0–0.6 scale,
expanding it if an observed interval requires more room.
Results 2 reads the extreme-group percentage and cutpoints from the pooled KM
exports, preserving the delivered 10% or 20% analysis and labeling it accordingly.

The script honors `COMPASS_DATA_ROOT`, `COMPASS_FIG_ROOT`, `COMPASS_RHINO_ROOT`,
`COMPASS_FEDERATED_RESULTS`, `COMPASS_FEDERATED_FIGURE_DATA` and
`COMPASS_FEDERATED_TRANSFER_NO_DFCI` as above. Poster-specific overrides:

```sh
COMPASS_ESMO_SECTIONS=2,3,4 \
COMPASS_ESMO_OUTPUT_ROOT=/path/to/poster-figures \
Rscript COMPASS/figures/ESMO_AI_Figures.R
```

To update the layout of an existing figure delivery locally, use its aggregate
CSV tables instead of rereading the cluster's analysis files:

```sh
COMPASS_ESMO_AGGREGATE_ROOT="$HOME/Desktop/ESMO_AI" \
COMPASS_ESMO_OUTPUT_ROOT="$HOME/Desktop/ESMO_AI_updated" \
Rscript COMPASS/figures/ESMO_AI_Figures.R
```

This mode needs the `results_*/data/*.csv` files from a previous run and uses the
same plot builders and checks. It preserves the saved estimates, cohort counts,
confidence intervals and KM cutoffs; it does not refit models. Use a separate
output directory to retain the original delivery. Only PDFs are generated;
PNG/SVG files from older runs are not updated.

Results 1 reads `survival_analysis/prediction_inputs_adt/aggregated_landmark0.csv`,
`LLM_NEPC_labels/full_manual_annotations.csv`, and
`primary_subtype_labels.parquet` from `COMPASS_LLM_ANNOTATIONS_PATH` (default:
the existing classifier path in `COMPASS_generate_figures_pipeline.R`). The
remaining sections can run without local phenotyping inputs. Missing required
panels, duplicate metrics or incomplete transfer CIs stop the run before
publishing the requested sections. Run with `--help` to list all settings.

| Toggle | Export folder (under `<COMPASS_FIG_ROOT>/ADT_figures/`; tables, legends and captions go to the same path under `ADT_figure_data/`) | Contents |
| --- | --- | --- |
| `cohort` | `01_cohort/` | Fig 1: consort, event-free survival, time to event, dx-to-treatment, record spans, site counts, `manuscript/01_cohort_overview` |
| `llm` | `02_llm/` | Fig 2: LLM NEPC/AVPC validation on the day-0 cohort (confusion matrix, enrichment, metric bars, subtype landscape) |
| `cohort_sensitivity` | `03_univariate/cohort_sensitivity/` | Fig 3 supplement: cohort forest and event incidence at day 180, ADT-vs-LLM and LLM-vs-regex metastatic labels |
| `federated_univariable` | `03_univariate/federated/` | Federated Fig 3: volcanoes (day 0/90/180), PSA/testosterone quintile KMs and trajectories, pooled and by site |
| `federated_multivariable` | `04_multivariate/federated/` | Federated Fig 4: elastic-net and XGBoost performance/importance, checklist cards |
| `transfer_survival` | `05_transfer_survival/` | Federated Fig 5: no-DFCI transfer KMs, discrimination, clinical comparison |
| `transfer_checklists` | `06_transfer_checklists/` | Federated Fig 6: transfer checklists, plus one panel per landmark x horizon |
| `checklist_augmentation` (opt-in) | `main/alternative_06/lm{0,90,180}_h1825/` | Notebook 07: five-year analyses at days 0/90/180, local Gleason/TSG augmentation, four-group risk |

`FIGURE_GROUPS` in `figure_workflow.R` maps each toggle to its prepared jobs
(`adt__platinum`, `cohort_overview`, `federated`) and export prefix; only the jobs
an enabled group needs are prepared. When every enabled group is federated-only,
the run needs no patient-level data. The group paths describe source selection; rendered figures go to `main/` or `supplement/`.

Predictive checklist alternatives retain their family subfolders under `supplement/`:
`federated_checklist_transfer/`, `generalized_federated_checklist_transfer/`, and
`S8_checklist_extension/`. Landmark/horizon variants retain their distinct filenames
inside those folders. Notebook 07's alternatives use the landmark folders below.

## Stages

`COMPASS_FIGURE_STAGE`:

- `all` (default) checks the notebook manifest, prepares changed figure sets, then
  renders missing or stale PDF and SVG figures.
- `prepare` builds the R graphics caches only.
- `render` re-exports figures from the last prepared snapshot without checking
  sources.

Figure tables (CSV/Markdown) go into each export folder's `data/`; graphics caches
stay under `COMPASS_FIGURE_DATA_ROOT`. Manuscript figures are written in both PDF and SVG formats.

## Settings

| Variable | Default |
| --- | --- |
| `COMPASS_DATA_ROOT` | `/data/gusev/USERS/jpconnor/data/CAIA/COMPASS` |
| `COMPASS_FIG_ROOT` | `/data/gusev/USERS/jpconnor/figures/CAIA/COMPASS` |
| `COMPASS_FIGURE_DATA_ROOT` | `<COMPASS_DATA_ROOT>/figure_data` |
| `COMPASS_RHINO_ROOT` | `<COMPASS_DATA_ROOT>/rhino_data_v2` |
| `COMPASS_FEDERATED_RESULTS`, `COMPASS_FEDERATED_TRANSFER_NO_DFCI`, `COMPASS_FEDERATED_FIGURE_DATA` | paths under `COMPASS_RHINO_ROOT`; set the same values for notebook 05 |
| `COMPASS_RENDER_PROFILE` | `fast` (200 dpi rasters); `publication` uses 600 dpi |
| `COMPASS_RENDER_OVERWRITE`, `COMPASS_PREPARE_OVERWRITE` | `false` |
| `COMPASS_RENDER_WORKERS`, `COMPASS_PREPARE_WORKERS` | `2`, `4` |
| `CAIA_PROJECT_ROOT` | used to locate the code when the script path cannot be inferred |

## Federated inputs

Under `COMPASS_RHINO_ROOT`:

```text
cox_federated_univariate_adt.csv     pooled univariate Cox (volcanoes)
within_site_cox_univariate/          per-site cohort counts (Fig 1 site counts)
federated_elastic_net_cox_{w,no}_DFCI/
federated_*/lm_<days>/{w,no}_DFCI/   per-landmark XGBoost, DeepHit and checklist exports
feature_set_models/                  adjust-only refits (Fig 6 checklists)
federated_transfer_adt_no_DFCI/      notebook 04 output (no-DFCI bundles scored on DFCI)
figure_data/                         aggregate KM curves and trajectories, pooled and within site
```

The older `rhino_results/` layout and the legacy `nvflare_within_site_*` folder names
still resolve.

## Alternative Figure 6: notebook 07

Run `COMPASS/notebooks/07_checklist_test.ipynb` after notebook 04. The analysis
implements [the November 7 plan](../federated_models/Figure_6_augmentation_analysis_plan.md)
in `federated_models/checklist_augmentation{,_inputs}.py`. It checks frozen item
definitions, patient scores/flags, counts and AUCs against notebook 04 and the
manuscript before fitting. Discrepancies stop the run with an audit table.

The primary analysis is fixed at ADT +90 days / 1,825 days. The notebook also runs
the requested day-0 and day-180 analyses, each with a five-year (1,825-day) horizon
after its own landmark and its own frozen generalized checklist. Day-90 manuscript
AUC targets and the original three thresholds are checked only at day 90; every
landmark reproduces its corresponding notebook-04 five-year transfer grid cell.
Nested patient-level
cross-validation fits age + Gleason, age + lab count, and their augmentation;
the genomic extension uses sequencing report +90 days, after ADT. Five-year
net risks, paired IPCW AUC differences, workload-matched screening metrics,
four-group risks, calibration and prespecified sensitivities are saved. NEPC
has frozen references and platinum-trained transfer only, with no NEPC-tuned
model. The default 1,000 patient bootstrap resamples repeat fitting and tuning;
`COMPASS_AUGMENTATION_BOOTSTRAP_MODE=conditional` with
`COMPASS_AUGMENTATION_BOOTSTRAP=200` provides a labeled diagnostic run.

Set `COMPASS_MANUSCRIPT_FIG_ROOT` to the figures repository containing `main/`
and `supplement/` to publish there directly. It defaults to
`<COMPASS_FIG_ROOT>/ADT_figures`. Both R entry scripts support this group:

```sh
export COMPASS_FIGURE_GROUPS=checklist_augmentation
export COMPASS_MANUSCRIPT_FIG_ROOT=/path/to/figures-repo
Rscript COMPASS/notebooks/final_figure_data_prep.R
Rscript COMPASS/notebooks/run_figure_render.R
```

All three landmarks have their own output subfolder under `main/alternative_06/`:
`lm0_h1825/`, `lm90_h1825/`, and `lm180_h1825/`. Each contains
`06_checklist_augmentation.{pdf,svg}` and separate b–d panels, with aggregate CSVs
and methods in its own `data/` subfolder.
`COMPASS_AUGMENTATION_LANDMARKS` defaults to `90,0,180` in both notebook and R;
set it to `0,180` for just the additions or `90` for the primary alone. Each
landmark has its own cohorts, fits, bootstrap diagnostics, and cache/receipt.
The existing Figure 6 is retained. This group consumes notebook 07's checksummed
aggregate receipt independently of notebook 05. Preparation caches its scenes;
rendering uses that snapshot without reading patient data or invoking Python.
The group is disabled by default so existing figure runs do not require a new
analysis. The notebook's last code cell runs both R stages with matching paths.

Protected day-90 outputs default to `<COMPASS_DATA_ROOT>/checklist_augmentation`,
overridden by `COMPASS_CHECKLIST_AUGMENTATION_ROOT`. Definitions, frozen
imputation values, source/code fingerprints and configuration versions are
recorded there. Patient inputs, item-level missingness, folds and OOF predictions
remain in `patient_level/`; only `aggregate/` files can enter figure exports.
Days 0 and 180 use `lm0_h1825/` and `lm180_h1825/` under this protected root.
Input preflight runs before archiving or creating outputs. Reruns archive the
selected landmark's previous tables, and an incomplete run
cannot be prepared by R. Failed folds/resamples remain visible, and CIs require
at least 90% valid resamples and ten valid estimates.

Optional inputs are deliberately explicit:

- `COMPASS_AUGMENTATION_GENE_COVERAGE`: CSV/Parquet keyed by
  `person_id` (or `DFCI_MRN`) and `REPORT_DT`, with 0/1/unknown
  `TP53_assessable`, `RB1_assessable`, `PTEN_assessable`. The coverage must
  support negative calls. Missing/unassessable genes exclude genomic rows.
  Optional `TP53_pathogenic`, `RB1_pathogenic`, `PTEN_pathogenic` and a nonempty
  `annotation_rule` define a separate pathogenic sensitivity before fitting.
- `COMPASS_AUGMENTATION_GLEASON`: existing Gleason timeline; revised cohorts
  require `source_note_date` as well as `gleason_date` and `gleason_total`.
  Missing note dates do not fall back to specimen dates. Optional
  `gleason_primary`/`gleason_secondary` enable Grade Group when consistent.
- `COMPASS_AUGMENTATION_DEATH`: verified patient death table with `DEATH_DATE`,
  enabling landmark survival exclusions and the competing-death sensitivity.

Absent optional sources produce unavailable analyses and an explicit genomic
panel placeholder. The notebook also supports `COMPASS_AUGMENTATION_LOCAL`,
`_BUNDLE`, `_SOMATIC`, `_SOMATIC_MANIFEST`, and `_LAB_MAPPING` path overrides.
Practical adoption criteria and decision curves require study-team choices;
the implementation does not choose an extension based on the best observed AUC.

Simple-score table keys are `gleason_score`, `gleason_ge_m`,
`gleason_ge_m_plus1`, `gleason_tsg_score`, and `gleason_tsg_two_genes`. For a
landmark's frozen M-of-N checklist, these add Gleason (N+1 items, fixed thresholds
M and M+1), then TSG status (N+2 items, training-only workload thresholds).
At day 90 these are the planned four-item ≥2/≥3 and five-item candidates.
Other landmarks retain their own M, N, item definitions and imputation values.
The sequencing-clock day-0 analysis accepts its anchoring report at prediction
time, requires that prediction time to be after ADT, and uses only earlier labs
and documented Gleason. The ADT-day-0 genomic sensitivity requires a pre-ADT report.

## Files

| File | Role |
| --- | --- |
| `figure_run.R` | Shared run logic for both entry scripts: toggles, paths, settings |
| `figure_workflow.R` | Cached prepare/render workflow, federated and manuscript figures, `FIGURE_GROUPS` |
| `COMPASS_generate_figures_pipeline.R` | Per-cohort DFCI figures (Figs 1-2) |
| `figure_supplements.R` | Cohort forest, cohort overview, metastatic-label prep and diagnostics |
| `checklist_supplement.R` | Predictive-checklist cards |
| `checklist_augmentation.R` | Notebook 07 aggregate-only cache and alternative Figure 6 panels |
| `figure_typography.R` | Shared fonts and sizes (also read by the Python plots) |
| `prepare_figure_data.py` | Python preparation called from notebook 05 |
| `retained_exports.txt` | Legacy curated export list |
