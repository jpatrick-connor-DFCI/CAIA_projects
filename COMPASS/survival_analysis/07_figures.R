#!/usr/bin/env Rscript
# COMPASS figures: cached preparation and rendering (stage 07, after notebook 06).
# Configured entirely through environment variables, e.g.
#   Rscript COMPASS/survival_analysis/07_figures.R
#   COMPASS_FIGURE_SCOPE=federated Rscript COMPASS/survival_analysis/07_figures.R
#   COMPASS_FIGURE_STAGE=render COMPASS_RENDER_PROFILE=publication COMPASS_RENDER_PDF=true Rscript ...
# Converted from 05_figures.Rmd; the former document text follows as comments.
#
# ---- Workflow --------------------------------------------------------------
#
# Run `06_prep_figure_data.ipynb` first to prepare the Polars tables and manifest.
# All Python execution belongs to that notebook. This script and its R helpers
# never invoke Python: they consume the notebook's prepared data only. Missing,
# changed, or incompatible inputs stop with instructions to rerun the notebook.
#
# Preparation and rendering are separate stages:
#
# 1. `06_prep_figure_data.ipynb` uses Polars to scan each selected arm's longitudinal
#    CSV once and cache patient metadata, all-lab record spans, selected lab
#    measurements, patient/bin raw and log means, and pre-treatment coverage.
# 2. R consumes those tables, calculates the existing survival statistics and
#    optional GAMs, and caches finished graphics objects. CSV/Markdown figure
#    tables are written during this stage; no images are generated.
# 3. Rendering loads the graphics objects and writes individual and multipanel PNGs and optional
#    PDFs. It does not read patient inputs, fit curves, or repeat plot calculations.
#
# `COMPASS_FIGURE_STAGE=all` (default) checks the notebook manifest and source
# metadata, prepares changed R figure sets, then renders missing or stale formats.
# `prepare` builds R graphics caches only; it does not prepare Python tables.
# `render` uses the last prepared snapshot without checking
# source data, and requires the same cohort/analysis settings. Use `all` for
# normal updates and `render` for DPI/PDF changes.
#
# `COMPASS_FIGURE_SCOPE=federated` processes only the federated forests, site cohort report,
# elastic-net/XGBoost performance and importance, aggregate KM curves and PSA/testosterone
# trajectories, platinum-to-NEPC/AVPC transfer figures, and federated manuscript Figures 3-5,
# without requiring longitudinal data or running other figure families.
#
# Every federated input defaults to the delivery under `COMPASS_RHINO_ROOT`
# (default `<COMPASS_DATA_ROOT>/rhino_results`):
#   federated_model_results/          pooled univariate Cox CSV, within-site Cox, and the
#                                     elastic-net/XGBoost bundles with and without DFCI
#   federated_transfer_adt_no_DFCI/   04's output (bundles trained without DFCI on DFCI endpoints)
#   misc_figure_data/                 aggregate KM curves and binned trajectories, pooled and within site
# `COMPASS_FEDERATED_RESULTS`, `COMPASS_FEDERATED_TRANSFER_NO_DFCI` and
# `COMPASS_FEDERATED_FIGURE_DATA` override the individual paths; set the same values for `06`.
#
# The site report reads `nvflare_within_site_univariate_cox/cox_within_site_all_sites_cohort.csv`
# (legacy name `nvflare_within_site_cox_univariate/` is still accepted)
# beside the federated results CSV. It exports cohort sizes, event counts, and observed
# event percentages at landmark 0 only in `01_cohort/site_incidence_lm000__platinum.{png,csv}`.
# The figure matches the cohort-comparison layout: incidence with 95% Wilson CIs
# on the left and patient counts with events overlaid on the right. Cells with
# fewer than 25 events are hatched/marked. Percentages are events divided by analyzed patients over available follow-up,
# not fixed-horizon cumulative incidence estimates. Missing site counts are reported
# as unavailable; forest plots still render if the site file is absent.
#
# Caches live under `COMPASS_FIGURE_DATA_ROOT` (default `COMPASS/figure_data`).
# They include patient data and need the same access restrictions as the analysis
# inputs. All RDS files, including per-image render receipts, live in that data
# cache; none are written beside figure exports. On the next run, recognized
# legacy `*.cache.rds` receipts in the figure tree are copied, verified, and moved
# into `COMPASS_FIGURE_DATA_ROOT/legacy_render_receipts`.
# The data-cache and figure-output directories must be separate and non-nested.
# Versioned caches are retained. Source identity uses absolute paths,
# file sizes and nanosecond modification times, plus code/settings versions.
# Use `FORCE = True` in `06_prep_figure_data.ipynb` after replacing sources while
# preserving timestamps and sizes. Here, `COMPASS_PREPARE_OVERWRITE=true`
# rebuilds R graphics caches only. Image receipts record the prepared snapshot and DPI;
# affected formats regenerate automatically. Existing images without a receipt
# are regenerated once.
#
# The notebook's Python kernel requires Polars. R needs its existing plotting
# packages plus `nanoparquet` or `arrow`; no Python installation is needed to run this script.
# The notebook controls `POLARS_MAX_THREADS`, defaulting to at most four and
# respecting the SLURM CPU allocation when available.
# `COMPASS_PREPARE_WORKERS` defaults to one to limit memory use;
# `COMPASS_RENDER_WORKERS` defaults to two and parallelizes individual panels on
# Linux. macOS and Windows render sequentially because native graphics libraries
# are not safe to fork.
#
# ---- Cohorts and outputs ---------------------------------------------------
#
# Defaults: all ADT, pre-ADT castrate patients included, platinum only, at
# landmarks 0/90/180. NEPC landmark figures are off by default; set
# `COMPASS_FIGURE_ENDPOINTS=nepc,platinum` to render both. The forests read
# existing day-180 results across six ADT configurations without enabling six
# full renders. ARPI uses its own input file.
# Retired AVPC endpoints and localized-adjuvant cohorts remain unavailable.
#
# Canonical ADT/platinum also generates three combined cohort-comparison figures:
# event incidence with 95% Wilson intervals plus patient/event counts across all
# six ADT definitions and both endpoints at day 180; and Stage-1 ADT-intent/LLM
# label overlap plus the four agreement-cell counts; and day-180 event-free KM
# curves for all six ADT definitions, one panel per endpoint (nested cohorts, so
# no log-rank). `05_cohort_comparison.ipynb` draws none of these; it reports
# the underlying tables only. The incidence denominator
# is each endpoint's aggregated landmark input, not a feature-specific Cox fit.
# It is the observed follow-up event fraction, not cumulative risk by day 180.
# Cells with fewer than 25 events are hatched/marked; missing cells stay blank.
# The label figure uses `mrn_lists/adt_intent_labels_model_cohort.csv` and
# `llm_met_labels_model_cohort.csv`, preserving missing labels explicitly.
# Run notebook `06` once after this update to prepare these small summary tables.
# Combined outputs are under `06_cohort_sensitivity/` (count tables in `06_cohort_sensitivity/data/`):
# `event_incidence_lm180__platinum__all__incl`, `metastatic_label_overlap__platinum__all__incl`
# and `cohort_km_cascade_lm180__platinum__all__incl`.
#
# Figure 1 also exports `01_cohort/figure1_cohort__<endpoint>__<subset>__<exclusion>`:
# a five-panel figure with the cohort-selection flow spanning the left column,
# and survival, record span, diagnosis-to-treatment timing, and time to platinum
# on the right. Figure 2v3 exports `02_llm/figure2v3_llm__platinum__all__incl`,
# with agreement, classification metrics, subtype composition, and platinum
# enrichment in a 2-by-2 layout. The existing `classifier_validation` and
# `subtype_platinum` paired exports remain available. The three metastatic-label
# confusion matrices also form `06_cohort_sensitivity/metastatic_label_agreement__<endpoint>__<subset>__<exclusion>`.
# These combined figures have lettered panels and retain each panel's counts,
# captions, and legends. All individual panels are exported alongside them.
# Compilation stays within each arm/cohort/endpoint; Figure 2v3 remains exclusive
# to canonical ADT/platinum. Run the default `COMPASS_FIGURE_STAGE=all` to rebuild
# the graphics caches and export these additions; `render` requires a matching prepared snapshot.
#
# The seven-panel DFCI overview is
# `03_univariate/internal/dfci_labs_overview_km_lm180__platinum__<subset>__<exclusion>`.
# The top row contains day-0/90/180 within-DFCI univariate volcanoes with common axes;
# the second row pairs the PSA log1p trajectory by platinum status with the PSA
# bottom-versus-top-20% KM curve; the third row does the same for testosterone.
# KM panels default to day 180. Set `COMPASS_COMBINED_KM_LANDMARK=0` or `90` to
# select another landmark; the filename and cached graphics record that choice.
# The overview requires all seven source panels and is restricted to platinum.
# Individual KM panels and the three-landmark KM sheets are also retained.
#
# Clinical combinations and risk stratification (ADT runs, under `04_multivariate/internal/`;
# co-mutation KM curves are univariate and go to `03_univariate/internal/km/`):
# `combinations_<gleason|gleason_somatic>_auc__...` shows the raw held-out test
# mean AUC(t) of every combination arm by landmark and model; the gleason_somatic
# cohort is anchored on the sequencing date (no paired deltas; CSV
# in `04_multivariate/internal/data/`). `04_multivariate/internal/riskstrat_<test|cv_oof>_<source>_lm<N>__...`
# pages the KM curves for the labs risk score and each clinical or comparison-arm
# stratifier (source `full`, Gleason and stage at ADT + landmark, or
# `<cohort>_<cox|xgboost>` matched cohorts; gleason_somatic is on the
# sequencing-date clock with TP53/PTEN/RB1 from its anchoring specimen);
# `withinstrat_<scheme>_<stratifier>_lm<N>` shows labs high vs low inside each
# clinical level. The test and cv_oof schemes are never pooled.
# `comutation_<trio_combinations|tp53_rb1|trio_burden>_lm<N>` gives TP53/PTEN/RB1
# co-mutation KMs on the sequencing-anchored gleason_somatic cohort (cv_oof file,
# falling back to test).
# All read `risk_stratified_patients_*.csv` from run_risk_stratification; rerun it
# if those files are missing, since older runs did not write them.
#
# Each arm root (`ADT/`, `ARPI/`) has six numbered section folders:
#   01_cohort/                 raw cohort numbers: Figure 1/Table 1, analysis sets,
#                              pre-ADT coverage, federated site incidence and cohort KM
#   02_llm/                    LLM NEPC/AVPC subtype labels and their KM curves
#   03_univariate/internal/    DFCI univariable labs (volcanoes) and somatic/Gleason
#                              associations; `km/` holds KM curves (co-mutation,
#                              somatic, Gleason) with lab KM curves in `km/<lab>/`;
#                              `trajectories/<lab>/` holds lab trajectories and distributions
#   03_univariate/federated/   pooled PSA/testosterone forests and volcanoes, with
#                              `km/<lab>/` and `trajectories/<lab>/` (pooled and by site)
#   04_multivariate/internal/  DFCI Elastic-Net/XGBoost discrimination, importance,
#                              somatic sensitivity, combinations, risk strata
#   04_multivariate/federated/ federated models, including the no-DFCI refits
#   05_transfer/               no-DFCI federated models scored at Dana-Farber
#   06_cohort_sensitivity/     cohort forests, endpoint incidence, metastatic-label
#                              comparisons, and `<subset>__<exclusion>/<section>/` for
#                              every run on a non-canonical cohort (anything but all__incl)
# Local filenames are `<figure>__<endpoint>__<subset>__<exclusion>.png`; federated
# ones are `<figure>__platinum.png`. Every export folder holds only PNG/PDF figures:
# CSV/Markdown tables and legends go in a `data/` subfolder of the same folder
# (e.g. `04_multivariate/internal/data/`), and older tables left beside figures are
# moved there when the catalog is rewritten. `<arm>/data/` holds `index.html` and
# `manifest.csv`.
# Compiled (multi-panel) figures sit beside their panels in the section folder; a
# compiled figure whose panels all appear in a canonical ADT/platinum manuscript
# figure is a duplicate and is not exported for that identity (the manuscript
# version is). Older `compiled/` subfolders are archived on the next render.
# Compiled figures retain panel letters, with no descriptive panel titles or overall supertitle. Counts formerly in a
# supertitle remain in panel subtitles or footnotes. Standalone panels stay in the
# section folder itself.
#
# The canonical ADT/platinum, cohort-overview and federated runs also write
# manuscript figures to a `manuscript/` subfolder of the section each summarizes
# (e.g. `ADT/03_univariate/internal/manuscript/03_dfci_univariable.{png,pdf}`). These
# additional exports generally use a 7.2-inch manuscript width, consistent 7–9 pt plot text,
# lettered panels without subplot titles, and separate `.md` figure legends in
# `manuscript/data/`. `manuscript/captions/` contains manuscript-ready legends as matching
# `figureN.md` and `figureN.txt` files (`federated_figureN.*` for the federated set). They always include a
# 600-dpi (or higher requested DPI) PNG and a PDF, even when `COMPASS_RENDER_PDF`
# is false. Individual panels are always retained.
#
# The numbered manuscript set is: `01_cohort_overview`, `02_llm_nepc_avpc`,
# `03_dfci_univariable`, `04_multivariable_labs` (held-out Elastic-Net/XGBoost
# discrimination versus age baselines),
# `06_dfci_cohort_sensitivity` (paired platinum/NEPC incidence and counts plus
# day-180 PSA/testosterone associations),
# `07_federated_incidence_associations` (landmark-0 site incidence and counts plus
# PSA/testosterone associations on natural-log hazard-ratio axes), and
# `08_federated_multivariable` (the local Figure 4 layout for the federated
# elastic-net Cox and XGBoost models: test performance, coefficients, and gains).
# A separate federated set, written by the federated job, is numbered on its own:
# `federated_03_univariable` (the Figure 3 layout from the pooled four-site univariable Cox
# results and the aggregate `misc_figure_data` PSA/testosterone trajectories and day-180
# extreme-quintile KM curves; the volcano y-axis spans the data, capped at -log10(P)=30),
# `federated_04_multivariable` (the Figure 8 layout, with the full models refit without
# Dana-Farber added as darker bars in the discrimination rows), and
# `federated_05_transfer_no_dfci` (the models trained without Dana-Farber scored on
# Dana-Farber platinum/NEPC/AVPC: a, mean AUC(t) of the full and age-only models; b, c, the full models
# vs. local out-of-fold Cox models of Gleason + age and of TP53/RB1/PTEN + altered-gene
# count + age, in the patients with a Gleason score before the ADT landmark / in sequenced
# patients on a sequencing-date clock).
# The local Figure 3 stays on Dana-Farber patient-level data. The cohort
# sensitivity components are assembled into one continuous multipanel figure for
# each analysis setting. Federated manuscript plots reuse the original subplot
# renderers: forests retain site colors and landmark/statistic facets; panel c
# labels only the outer log-hazard-ratio ticks because its four facets repeat the
# same scale, plus 0 at the dashed null line; elastic-net and XGBoost
# retain the local blue/gold model/baseline bars and lab-category colors. Incomplete source groups are not assembled. The DFCI
# overview uses the same configurable extreme-decile KM landmark as its larger
# compiled counterpart. Figure titles remain in filenames/legends, not supertitles.
# The cohort overview uses a 7.2 × 7.2-inch canvas; panels b–e omit their
# patient-count subtitles because those counts are retained in the external legend.
# In the DFCI univariable manuscript figure, the three landmark volcanoes use an
# expanded first row, compact horizontal spacing, a shared y-axis displayed
# through -log10(P)=20, centered landmark headers, and one horizontal category legend.
# The day-180 Albumin-minimum annotation receives a fixed upper-left nudge so it
# does not overlap the Testosterone-maximum label. Panels e and g use a two-line
# platinum-free-probability y-axis title whose shorter vertical extent remains
# separate from the panel letter.
# Trajectory patient-count/scale subtitles are kept in the external legend, and
# KM log-rank values are drawn inside their panels. The manuscript figure fixes
# both KM panels at the 180-day landmark.
# The multivariable-labs manuscript figure includes a held-out AUC(t) panel
# followed by three-landmark Elastic-Net coefficient and XGBoost gain rows;
# the panel-b and panel-c landmark headers are bold and centered above their
# corresponding Elastic-Net coefficient and XGBoost importance subplots. The
# importance rows share the source lab-category colors. Its 8.5-inch canvas
# provides additional horizontal space for the three-column coefficient and gain rows.
# All compiled figures use lowercase bold panel letters and bold x- and y-axis
# titles; tick labels and legend text retain their existing weight.
#
# Manuscript figures are built from the source ggplots or exported aggregate
# tables during `prepare`/`all`, then cached as graphics only. Run `all` after
# updating the R files; `render` alone cannot rebuild stale scenes. Cohort counts,
# original panel captions and supplied federated population/model-input audit
# notes are retained in the accompanying legends. The federated XGBoost figure
# uses a 7.2 × 10.2-inch canvas; its model-input audit remains in the external
# legend rather than as a caution footer on the plotted page. Downloaded PNGs alone cannot provide vector
# replacements for the original source plots.
# Each arm gets a thumbnail `data/index.html` and `data/manifest.csv` with relative paths,
# titles, endpoint, cohort, and landmark metadata. No duplicate thumbnail files are made.
#
# R compiles classifier validation into one figure and subtype/enrichment into one;
# AUC(t) into paired comparisons; importance into one three-landmark sheet per
# model; lab-extreme KM into one three-landmark sheet per lab; and mutation KM into
# pages of at most four panels. Model fits, counts, time origins, and tests are unchanged.
# Local univariate lab volcanoes retain individual day-0/90/180 plots and also
# export `03_univariate/internal/labs_univariate_all_landmarks__<endpoint>__<subset>__<exclusion>`.
# The combined figure uses three side-by-side panels on a 24 × 9-inch canvas,
# common axes, a shared category legend, and separate per-landmark significance
# counts. Larger feature labels (approximately 12 pt) and axis text are preserved
# in both formats. The cohort and label figures described above also retain their
# individual panels, as do the lab KM curves; other compiled figure families do not export component images.
#
# For an existing canonical ADT/platinum PNG download without the graphics cache,
# `Rscript figure_workflow.R <input-ADT-dir> <new-parent>/ADT [0|90|180]` removes
# the compiled supertitles and assembles the new overview from the rendered panels.
# It preserves raster data/geometry, writes a separate tree and refreshed catalog,
# and exports the overview as PNG plus a PDF containing the existing raster panels.
# Normal pipeline runs rebuild figures from the underlying R plots instead.
#
# After a successful render, superseded `by_figure` exports for regenerated cohorts,
# and their files in the retired pre-section layout (`cohort/`, `labels/`,
# `associations/`, `prediction/`, `trajectories/`, `federated/`, top-level
# `compiled/` and `manuscript figures/`), are copied, checksum-verified, and archived
# in `COMPASS_FIGURE_DATA_ROOT/previous_exports/`, as are section `compiled/` folders,
# figures left in `03_univariate/` from before the `km/`/`trajectories/` split, and the
# retired internal Gleason sensitivity (`sensitivity_gleason_auc__*`, manuscript Figure 5);
# only then are those old copies
# and emptied folders removed. Unselected cohorts remain
# untouched. Failed renders do not trigger archival. Archives are recoverable in data.
#
# Figure 2 runs once for canonical ADT/platinum. Lab figures run on each selected
# platinum cohort, for PSA/testosterone by default. Trajectories retain day-zero
# boundaries, 180-day bins, and log transformation before averaging. Pre-ADT
# coverage uses five years and full-stratum denominators. Lab KM curves compare
# bottom versus top 10% without splitting ties (federated aggregate KM keeps the
# site-exported 20% cutoffs); both zoom the y axis because platinum is rare. Somatic and Gleason curves retain
# their specimen/score date origins. GAMs, non-androgen lab panels and the older
# ADT-intent comparison are disabled by default.
#
# The metastatic supplement defaults to three heatmaps: ADT/LLM versus maximum
# regex stage across the record, and ADT versus LLM. Set
# `COMPASS_METASTATIC_EXTRA_PANELS=1` to restore coverage, ICD burden, survival and
# trajectory diagnostics. These use canonical ADT/platinum and the longitudinal
# analysis anchor, with supplied ADT/LLM labels preserved.
#
# The federated HR output contains only PSA/testosterone forest plots at days
# 0/90/180: mean, minimum, maximum, and last value, with delta and observation-count
# features excluded. Two compiled forests (one per analyte) show all sites and the
# federated result together, with statistic columns and landmark rows on a 16:9
# landscape canvas (one analyte per PowerPoint slide). Plots show
# HR per SD and 95% CIs on a common log axis per analyte. Exact p/q values and
# modeled/observed counts remain in the adjacent CSV. Symbols distinguish nominal-only p < 0.05,
# FDR q < 0.05, and nonsignificance. q-values are not recalculated on this subset.
# Finite positive HR estimates are retained; missing/invalid estimates or CIs are
# explicitly marked instead of applying the former volcano magnitude filters.
# Both across-site and individual within-site HRs are plotted, without re-pooling
# site estimates. Within-site forests read
# `nvflare_within_site_univariate_cox/cox_within_site_all_sites_results.csv` beside
# the across-site results CSV. Each uses its own supplied p/q values.
# No matching local results are required. Outputs are
# `03_univariate/federated/{psa,testosterone}_forest__platinum.{png,csv}` plus the day-0
# incidence figure/table. Supplied p/q values are not recomputed or pooled.
# The federated population is not assumed to include every displayed site: matching
# patient/event counts across landmarks are reported as inferred membership only;
# ambiguous or unavailable membership is explicitly unverified. When pooled counts
# exceed the sum over every within-site export at each landmark (a site such as MSK
# that pooled but has no within-site results), the excess is reported as unattributed.
# These forests run when canonical ADT/platinum is selected, or scope
# is `federated`. Missing optional sources are reported and skipped. Failures in
# one figure set do not prevent preparation/rendering of other available sets.
#
# Federated multivariable models are read beside the Cox results: elastic-net Cox
# from `federated_cox_multivariate/cox_federated_elasticnet_{metrics,coefficients}_adt.csv`
# and XGBoost from `federated_xgboost/xgboost_federated_{metrics,importance}_adt.csv`.
# Each model gets two compiled figures built with the local PROFILE Figure 4
# plotting functions, in `04_multivariate/federated/`. `{elasticnet,xgboost}_performance__platinum` pairs
# held-out test mean AUC(t) for labs versus the age baseline at days
# 0/90/180. `elasticnet_coefficients__platinum` shows the three landmarks' top 15
# nonzero coefficients by absolute value, and `xgboost_importance__platinum` the top
# 15 positive-gain features. Both use the local labels, clinical-category colors,
# and age/body-height display exclusions. Unlike the selected HR forests, these
# panels include all supplied feature statistics, including deltas, observation
# counts, and `__<stat>__missing` indicators (shown as "<stat> missing").
# No models are refitted; training/tuning-CV metrics are not substituted for test
# performance. The horizon exports lack a train/test identifier and are not plotted.
# CSV sidecars retain source metadata, counts, all feature values, and display flags.
# Identifier features with nonzero gain or coefficients trigger a warning and a
# figure caption; they are not silently removed or treated as validated predictors.
# Manuscript Figure 8 requires all four captured CSVs.
# The same exports refit without Dana-Farber (`federated_cox_multivariate_no_DFCI/`,
# `federated_xgboost_no_DFCI/`) render as `04_multivariate/federated/{elasticnet,xgboost}_*_no_dfci__platinum`
# and feed federated manuscript Figure 4.
#
# Aggregate KM curves and trajectories (`COMPASS_FEDERATED_FIGURE_DATA`, default
# `misc_figure_data/`) need `manifest.json` (bin width, minimum bin size, sites and per-site
# counts) plus `cohort_km.csv`, `lab_extreme_{km,summary}.csv` and
# `lab_trajector{ies,y_cohort}.csv`, each with a `within_site_` twin. They render
# `01_cohort/federated_cohort_km` and, in `03_univariate/federated/`, `trajectories/<lab>/federated_trajectory_<lab>`, `km/<lab>/federated_km_quintile_<lab>_landmark<N>`
# (pooled cutoffs) and `_by_site` pages (one panel per site, site-specific cutoffs),
# with `federated_volcano_landmark<N>` from the pooled univariate CSV. The local
# trajectory and KM plotters are reused; nothing is refitted. Missing files are reported and skipped.
#
# Federated transfer (`COMPASS_FEDERATED_TRANSFER_NO_DFCI`, default
# `federated_transfer_adt_no_DFCI/`, 04's `OUTPUT_DIR`) reads `transfer_metrics.csv` and
# `transfer_auc_t.csv` written by `04_federated_transfer.ipynb`: the platinum-trained
# federated XGBoost and elastic-net bundles refit without Dana-Farber and scored on the
# local ADT cohort for platinum, NEPC, and AVPC, an external validation.
# `05_transfer/transfer_discrimination_no_dfci__platinum` shows IPCW mean AUC(t) and the
# paired gain over the age-only model by landmark, faceted by endpoint. CSV sidecars keep
# all metrics (`transfer_discrimination_no_dfci__platinum.csv`) and per-horizon AUC(t)
# (`transfer_auc_t_no_dfci__platinum.csv`). `transfer_clinical_comparison.csv`
# (04's Gleason / TP53-RB1-PTEN / Gleason + TP53-RB1-PTEN section) renders
# `05_transfer/transfer_clinical_comparison_no_dfci__platinum` and feeds Federated Figure 5 b-d;
# without it, Figure 5 is drawn with panel a only. Missing transfer outputs are reported and skipped.
# Missing optional XGBoost files are reported and skipped independently. Run `06`
# again after this update to register both files (including absent paths), then
# run `07_figures.R` with stage `all`. All Python remains in `06`; graphics caches stay in data.

# ---- Configuration ----------------------------------------------------------
options(readr.num_threads = 1L, readr.read_lazy = FALSE)
# Absolute code root (same as GAM/04_gam.ipynb) so the script runs from any
# working directory; set CAIA_PROJECT_ROOT to use a different checkout.
PROJECT_ROOT <- Sys.getenv("CAIA_PROJECT_ROOT", "/data/gusev/USERS/jpconnor/code/CAIA")
pipeline_path <- normalizePath(file.path(PROJECT_ROOT, "COMPASS", "survival_analysis",
                                         "COMPASS_generate_figures_pipeline.R"), mustWork = TRUE)
source(pipeline_path)
source(file.path(dirname(pipeline_path), "figure_workflow.R"))

env_flag <- function(name, default = FALSE) {
  value <- tolower(Sys.getenv(name, if (default) "true" else "false"))
  if (!value %in% c("true", "false", "1", "0", "yes", "no")) stop(name, " must be true or false")
  value %in% c("true", "1", "yes")
}
env_values <- function(name, default) trimws(strsplit(Sys.getenv(name, default), ",", fixed = TRUE)[[1]])
env_workers <- function(name, default) {
  value <- suppressWarnings(as.integer(Sys.getenv(name, as.character(default))))
  if (is.na(value) || value < 1L) stop(name, " must be a positive integer")
  value
}
NEPC_PROJ_PATH <- Sys.getenv("COMPASS_DATA_ROOT", "/data/gusev/USERS/jpconnor/data/CAIA/COMPASS")
FIG_ROOT <- Sys.getenv("COMPASS_FIG_ROOT", "/data/gusev/USERS/jpconnor/figures/CAIA/COMPASS")
FIGURE_DATA_ROOT <- Sys.getenv("COMPASS_FIGURE_DATA_ROOT", file.path(NEPC_PROJ_PATH, "figure_data"))
FIGURE_STAGE <- match.arg(Sys.getenv("COMPASS_FIGURE_STAGE", "all"), c("all", "prepare", "render"))
FIGURE_SCOPE <- match.arg(Sys.getenv("COMPASS_FIGURE_SCOPE", "all"), c("all", "federated"))
# Platinum only by default; set COMPASS_FIGURE_ENDPOINTS=nepc,platinum to restore
# the NEPC landmark figures alongside it (or =nepc for that endpoint alone).
ENDPOINTS <- env_values("COMPASS_FIGURE_ENDPOINTS", "platinum")
ADT_COHORT_OPTIONS <- c("adt", "adt_noprecastrate", "adt_metastatic_adt",
  "adt_metastatic_adt_noprecastrate", "adt_metastatic_llm", "adt_metastatic_llm_noprecastrate")
COHORTS <- env_values("COMPASS_FIGURE_COHORTS", "adt")
stopifnot(length(COHORTS) > 0L, length(ENDPOINTS) > 0L,
          all(COHORTS %in% SUPPORTED_COHORTS), all(ENDPOINTS %in% SUPPORTED_ENDPOINTS),
          !anyDuplicated(COHORTS), !anyDuplicated(ENDPOINTS))
COHORT_FOREST_CONFIG <- list(script = file.path(dirname(pipeline_path), "figure_supplements.R"),
                             cohorts = ADT_COHORT_OPTIONS, landmark = 180L)
PLOT_NON_ANDROGEN_LAB_FIGURES <- env_flag("COMPASS_NON_ANDROGEN_LABS")
PLOT_GAM_TRAJECTORIES <- env_flag("COMPASS_GAM_TRAJECTORIES")
PLOT_ADT_INTENT_SUPPLEMENT <- env_flag("COMPASS_ADT_INTENT_SUPPLEMENT")
PLOT_METASTATIC_LABEL_SUPPLEMENT <- env_flag("COMPASS_METASTATIC_LABEL_SUPPLEMENT", TRUE)
PLOT_FEDERATED_SUPPLEMENT <- env_flag("COMPASS_FEDERATED_SUPPLEMENT", TRUE)
METASTATIC_SUPPLEMENT <- list(
  prepare_script = file.path(dirname(pipeline_path), "figure_supplements.R"),
  r_script = file.path(dirname(pipeline_path), "figure_supplements.R"),
  intent = file.path(NEPC_PROJ_PATH, "mrn_lists", "adt_intent_labels_model_cohort.csv"),
  stage = Sys.getenv("COMPASS_REGEX_STAGE_PATH", file.path(
    Sys.getenv("PROFILE_DATA_PATH", "/data/gusev/USERS/jpconnor/data/PROFILE_DATA"),
    "CANCER_ANNOTATIONS", "CANCER_STAGE_NOTE_LEVEL.parquet")),
  llm = Sys.getenv("LLM_MET_LABELS_PATH", "/data/gusev/USERS/jpconnor/data/LLM_annotations/LLM_met_diagnosis/met_dx_labels.parquet"),
  icd = file.path(NEPC_PROJ_PATH, "prostate_icd_data.csv"))
RHINO_ROOT <- Sys.getenv("COMPASS_RHINO_ROOT", file.path(NEPC_PROJ_PATH, "rhino_results"))
FEDERATED_SUPPLEMENT <- list(
  results = Sys.getenv("COMPASS_FEDERATED_RESULTS", file.path(
    RHINO_ROOT, "federated_model_results", "cox_federated_univariate_adt.csv")),
  transfer_no_dfci = Sys.getenv("COMPASS_FEDERATED_TRANSFER_NO_DFCI",
    file.path(RHINO_ROOT, "federated_transfer_adt_no_DFCI")),
  figure_data = Sys.getenv("COMPASS_FEDERATED_FIGURE_DATA", file.path(RHINO_ROOT, "misc_figure_data")))
RENDER_PROFILE <- match.arg(tolower(Sys.getenv("COMPASS_RENDER_PROFILE", "fast")), c("fast", "publication"))
RENDER_DPI <- if (RENDER_PROFILE == "publication") 600 else 200
RENDER_PDF <- env_flag("COMPASS_RENDER_PDF")
RENDER_OVERWRITE <- env_flag("COMPASS_RENDER_OVERWRITE")
PREPARE_OVERWRITE <- env_flag("COMPASS_PREPARE_OVERWRITE")
RENDER_WORKERS <- env_workers("COMPASS_RENDER_WORKERS", 2L)
PREPARE_WORKERS <- env_workers("COMPASS_PREPARE_WORKERS", 1L)
source(file.path(dirname(pipeline_path), "figure_supplements.R"))

figure_config <- list(
  data_root = NEPC_PROJ_PATH, cache_root = FIGURE_DATA_ROOT, fig_root = FIG_ROOT,
  cohorts = COHORTS, endpoints = ENDPOINTS, scope = FIGURE_SCOPE,
  labs = if (PLOT_NON_ANDROGEN_LAB_FIGURES) names(CATEGORY_MAP) else ANDROGEN,
  classifier_path = DEFAULT_LLM_ANNOTATIONS_PATH,
  forest_cohorts = COHORT_FOREST_CONFIG$cohorts, forest_landmark = COHORT_FOREST_CONFIG$landmark,
  gam = PLOT_GAM_TRAJECTORIES, adt_intent = PLOT_ADT_INTENT_SUPPLEMENT,
  metastatic = PLOT_METASTATIC_LABEL_SUPPLEMENT, metastatic_extra = metastatic_extra_panels_enabled(),
  metastatic_sources = METASTATIC_SUPPLEMENT[c("intent", "stage", "llm", "icd")],
  federated = PLOT_FEDERATED_SUPPLEMENT,
  federated_path = FEDERATED_SUPPLEMENT$results,
  federated_transfer_no_dfci_path = FEDERATED_SUPPLEMENT$transfer_no_dfci,
  federated_figure_data_path = FEDERATED_SUPPLEMENT$figure_data)

# ---- Prepare and render -----------------------------------------------------
figure_run <- run_cached_figure_workflow(
  figure_config, pipeline_path, FIGURE_STAGE,
  prepare_workers = PREPARE_WORKERS, render_workers = RENDER_WORKERS,
  dpi = RENDER_DPI, pdf = RENDER_PDF,
  prepare_overwrite = PREPARE_OVERWRITE, render_overwrite = RENDER_OVERWRITE,
  metastatic_config = METASTATIC_SUPPLEMENT,
  forest_config = COHORT_FOREST_CONFIG,
  federated_config = FEDERATED_SUPPLEMENT)
