# COMPASS figure workflow

!/usr/bin/env Rscript
COMPASS figures: cached preparation and rendering (stage 07, after notebook 06).
Configured entirely through environment variables, e.g.
  Rscript COMPASS/survival_analysis/07_figures/all.R
  COMPASS_FIGURE_SCOPE=federated Rscript COMPASS/survival_analysis/07_figures/all.R
  COMPASS_FIGURE_STAGE=render COMPASS_RENDER_PROFILE=publication COMPASS_RENDER_PDF=true Rscript ...
Converted from 05_figures.Rmd; the former document text follows as comments.

---- Workflow --------------------------------------------------------------

Run `06_prep_figure_data.ipynb` first to prepare the Polars tables and manifest.
All Python execution belongs to that notebook. This script and its R helpers
never invoke Python: they consume the notebook's prepared data only. Missing,
changed, or incompatible inputs stop with instructions to rerun the notebook.

Preparation and rendering are separate stages:

1. `06_prep_figure_data.ipynb` uses Polars to scan each selected arm's longitudinal
   CSV once and cache patient metadata, all-lab record spans, selected lab
   measurements, patient/bin raw and log means, and pre-treatment coverage.
2. R consumes those tables, calculates the existing survival statistics and
   optional GAMs, and caches finished graphics objects. CSV/Markdown figure
   tables are written during this stage; no images are generated.
3. Rendering loads the graphics objects and writes individual and multipanel PNGs and optional
   PDFs. It does not read patient inputs, fit curves, or repeat plot calculations.

`COMPASS_FIGURE_STAGE=all` (default) checks the notebook manifest and source
metadata, prepares changed R figure sets, then renders missing or stale formats.
`prepare` builds R graphics caches only; it does not prepare Python tables.
`render` uses the last prepared snapshot without checking
source data, and requires the same cohort/analysis settings. Use `all` for
normal updates and `render` for DPI/PDF changes.

`COMPASS_FIGURE_SCOPE=federated` processes only the federated forests, site cohort report,
elastic-net/XGBoost performance and importance, aggregate KM curves and PSA/testosterone
trajectories, platinum-to-NEPC/AVPC transfer figures, and federated manuscript Figures 3-5,
without requiring longitudinal data or running other figure families.

Every federated input defaults to the delivery under `COMPASS_RHINO_ROOT`
(default `<COMPASS_DATA_ROOT>/rhino_data_v2`):
  cox_federated_univariate_adt.csv  pooled univariate Cox results
  within_site_cox_univariate/       within-site Cox results and per-site cohort counts
  federated_elastic_net_cox_{w,no}_DFCI/
                                    elastic-net Cox exports with and without DFCI
  federated_*/lm_<days>/{w,no}_DFCI/ per-landmark XGBoost, DeepHit and checklist exports
  federated_transfer_adt_no_DFCI/   04a's output (bundles trained without DFCI on DFCI endpoints)
  figure_data/                      aggregate KM curves and binned trajectories, pooled and within site
The earlier rhino_results layout (federated_model_results/, misc_figure_data/) still
resolves when the paths below point at it.
`COMPASS_FEDERATED_RESULTS`, `COMPASS_FEDERATED_TRANSFER_NO_DFCI` and
`COMPASS_FEDERATED_FIGURE_DATA` override the individual paths; set the same values for `06`.

The site report reads `within_site_cox_univariate/cox_within_site_all_sites_cohort.csv`
(legacy names `nvflare_within_site_{univariate_cox,cox_univariate}/` are still accepted)
beside the federated results CSV. It exports cohort sizes, event counts, and observed
event percentages at landmark 0 only in `01_cohort/site_incidence_lm000__platinum.{png,csv}`.
The figure matches the cohort-comparison layout: incidence with 95% Wilson CIs
on the left and patient counts with events overlaid on the right. Cells with
fewer than 25 events are hatched/marked. Percentages are events divided by analyzed patients over available follow-up,
not fixed-horizon cumulative incidence estimates. Missing site counts are reported
as unavailable; forest plots still render if the site file is absent.

Checklist supplement (Figure 4S): `06` flattens the predictive-checklist runs at
`COMPASS_CHECKLIST_LABS_RUN` (labs only) and `COMPASS_CHECKLIST_TRIO_RUN` (labs + Gleason +
somatic), defaulting to `<COMPASS_DATA_ROOT>/survival_analysis/predictive_checklists/{full_adt,gleason_trio_labs}`
(fit with horizons=[1825]); 07 draws the 5-year platinum and NEPC checklists per landmark with
held-out metrics. Skipped when no run exists; `COMPASS_CHECKLIST_SUPPLEMENT=false` disables it.
Caches live under `COMPASS_FIGURE_DATA_ROOT` (default `COMPASS/figure_data`).
They include patient data and need the same access restrictions as the analysis
inputs. All RDS files, including per-image render receipts, live in that data
cache; none are written beside figure exports. On the next run, recognized
legacy `*.cache.rds` receipts in the figure tree are copied, verified, and moved
into `COMPASS_FIGURE_DATA_ROOT/legacy_render_receipts`.
The data-cache and figure-output directories must be separate and non-nested.
Versioned caches are retained. Source identity uses absolute paths,
file sizes and nanosecond modification times, plus code/settings versions.
Use `FORCE = True` in `06_prep_figure_data.ipynb` after replacing sources while
preserving timestamps and sizes. Here, `COMPASS_PREPARE_OVERWRITE=true`
rebuilds R graphics caches only. Image receipts record the prepared snapshot and DPI;
affected formats regenerate automatically. Existing images without a receipt
are regenerated once.

The notebook's Python kernel requires Polars. R needs its existing plotting
packages plus `nanoparquet` or `arrow`; no Python installation is needed to run this script.
The notebook controls `POLARS_MAX_THREADS`, defaulting to at most four and
respecting the SLURM CPU allocation when available.
`COMPASS_PREPARE_WORKERS` defaults to one to limit memory use;
`COMPASS_RENDER_WORKERS` defaults to two and parallelizes individual panels on
Linux. macOS and Windows render sequentially because native graphics libraries
are not safe to fork.

---- Cohorts and outputs ---------------------------------------------------

Defaults: all ADT, pre-ADT castrate patients included, platinum only, at
landmarks 0/90/180. NEPC landmark figures are off by default; set
`COMPASS_FIGURE_ENDPOINTS=nepc,platinum` to render both. The forests read
existing day-180 results across six ADT configurations without enabling six
full renders. ARPI uses its own input file.
Retired AVPC endpoints and localized-adjuvant cohorts remain unavailable.

Canonical ADT/platinum also generates three combined cohort-comparison figures:
event incidence with 95% Wilson intervals plus patient/event counts across all
six ADT definitions and both endpoints at day 180; and Stage-1 ADT-intent/LLM
label overlap plus the four agreement-cell counts; and day-180 event-free KM
curves for all six ADT definitions, one panel per endpoint (nested cohorts, so
no log-rank). `05_cohort_comparison.ipynb` draws none of these; it reports
the underlying tables only. The incidence denominator
is each endpoint's aggregated landmark input, not a feature-specific Cox fit.
It is the observed follow-up event fraction, not cumulative risk by day 180.
Cells with fewer than 25 events are hatched/marked; missing cells stay blank.
The label figure uses `mrn_lists/adt_intent_labels_model_cohort.csv` and
`llm_met_labels_model_cohort.csv`, preserving missing labels explicitly.
Run notebook `06` once after this update to prepare these small summary tables.
Combined outputs are under `06_cohort_sensitivity/` (count tables in `06_cohort_sensitivity/data/`):
`event_incidence_lm180__platinum__all__incl`, `metastatic_label_overlap__platinum__all__incl`
and `cohort_km_cascade_lm180__platinum__all__incl`.

Figure 1 also exports `01_cohort/figure1_cohort__<endpoint>__<subset>__<exclusion>`:
a five-panel figure with the cohort-selection flow spanning the left column,
and survival, record span, diagnosis-to-treatment timing, and time to platinum
on the right. Canonical ADT/platinum adds `figure1d_dfci_data` (DFCI patients with
ICD-10 prostate cancer, ADT, a pre-ADT Gleason score, tumor sequencing, and the
day-0 prediction cohort; Gleason/sequencing counted within the day-0 cohort from
`prediction_inputs_adt/somatic_gleason/build_manifest.json`) and
`figure1e_site_counts` (day-0 platinum events/patients per federated site from the
within-site cohort export; skipped when absent), both in manuscript Figure 1.
Figure 2v3 is drawn on three cohorts, each restricted to patients with an LLM label:
`icd` (every ICD-10 C61 patient), `adt` (ADT_EXPOSED == 1) and `lm0` (the ADT day-0
prediction cohort). Platinum status is any dated platinum ever
(`mrn_lists/platinum_any_MRN_list.csv`, written by compile_COMPASS_cohort_data.py).
Each exports `02_llm/figure2v3_llm_<cohort>__platinum__all__incl`, with agreement,
classification metrics, subtype composition, and platinum enrichment in a 2-by-2
layout, plus `classifier_validation_<cohort>` and `subtype_platinum_<cohort>` paired
exports. The three metastatic-label
confusion matrices also form `06_cohort_sensitivity/metastatic_label_agreement__<endpoint>__<subset>__<exclusion>`.
These combined figures have lettered panels and retain each panel's counts,
captions, and legends. All individual panels are exported alongside them.
Compilation stays within each arm/cohort/endpoint; Figure 2v3 remains exclusive
to canonical ADT/platinum. Run the default `COMPASS_FIGURE_STAGE=all` to rebuild
the graphics caches and export these additions; `render` requires a matching prepared snapshot.

The seven-panel DFCI overview is
`03_univariate/internal/dfci_labs_overview_km_lm180__platinum__<subset>__<exclusion>`.
The top row contains day-0/90/180 within-DFCI univariate volcanoes with common axes;
the second row pairs the PSA log1p trajectory by platinum status with the PSA
bottom-versus-top-10% KM curve; the third row does the same for testosterone.
KM panels default to day 180. Set `COMPASS_COMBINED_KM_LANDMARK=0` or `90` to
select another landmark; the filename and cached graphics record that choice.
The overview requires all seven source panels and is restricted to platinum.
Individual KM panels and the three-landmark KM sheets are also retained.
The DFCI supplement `03_univariate/internal/dfci_supplement_labs_lm180__platinum__<subset>__<exclusion>`
pairs each supplementary lab's trajectory (log1p for LDH and alkaline phosphatase,
raw for hemoglobin and albumin) with its bottom-versus-top-10% KM curve. Its labs
default to LDH, hemoglobin, albumin and alkaline phosphatase; set
`COMPASS_SUPPLEMENT_LABS` (comma-separated canonical names) to change them, here and
in notebook 06.

Clinical combinations and risk stratification (ADT runs, under `04_multivariate/internal/`;
co-mutation KM curves are univariate and go to `03_univariate/internal/km/`):
`combinations_<gleason|gleason_somatic>_auc__...` shows the raw held-out test
mean AUC(t) of every combination arm by landmark and model; the gleason_somatic
cohort is anchored on the sequencing date (no paired deltas; CSV
in `04_multivariate/internal/data/`). `04_multivariate/internal/riskstrat_<test|cv_oof>_<source>_lm<N>__...`
pages the KM curves for the labs risk score and each clinical or comparison-arm
stratifier (source `full`, Gleason and stage at ADT + landmark, or
`<cohort>_<cox|xgboost>` matched cohorts; gleason_somatic is on the
sequencing-date clock with TP53/PTEN/RB1 from its anchoring specimen);
`withinstrat_<scheme>_<stratifier>_lm<N>` shows labs high vs low inside each
clinical level. The test and cv_oof schemes are never pooled.
`comutation_<trio_combinations|tp53_rb1|trio_burden>_lm<N>` gives TP53/PTEN/RB1
co-mutation KMs on the sequencing-anchored gleason_somatic cohort (cv_oof file,
falling back to test).
All read `risk_stratified_patients_*.csv` from run_risk_stratification; rerun it
if those files are missing, since older runs did not write them.

Each arm root (`ADT/`, `ARPI/`) has six numbered section folders:
  01_cohort/                 raw cohort numbers: Figure 1/Table 1, analysis sets,
                             pre-ADT coverage, federated site incidence and cohort KM
  02_llm/                    LLM NEPC/AVPC subtype labels and their KM curves
  03_univariate/internal/    DFCI univariable labs (volcanoes) and somatic/Gleason
                             associations; `km/` holds KM curves (co-mutation,
                             somatic, Gleason) with lab KM curves in `km/<lab>/`;
                             `trajectories/<lab>/` holds lab trajectories and distributions
  03_univariate/federated/   pooled PSA/testosterone forests and volcanoes, with
                             `km/<lab>/` and `trajectories/<lab>/` (pooled and by site)
  04_multivariate/internal/  DFCI Elastic-Net/XGBoost discrimination, importance,
                             somatic sensitivity, combinations, risk strata
  04_multivariate/federated/ federated models, including the no-DFCI refits
  05_transfer/               no-DFCI federated models scored at Dana-Farber
  06_cohort_sensitivity/     cohort forests, endpoint incidence, metastatic-label
                             comparisons, and `<subset>__<exclusion>/<section>/` for
                             every run on a non-canonical cohort (anything but all__incl)
Local filenames are `<figure>__<endpoint>__<subset>__<exclusion>.png`; federated
ones are `<figure>__platinum.png`. Every export folder holds only PNG/PDF figures:
CSV/Markdown tables and legends go in a `data/` subfolder of the same folder
(e.g. `04_multivariate/internal/data/`), and older tables left beside figures are
moved there when the catalog is rewritten. `<arm>/data/` holds `index.html` and
`manifest.csv`.
Compiled (multi-panel) figures sit beside their panels in the section folder; a
compiled figure whose panels all appear in a canonical ADT/platinum manuscript
figure is a duplicate and is not exported for that identity (the manuscript
version is). Older `compiled/` subfolders are archived on the next render.
Compiled figures retain panel letters, with no descriptive panel titles or overall supertitle. Counts formerly in a
supertitle remain in panel subtitles or footnotes. Standalone panels stay in the
section folder itself.

The canonical ADT/platinum, cohort-overview and federated runs also write
manuscript figures to a `manuscript/` subfolder of the section each summarizes
(e.g. `ADT/03_univariate/internal/manuscript/03_dfci_univariable.{png,pdf}`). These
additional exports generally use a 7.2-inch manuscript width, consistent 7–9 pt plot text,
lettered panels without subplot titles, and separate `.md` figure legends in
`manuscript/data/`. `manuscript/captions/` contains manuscript-ready legends as matching
`figureN.md` and `figureN.txt` files (`federated_figureN.*` for the federated set). They always include a
600-dpi (or higher requested DPI) PNG and a PDF, even when `COMPASS_RENDER_PDF`
is false. Individual panels are always retained.

The numbered manuscript set is: `01_cohort_overview`, `02_llm_nepc_avpc__<icd|adt|lm0>`
(one per Figure 2 cohort, captions `figure2_<cohort>`, with a cohort header line),
`03_dfci_univariable`, `04_multivariable_labs` (held-out Elastic-Net/XGBoost
discrimination versus age baselines),
`06_dfci_cohort_sensitivity` (paired platinum/NEPC incidence and counts plus
day-180 PSA/testosterone associations),
`07_federated_incidence_associations` (landmark-0 site incidence and counts plus
PSA/testosterone associations on natural-log hazard-ratio axes), and
`08_federated_multivariable` (the local Figure 4 layout for the federated
elastic-net Cox and XGBoost models: test performance, coefficients, and gains).
A separate federated set, written by the federated job, is numbered on its own:
`federated_03_univariable` (the Figure 3 layout from the pooled four-site univariable Cox
results and the aggregate `figure_data` PSA/testosterone trajectories and day-180
extreme-decile KM curves; the volcano y-axis spans the data, capped at -log10(P)=30;
an arrow under the color key marks positive log HR as higher risk/faster time to platinum),
`federated_04_multivariable` (the Figure 8 layout, with the full models refit without
Dana-Farber added as darker bars in the discrimination rows), and
`federated_05_transfer_no_dfci` (the models trained without Dana-Farber scored on
Dana-Farber platinum/AVPC/NEPC, in that order: a, mean AUC(t) of the full models; b-d, mean
AUC(t) of the full models, local out-of-fold Cox models of Gleason + age (b),
TP53/RB1/PTEN + age (c) or Gleason + TP53/RB1/PTEN + age (d), and combined Cox models of
each transferred score + the same covariates without age (the full models include age),
in patients with a Gleason score before the ADT landmark (b), sequenced patients (c) and
sequenced patients with a Gleason score (d); c and d on a sequencing-date clock; facet
strips give day-0 patients and events; e-g, platinum/AVPC/NEPC KM curves for the top vs
bottom 10% of the day-180 XGBoost score; h, 5-year event-free probability by score
quartile, Gleason score and six mutually exclusive mutation groups (0 altered, TP53 altered, PTEN altered, RB1 altered, 2 altered, 3 altered)). `federated_05_transfer_no_dfci__no_d`
(caption `federated_figure5_no_d`) drops panel d, so its KM curves are d-f and the gradient g.
The local Figure 3 stays on Dana-Farber patient-level data. The cohort
sensitivity components are assembled into one continuous multipanel figure for
each analysis setting. Federated manuscript plots reuse the original subplot
renderers: forests retain site colors and landmark/statistic facets; panel c
labels only the outer log-hazard-ratio ticks because its four facets repeat the
same scale, plus 0 at the dashed null line; elastic-net and XGBoost
retain the local blue/gold model/baseline bars and lab-category colors. Incomplete source groups are not assembled. The DFCI
overview uses the same configurable extreme-decile KM landmark as its larger
compiled counterpart. Figure titles remain in filenames/legends, not supertitles.
The cohort overview uses a 7.2 × 9.4-inch canvas (panel g is omitted, and f spans the
row, without the federated within-site export); panels b–g omit their subtitles
because those counts and notes are retained in the external legend.
In the DFCI univariable manuscript figure, the three landmark volcanoes use an
expanded first row, compact horizontal spacing, a shared y-axis displayed
through -log10(P)=20, centered landmark headers, and one horizontal category legend.
The day-180 Albumin-minimum annotation receives a fixed upper-left nudge so it
does not overlap the Testosterone-maximum label. Panels e and g use a two-line
platinum-free-probability y-axis title whose shorter vertical extent remains
separate from the panel letter.
Trajectory patient-count/scale subtitles are kept in the external legend, and
KM log-rank values are drawn inside their panels. The manuscript figure fixes
both KM panels at the 180-day landmark.
The multivariable-labs manuscript figure includes a held-out AUC(t) panel
followed by three-landmark Elastic-Net coefficient and XGBoost gain rows;
the panel-b and panel-c landmark headers are bold and centered above their
corresponding Elastic-Net coefficient and XGBoost importance subplots. The
importance rows share the source lab-category colors. Its 8.5-inch canvas
provides additional horizontal space for the three-column coefficient and gain rows.
All compiled figures use lowercase bold panel letters and bold x- and y-axis
titles; tick labels and legend text retain their existing weight.

Manuscript figures are built from the source ggplots or exported aggregate
tables during `prepare`/`all`, then cached as graphics only. Run `all` after
updating the R files; `render` alone cannot rebuild stale scenes. Cohort counts,
original panel captions and supplied federated population/model-input audit
notes are retained in the accompanying legends. The federated XGBoost figure
uses a 7.2 × 10.2-inch canvas; its model-input audit remains in the external
legend rather than as a caution footer on the plotted page. Downloaded PNGs alone cannot provide vector
replacements for the original source plots.
Each arm gets a thumbnail `data/index.html` and `data/manifest.csv` with relative paths,
titles, endpoint, cohort, and landmark metadata. No duplicate thumbnail files are made.

R compiles classifier validation into one figure and subtype/enrichment into one;
AUC(t) into paired comparisons; importance into one three-landmark sheet per
model; lab-extreme KM into one three-landmark sheet per lab; and mutation KM into
pages of at most four panels. Model fits, counts, time origins, and tests are unchanged.
Local univariate lab volcanoes retain individual day-0/90/180 plots and also
export `03_univariate/internal/labs_univariate_all_landmarks__<endpoint>__<subset>__<exclusion>`.
The combined figure uses three side-by-side panels on a 24 × 9-inch canvas,
common axes, a shared category legend, and separate per-landmark significance
counts. Larger feature labels (approximately 12 pt) and axis text are preserved
in both formats. The cohort and label figures described above also retain their
individual panels, as do the lab KM curves; other compiled figure families do not export component images.

For an existing canonical ADT/platinum PNG download without the graphics cache,
`Rscript figure_workflow.R <input-ADT-dir> <new-parent>/ADT [0|90|180]` removes
the compiled supertitles and assembles the new overview from the rendered panels.
It preserves raster data/geometry, writes a separate tree and refreshed catalog,
and exports the overview as PNG plus a PDF containing the existing raster panels.
Normal pipeline runs rebuild figures from the underlying R plots instead.

After a successful render, superseded `by_figure` exports for regenerated cohorts,
and their files in the retired pre-section layout (`cohort/`, `labels/`,
`associations/`, `prediction/`, `trajectories/`, `federated/`, top-level
`compiled/` and `manuscript figures/`), are copied, checksum-verified, and archived
in `COMPASS_FIGURE_DATA_ROOT/previous_exports/`, as are section `compiled/` folders,
figures left in `03_univariate/` from before the `km/`/`trajectories/` split, and the
retired internal Gleason sensitivity (`sensitivity_gleason_auc__*`, manuscript Figure 5);
only then are those old copies
and emptied folders removed. Unselected cohorts remain
untouched. Failed renders do not trigger archival. Archives are recoverable in data.

Figure 2 runs once for canonical ADT/platinum (three cohort variants). Lab figures run on
each selected platinum cohort, for PSA/testosterone and the supplementary labs by default. Trajectories retain day-zero
boundaries, 180-day bins, and log transformation before averaging. Pre-ADT
coverage uses five years and full-stratum denominators. Lab KM curves compare
bottom versus top 10% without splitting ties (federated aggregate KM uses the
site-exported `extreme_fraction`, 10% by default; older 20% `q20`/`q80` exports
are still read); both zoom the y axis because platinum is rare. Somatic and Gleason curves retain
their specimen/score date origins. GAMs, non-androgen lab panels and the older
ADT-intent comparison are disabled by default.

The metastatic supplement defaults to three heatmaps: ADT/LLM versus maximum
regex stage across the record, and ADT versus LLM. Set
`COMPASS_METASTATIC_EXTRA_PANELS=1` to restore coverage, ICD burden, survival and
trajectory diagnostics. These use canonical ADT/platinum and the longitudinal
analysis anchor, with supplied ADT/LLM labels preserved.

The federated HR output contains only PSA/testosterone forest plots at days
0/90/180: mean, minimum, maximum, and last value, with delta and observation-count
features excluded. Two compiled forests (one per analyte) show all sites and the
federated result together, with statistic columns and landmark rows on a 16:9
landscape canvas (one analyte per PowerPoint slide). Plots show
HR per SD and 95% CIs on a common log axis per analyte. Exact p/q values and
modeled/observed counts remain in the adjacent CSV. Symbols distinguish nominal-only p < 0.05,
FDR q < 0.05, and nonsignificance. q-values are not recalculated on this subset.
Finite positive HR estimates are retained; missing/invalid estimates or CIs are
explicitly marked instead of applying the former volcano magnitude filters.
Both across-site and individual within-site HRs are plotted, without re-pooling
site estimates. Within-site forests read
`within_site_cox_univariate/cox_within_site_all_sites_results.csv` beside
the across-site results CSV. Each uses its own supplied p/q values.
No matching local results are required. Outputs are
`03_univariate/federated/{psa,testosterone}_forest__platinum.{png,csv}` plus the day-0
incidence figure/table. Supplied p/q values are not recomputed or pooled.
The federated population is not assumed to include every displayed site: matching
patient/event counts across landmarks are reported as inferred membership only;
ambiguous or unavailable membership is explicitly unverified. When pooled counts
exceed the sum over every within-site export at each landmark (a site such as MSK
that pooled but has no within-site results), the excess is reported as unattributed.
These forests run when canonical ADT/platinum is selected, or scope
is `federated`. Missing optional sources are reported and skipped. Failures in
one figure set do not prevent preparation/rendering of other available sets.

Federated multivariable models are read beside the Cox results: elastic-net Cox
from `cox_federated_elasticnet_{metrics,coefficients}_adt.csv` and XGBoost from
`xgboost_federated_{metrics,importance}_adt.csv`, either in a v2 variant directory
(`federated_elastic_net_cox_w_DFCI/`) or one per landmark (`<model>/lm_<days>/w_DFCI/`,
read together), each optionally under `adt/`. The legacy `federated_cox_multivariate/`
and `federated_xgboost/` directories take precedence when present.
Each model gets two compiled figures built with the local PROFILE Figure 4
plotting functions, in `04_multivariate/federated/`. `{elasticnet,xgboost}_performance__platinum` pairs
held-out test mean AUC(t) for labs versus the age baseline at days
0/90/180. `elasticnet_coefficients__platinum` shows the three landmarks' top 15
nonzero coefficients by absolute value, and `xgboost_importance__platinum` the top
15 positive-gain features. Both use the local labels, clinical-category colors,
and age/body-height display exclusions. Unlike the selected HR forests, these
panels include all supplied feature statistics, including deltas, observation
counts, and `__<stat>__missing` indicators (shown as "<stat> missing").
No models are refitted; training/tuning-CV metrics are not substituted for test
performance. The horizon exports lack a train/test identifier and are not plotted.
CSV sidecars retain source metadata, counts, all feature values, and display flags.
Identifier features with nonzero gain or coefficients trigger a warning and a
figure caption; they are not silently removed or treated as validated predictors.
Manuscript Figure 8 requires all four captured CSVs.
The same exports refit without Dana-Farber (`no_DFCI` in place of `w_DFCI`; legacy
`<directory>_no_DFCI/`) render as `04_multivariate/federated/{elasticnet,xgboost}_*_no_dfci__platinum`
and feed federated manuscript Figure 4.
When a federated DeepHit metrics export is present (legacy `fast_federated_deephit/`
or the v2 per-landmark `deephit_federated_metrics_adt.csv` files), its dynamic arm
appears in the federated multivariate discrimination panel and Figure 4;
missing DeepHit results leave the Cox/XGBoost figure renderable.
The v2 predictive checklists (`federated_predictive_checklists/lm_<days>/w_DFCI/`)
render as `04_multivariate/federated/checklist_supplement__platinum`: held-out
test AUC(t) per landmark above the 1-year and 5-year checklist cards, one column
per landmark, with `_items`/`_metrics` sidecars. The no-DFCI checklists appear only
in the transfer supplement.

Aggregate KM curves and trajectories (`COMPASS_FEDERATED_FIGURE_DATA`, default
`figure_data/`) need `manifest.json` (bin width, minimum bin size, sites and per-site
counts) plus `cohort_km.csv`, `lab_extreme_{km,summary}.csv` and
`lab_trajector{ies,y_cohort}.csv`, each with a `within_site_` twin. They render
`01_cohort/federated_cohort_km` and, in `03_univariate/federated/`, `trajectories/<lab>/federated_trajectory_<lab>`, `km/<lab>/federated_km_quintile_<lab>_landmark<N>`
(pooled cutoffs) and `_by_site` pages (one panel per site, site-specific cutoffs),
with `federated_volcano_landmark<N>` from the pooled univariate CSV. The local
trajectory and KM plotters are reused; nothing is refitted. Missing files are reported and skipped.

Federated transfer (`COMPASS_FEDERATED_TRANSFER_NO_DFCI`, default
`federated_transfer_adt_no_DFCI/`, 04a's `OUTPUT_DIR`) reads `transfer_metrics.csv` and
`transfer_auc_t.csv` written by `04a_transfer_updated.ipynb`: the platinum-trained
federated XGBoost and elastic-net bundles refit without Dana-Farber and scored on the
local ADT cohort for platinum, NEPC, and AVPC, an external validation.
`05_transfer/transfer_discrimination_no_dfci__platinum` shows IPCW mean AUC(t) and the
paired gain over the age-only model by landmark, faceted by endpoint. CSV sidecars keep
all metrics (`transfer_discrimination_no_dfci__platinum.csv`) and per-horizon AUC(t)
(`transfer_auc_t_no_dfci__platinum.csv`). `transfer_clinical_comparison.csv`
(04's Gleason / TP53-RB1-PTEN / Gleason + TP53-RB1-PTEN section) renders
`05_transfer/transfer_clinical_comparison_no_dfci__platinum` and feeds Federated Figure 5 b-d;
without it, Figure 5 is drawn with panel a only. `transfer_km_patients.csv` (04's patient-level
transferred labs scores with Gleason and TP53/RB1/PTEN status on the ADT and sequencing clocks)
renders `05_transfer/km/`: score × TP53/RB1/PTEN and top-vs-bottom-10% score KM curves per
endpoint, a head-to-head grid (score | Gleason | TP53/RB1/PTEN) and the 5-year event-free
gradient (score quartile, Gleason, six TP53/PTEN/RB1 alteration groups) per bundle × landmark. Hazard-ratio
forests and score × Gleason curves are no longer drawn. Only aggregate tables are exported
(`transfer_km_{curves,summary,gradient}_no_dfci`); they feed Federated Figure 5 e-h
(XGBoost, 180 days). Missing transfer outputs are reported and skipped.
`all_models/transfer_metrics.csv`, when present, adds the horizon-specific
predictive checklist supplement at `05_transfer/transfer_predictive_checklist_supplement__platinum`.
With `all_models/transfer_auc_t.csv`, a second supplement compares the frozen
Cox, XGBoost, landmark/dynamic DeepHit and checklist AUC(t) on common DFCI horizons.
Missing optional XGBoost files are reported and skipped independently. Run `06`
again after this update to register both files (including absent paths), then
run `07_figures/all.R` with stage `all`. All Python remains in `06`; graphics caches stay in data.


## Entry points

Run `Rscript COMPASS/survival_analysis/07_figures/all.R` for the complete workflow, or run a numbered script in this directory for one manuscript figure and its section supplements. Run notebook `06_prep_figure_data.ipynb` first. Set `CAIA_PROJECT_ROOT` when using a checkout outside the cluster default. Each script honors the same `COMPASS_*` settings and `COMPASS_FIGURE_STAGE` (`all`, `prepare`, or `render`). Figure data and legends remain alongside exported figures in their `data/` folders; patient and graphics caches remain under `COMPASS_FIGURE_DATA_ROOT`.

The entry points also work with `source("/absolute/path/to/07_figures/<figure>.R")`; they infer the code root from the sourced file when `CAIA_PROJECT_ROOT` is unset.

The numbered scripts share the same preparation functions. Preparing one figure may compute additional source panels needed by its manuscript layout; rendering is restricted to that figure family. Run `all.R` to render and reconcile the full catalog.
