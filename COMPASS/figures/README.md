# COMPASS figure workflow

!/usr/bin/env Rscript
COMPASS figures: cached preparation and rendering (stage 07, after notebook 05).
Configured entirely through environment variables, e.g.
  Rscript COMPASS/figures/all.R
  COMPASS_FIGURE_SCOPE=federated Rscript COMPASS/figures/all.R
  COMPASS_FIGURE_STAGE=render Rscript ...
Converted from 05_figures.Rmd; the former document text follows as comments.

---- Workflow --------------------------------------------------------------

Run `05_prep_figure_data.ipynb` first to prepare the Polars tables and manifest.
All Python execution belongs to that notebook. This script and its R helpers
never invoke Python: they consume the notebook's prepared data only. Missing,
changed, or incompatible inputs stop with instructions to rerun the notebook.

Preparation and rendering are separate stages:

1. `05_prep_figure_data.ipynb` uses Polars to scan each selected arm's longitudinal
   CSV once and cache patient metadata, all-lab record spans, selected lab
   measurements, patient/bin raw and log means, and pre-treatment coverage.
2. R consumes those tables, calculates the existing survival statistics and
   optional GAMs, and caches finished graphics objects. CSV/Markdown figure
   tables are written during this stage; no images are generated.
3. Rendering loads the graphics objects and writes individual and multipanel PDFs.
   It does not read patient inputs, fit curves, or repeat plot calculations.

`COMPASS_FIGURE_STAGE=all` (default) checks the notebook manifest and source
metadata, prepares changed R figure sets, then renders missing or stale PDFs.
`prepare` builds R graphics caches only; it does not prepare Python tables.
`render` uses the last prepared snapshot without checking
source data, and requires the same cohort/analysis settings. Use `all` for
normal updates and `render` to re-export PDFs from the last snapshot.

`COMPASS_FIGURE_SCOPE=federated` processes only the federated forests, site cohort report,
elastic-net/XGBoost performance and importance, aggregate KM curves and PSA/testosterone
trajectories, platinum-to-NEPC/AVPC transfer figures, and federated manuscript Figures 3-6,
without requiring longitudinal data or running other figure families.

Every federated input defaults to the delivery under `COMPASS_RHINO_ROOT`
(default `<COMPASS_DATA_ROOT>/rhino_data_v2`):
  cox_federated_univariate_adt.csv  pooled univariate Cox results
  within_site_cox_univariate/       within-site Cox results and per-site cohort counts
  federated_elastic_net_cox_{w,no}_DFCI/
                                    elastic-net Cox exports with and without DFCI
  federated_*/lm_<days>/{w,no}_DFCI/ per-landmark XGBoost, DeepHit and checklist exports
  federated_transfer_adt_no_DFCI/   04's output (bundles trained without DFCI on DFCI endpoints)
  figure_data/                      aggregate KM curves and binned trajectories, pooled and within site
The earlier rhino_results layout (federated_model_results/, misc_figure_data/) still
resolves when the paths below point at it.
`COMPASS_FEDERATED_RESULTS`, `COMPASS_FEDERATED_TRANSFER_NO_DFCI` and
`COMPASS_FEDERATED_FIGURE_DATA` override the individual paths; set the same values for `05`.

The site report reads `within_site_cox_univariate/cox_within_site_all_sites_cohort.csv`
(legacy names `nvflare_within_site_{univariate_cox,cox_univariate}/` are still accepted)
beside the federated results CSV. It exports cohort sizes, event counts, and observed
event percentages at landmark 0 only in `01_cohort/site_incidence_lm000__platinum.{pdf,csv}`.
The figure matches the cohort-comparison layout: incidence with 95% Wilson CIs
on the left and patient counts with events overlaid on the right. Cells with
fewer than 25 events are hatched/marked. Percentages are events divided by analyzed patients over available follow-up,
not fixed-horizon cumulative incidence estimates. Missing site counts are reported
as unavailable; forest plots still render if the site file is absent.

Checklist supplement (Figure 4S): `05` flattens the predictive-checklist runs at
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
Use `FORCE = True` in `05_prep_figure_data.ipynb` after replacing sources while
preserving timestamps and sizes. Here, `COMPASS_PREPARE_OVERWRITE=true`
rebuilds R graphics caches only. Image receipts record the prepared snapshot and page size;
affected PDFs regenerate automatically. Existing images without a receipt
are regenerated once.

The notebook's Python kernel requires Polars. R needs its existing plotting
packages plus `nanoparquet` or `arrow`; no Python installation is needed to run this script.
All figure writers prefer Arial, with shared point sizes from
`survival_common/figure_typography.json`: body text, annotations, tick labels,
captions and legends are 8 pt; axes and facet headings are 9 pt bold;
subtitles are 9 pt; titles are 11 pt bold; panel letters are 12 pt bold.
The same settings apply to individual panels, manuscript composites,
checklist supplements and the Python survival plots. If Arial is unavailable,
the writers try Liberation Sans, then DejaVu Sans, then the default sans-serif
font, and report the substitution once per session. Sizes and styling stay the
same. R uses `systemfonts` to check installed families; without it, R uses the
default sans-serif font. The Cairo PDF device explicitly selects the resolved
font for text and mathematical symbols. PDF exports embed that font.

For actual Arial output, install Arial on the rendering host and restart the
Python/R session before rebuilding. A font installed on a laptop is not
automatically available to a remote notebook kernel. After updating this code,
restart the notebook kernel (or reload `survival_common.plotting`) to replace
the earlier strict font check.

Use `COMPASS_FIGURE_STAGE=all` after typography changes so the pipeline measures
and rebuilds the graphics caches before exporting. The font configuration is
part of the cache signature, along with the resolved font.

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
no log-rank). `02a_cohort_comparison.ipynb` draws none of these; it reports
the underlying tables only. The incidence denominator
is each endpoint's aggregated landmark input, not a feature-specific Cox fit.
It is the observed follow-up event fraction, not cumulative risk by day 180.
Cells with fewer than 25 events are hatched/marked; missing cells stay blank.
The label figure uses `mrn_lists/adt_intent_labels_model_cohort.csv` and
`llm_met_labels_model_cohort.csv`, preserving missing labels explicitly.
Run notebook `05` once after this update to prepare these small summary tables.
Combined outputs are under `03_univariate/cohort_sensitivity/` (count tables in `03_univariate/cohort_sensitivity/data/`):
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
confusion matrices also form `03_univariate/cohort_sensitivity/metastatic_label_agreement__<endpoint>__<subset>__<exclusion>`.
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
in notebook 05.

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

Each arm root (`ADT/`, `ARPI/`) has five numbered section folders:
  01_cohort/                 raw cohort numbers: Figure 1/Table 1, analysis sets,
                             pre-ADT coverage, federated site incidence and cohort KM
  02_llm/                    LLM NEPC/AVPC subtype labels and their KM curves
  03_univariate/internal/    DFCI univariable labs (volcanoes) and somatic/Gleason
                             associations; `km/` holds KM curves (co-mutation,
                             somatic, Gleason) with lab KM curves in `km/<lab>/`;
                             `trajectories/<lab>/` holds lab trajectories and distributions
  03_univariate/federated/   pooled PSA/testosterone forests and volcanoes, with
                             `km/<lab>/` and `trajectories/<lab>/` (pooled and by site)
  03_univariate/cohort_sensitivity/  Figure 3 supplement: cohort forests,
                             endpoint incidence, metastatic-label comparisons,
                             and `<subset>__<exclusion>/<section>/` for other cohorts
  04_multivariate/internal/  DFCI Elastic-Net/XGBoost discrimination, importance,
                             somatic sensitivity, combinations, risk strata
  04_multivariate/federated/ federated models, including the no-DFCI refits
  05_transfer_survival/      no-DFCI survival models scored at Dana-Farber
  06_transfer_checklists/    predictive checklist transfer figures
Local filenames are `<figure>__<endpoint>__<subset>__<exclusion>.pdf`; federated
ones are `<figure>__platinum.pdf`. Every export folder holds only PDF figures:
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
(e.g. `ADT/04_multivariate/internal/manuscript/04_multivariable_labs.pdf`). These
additional exports generally use a 7.2-inch manuscript width, consistent 7–9 pt plot text,
lettered panels without subplot titles, and separate `.md` figure legends in
`manuscript/data/`. `manuscript/captions/` contains manuscript-ready legends as matching
`figureN.md` and `figureN.txt` files (`federated_figureN.*` for the federated set). They always include a
vector PDF on every render. Individual panels are always retained.

The mean-only Figure 3, transfer Figure 5 and checklist Figure 6 designs export to
`manuscript/` within their respective sections:

- `ADT/03_univariate/internal/manuscript/03_dfci_univariable.pdf`
- `ADT/03_univariate/federated/manuscript/federated_03_univariable.pdf`
- `ADT/05_transfer_survival/manuscript/federated_05_transfer_no_dfci.pdf`
- `ADT/05_transfer_survival/manuscript/federated_05_transfer_no_dfci__no_d.pdf`
- `ADT/06_transfer_checklists/manuscript/federated_06_checklist_no_dfci.pdf`
- `ADT/05_transfer_survival/manuscript/federated_s1_feature_set.pdf`

Their tables/legends and captions use `manuscript/data/` and
`manuscript/captions/`. Run stage `all` once after this path change to rebuild the cached destinations and replace the previous manuscript exports.

The numbered manuscript set is: `01_cohort_overview`, `02_llm_nepc_avpc__<icd|adt|lm0>`
(one per Figure 2 cohort, captions `figure2_<cohort>`, with a cohort header line),
`03_dfci_univariable`, `03_cohort_sensitivity_supplement` (paired platinum/NEPC
incidence and counts plus day-180 PSA/testosterone associations),
`04_multivariable_labs` (held-out Elastic-Net/XGBoost discrimination versus age
baselines),
`07_federated_incidence_associations` (landmark-0 site incidence and counts plus
PSA/testosterone associations on natural-log hazard-ratio axes), and
`08_federated_multivariable` (the local Figure 4 layout for the federated
elastic-net Cox and XGBoost models: test performance, coefficients, and gains).
A separate federated set, written by the federated job, is numbered on its own:
`federated_03_univariable` (the Figure 3 layout from the pooled four-site univariable Cox
results and the aggregate `figure_data` PSA/testosterone trajectories and day-180
extreme-decile KM curves; mean-only association panels show risk-oriented HR per SD on a logarithmic axis
in the labeled higher/lower direction against -log10(q), capped at 30 with triangles),
`federated_04_multivariable` (pooled four-site models, without the XGBoost age-only baseline;
its feature panels rank up to 15 nonzero coefficients or positive gains after excluding age,
height, identifiers, observation counts and missingness indicators). The companion
`federated_04_multivariable__full_features` includes both age baselines and every supplied
feature, including zero-valued features and any identifiers, for audit; and
`federated_05_transfer_no_dfci` (the XGBoost model trained without Dana-Farber scored on
Dana-Farber platinum/AVPC/NEPC, in that order: a, mean AUC(t) of the full model at each
landmark, with facet strips giving day-0 patients and events; b-d, platinum/AVPC/NEPC KM
curves for the top vs bottom 10% of the day-180 score on the ADT clock; e, 5-year platinum
probability (1 - KM) from sequencing date +180 days in sequenced patients, by top vs bottom
10% of the score, Gleason ≤7 vs ≥8, and 0 vs ≥1 of TP53/RB1/PTEN altered. The deciles in e
are recomputed within sequenced patients) and
`federated_06_checklist_no_dfci` (the unit-weight 5-year platinum checklist from +180 days,
trained without Dana-Farber: a, checklist items and held-out metrics at the training sites;
b, 5-year IPCW AUC at Dana-Farber, which equals balanced accuracy for a yes/no rule, of the
checklist, Gleason ≥8 and any TP53/RB1/PTEN alteration, in the same sequenced patients with
a Gleason score on the sequencing clock; c-e, platinum/AVPC/NEPC KM curves by checklist
status from the 180-day landmark on the ADT clock).

`federated_s1_feature_set` (caption `federated_figure_s1`) compares the regular models with
the feature-set refits in `rhino_data_v2/feature_set_models/`, where missingness flags and
observation counts adjust the fit but cannot enter a score or checklist: a, held-out platinum
AUC of models trained with Dana-Farber (elastic-net Cox, XGBoost, 1-year and 5-year
checklists); b-d, the 5-year feature-set checklists trained without Dana-Farber at +0, +90
and +180; e, horizon-matched AUC of the frozen no-DFCI checklists at Dana-Farber, feature set
vs regular; f-h, platinum/AVPC/NEPC KM curves by feature-set checklist status from +180 on
the ADT clock. Panel a labels the feature-set XGBoost slot "not yet available" until
`feature_set_models/w_DFCI/federated_xgboost/xgboost_federated_metrics_adt.csv` exists; the
points then appear after rerunning 05, with no code change.

`federated_05_transfer_no_dfci__no_d` (caption `federated_figure5_no_d`) keeps the fuller
layout for both model families: a, mean AUC(t) of the full models; b-c, the full models,
local out-of-fold Cox models of Gleason + age (b) or TP53/RB1/PTEN + age (c), and combined
Cox models of each transferred score + the same covariates without age (the full models
include age), in patients with a Gleason score before the ADT landmark (b) and sequenced
patients (c, on a sequencing-date clock); d-f, the KM curves; g, 5-year event-free
probability by score quartile, Gleason score and six mutually exclusive mutation groups
(WT, TP53, PTEN, RB1, 2 MT, 3 MT).

Figure 5 reads existing 04 exports: rerun `05_prep_figure_data.ipynb`, which registers the
checklist tables in the manifest, then `COMPASS/figures/federated_05_transfer_no_dfci.R`. Figure 6
needs the checklist cells of `04_federated_transfer.ipynb` first, then
`05_prep_figure_data.ipynb`, then `COMPASS/figures/federated_06_checklist_no_dfci.R`. Without the
checklist tables it is skipped with a message. Figure S1 needs the last section of
`04_federated_transfer.ipynb` (output `federated_transfer_adt_no_DFCI/feature_set_checklists/`),
then `05_prep_figure_data.ipynb`, then `COMPASS/figures/federated_s1_feature_set.R`, and is
skipped the same way without those tables.

Figure 5 discrimination panels now default to horizontal confidence intervals,
with directly labeled model rows and one color key for the landmarks. All three
landmarks remain included. `COMPASS_TRANSFER_LAYOUT=scatter`
restores the original point/shape layout; `COMPASS_TRANSFER_LANDMARKS=180` focuses
the discrimination and clinical-comparison panels on day 180, including its actual
patient/event counts. These settings enter the graphics-cache signature and the
captions. KM/gradient panels remain at day 180 with their original time origins.

Two additional Figure 5 alternatives are exported automatically to
`05_transfer_survival/manuscript/`:

- `federated_05_transfer_no_dfci__mean_landmarks_set1`: three panels showing
  federated scores in the full cohort, Gleason comparisons, and somatic comparisons.
- `federated_05_transfer_no_dfci__mean_landmarks_set2`: two panels showing
  federated scores in the full cohort and comparisons in the combined Gleason +
  somatic cohort (clinical only, score only, and score + clinical covariates).

Each labeled point is `(mean_auc_day0 + mean_auc_day90 + mean_auc_day180) / 3`,
computed separately for each endpoint, cohort and model category. These variants
always use all three landmarks, independently of `COMPASS_TRANSFER_LANDMARKS`.
Missing or nonfinite landmark estimates make the average unavailable. Clinical-only
rows repeated across bundles are counted once per landmark. No confidence interval
is drawn: the overlapping landmark cohorts require joint bootstrap information to
estimate uncertainty for their average. Source AUCs and patient/event counts at each
landmark are saved beside each figure in `data/`; patient counts are not summed.
These compact variants omit the KM and gradient rows. The aggregate preview script
writes the same designs as `figure5_mean_landmarks_set1` and `figure5_mean_landmarks_set2`.

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
In both univariable manuscript figures, the three landmark association plots use an
expanded first row, compact horizontal spacing, a shared y-axis displayed
through -log10(q)=30 (larger values capped with triangles), centered landmark
headers, and one horizontal category legend. Only the mean statistic is included.
Benjamini-Hochberg q-values are recalculated per endpoint/analysis/landmark across
all available mean-feature tests with valid P values, before height, effect-size,
CI, or annotation display filters. Raw two-sided P values are unchanged. The
association tables retain `q_value_original`, `q_value_mean`, `n_mean_tests`, and
the original signed coefficient; `q_value` now refers to the mean-only adjustment.
Negative coefficients are displayed as positive magnitudes with "Lower" labels:
HR becomes 1/HR and confidence limits become [1/upper, 1/lower]. This reverses
the direction of a one-SD change in the modeled feature, rather than transforming
the lab value to its reciprocal. Directions are descriptive and selected from
the fitted coefficients. The horizontal threshold is fixed at q=0.05. The
separate site/cohort forest supplements retain their original statistics and
supplied q-values. Panels e and g use a two-line
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
legend rather than as a caution footer on the plotted page.
Each arm gets a `data/index.html` that embeds each PDF, and a `data/manifest.csv` with relative paths,
titles, endpoint, cohort, and landmark metadata. No duplicate preview files are made.

R compiles classifier validation into one figure and subtype/enrichment into one;
AUC(t) into paired comparisons; importance into one three-landmark sheet per
model; lab-extreme KM into one three-landmark sheet per lab; and mutation KM into
pages of at most four panels. Model fits, counts, time origins, and tests are unchanged.
Local univariate lab volcanoes retain individual day-0/90/180 plots and also
export `03_univariate/internal/labs_univariate_all_landmarks__<endpoint>__<subset>__<exclusion>`.
The combined figure uses three side-by-side panels on a 24 × 9-inch canvas,
common axes, a shared category legend, and separate per-landmark significance
counts. Larger feature labels (approximately 12 pt) and axis text are preserved.
The cohort and label figures described above also retain their
individual panels, as do the lab KM curves; other compiled figure families do not export component images.

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
`03_univariate/federated/{psa,testosterone}_forest__platinum.{pdf,csv}` plus the day-0
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
`federated_transfer_adt_no_DFCI/`, 04's `OUTPUT_DIR`) reads `transfer_metrics.csv` and
`transfer_auc_t.csv` written by `04_federated_transfer.ipynb`: the platinum-trained
federated XGBoost and elastic-net bundles refit without Dana-Farber and scored on the
local ADT cohort for platinum, NEPC, and AVPC, an external validation.
`05_transfer_survival/transfer_discrimination_no_dfci__platinum` shows IPCW mean AUC(t) and the
paired gain over the age-only model by landmark, faceted by endpoint. CSV sidecars keep
all metrics (`transfer_discrimination_no_dfci__platinum.csv`) and per-horizon AUC(t)
(`transfer_auc_t_no_dfci__platinum.csv`). `transfer_clinical_comparison.csv`
(04's Gleason / TP53-RB1-PTEN / Gleason + TP53-RB1-PTEN section) renders
`05_transfer_survival/transfer_clinical_comparison_no_dfci__platinum` and feeds Federated Figure 5 b-d;
without it, Figure 5 is drawn with panel a only. `transfer_km_patients.csv` (04's patient-level
transferred labs scores with Gleason and TP53/RB1/PTEN status on the ADT and sequencing clocks)
renders `05_transfer_survival/km/`: score × TP53/RB1/PTEN and top-vs-bottom-10% score KM curves per
endpoint, a head-to-head grid (score | Gleason | TP53/RB1/PTEN) and the 5-year event-free
gradient (score quartile, Gleason, six TP53/PTEN/RB1 alteration groups) per bundle × landmark, all on the sequencing clock (sequenced patients). Hazard-ratio
forests and score × Gleason curves are no longer drawn. Only aggregate tables are exported
(`transfer_km_{curves,summary,gradient}_no_dfci`); they feed Federated Figure 5 e-h
(XGBoost, 180 days). Missing transfer outputs are reported and skipped.
`all_models/transfer_metrics.csv`, when present, adds the horizon-specific
predictive checklist supplement at `06_transfer_checklists/transfer_predictive_checklist_supplement__platinum`.
With `all_models/transfer_auc_t.csv`, a second supplement compares the frozen
Cox, XGBoost, landmark/dynamic DeepHit and checklist AUC(t) on common DFCI horizons.
Missing optional XGBoost files are reported and skipped independently. Run `05`
again after this update to register both files (including absent paths), then
run `COMPASS/figures/all.R` with stage `all`. All Python remains in `05`; graphics caches stay in data.

The transfer notebooks' additional **locally trained** score + Gleason + mutation
checklists write `clinical_checklists/checklist_{cells,items_long,metrics}.csv`.
These fit 1-year and 5-year horizons using each frozen no-DFCI Cox/XGBoost score on
the sequencing clock, Gleason and TP53/RB1/PTEN status. Existing patient splits
isolate local held-out patients from threshold selection, calibration and censoring
estimation. When endpoint inputs are absent, the runner uses the existing platinum
split and records its provenance; target outcomes remain endpoint-specific.
Each family × endpoint gets a supplemental figure with
held-out AUC(t) intervals and both horizons' M-of-N cards across landmarks:
`06_transfer_checklists/transfer_clinical_checklist_<family>_<endpoint>__platinum`.
The title identifies the evaluated endpoint; `__platinum` follows the source
federated-model figure convention. These fits are separate from the frozen
federated-checklist transfer supplement. Missing optional tables skip the figure;
unsupported cells show their reasons. Rerun 05 after training to refresh the
registered source fingerprints before rendering Figure 5 or `all.R`.


## Entry points

To compare designs using an existing aggregate export tree without the cluster
inputs, run from the repository root:

```sh
Rscript COMPASS/figures/preview_03_05.R /path/to/ADT
```

This writes PDF previews for the one-sided mean association plots, a selected
lab forest, all-landmark and day-180 transfer intervals, and paired day-180 AUC
gains. The gain intervals use exported paired-bootstrap differences. Preview
images go to `03_univariate/federated/manuscript/` and
`05_transfer_survival/manuscript/` within the supplied ADT directory, with
preview tables and notes in each folder's `data/` subdirectory. An optional
second argument selects a different output ADT root using the same structure.
FDR values use the available exported means; older exports have already passed
display filters, so final manuscript q-values require the raw-input production
run. Use stage `all`, not `render`, to rebuild the updated scenes.

Run `Rscript COMPASS/figures/all.R` for the complete workflow, or run a numbered script in this directory for one manuscript figure and its section supplements. Run notebook `05_prep_figure_data.ipynb` first. Set `CAIA_PROJECT_ROOT` when using a checkout outside the cluster default. Each script honors the same `COMPASS_*` settings and `COMPASS_FIGURE_STAGE` (`all`, `prepare`, or `render`). Figure data and legends remain alongside exported figures in their `data/` folders; patient and graphics caches remain under `COMPASS_FIGURE_DATA_ROOT`.

`retained_exports.txt` records the curated PDF output set based on `Desktop/ADT` on 2026-10-26, with cohort sensitivity relocated under Figure 3. Every figure entry point renders only paths in that list under `COMPASS_FIG_ROOT/ADT`; this includes `all.R`. Prepared graphics caches and data sidecars may still contain inputs for assembling the retained figures. To restore an intentionally deleted render, add its relative PDF path to the list.

The entry points also work with `source("/absolute/path/to/COMPASS/figures/<figure>.R")`; they infer the code root from the sourced file when `CAIA_PROJECT_ROOT` is unset.

The numbered scripts share the same preparation functions. Preparing one figure may compute additional source panels needed by its manuscript layout; rendering is restricted to that figure family. Run `all.R` to render and reconcile the full catalog.
