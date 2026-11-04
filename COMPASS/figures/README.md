# COMPASS figures

Manuscript figures for the ADT cohort, platinum endpoint. Only the PDFs listed in
`retained_exports.txt` (the curated `Desktop/ADT` export) are rendered; everything
else in the export tree is archived by a full run.

## Running

1. Run `COMPASS/notebooks/05_prep_figure_data.ipynb`. It is the only step that runs
   Python: it calls `prepare_figure_data.py` to scan the cohort data and federated
   deliveries into Polars tables plus a manifest under `COMPASS_FIGURE_DATA_ROOT`.
2. Run the two R entry scripts (both in `COMPASS/notebooks/`; the code they source
   stays in this folder):

   ```sh
   Rscript COMPASS/notebooks/final_figure_data_prep.R   # R-side data prep / cache
   Rscript COMPASS/notebooks/run_figure_render.R        # draw PDFs, write figure data
   ```

   Flip the `FIGURES` toggles at the top of `figure_run.R` to run a subset, or
   override them without editing the file:

   ```sh
   COMPASS_FIGURE_GROUPS=transfer_survival,transfer_checklists Rscript COMPASS/notebooks/run_figure_render.R
   ```

The R code never calls Python. Missing or changed prepared inputs stop with a
message to rerun notebook 05.

## Figure groups

| Toggle | Export folder (under `<COMPASS_FIG_ROOT>/ADT_figures/`; tables, legends and captions go to the same path under `ADT_figure_data/`) | Contents |
| --- | --- | --- |
| `cohort` | `01_cohort/` | Fig 1: consort, event-free survival, time to event, dx-to-treatment, record spans, site counts, `manuscript/01_cohort_overview` |
| `llm` | `02_llm/` | Fig 2: LLM NEPC/AVPC validation on the day-0 cohort (confusion matrix, enrichment, metric bars, subtype landscape) |
| `cohort_sensitivity` | `03_univariate/cohort_sensitivity/` | Fig 3 supplement: cohort forest and event incidence at day 180, ADT-vs-LLM and LLM-vs-regex metastatic labels |
| `federated_univariable` | `03_univariate/federated/` | Federated Fig 3: volcanoes (day 0/90/180), PSA/testosterone quintile KMs and trajectories, pooled and by site |
| `federated_multivariable` | `04_multivariate/federated/` | Federated Fig 4: elastic-net and XGBoost performance/importance, checklist cards |
| `transfer_survival` | `05_transfer_survival/` | Federated Fig 5: no-DFCI transfer KMs, discrimination, clinical comparison |
| `transfer_checklists` | `06_transfer_checklists/` | Federated Fig 6: transfer checklists, plus one panel per landmark x horizon |

`FIGURE_GROUPS` in `figure_workflow.R` maps each toggle to its prepared jobs
(`adt__platinum`, `cohort_overview`, `federated`) and export prefix; only the jobs
an enabled group needs are prepared. When every enabled group is federated-only,
the run needs no patient-level data. Archiving retired exports and flattening the
tree happen only when all groups are enabled.

## Stages

`COMPASS_FIGURE_STAGE`:

- `all` (default) checks the notebook manifest, prepares changed figure sets, then
  renders missing or stale PDFs.
- `prepare` builds the R graphics caches only.
- `render` re-exports PDFs from the last prepared snapshot without checking
  sources.

Figure tables (CSV/Markdown) go into each export folder's `data/`; graphics caches
stay under `COMPASS_FIGURE_DATA_ROOT`. Outputs are PDF only.

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

## Files

| File | Role |
| --- | --- |
| `figure_run.R` | Shared run logic for both entry scripts: toggles, paths, settings |
| `figure_workflow.R` | Cached prepare/render workflow, federated and manuscript figures, `FIGURE_GROUPS` |
| `COMPASS_generate_figures_pipeline.R` | Per-cohort DFCI figures (Figs 1-2) |
| `figure_supplements.R` | Cohort forest, cohort overview, metastatic-label prep and diagnostics |
| `checklist_supplement.R` | Predictive-checklist cards |
| `figure_typography.R` | Shared fonts and sizes (also read by the Python plots) |
| `prepare_figure_data.py` | Python preparation called from notebook 05 |
| `retained_exports.txt` | The rendered PDF set; add a path to restore a figure |
