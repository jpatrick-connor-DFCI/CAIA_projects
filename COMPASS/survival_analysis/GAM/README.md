# Independent GAM lab analysis

This directory owns every GAM-derived lab feature and analysis. The main
`COMPASS/survival_analysis` pipeline does not load GAM outputs; it models only
the aggregate lab statistics already present in `aggregated_landmark*.csv`.

## Data boundary

The main pipeline produces the cohort artifacts consumed here:

- `aggregated_landmark{D}.csv` for patient splits, outcomes, age, and the
  matching `<LAB>__n_observations` adjustment columns
- `pre_treatment_lab_long_landmark{D}.csv` for the longitudinal lab values
- `canonical_labs_train_val.csv` for the eligible lab set
- `build_manifest.json` and the split-assignment files for provenance

The GAM workflow treats that prediction-input directory as read-only. Stage A
writes trajectory features, fitted curves, and diagnostics to a GAM-specific
feature directory. Stage B joins those features to cohort metadata in memory,
selects only `__gam_*` columns using train/validation coverage, and writes its
selection and Cox results to a GAM-specific result directory.

No GAM CSV should be placed in `prediction_inputs_*`, and the main Python
survival models do not merge one if it is present.

## Run

Open `04_gam.ipynb` with an R kernel containing `mgcv` and `data.table`.
Configure `DATA_VARIANT`, `ARM`, `LANDMARK_DAYS`, and `N_WORKERS`, then run the
notebook after `01_preprocessing.ipynb` has built prediction inputs for the
matching `DATA_VARIANT`.

The default notebook output layout is:

```text
<data-root>/survival_analysis/GAM/<arm>/
├── features/
│   ├── gam_trajectory_features_landmark{D}.csv
│   ├── gam_fit_diagnostics_landmark{D}.csv
│   └── gam_trajectory_curves_landmark{D}.csv
└── nonlinearity/
    ├── gam_feature_selection_landmark{D}.csv
    └── gam_cox_nonlinearity_landmark{D}.csv
```

The Cox output includes both the linear GAM-feature association
(`coef_linear`, `p_linear`, `q_linear`) and the smooth-versus-linear comparison
(`edf`, `p_lrt`, `q_lrt`, `delta_aic`).

The notebook's final section writes figures to the shared figure repository at
`<figure-root>/<ARM>/gam_trajectories/`. It exports
`<arm>_gam_trajectory_curves.png`, one overall GAM-Cox volcano per landmark,
and one volcano per available landmark × lab-type combination. Volcano
filenames follow `<arm>_gam_cox_volcano_landmark{D}.png` and
`<arm>_gam_cox_volcano_landmark{D}_{lab-type}.png`.

## Smoke tests

Run from the repository root:

```sh
Rscript COMPASS/survival_analysis/GAM/tests/test_gam_trajectory_features_smoke.R
Rscript COMPASS/survival_analysis/GAM/tests/test_gam_cox_nonlinearity_smoke.R
```
