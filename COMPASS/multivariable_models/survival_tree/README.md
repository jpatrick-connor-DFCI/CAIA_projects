# survival_tree (local DFCI)

A shallow survival tree (depth 1 to 3) fit on the local DFCI ADT cohort for **time to platinum**. The same out-of-fold
scores are then evaluated for **time to NEPC** and **time to AVPC**. This is the single-site counterpart of
`caia-project-compass/rhino_scripts/feature_set_survival_tree`. That app's README gives the full model description.

Notebook: `COMPASS/notebooks/03d_shallow_decision_tree.ipynb`. It runs this package and scores the federated
(no-DFCI) tree bundle on the same rows.

## Model

The model code is not reimplemented here. `_vendor/` holds byte-identical copies of the federated app's modules:

- `survival_tree_core.py`: rpart `method = "exp"`. Time is rescaled to Λ0(T), splits use the Poisson deviance gain, and the `cp = 0` collapse is applied. Leaves carry a shrunken rate and a KM curve at 7-day units.
- `checklist_core.py` and `federated_checklist.py`: the checklist items.

The frames come from `survival_common.federated_endpoint_transfer` (`to_federated_long` → `apply_exclusion` →
`build_landmark_frame`). The features are therefore named and computed exactly as in the federated bundles, and
they are identical across endpoints.

- **Items.** The candidate features are age plus `<lab>__{mean,min,max,last,delta}` (allowlist), keeping labs with coverage ≥ 20 %.
  - Each feature's deciles on the fit rows are rounded to clinical cutoffs, giving `<=` and `>=` items.
  - A missing value makes the item false.
  - Items that are constant on the fit rows are dropped.
  - Every fit scope (each inner fold and each refit) builds its own items from its own rows.
- **Configs.** `both` may split on every item. `baseline` may split on age items only.
- **Selection.** Inner 5-fold CV over `max_depth ∈ {1,2,3}` × `min_leaf_frac ∈ {0.02, 0.05, 0.10}`.
  - The key is mean AUC(t) when some candidate has it in every fold, otherwise Harrell's C.
  - Ties go to higher C, more valid folds, fewer leaves, the larger frac, then the smaller depth.
  - The selected tree is refit on the selection rows.

## Cross-fitting

This follows `COMPASS/federated_models/checklist_extension.py`:

1. One patient-level 5-fold split, stratified on any platinum event, is shared by every landmark and endpoint.
2. For each outer fold k, the tree is selected and fit on the **platinum rows of the other folds only**. It is then
   applied to fold k's platinum, NEPC and AVPC rows.
3. NEPC and AVPC labels never enter fitting or selection. `test_transfer_labels_never_reach_the_trees` checks this
   by permuting them.
4. A refit on the whole platinum cohort (`fold = "all"`) is the tree one would report.
   - It is descriptive only: its in-sample metrics are not computed.
   - It is exported as a `survival_tree` bundle in the federated schema.

**Scores.** The risk score (used for C and the Cox slope) is the leaf's shrunken rate. AUC(t) and Brier score the
leaf KM failure `1 − S_leaf(t)` on the federated week grid.

Each fold tree has its own Λ0, so rates from different folds are on slightly different scales when pooled for C.
`S_leaf(t)` is absolute, so AUC(t) is not affected.

**Inference.** Both configs are scored on the same patients, so the bootstrap deltas against `baseline` are paired.
The bootstrap is conditional on the fold trees: it does not include re-selection.

## Outputs

| File | Content |
| --- | --- |
| `local_survival_tree_metrics.csv` | per endpoint × landmark × config: C with CI, Uno's C, mean AUC(t) with CI, IBS, Cox slope, and paired Δ vs `baseline` |
| `local_survival_tree_auc_t.csv` | AUC(t) on the valid week grid |
| `local_survival_tree_selection.csv` | selected depth, `min_leaf` and splits per landmark × outer fold (`0`–`4`, `all`) × config |
| `local_survival_tree_leaves.csv` | leaf rule, n, events, rate, relative rate, and S(365d) and S(1825d) with log-log bands |
| `local_survival_tree_inner_cv.csv` | inner-fold rows and per-candidate summaries |
| `local_survival_tree_patients.csv` | **patient level**: out-of-fold leaf and score per endpoint. Keep it under the protected data root |
| `local_survival_tree_model.json` | the `fold = "all"` refits as a `caia-federated-model-bundle` (`training_sites: ["DFCI"]`, `local_refit: true`) |

The bundle scores through `survival_common.federated_inference.predict_tree` and `fet.evaluate_transfer`, like a
federated tree. **Do not** score it on DFCI as a validation: it was fit on those rows.

## CLI

```sh
python -m COMPASS.multivariable_models.survival_tree --output-dir <dir> \
    [--data <longitudinal_prediction_data_adt.csv>] [--lab-mapping <json>] \
    [--config <json overriding TreeConfig>] [--n-bootstrap 200] [--seed 0]
```

On the full grid, each (landmark, outer fold) runs 2 configs × 3 fracs × 5 inner folds. On 300 synthetic patients
the test configuration (2 landmarks, 20 bootstraps) takes about 3 s.

## Vendoring rule

Edit `_vendor/*.py` only in `rhino_scripts/feature_set_survival_tree/` (the checklist modules in
`feature_set_predictive_checklist/`). Then copy them here and update the sha256 values in `_vendor/VENDORED.json`.
`tests/test_local_survival_tree.py` checks the manifest. When `caia-project-compass` is checked out beside this repo,
it also checks byte equality with the federated app.

## Tests

```sh
python -m pytest tests/test_local_survival_tree.py
```

The tests cover:

- vendoring;
- items: rounded deciles, no constant items, missing values make an item false, and `baseline` splits on age only;
- fold trees equal a refit on the other folds' platinum rows;
- NEPC and AVPC labels permuted → identical trees and platinum scores;
- C and every AUC(t) point equal `harrell_c` and sksurv on the out-of-fold scores;
- leaf-table sanity (lower ≤ S ≤ upper, leaf n sums to the fit n);
- selection tie-breaks;
- the bundle round trip through `federated_inference` and `evaluate_transfer`.
