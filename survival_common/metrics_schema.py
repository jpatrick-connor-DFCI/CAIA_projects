"""One canonical column schema for every `*_metrics*.csv` the pipeline writes.

Before this module the three model families each spelled the same quantities
differently -- elastic-net Cox wrote `test_c_index` / `n_events_train_val` /
`landmark_days`, XGBoost wrote `c_index` / `n_train_val_events` /
`landmark_day`, and Dynamic-DeepHit wrote `c_index` and named the endpoint
column `event`. Downstream readers (the R figure pipeline, `summarize_outputs`)
had to carry per-family candidate lists to paper over it.

The canonical spelling is the elastic-net Cox one, since it already
distinguished train-side from held-out metrics. Families that compute no
train-side twin emit the column as NaN rather than omitting it, so a reader can
assume presence and test `is.finite()` / `notna()` instead of testing for the
column.

This is a hard cutover: metrics CSVs written before it lack the canonical
columns and must be refit.
"""

from __future__ import annotations

# Which model family produced the row. One value per writer.
MODEL_ELASTIC_NET_COX = "elastic_net_cox"
MODEL_XGBOOST = "xgboost"
MODEL_DYNAMIC_DEEPHIT = "dynamic_deephit"
MODEL_SURVLATENT_ODE = "survlatent_ode"

# Patient-subset restriction the run was fit under (see COHORT_SPECS in
# compass_pipeline). "all" is the unrestricted arm cohort.
DEFAULT_COHORT = "all"

# Identity of the run: which model, which patient subset, which endpoint,
# which landmark, and which covariate configuration.
IDENTITY_COLUMNS = (
    "model",
    "cohort",
    "endpoint",
    "landmark_days",
    "config",
)

# Cohort sizes and event counts, train/validation block and held-out test.
COUNT_COLUMNS = (
    "n_train_val",
    "n_test",
    "n_events_train_val",
    "n_events_test",
)

# Discrimination and calibration. The `train_val_*` twins are NaN for families
# that only score the held-out block.
PERFORMANCE_COLUMNS = (
    "train_val_c_index",
    "test_c_index",
    "train_val_mean_auc_t",
    "test_mean_auc_t",
    "test_integrated_brier",
)

# The full block every metrics frame must carry. Family-specific columns
# (selected_penalizer, xgb_params, selected_hidden_dim, per-horizon *_auc_h*,
# timeline diagnostics) are additive and deliberately not standardized.
CANONICAL_METRIC_COLUMNS = IDENTITY_COLUMNS + COUNT_COLUMNS + PERFORMANCE_COLUMNS


def canonical_identity(
    *,
    model: str,
    cohort: str | None,
    endpoint: str,
    landmark_days: int,
    config: str,
) -> dict[str, object]:
    """Build the identity block, defaulting an unset cohort to "all"."""
    return {
        "model": model,
        "cohort": str(cohort) if cohort else DEFAULT_COHORT,
        "endpoint": endpoint,
        "landmark_days": int(landmark_days),
        "config": config,
    }


def missing_canonical_columns(columns) -> list[str]:
    """Canonical columns absent from `columns`, in schema order."""
    present = set(columns)
    return [c for c in CANONICAL_METRIC_COLUMNS if c not in present]


def order_canonical_first(metrics):
    """Put the canonical schema block at the front, family columns after.

    Raises if any canonical column is absent, so a writer that drifts from the
    schema fails at write time rather than producing a CSV the figure pipeline
    silently reads as all-NA. Ordering itself is cosmetic -- readers select by
    name -- but it makes the shared block visible when eyeballing a metrics CSV.
    """
    missing = missing_canonical_columns(metrics.columns)
    if missing:
        raise ValueError(
            "Metrics frame is missing canonical columns "
            f"{missing}; every writer must emit {list(CANONICAL_METRIC_COLUMNS)}."
        )
    rest = [c for c in metrics.columns if c not in set(CANONICAL_METRIC_COLUMNS)]
    return metrics.loc[:, list(CANONICAL_METRIC_COLUMNS) + rest]
