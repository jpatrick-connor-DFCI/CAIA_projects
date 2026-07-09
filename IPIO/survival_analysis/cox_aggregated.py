"""IPIO-specific adapter for shared landmarked survival-analysis code.

The generic Cox model mechanics live in ``survival_common.cox_models`` and
``survival_common.cox_engine``. This module keeps the IPIO endpoint/schema
constants, baseline-covariate discovery, and per-landmark context assembly used
by the runnable scripts.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SURVIVAL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SURVIVAL_DIR.parent
DATA_PREPROCESSING_DIR = PROJECT_DIR / "data_preprocessing"
if str(DATA_PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PREPROCESSING_DIR))

from _paths import ensure_survival_common_on_path  # noqa: E402

ensure_survival_common_on_path()

from survival_common.cohort import (  # noqa: E402,F401
    AGE_COL,
    ID_COL,
    build_feature_matrix,
    build_landmark_availability_table,
    build_landmark_merged,
    build_pre_treatment_lab_long,
    make_outcome_df,
    normalize_landmark_days,
)
from survival_common.cox_engine import (  # noqa: E402,F401
    _make_survival_array,
    apply_auc_admin_censoring as _apply_auc_admin_censoring,
    benjamini_hochberg,
    build_coxnet_xy as _build_coxnet_xy,
    compute_ipcw_auc_t,
    coxnet_coefficients,
    coxnet_survival_at_horizons,
    duration_to_auc_units as _duration_to_auc_units,
    fit_cox_with_fallback,
    fit_coxnet_with_fallback,
    make_cv_splitter,
    matching_n_obs_feature,
    parse_feature_name,
    require_lifelines,
    require_sksurv,
    score_coxnet_model,
    summarize_fold_failures as _summarize_fold_failures,
)
from survival_common.cox_models import (  # noqa: E402
    build_endpoint_horizon_grids as _shared_build_endpoint_horizon_grids,
    build_model_matrices as _shared_build_model_matrices,
    fit_final_multivariable_model as _shared_fit_final_multivariable_model,
    load_build_manifest as _shared_load_build_manifest,
    load_prebuilt_landmark as _shared_load_prebuilt_landmark,
    normalize_endpoints as _shared_normalize_endpoints,
    print_top_hits as _shared_print_top_hits,
    run_univariate_nobs_adjusted_associations as _shared_run_univariate_nobs_adjusted_associations,
    select_feature_columns as _shared_select_feature_columns,
    tune_multivariable_model as _shared_tune_multivariable_model,
)
from survival_common.helper import (  # noqa: E402,F401
    assert_disjoint_folds,
    assert_no_test_leakage,
    compute_brier,
    compute_horizon_grid,
    horizon_grid_frame,
    select_canonical_labs,
)

BASE = SURVIVAL_DIR
DATA_PATH = Path("/data/gusev/USERS/jpconnor/data/CAIA/IPIO/")
RESULTS = Path("/data/gusev/USERS/jpconnor/data/CAIA/IPIO/survival_analysis")

DEFAULT_SEED = 42
DEFAULT_TEST_FRAC = 0.20
DEFAULT_N_FOLDS = 5
DEFAULT_LANDMARK_DAYS = [0, 90]
DEFAULT_MIN_PATIENT_COVERAGE = 0.20
DEFAULT_MIN_EVENTS_PER_FEATURE = 10
DEFAULT_AUC_MAX_TIME_UNITS = 260
DEFAULT_CV_PENALIZERS = [
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.0075,
    0.01,
    0.025,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.60,
    0.80,
    1.20,
    1.60,
    2.40,
    3.20,
    4.80,
    6.40,
]
DEFAULT_CV_L1_RATIOS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
DEFAULT_AUC_QUANTILES = (0.25, 0.375, 0.50, 0.625, 0.75)
DEFAULT_AUC_TIME_UNIT_DAYS = 7
HORIZON_GRID_FILENAME = "cox_agg_horizon_grid.csv"
CANONICAL_LABS_FOLDS_FILENAME = "cox_agg_canonical_labs_folds.csv"

ENDPOINTS = {
    "irae": {
        "duration_col": "t_irae",
        "event_col": "IRAE",
        "description": "Time from IO start to first immune-related adverse event",
    },
}

# Non-feature outcome/metadata columns, independent of the age column. The age
# column is added at call time via outcome_columns() so a runtime --age-col
# override (which rebinds AGE_COL after import) is honored — a bare AGE_COL baked
# into this literal would freeze the import-time default and silently leak the
# real age column into the feature set.
OUTCOME_METADATA_COLUMNS = {
    "FIRST_RECORD_DATE",
    "FIRST_TREATMENT_DATE",
    "FIRST_TREATMENT",
    "LAST_CONTACT_DATE",
    "IO_START",
    "LAST_DATE",
    "IRAE",
    "t_irae",
    "t_irae_from_first_record",
    "split",
}


def outcome_columns() -> set[str]:
    """Non-feature columns, including the currently configured age column.

    Resolves ``AGE_COL`` at call time so a runtime ``--age-col`` override is
    respected. Callers must use this rather than a module-level set literal.
    """
    return OUTCOME_METADATA_COLUMNS | {AGE_COL}

BASELINE_STATIC_FIXED = ("GENDER_MALE", "pd1pdl1", "ctla4")
BASELINE_STATIC_PREFIX = "CANCER_TYPE_"
GENOMIC_FEATURE_RE = re.compile(r"^[A-Za-z0-9]+_(SV|SNV|AMP|DEL)$")


def baseline_covariate_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in BASELINE_STATIC_FIXED if c in df.columns]
    cols += sorted(c for c in df.columns if c.startswith(BASELINE_STATIC_PREFIX))
    return cols


def genomic_feature_columns(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if GENOMIC_FEATURE_RE.match(str(c)))


def normalize_endpoints(raw_endpoints: list[str]) -> list[str]:
    return _shared_normalize_endpoints(raw_endpoints, ENDPOINTS)


def select_feature_columns(
    data: pd.DataFrame,
    raw_feature_cols: list[str],
    *,
    min_patient_coverage: float,
    restrict_to_labs: list[str] | None = None,
    always_include: list[str] | None = None,
) -> tuple[list[str], pd.DataFrame]:
    return _shared_select_feature_columns(
        data,
        raw_feature_cols,
        min_patient_coverage=min_patient_coverage,
        restrict_to_labs=restrict_to_labs,
        always_include=always_include,
    )


def build_model_matrices(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    duration_col: str,
    event_col: str,
    static_covariate_cols: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    return _shared_build_model_matrices(
        train_df,
        eval_df,
        feature_cols=feature_cols,
        duration_col=duration_col,
        event_col=event_col,
        static_covariate_cols=static_covariate_cols,
        age_col=AGE_COL,
    )


def run_univariate_nobs_adjusted_associations(
    data: pd.DataFrame,
    *,
    feature_cols: list[str],
    endpoint: str,
    min_events_per_feature: int,
    fallback_penalizer: float,
    static_covariate_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    return _shared_run_univariate_nobs_adjusted_associations(
        data,
        feature_cols=feature_cols,
        endpoint=endpoint,
        min_events_per_feature=min_events_per_feature,
        fallback_penalizer=fallback_penalizer,
        static_covariate_cols=static_covariate_cols,
        endpoint_map=ENDPOINTS,
        age_col=AGE_COL,
    )


def tune_multivariable_model(
    train_val: pd.DataFrame,
    *,
    raw_feature_cols: list[str],
    endpoint: str,
    penalizers: list[float],
    l1_ratios: list[float],
    n_folds: int,
    seed: int,
    auc_time_unit_days: int,
    auc_max_time_units: int | None,
    pre_treatment_lab_df: pd.DataFrame,
    horizon_grid: np.ndarray,
    min_patient_coverage: float,
    static_covariate_cols: tuple[str, ...] = (),
    always_include_feature_cols: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    return _shared_tune_multivariable_model(
        train_val,
        raw_feature_cols=raw_feature_cols,
        endpoint=endpoint,
        penalizers=penalizers,
        l1_ratios=l1_ratios,
        n_folds=n_folds,
        seed=seed,
        auc_time_unit_days=auc_time_unit_days,
        auc_max_time_units=auc_max_time_units,
        pre_treatment_lab_df=pre_treatment_lab_df,
        horizon_grid=horizon_grid,
        min_patient_coverage=min_patient_coverage,
        static_covariate_cols=static_covariate_cols,
        always_include_feature_cols=always_include_feature_cols,
        endpoint_map=ENDPOINTS,
        id_col=ID_COL,
        age_col=AGE_COL,
    )


def fit_final_multivariable_model(
    train_val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_cols: list[str],
    endpoint: str,
    penalizer: float,
    l1_ratio: float,
    split_stratification: str,
    cv_stratification: str,
    auc_time_unit_days: int,
    auc_max_time_units: int | None,
    horizon_grid: np.ndarray,
    canonical_labs: list[str],
    static_covariate_cols: tuple[str, ...] = (),
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _shared_fit_final_multivariable_model(
        train_val,
        test,
        feature_cols=feature_cols,
        endpoint=endpoint,
        penalizer=penalizer,
        l1_ratio=l1_ratio,
        split_stratification=split_stratification,
        cv_stratification=cv_stratification,
        auc_time_unit_days=auc_time_unit_days,
        auc_max_time_units=auc_max_time_units,
        horizon_grid=horizon_grid,
        canonical_labs=canonical_labs,
        static_covariate_cols=static_covariate_cols,
        endpoint_map=ENDPOINTS,
        auc_quantiles=DEFAULT_AUC_QUANTILES,
        id_col=ID_COL,
        age_col=AGE_COL,
    )


def print_top_hits(df: pd.DataFrame, *, endpoint: str, label: str = "univariate") -> None:
    _shared_print_top_hits(df, endpoint=endpoint, label=label)


def _load_build_manifest(inputs_dir: Path) -> dict:
    from build_prediction_inputs import BUILD_MANIFEST_FILENAME

    return _shared_load_build_manifest(
        inputs_dir,
        manifest_filename=BUILD_MANIFEST_FILENAME,
    )


def _load_prebuilt_landmark(
    inputs_dir: Path,
    landmark_day: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from build_prediction_inputs import (
        aggregated_filename,
        pre_treatment_lab_filename,
    )

    return _shared_load_prebuilt_landmark(
        inputs_dir,
        landmark_day,
        aggregated_filename=aggregated_filename,
        pre_treatment_lab_filename=pre_treatment_lab_filename,
        id_col=ID_COL,
    )


@dataclass
class LandmarkContext:
    landmark_day: int
    merged: pd.DataFrame
    train_val: pd.DataFrame
    test: pd.DataFrame
    pre_treatment_lab_df: pd.DataFrame
    raw_feature_cols: list[str]
    univariate_data: pd.DataFrame
    split_stratification: str
    canonical_labs: list[str]
    selected_feature_cols: list[str]
    feature_meta_selected: pd.DataFrame
    always_include_feature_cols: tuple[str, ...] = ()


def prepare_landmark_context(
    inputs_dir: Path,
    landmark_day: int,
    *,
    min_patient_coverage: float,
) -> LandmarkContext:
    print(f"\n##### LANDMARK ANALYSES: +{landmark_day} DAYS #####")
    merged, train_val, test, pre_treatment_lab_df = _load_prebuilt_landmark(
        inputs_dir, landmark_day
    )

    excluded = outcome_columns() | set(baseline_covariate_columns(merged))
    raw_feature_cols = [c for c in merged.columns if c not in excluded]
    always_include_feature_cols = tuple(
        c for c in genomic_feature_columns(merged) if c in raw_feature_cols
    )
    univariate_data = merged.copy()
    split_stratification = "prebuilt"

    assert_no_test_leakage(
        test_mrns=test.index,
        train_mrns=train_val.index,
        context=f"prepare_landmark_context[landmark+{landmark_day}d]",
    )

    canonical_labs = select_canonical_labs(
        pre_treatment_lab_df,
        mrns=train_val.index,
        min_coverage=min_patient_coverage,
        id_col=ID_COL,
    )
    selected_feature_cols, feature_meta = select_feature_columns(
        train_val,
        raw_feature_cols,
        min_patient_coverage=min_patient_coverage,
        restrict_to_labs=canonical_labs,
        always_include=list(always_include_feature_cols),
    )
    feature_meta_selected = feature_meta.loc[
        feature_meta["selected"],
        ["feature", "lab_name", "feature_stat", "coverage", "unique_non_missing"],
    ].copy()
    feature_meta_selected.insert(0, "landmark_days", landmark_day)

    print(f"Full cohort: {len(merged)} patients")
    print(f"Train/val (Arm 2): {len(train_val)} patients")
    print(f"Test (Arm 2):      {len(test)} patients")
    print(f"Canonical labs (train_val): {len(canonical_labs)}")
    print(f"Selected summary-lab features (train_val pre-filter): {len(selected_feature_cols)}")
    if always_include_feature_cols:
        n_selected_genomics = len(set(always_include_feature_cols).intersection(selected_feature_cols))
        print(
            f"Genomic indicators detected: {len(always_include_feature_cols)} "
            f"({n_selected_genomics} selected after coverage/variability filters)"
        )

    return LandmarkContext(
        landmark_day=landmark_day,
        merged=merged,
        train_val=train_val,
        test=test,
        pre_treatment_lab_df=pre_treatment_lab_df,
        raw_feature_cols=raw_feature_cols,
        univariate_data=univariate_data,
        split_stratification=split_stratification,
        canonical_labs=canonical_labs,
        selected_feature_cols=selected_feature_cols,
        feature_meta_selected=feature_meta_selected,
        always_include_feature_cols=always_include_feature_cols,
    )


def build_endpoint_horizon_grids(
    landmark_day: int,
    *,
    endpoints: list[str],
    auc_horizons_by_landmark: dict,
    auc_quantiles: tuple[float, ...],
    auc_time_unit_days: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    return _shared_build_endpoint_horizon_grids(
        landmark_day,
        endpoints=endpoints,
        auc_horizons_by_landmark=auc_horizons_by_landmark,
        auc_quantiles=auc_quantiles,
        auc_time_unit_days=auc_time_unit_days,
    )
