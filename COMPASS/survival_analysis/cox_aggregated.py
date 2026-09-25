"""PROFILE-specific adapter for shared landmarked survival-analysis code.

The generic Cox model mechanics live in ``survival_common.cox_models`` and
``survival_common.cox_engine``. This module keeps the PROFILE endpoint/schema
constants, optional stage cohort restrictions, and the per-landmark
context assembly used by the runnable scripts.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SURVIVAL_DIR = Path(__file__).resolve().parent
SURVIVAL_PARENT = SURVIVAL_DIR.parent
REPO_ROOT = SURVIVAL_PARENT.parent
DATA_PREPROCESSING_DIR = SURVIVAL_PARENT / "data_preprocessing"
for _p in (str(REPO_ROOT), str(SURVIVAL_PARENT), str(DATA_PREPROCESSING_DIR), str(SURVIVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
    compute_out_of_fold_risk_scores as _shared_compute_out_of_fold_risk_scores,
    compute_fixed_feature_out_of_fold_risk_scores as _shared_compute_fixed_feature_out_of_fold_risk_scores,
    tune_multivariable_model as _shared_tune_multivariable_model,
)
from survival_common.helper import (  # noqa: E402,F401
    DEFAULT_AUC_MAX_TIME_UNITS,
    assert_disjoint_folds,
    assert_no_test_leakage,
    compute_brier,
    compute_horizon_grid,
    horizon_grid_frame,
    select_canonical_labs,
)

BASE = Path(__file__).resolve().parent
DATA_PATH = Path("/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/")
RESULTS = Path("/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis")

DEFAULT_SEED = 42
DEFAULT_TEST_FRAC = 0.20
DEFAULT_N_FOLDS = 5
DEFAULT_LANDMARK_DAYS = [0, 90, 180]
DEFAULT_MIN_PATIENT_COVERAGE = 0.20
DEFAULT_MIN_EVENTS_PER_FEATURE = 10
# DEFAULT_AUC_MAX_TIME_UNITS is re-exported from survival_common.helper so the
# builders' horizon grid and the runners' evaluation cap cannot drift apart.
# Elastic-net CV grid aligned with the clinical text embedding project
# (clinical_text_embedding_project/v2/pipelines/training/slurm_array_utils.py:
# DEFAULT_ALPHAS / DEFAULT_LOW_ALPHAS / DEFAULT_L1_RATIOS). Both projects fit
# with sksurv CoxnetSurvivalAnalysis + penalty_factor, so the alpha scale is
# directly comparable. The low-alpha block is the embedding project's adaptive
# refinement, prepended here because this runner searches one flat grid rather
# than re-running on a boundary hit.
DEFAULT_CV_PENALIZERS = (
    np.logspace(-5, -4, 6).tolist()[:-1] + np.logspace(-4, 0, 10).tolist()
)
DEFAULT_CV_L1_RATIOS = [0.5, 1.0]
DEFAULT_AUC_QUANTILES = (0.25, 0.375, 0.50, 0.625, 0.75)
DEFAULT_AUC_TIME_UNIT_DAYS = 7
# Mirrors IPIO.survival_analysis.cox_aggregated.DEFAULT_MIN_GENOMIC_PREVALENCE
# (Finding 10): a genomic mutation indicator present in fewer than this
# fraction of train_val patients is dropped, since near-zero-prevalence
# binary indicators are essentially unfittable and unstable across folds.
DEFAULT_MIN_GENOMIC_PREVALENCE = 0.025
# The genomic arm is SNV-only. GENOMIC_FEATURE_RE matches what is testable;
# ANY_VARIANT_RE matches every variant class so non-SNV columns can be kept
# out of the lab feature set rather than falling through into it.
GENOMIC_FEATURE_RE = re.compile(r"^([A-Za-z0-9.\-]+)_(SNV)$")
ANY_VARIANT_RE = re.compile(r"^([A-Za-z0-9.\-]+)_(SV|SNV|AMP|DEL)$")
HORIZON_GRID_FILENAME = "cox_agg_horizon_grid.csv"
CANONICAL_LABS_FOLDS_FILENAME = "cox_agg_canonical_labs_folds.csv"

# Component-based feature-set tokens (Plan §2a): each token is a combination
# of {labs, gleason, somatic} drawn from somatic_gleason_features.csv plus the
# usual lab summary columns. The legacy "somatic_gleason" token is unchanged
# and keeps testing every feature in the manifest (labs + gleason + somatic
# together, but resolved as a single static-feature block rather than through
# this component map) so existing sensitivity outputs and Figure 4 keep their
# meaning.
FEATURE_SET_COMPONENTS: dict[str, frozenset[str]] = {
    "gleason": frozenset({"gleason"}),
    "somatic": frozenset({"somatic"}),
    "gleason_labs": frozenset({"gleason", "labs"}),
    "somatic_labs": frozenset({"somatic", "labs"}),
    "gleason_somatic": frozenset({"gleason", "somatic"}),
    "gleason_somatic_labs": frozenset({"gleason", "somatic", "labs"}),
}
# Feature sets that carry no lab summary columns at all, so canonical-lab
# selection and the lab coverage gate must be skipped entirely rather than
# just failing to find a canonical lab.
NO_LABS_FEATURE_SETS: frozenset[str] = frozenset(
    {"somatic_gleason", "text"}
    | {token for token, components in FEATURE_SET_COMPONENTS.items() if "labs" not in components}
)


def feature_set_has_labs_component(feature_set: str) -> bool:
    """True if `feature_set` tests lab summary columns at all.

    Used to decide whether canonical-lab selection and the lab coverage gate
    apply, generalizing the old `feature_set in {"somatic_gleason", "text"}`
    check to the new component-based tokens.
    """
    token = str(feature_set).lower().replace("-", "_")
    return token not in NO_LABS_FEATURE_SETS

ENDPOINTS = {
    "platinum": {
        "duration_col": "t_platinum",
        "event_col": "PLATINUM",
        "description": "Time from the treatment anchor (first ARPI/taxane/radium-223 exposure = time 0) to first platinum exposure",
    },
    "nepc": {
        "duration_col": "t_nepc",
        "event_col": "NEPC",
        "description": "Time from the treatment anchor (first ADT exposure = time 0) to the "
                        "first documented NEPC criterion (NEPC-only, independent of AVPC "
                        "criteria), from the longitudinal AVPC/NEPC criteria timeline",
    },
    "avpc": {
        "duration_col": "t_avpc",
        "event_col": "AVPC",
        "description": "Time from the treatment anchor (first ADT exposure = time 0) to "
                        "first meeting AVPC (>=4 Aparicio criteria, independent of any NEPC "
                        "feature), from the longitudinal AVPC/NEPC criteria timeline",
    },
}

# Non-feature outcome/metadata columns, independent of the age column. The age
# column is added at call time via outcome_columns() so a runtime --age-col
# override (which rebinds AGE_COL after import) is honored — a bare AGE_COL baked
# into this literal would freeze the import-time default and silently leak the
# real age column into the feature set.
OUTCOME_METADATA_COLUMNS = {
    "GENDER",
    "FIRST_RECORD_DATE",
    "DIAGNOSIS_DATE",
    "TREATMENT_ANCHOR_DATE",
    "LAST_CONTACT_DATE",
    "PLATINUM_DATE",
    "PLATINUM",
    "DEATH",
    "EITHER",
    "t_diagnosis",
    "t_platinum",
    "t_platinum_from_first_record",
    "t_last_contact",
    "t_last_contact_from_first_record",
    "t_death",
    "t_death_from_first_record",
    "t_either",
    "split",
    # Endpoint and audit columns must never enter the feature matrix. The
    # AVPC_NEPC fields remain cohort-level audit metadata, but the joint label
    # is no longer a modeled endpoint.
    "NEPC_DATE",
    "NEPC",
    "t_nepc",
    "t_nepc_from_first_record",
    "NEPC_DATE_SOURCE",
    "NEPC_DATE_PRECISION",
    "NEPC_LABEL_SOURCE",
    "AVPC_NEPC",
    "t_avpc_nepc",
    "t_avpc_nepc_from_first_record",
    "AVPC_NEPC_DATE",
    "AVPC_NEPC_DATE_SOURCE",
    "AVPC_NEPC_DATE_PRECISION",
    "AVPC_NEPC_LABEL_SOURCE",
    "AVPC",
    "AVPC_DATE",
    "t_avpc",
    "t_avpc_from_first_record",
    "AVPC_N_CRITERIA",
    "NEPC_TIMELINE",
    "NEPC_TIMELINE_DATE",
    # Panel version is an adjustment covariate, never a tested feature. The raw
    # string column is listed here; the reference-coded PANEL_VERSION_* dummies
    # are stripped from the feature universe by outcome_columns() below.
    "PANEL_VERSION",
}

PANEL_VERSION_COL = "PANEL_VERSION"
PANEL_VERSION_FEATURE_PREFIX = "PANEL_VERSION_"


def panel_version_covariate_columns(df: pd.DataFrame) -> list[str]:
    """Reference-coded panel-version dummies written by build_genomic_inputs.py.

    Empty when the inputs predate the panel-version change, so older prebuilt
    inputs keep working (the fits then just have no version adjustment).
    """
    return sorted(
        c for c in df.columns if str(c).startswith(PANEL_VERSION_FEATURE_PREFIX)
    )


def outcome_columns() -> set[str]:
    """Non-feature columns, including the currently configured age column.

    Resolves ``AGE_COL`` at call time so a runtime ``--age-col`` override is
    respected. Callers must use this rather than a module-level set literal.
    """
    return OUTCOME_METADATA_COLUMNS | {AGE_COL}


def non_feature_columns(df: pd.DataFrame) -> set[str]:
    """outcome_columns() plus this frame's panel-version covariate dummies."""
    return outcome_columns() | set(panel_version_covariate_columns(df))


def normalize_endpoints(raw_endpoints: list[str]) -> list[str]:
    return _shared_normalize_endpoints(raw_endpoints, ENDPOINTS)


def select_feature_columns(
    data: pd.DataFrame,
    raw_feature_cols: list[str],
    *,
    min_patient_coverage: float,
    restrict_to_labs: list[str] | None = None,
    always_include: list[str] | None = None,
    genomic_feature_cols: list[str] | None = None,
    min_genomic_prevalence: float | None = None,
) -> tuple[list[str], pd.DataFrame]:
    return _shared_select_feature_columns(
        data,
        raw_feature_cols,
        min_patient_coverage=min_patient_coverage,
        restrict_to_labs=restrict_to_labs,
        always_include=always_include,
        genomic_feature_cols=genomic_feature_cols,
        min_genomic_prevalence=min_genomic_prevalence,
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
    baseline_covariate_cols: tuple[str, ...] = (),
    genomic_feature_cols: list[str] | tuple[str, ...] | None = None,
    model_type: str = "cox",
) -> pd.DataFrame:
    return _shared_run_univariate_nobs_adjusted_associations(
        data,
        feature_cols=feature_cols,
        endpoint=endpoint,
        min_events_per_feature=min_events_per_feature,
        fallback_penalizer=fallback_penalizer,
        baseline_covariate_cols=baseline_covariate_cols,
        genomic_feature_cols=genomic_feature_cols,
        endpoint_map=ENDPOINTS,
        age_col=AGE_COL,
        model_type=model_type,
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
    genomic_feature_cols: tuple[str, ...] = (),
    min_genomic_prevalence: float | None = DEFAULT_MIN_GENOMIC_PREVALENCE,
    restrict_to_canonical_labs: bool = True,
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
        genomic_feature_cols=genomic_feature_cols,
        min_genomic_prevalence=min_genomic_prevalence,
        restrict_to_canonical_labs=restrict_to_canonical_labs,
        endpoint_map=ENDPOINTS,
        id_col=ID_COL,
        age_col=AGE_COL,
    )


def compute_out_of_fold_risk_scores(
    cohort: pd.DataFrame,
    *,
    raw_feature_cols: list[str],
    endpoint: str,
    penalizers: list[float],
    l1_ratios: list[float],
    outer_folds: int,
    inner_folds: int,
    seed: int,
    auc_time_unit_days: int,
    auc_max_time_units: int | None,
    pre_treatment_lab_df: pd.DataFrame,
    horizon_grid: np.ndarray,
    min_patient_coverage: float,
    static_covariate_cols: tuple[str, ...] = (),
    always_include_feature_cols: tuple[str, ...] = (),
    genomic_feature_cols: tuple[str, ...] = (),
    min_genomic_prevalence: float | None = DEFAULT_MIN_GENOMIC_PREVALENCE,
    restrict_to_canonical_labs: bool = True,
) -> pd.DataFrame:
    return _shared_compute_out_of_fold_risk_scores(
        cohort,
        raw_feature_cols=raw_feature_cols,
        endpoint=endpoint,
        penalizers=penalizers,
        l1_ratios=l1_ratios,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        seed=seed,
        auc_time_unit_days=auc_time_unit_days,
        auc_max_time_units=auc_max_time_units,
        pre_treatment_lab_df=pre_treatment_lab_df,
        horizon_grid=horizon_grid,
        min_patient_coverage=min_patient_coverage,
        static_covariate_cols=static_covariate_cols,
        always_include_feature_cols=always_include_feature_cols,
        genomic_feature_cols=genomic_feature_cols,
        min_genomic_prevalence=min_genomic_prevalence,
        restrict_to_canonical_labs=restrict_to_canonical_labs,
        endpoint_map=ENDPOINTS,
        id_col=ID_COL,
        age_col=AGE_COL,
    )


def compute_fixed_feature_out_of_fold_risk_scores(
    cohort: pd.DataFrame,
    *,
    feature_cols: list[str],
    endpoint: str,
    penalizer: float,
    l1_ratio: float,
    outer_folds: int,
    seed: int,
    static_covariate_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    return _shared_compute_fixed_feature_out_of_fold_risk_scores(
        cohort,
        feature_cols=feature_cols,
        endpoint=endpoint,
        penalizer=penalizer,
        l1_ratio=l1_ratio,
        outer_folds=outer_folds,
        seed=seed,
        static_covariate_cols=static_covariate_cols,
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
    genomic_feature_cols: tuple[str, ...] = ()


def compute_shared_canonical_labs(
    inputs_dir: Path,
    landmark_days: list[int],
    *,
    min_patient_coverage: float,
) -> list[str]:
    """Canonical lab set shared across all landmarks (their intersection).

    For each landmark, derive the coverage-based canonical labs on that
    landmark's train_val block and pre-landmark long table, then intersect so
    the returned set is the labs that clear the coverage/variability bar at
    EVERY landmark. Passing this into ``prepare_landmark_context`` via
    ``canonical_labs_override`` makes every landmark test an identical lab list,
    so associations are comparable across landmarks (e.g. +0d vs +90d).
    """
    per_landmark: list[set[str]] = []
    for landmark_day in landmark_days:
        _, train_val, _, pre_treatment_lab_df = _load_prebuilt_landmark(
            inputs_dir, landmark_day
        )
        labs = select_canonical_labs(
            pre_treatment_lab_df,
            mrns=train_val.index,
            min_coverage=min_patient_coverage,
            id_col=ID_COL,
        )
        per_landmark.append(set(str(lab) for lab in labs))
        print(f"  [shared-canonical-labs] landmark +{landmark_day}d: {len(labs)} canonical labs")

    shared = set.intersection(*per_landmark) if per_landmark else set()
    shared_sorted = sorted(shared)
    print(
        f"  [shared-canonical-labs] intersection across landmarks "
        f"{landmark_days}: {len(shared_sorted)} labs"
    )
    if not shared_sorted:
        raise ValueError(
            "Shared canonical lab set is empty: no lab clears the coverage/"
            "variability bar at every requested landmark. Lower "
            "--min-patient-coverage or drop a landmark."
        )
    return shared_sorted


def prepare_landmark_context(
    inputs_dir: Path,
    landmark_day: int,
    *,
    min_patient_coverage: float,
    canonical_labs_override: list[str] | None = None,
    feature_set: str = "labs",
) -> LandmarkContext:
    print(f"\n##### LANDMARK ANALYSES: +{landmark_day} DAYS #####")
    merged, train_val, test, pre_treatment_lab_df = _load_prebuilt_landmark(
        inputs_dir, landmark_day
    )

    feature_set = str(feature_set).lower().replace("-", "_")
    _known_feature_sets = (
        {"labs", "somatic_gleason", "genomic", "text", "labs_text"}
        | set(FEATURE_SET_COMPONENTS)
    )
    if feature_set not in _known_feature_sets:
        raise ValueError(
            f"Unsupported feature set {feature_set!r}; expected one of "
            f"{sorted(_known_feature_sets)}."
        )

    always_include_feature_cols: tuple[str, ...] = ()
    genomic_feature_cols: tuple[str, ...] = ()
    if feature_set == "somatic_gleason":
        manifest_path = inputs_dir / "somatic_gleason_features.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing {manifest_path}. Run build_somatic_gleason_inputs.py first."
            )
        feature_manifest = pd.read_csv(manifest_path)
        if "feature" not in feature_manifest.columns:
            raise ValueError(f"{manifest_path} is missing the 'feature' column.")
        declared_features = [
            str(feature)
            for feature in feature_manifest["feature"].dropna().drop_duplicates()
        ]
        missing_features = [
            feature for feature in declared_features if feature not in merged.columns
        ]
        if missing_features:
            raise ValueError(
                f"Landmark +{landmark_day} inputs are missing {len(missing_features)} "
                f"declared somatic/Gleason features; first values: {missing_features[:10]}"
            )
        raw_feature_cols = declared_features
        always_include_feature_cols = tuple(raw_feature_cols)
        # The manifest's 'feature_kind' column (written by
        # build_somatic_gleason_inputs.py) distinguishes genuine binary
        # somatic mutation indicators ("somatic_binary") from the Gleason
        # score ("gleason_continuous"), the tested-coverage flag, and PRS
        # rows -- so unlike IPIO's naming-convention regex, COMPASS can read
        # the genomic subset directly off the manifest (Finding 10).
        if "feature_kind" in feature_manifest.columns:
            genomic_feature_cols = tuple(
                str(feature)
                for feature in feature_manifest.loc[
                    feature_manifest["feature_kind"] == "somatic_binary", "feature"
                ]
                .dropna()
                .drop_duplicates()
                if str(feature) in raw_feature_cols
            )
    elif feature_set == "genomic":
        # genomic_aggregated.csv carries <GENE>_SNV indicators alongside the lab
        # summary features. Scoping genomic_feature_cols here is what arms the
        # 2.5% prevalence floor in select_feature_columns -- that filter is
        # guarded on a non-empty tuple, so leaving it empty (as the "labs"
        # branch does) silently disables the floor and routes the indicators
        # through the lab n_obs-adjusted univariate model instead of the
        # static-feature path.
        excluded = non_feature_columns(merged)
        raw_feature_cols = [
            c
            for c in merged.columns
            if c not in excluded
            and (ANY_VARIANT_RE.match(str(c)) is None or GENOMIC_FEATURE_RE.match(str(c)))
        ]
        genomic_feature_cols = tuple(
            c for c in raw_feature_cols if GENOMIC_FEATURE_RE.match(str(c))
        )
        always_include_feature_cols = genomic_feature_cols
        if not genomic_feature_cols:
            raise ValueError(
                f"Landmark +{landmark_day} inputs declare no <GENE>_SNV columns; "
                "the genomic feature set has nothing to test. Confirm "
                "build_genomic_inputs.py wrote genomic_aggregated.csv here."
            )
    elif feature_set in {"text", "labs_text"}:
        # Pooled note-embedding columns, declared by the manifest that
        # build_text_embedding_inputs.py writes beside the landmark frames.
        #
        # always_include is load-bearing here, not a convenience. The pooled
        # columns are named <TYPE>_EMBEDDING_<i> with no "__" separator, so
        # parse_feature_name() reads each as a lab named after the whole column
        # with stat "value". select_feature_columns' restrict_to_labs gate would
        # then drop all of them silently -- leaving an age-only model that looks
        # like a successful text fit. Exempting them from the lab gate is what
        # prevents that; coverage and variability filters still apply.
        manifest_path = inputs_dir / "text_embedding_features.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing {manifest_path}. Run build_text_embedding_inputs.py first."
            )
        feature_manifest = pd.read_csv(manifest_path)
        if "feature" not in feature_manifest.columns:
            raise ValueError(f"{manifest_path} is missing the 'feature' column.")
        text_features = [
            str(feature)
            for feature in feature_manifest["feature"].dropna().drop_duplicates()
        ]
        missing_text = [f for f in text_features if f not in merged.columns]
        if missing_text:
            raise ValueError(
                f"Landmark +{landmark_day} inputs are missing {len(missing_text)} "
                f"declared text features; first values: {missing_text[:10]}"
            )
        if feature_set == "text":
            raw_feature_cols = text_features
        else:
            lab_cols = [
                c
                for c in merged.columns
                if c not in non_feature_columns(merged) and c not in set(text_features)
            ]
            raw_feature_cols = lab_cols + text_features
        always_include_feature_cols = tuple(text_features)
        # genomic_feature_cols stays empty: DEFAULT_MIN_GENOMIC_PREVALENCE is a
        # floor on binary mutation indicators (value == 1 in >= X% of patients).
        # Applied to dense continuous embedding dimensions it would be
        # meaningless and would drop essentially every column.
    elif feature_set in FEATURE_SET_COMPONENTS:
        # Component-based Gleason/somatic/labs arms (Plan §2a). Unlike the
        # legacy "somatic_gleason" token (every manifest feature together),
        # each of these picks a subset of {gleason, somatic} components by
        # feature_kind, optionally combined with the normal lab columns.
        components = FEATURE_SET_COMPONENTS[feature_set]
        manifest_path = inputs_dir / "somatic_gleason_features.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing {manifest_path}. Run build_somatic_gleason_inputs.py first."
            )
        feature_manifest = pd.read_csv(manifest_path)
        for required_col in ("feature", "feature_kind"):
            if required_col not in feature_manifest.columns:
                raise ValueError(f"{manifest_path} is missing the '{required_col}' column.")

        component_feature_kind = {"gleason": "gleason_continuous", "somatic": "somatic_binary"}
        static_feature_cols: list[str] = []
        somatic_cols: list[str] = []
        for component in ("gleason", "somatic"):
            if component not in components:
                continue
            kind = component_feature_kind[component]
            cols = [
                str(feature)
                for feature in feature_manifest.loc[
                    feature_manifest["feature_kind"] == kind, "feature"
                ]
                .dropna()
                .drop_duplicates()
            ]
            missing = [c for c in cols if c not in merged.columns]
            if missing:
                raise ValueError(
                    f"Landmark +{landmark_day} inputs are missing {len(missing)} "
                    f"declared {component} features; first values: {missing[:10]}"
                )
            if not cols:
                raise ValueError(
                    f"somatic_gleason_features.csv declares no {kind!r} features; "
                    f"feature set {feature_set!r} has nothing to test for the "
                    f"{component} component."
                )
            static_feature_cols.extend(cols)
            if component == "somatic":
                somatic_cols = cols

        always_include_feature_cols = tuple(static_feature_cols)
        genomic_feature_cols = tuple(somatic_cols)
        if "labs" in components:
            lab_cols = [
                c
                for c in merged.columns
                if c not in non_feature_columns(merged) and c not in set(static_feature_cols)
            ]
            raw_feature_cols = lab_cols + static_feature_cols
        else:
            raw_feature_cols = static_feature_cols
    else:
        raw_feature_cols = [
            c for c in merged.columns if c not in non_feature_columns(merged)
        ]
    univariate_data = merged.copy()
    split_stratification = "prebuilt"

    assert_no_test_leakage(
        test_mrns=test.index,
        train_mrns=train_val.index,
        context=f"prepare_landmark_context[landmark+{landmark_day}d]",
    )

    if not feature_set_has_labs_component(feature_set):
        # This arm models no lab summaries, so there is no canonical lab set
        # to select. "labs_text" and the "*_labs" component tokens DO carry
        # labs and therefore fall through to the normal coverage-based
        # selection below.
        canonical_labs = []
    elif canonical_labs_override is not None:
        # Shared-canonical-labs arm: the caller has fixed the canonical set (e.g.
        # the intersection across all landmarks) so every landmark tests an
        # identical lab list. Skip the per-landmark coverage-based selection.
        canonical_labs = sorted(str(lab) for lab in canonical_labs_override)
        print(f"  [shared-canonical-labs] using {len(canonical_labs)} caller-supplied labs")
    else:
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
        restrict_to_labs=(
            canonical_labs if feature_set_has_labs_component(feature_set) else []
        ),
        always_include=list(always_include_feature_cols),
        genomic_feature_cols=list(genomic_feature_cols),
        min_genomic_prevalence=DEFAULT_MIN_GENOMIC_PREVALENCE,
    )
    feature_meta_selected = feature_meta.loc[
        feature_meta["selected"],
        ["feature", "lab_name", "feature_stat", "coverage", "unique_non_missing"],
    ].copy()
    feature_meta_selected.insert(0, "landmark_days", landmark_day)

    print(f"Full cohort: {len(merged)} patients")
    print(f"Train/val (Arm 2): {len(train_val)} patients")
    print(f"Test (Arm 2):      {len(test)} patients")
    print(f"Feature set: {feature_set}")
    print(f"Canonical labs (train_val): {len(canonical_labs)}")
    feature_label = {
        "somatic_gleason": "static somatic/Gleason",
        "text": "pooled note-embedding",
        "labs_text": "summary-lab + note-embedding",
        "gleason": "Gleason",
        "somatic": "somatic",
        "gleason_labs": "Gleason + summary-lab",
        "somatic_labs": "somatic + summary-lab",
        "gleason_somatic": "Gleason + somatic",
        "gleason_somatic_labs": "Gleason + somatic + summary-lab",
    }.get(feature_set, "summary-lab")
    print(f"Selected {feature_label} features (train_val pre-filter): {len(selected_feature_cols)}")

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
        genomic_feature_cols=genomic_feature_cols,
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
