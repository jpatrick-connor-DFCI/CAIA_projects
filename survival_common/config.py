"""Configuration hooks for project-specific survival analyses."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Any, Callable, List, Tuple

import pandas as pd


CovariateHook = Callable[[Any, Namespace, Any], Tuple[str, ...]]
FeatureHook = Callable[[Any, Namespace, Any], List[str]]
KwargsHook = Callable[[Namespace], dict]
ParserHook = Callable[[ArgumentParser, Any], None]
# XGBoost-specific: builds the per-endpoint feature-selection metadata frame
# written to landmark_xgboost*_feature_selection.csv. Signature:
#   (ctx, endpoint, landmark_day, selected_features, feature_meta) -> DataFrame
XGBFeatureMetaHook = Callable[[Any, str, int, list, Any], "pd.DataFrame"]
# XGBoost-specific: builds the per-endpoint held-out risk frame written to
# landmark_xgboost*_patient_risks.csv. Signature:
#   (test, endpoint, landmark_day, duration, event, risk, id_col) -> DataFrame
XGBRiskFrameHook = Callable[[Any, str, int, Any, Any, Any, str], "pd.DataFrame"]


def no_context_kwargs(_args: Namespace) -> dict:
    return {}


def no_static_covariates(_ctx: Any, _args: Namespace, _cox: Any) -> tuple[str, ...]:
    return ()


def no_baseline_features(_ctx: Any, _args: Namespace, _cox: Any) -> list[str]:
    return []


def no_extra_args(_parser: ArgumentParser, _cox: Any) -> None:
    return None


@dataclass(frozen=True)
class CoxProjectConfig:
    """Project-specific hooks for shared Cox runners.

    The shared runners own CLI flow, per-landmark loops, output writing, and
    model orchestration. Project configs only describe study-specific pieces:
    endpoint defaults and static/baseline covariates.
    """

    name: str
    default_endpoints: tuple[str, ...]
    univariate_description: str
    multivariable_description: str
    baseline_help: str
    baseline_feature_count_column: str | None = None
    prepare_context_kwargs: KwargsHook = no_context_kwargs
    static_covariates: CovariateHook = no_static_covariates
    baseline_feature_cols: FeatureHook = no_baseline_features
    # Adds the project's extra CLI flags; the same flags apply to both the
    # univariate and multivariable runners (no project has needed to differ).
    add_cli_args: ParserHook = no_extra_args
    # --- XGBoost-arm hooks (used only by survival_common.xgboost_runners) ---
    # Whether the XGBoost baseline arm feeds config.baseline_feature_cols(...)
    # through the *feature* path (tree splits + missing indicators; COMPASS's
    # cancer-stage baseline) rather than the *static covariate* path (IPIO's
    # GENDER/therapy/cancer-type baseline). When False the baseline arm uses
    # feature_cols=[] and config.static_covariates(...) as static covariates.
    xgb_baseline_as_features: bool = False
    # feature_stat label stamped on the baseline feature-selection frame.
    xgb_baseline_feature_stat: str = "static"
    # Builds the per-endpoint XGBoost feature-selection and patient-risk frames.
    # Defaults preserve IPIO's schema; COMPASS overrides for its wider schema.
    xgb_feature_meta_frame: XGBFeatureMetaHook | None = None
    xgb_risk_frame: XGBRiskFrameHook | None = None


def existing_columns(df: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> list[str]:
    """Return configured columns that are present in a frame, preserving order."""
    return [col for col in columns if col in df.columns]
