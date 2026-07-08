"""COMPASS PROFILE Cox runner configuration."""

from __future__ import annotations

import argparse
from typing import Any

from survival_common.config import CoxProjectConfig


def _context_kwargs(args: argparse.Namespace) -> dict:
    return {
        "restrict_to_stage": args.restrict_to_stage,
    }


def _baseline_feature_cols(ctx: Any, _args: argparse.Namespace, cox: Any) -> list[str]:
    return cox.stage_feature_columns(ctx.merged)


def _add_profile_args(parser: argparse.ArgumentParser, _cox: Any) -> None:
    parser.add_argument(
        "--restrict-to-stage",
        action="store_true",
        help=(
            "Restrict the cohort to stage-available patients (non-missing "
            "CANCER_STAGE_*) before fitting/evaluating. Errors if no stage "
            "columns are present."
        ),
    )


CONFIG = CoxProjectConfig(
    name="compass_profile",
    default_endpoints=("platinum",),
    univariate_description="Univariate (n_obs-adjusted) Cox associations on COMPASS PROFILE landmarked lab features.",
    multivariable_description="Multivariable elastic-net Cox (+ age/static baseline) on COMPASS PROFILE landmarked lab features.",
    baseline_help=(
        "Fit an age(+cancer-stage)-only Cox model (no labs, no CV) on "
        "the same horizon grid for benchmarking."
    ),
    baseline_feature_count_column="n_stage_cols",
    prepare_context_kwargs=_context_kwargs,
    baseline_feature_cols=_baseline_feature_cols,
    add_cli_args=_add_profile_args,
)
