"""COMPASS PROFILE Cox runner configuration."""

from __future__ import annotations

import argparse

from typing import Any

from survival_common.config import CoxProjectConfig


def _static_covariates(ctx: Any, _args: argparse.Namespace, cox: Any) -> tuple[str, ...]:
    """Panel-version dummies and the somatic burden count, adjusted for in every fit but never tested.

    Each is empty unless the inputs carry it: panel version for genomic inputs
    built after it was carried through, the burden count only for the
    any-alteration sequencing screen. Every other run is unchanged.
    """
    return tuple(cox.panel_version_covariate_columns(ctx.merged)) + tuple(
        cox.somatic_burden_covariate_columns(ctx.merged)
    )


def _prepare_context_kwargs(args: argparse.Namespace) -> dict:
    return {
        "feature_set": getattr(args, "feature_set", "labs"),
        "exclude_vitals": getattr(args, "exclude_vitals", False),
        "feature_stats": getattr(args, "feature_stats", "all"),
    }


def _add_cli_args(parser: argparse.ArgumentParser, _cox) -> None:
    parser.add_argument(
        "--exclude-vitals", action="store_true",
        help="Exclude the seven canonical vital measurements from lab candidates (labs feature set only).",
    )
    parser.add_argument(
        "--feature-stats", choices=["all", "mean-n-obs"], default="all",
        help="Lab summary candidates: all statistics, or mean + n_observations; both retain train-derived missingness indicators.",
    )
    parser.add_argument(
        "--feature-set",
        choices=[
            "labs",
            "somatic-gleason",
            "genomic",
            "gleason",
            "somatic",
            "gleason-labs",
            "somatic-labs",
            "gleason-somatic",
            "gleason-somatic-labs",
        ],
        default="labs",
        help=(
            "Run the standard lab features, the separate somatic + Gleason "
            "feature inputs (legacy 'somatic-gleason', every manifest feature "
            "together), the SNV-only genomic arm, or one of the "
            "component-based Gleason/somatic/labs arms ('gleason', "
            "'somatic', 'gleason-labs', 'somatic-labs', 'gleason-somatic', "
            "'gleason-somatic-labs'). The Gleason/somatic arms read "
            "somatic_gleason_features.csv from build_somatic_gleason_inputs.py."
        ),
    )


CONFIG = CoxProjectConfig(
    name="compass_profile",
    default_endpoints=("platinum",),
    univariate_description="Univariate (n_obs-adjusted) Cox associations on COMPASS PROFILE landmarked lab features.",
    multivariable_description="Multivariable elastic-net Cox (+ age baseline) on COMPASS PROFILE landmarked lab features.",
    baseline_help=(
        "Fit an age-only Cox model (no labs, no CV) on "
        "the same horizon grid for benchmarking."
    ),
    prepare_context_kwargs=_prepare_context_kwargs,
    static_covariates=_static_covariates,
    add_cli_args=_add_cli_args,
)
