"""COMPASS PROFILE Cox runner configuration."""

from __future__ import annotations

import argparse

from typing import Any

from survival_common.config import CoxProjectConfig


def _static_covariates(ctx: Any, _args: argparse.Namespace, cox: Any) -> tuple[str, ...]:
    """Panel-version dummies, adjusted for in every fit but never tested.

    Empty for lab-only runs and for genomic inputs built before panel version
    was carried through, so those runs are unchanged.
    """
    return tuple(cox.panel_version_covariate_columns(ctx.merged))


def _prepare_context_kwargs(args: argparse.Namespace) -> dict:
    return {"feature_set": args.feature_set}


def _add_cli_args(parser: argparse.ArgumentParser, _cox) -> None:
    parser.add_argument(
        "--feature-set",
        choices=["labs", "somatic-gleason", "genomic"],
        default="labs",
        help=(
            "Run the standard lab features, the separate somatic + Gleason "
            "feature inputs, or the SNV-only genomic arm."
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
