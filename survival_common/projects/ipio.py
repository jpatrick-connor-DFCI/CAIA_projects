"""IPIO Cox runner configuration."""

from __future__ import annotations

import argparse
from typing import Any

from survival_common.config import CoxProjectConfig


def _static_covariates(ctx: Any, _args: argparse.Namespace, cox: Any) -> tuple[str, ...]:
    return tuple(cox.baseline_covariate_columns(ctx.merged))


CONFIG = CoxProjectConfig(
    name="ipio",
    default_endpoints=("irae",),
    univariate_description="Univariate (n_obs-adjusted) Cox associations on IPIO landmarked lab features.",
    multivariable_description="Multivariable elastic-net Cox (+ age/baseline-covariate baseline) on IPIO landmarked lab features.",
    baseline_help=(
        "Fit an age(+baseline-covariates)-only Cox model (no labs, no CV) on "
        "the same horizon grid for benchmarking."
    ),
    static_covariates=_static_covariates,
)
