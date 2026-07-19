"""COMPASS PROFILE Cox runner configuration."""

from __future__ import annotations

from survival_common.config import CoxProjectConfig


CONFIG = CoxProjectConfig(
    name="compass_profile",
    default_endpoints=("platinum",),
    univariate_description="Univariate (n_obs-adjusted) Cox associations on COMPASS PROFILE landmarked lab features.",
    multivariable_description="Multivariable elastic-net Cox (+ age baseline) on COMPASS PROFILE landmarked lab features.",
    baseline_help=(
        "Fit an age-only Cox model (no labs, no CV) on "
        "the same horizon grid for benchmarking."
    ),
)
