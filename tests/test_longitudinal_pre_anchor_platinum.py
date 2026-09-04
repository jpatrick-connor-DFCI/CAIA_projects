"""Longitudinal trajectory panels must exclude pre-anchor platinum exposure.

The platinum endpoint gates its cohort on ``t_platinum > 0`` in
``survival_common.cohort.make_outcome_df``, so its ``aggregated_landmark*.csv``
files carry no patient treated with platinum before the landmark. The NEPC and
AVPC endpoints deliberately skip that gate ("pre-anchor platinum exposure is
irrelevant here"), so their aggregated CSVs *do* retain such patients.

``aggregated_landmark_mrns`` in COMPASS_generate_figures_pipeline.R is the sole
cohort gate for the Figure 7 trajectory panels, and the GAM block re-reads the
same CSVs for its own strata. Both must drop pre-anchor platinum patients, or a
NEPC/AVPC longitudinal figure plots patients whose PLATINUM==1 label describes
treatment history rather than an incident post-landmark event -- and the
"Platinum" trace stops matching the prediction/univariate/multivariate cohort.

Durations in these CSVs are already landmark-rebased, so ``t_platinum <= 0`` is
exactly the pre-landmark exposure to exclude.
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_R = (
    REPO_ROOT / "COMPASS" / "survival_analysis" / "COMPASS_generate_figures_pipeline.R"
)

pytestmark = pytest.mark.skipif(
    shutil.which("Rscript") is None, reason="Rscript is not available"
)


def _extract_nested_function(name: str) -> str:
    """Return a function defined at any indent level, by brace balance.

    ``aggregated_landmark_mrns`` lives inside ``generate_figures``, so the
    column-0 extractor in test_figure_pipeline_endpoints.py does not reach it.
    """
    lines = PIPELINE_R.read_text().split("\n")
    start = next(
        i
        for i, line in enumerate(lines)
        if line.strip().startswith(f"{name} <- function")
    )
    depth = 0
    for i in range(start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth == 0 and i > start:
            return textwrap.dedent("\n".join(lines[start : i + 1]))
    raise AssertionError(f"unterminated function: {name}")


def _run_r(script: str, tmp_path: Path) -> str:
    path = tmp_path / "check.R"
    path.write_text(script)
    result = subprocess.run(
        ["Rscript", str(path)], capture_output=True, text=True, cwd=tmp_path
    )
    if result.returncode != 0:
        pytest.fail(f"Rscript failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout


def _harness(body: str, landmarks: str = "c(0)") -> str:
    return textwrap.dedent(
        """
        suppressPackageStartupMessages({{library(dplyr); library(readr)}})
        {fn}
        INPUTS_DIR <- tempdir()
        LANDMARKS <- {landmarks}
        unlink(list.files(INPUTS_DIR, "aggregated_landmark", full.names = TRUE))
        {body}
        """
    ).format(
        fn=_extract_nested_function("aggregated_landmark_mrns"),
        landmarks=landmarks,
        body=textwrap.dedent(body),
    )


def test_drops_platinum_at_or_before_the_landmark(tmp_path):
    """t_platinum <= 0 is pre-landmark exposure and must not reach the figure.

    p2's platinum predates the anchor and p4's lands exactly on it; both are
    excluded. p1 (incident platinum) and p3 (never platinum) are kept.
    """
    script = _harness(
        """
        write_csv(tibble(
          DFCI_MRN   = c("p1", "p2", "p3", "p4"),
          PLATINUM   = c(1, 1, 0, 1),
          t_platinum = c(200, -30, 500, 0)
        ), file.path(INPUTS_DIR, "aggregated_landmark0.csv"))
        cat(paste(sort(aggregated_landmark_mrns("DFCI_MRN")), collapse = ","), "\\n")
        """
    )
    assert _run_r(script, tmp_path).strip() == "p1,p3"


def test_exclusion_holds_across_every_landmark(tmp_path):
    """The union over landmarks must not re-admit a patient excluded elsewhere.

    p2 is pre-anchor at both landmarks. Were the filter applied to only one
    CSV, the union would let p2 back into the plotted cohort.
    """
    script = _harness(
        """
        write_csv(tibble(
          DFCI_MRN   = c("p1", "p2", "p3"),
          PLATINUM   = c(1, 1, 0),
          t_platinum = c(200, -30, 500)
        ), file.path(INPUTS_DIR, "aggregated_landmark0.csv"))
        write_csv(tibble(
          DFCI_MRN   = c("p1", "p2", "p5"),
          PLATINUM   = c(1, 1, 1),
          t_platinum = c(110, -120, 300)
        ), file.path(INPUTS_DIR, "aggregated_landmark90.csv"))
        cat(paste(sort(aggregated_landmark_mrns("DFCI_MRN")), collapse = ","), "\\n")
        """,
        landmarks="c(0, 90)",
    )
    assert _run_r(script, tmp_path).strip() == "p1,p3,p5"


def test_missing_timing_columns_keep_the_cohort(tmp_path):
    """An aggregated CSV without t_platinum must warn, not silently empty out."""
    script = _harness(
        """
        write_csv(tibble(DFCI_MRN = c("a", "b"), PLATINUM = c(1, 0)),
                  file.path(INPUTS_DIR, "aggregated_landmark0.csv"))
        cat(paste(sort(aggregated_landmark_mrns("DFCI_MRN")), collapse = ","), "\\n")
        """
    )
    assert _run_r(script, tmp_path).strip() == "a,b"


def test_fully_excluded_cohort_returns_null(tmp_path):
    """No eligible patient must return NULL so the caller skips the figure.

    Returning an empty vector instead would filter the trajectory frame to zero
    rows and emit an empty panel.
    """
    script = _harness(
        """
        write_csv(tibble(DFCI_MRN = "x", PLATINUM = 1, t_platinum = -5),
                  file.path(INPUTS_DIR, "aggregated_landmark0.csv"))
        cat(is.null(aggregated_landmark_mrns("DFCI_MRN")), "\\n")
        """
    )
    assert _run_r(script, tmp_path).strip() == "TRUE"


def test_gam_block_also_filters_pre_anchor_platinum():
    """The GAM strata re-read the aggregated CSVs and need the same exclusion."""
    source = PIPELINE_R.read_text()
    start = source.index("agg_raw <- read_csv(")
    block = source[start : start + 2000]
    assert '"t_platinum"' in block, "GAM read must select t_platinum"
    assert "agg_pre_anchor" in block, "GAM block must compute the pre-anchor mask"
    assert (
        "agg_raw[!agg_pre_anchor, , drop = FALSE]" in block
    ), "GAM strata must be built from the filtered frame"
    assert (
        "gam_curves %>% filter(DFCI_MRN %in% eligible_gam_mrns)" in source
    ), "GAM curves must be restricted to the platinum-eligible cohort"
