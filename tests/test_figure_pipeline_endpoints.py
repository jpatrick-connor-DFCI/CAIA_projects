"""The R figure pipeline's endpoint wiring: results-tree paths, output
separation, and per-endpoint metric selection.

Grounded in COMPASS/survival_analysis/COMPASS_generate_figures_pipeline.R
(read_endpoint_performance + the ENDPOINT_SUFFIXES/FIG_ROOT plumbing in
generate_figures) and compass_pipeline.make_runs, whose ``output_suffix`` these
paths must mirror: "" for platinum, "_nepc" for NEPC.

Two invariants matter here. First, the *suffix agreement*: the figure pipeline
reads local_runs_*/prediction_inputs_* trees that Python writes, so a drift
between the two suffix schemes silently points figures at the wrong cohort.
Second, the *filter*: metrics files carry both endpoints in one CSV, so
selecting rows by endpoint must return that endpoint's row and nothing else.
The dplyr data-masking form matters -- a bare `endpoint` inside filter() binds
to the data column rather than the argument, which keeps every row and makes a
NEPC panel silently display platinum numbers.
"""

from __future__ import annotations

import re
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


def _pipeline_source() -> str:
    return PIPELINE_R.read_text()


def _extract_function(name: str) -> str:
    """Return one top-level R function definition, by brace balance.

    Counting braces rather than stopping at the first ``}`` keeps nested
    helpers (read_endpoint_performance defines first_numeric) intact.
    """
    lines = _pipeline_source().split("\n")
    start = next(
        i for i, line in enumerate(lines) if line.startswith(f"{name} <- function")
    )
    depth = 0
    for i in range(start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth == 0 and i > start:
            return "\n".join(lines[start : i + 1])
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


def _r_literal(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def test_endpoint_suffixes_match_make_runs():
    """ENDPOINT_SUFFIXES must mirror compass_pipeline.make_runs's output_suffix."""
    source = _pipeline_source()
    block = re.search(r"ENDPOINT_SUFFIXES <- c\((.*?)\)", source, re.S)
    assert block, "ENDPOINT_SUFFIXES not found"
    body = block.group(1)
    assert re.search(r'platinum\s*=\s*""', body), "platinum suffix must be empty"
    assert re.search(r'nepc\s*=\s*"_nepc"', body), "nepc suffix must be _nepc"


def test_endpoint_filter_is_not_data_masked():
    """The endpoint filter must not compare a column against itself.

    Inside dplyr::filter, a bare `endpoint` resolves to the data column, so
    `tolower(endpoint)` would be an elementwise self-comparison that keeps every
    row. The comparison must therefore come from the environment (.env$... or a
    distinctly named local), never a bare parameter sharing the column's name.
    """
    fn = _extract_function("read_endpoint_performance")
    filter_lines = [line for line in fn.split("\n") if "filter(" in line]
    assert filter_lines, "expected a filter() call"
    joined = "\n".join(filter_lines)
    assert "== tolower(endpoint)" not in joined, (
        "bare `endpoint` inside filter() is data-masked to the column; "
        "use .env$ or a distinctly named local"
    )
    assert ".env$" in joined or "wanted_endpoint" in joined


@pytest.mark.parametrize(
    ("endpoint", "expected_auc"), [("platinum", 0.71), ("nepc", 0.64)]
)
def test_reads_the_requested_endpoints_row(tmp_path, endpoint, expected_auc):
    """A metrics CSV holding both endpoints must yield the requested one."""
    script = textwrap.dedent(
        f"""
        suppressPackageStartupMessages({{library(dplyr); library(readr)}})
        {_extract_function("read_endpoint_performance")}

        path <- file.path(tempdir(), "metrics.csv")
        write_csv(tibble(
          endpoint = c("platinum", "nepc"),
          test_mean_auc_t = c(0.71, 0.64),
          test_c_index = c(0.69, 0.62),
          test_integrated_brier = c(0.15, 0.19)
        ), path)

        got <- read_endpoint_performance(
          path, {_r_literal(endpoint)},
          "test_mean_auc_t", "test_c_index", "test_integrated_brier"
        )
        cat(sprintf("%.4f\\n", got[["auc"]]))
        """
    )
    assert float(_run_r(script, tmp_path).strip()) == pytest.approx(expected_auc)


def test_absent_endpoint_does_not_fall_back(tmp_path):
    """A platinum-only file queried for NEPC yields NA, never platinum's row."""
    script = textwrap.dedent(
        f"""
        suppressPackageStartupMessages({{library(dplyr); library(readr)}})
        {_extract_function("read_endpoint_performance")}

        path <- file.path(tempdir(), "platinum_only.csv")
        write_csv(tibble(
          endpoint = "platinum", mean_auc_t = 0.9,
          c_index = 0.9, integrated_brier = 0.1
        ), path)

        got <- read_endpoint_performance(
          path, "nepc", "mean_auc_t", "c_index", "integrated_brier"
        )
        cat(all(is.na(got)), "\\n")
        """
    )
    assert _run_r(script, tmp_path).strip() == "TRUE"


def test_endpoint_match_is_case_insensitive(tmp_path):
    """Stored endpoint values are matched case-insensitively."""
    script = textwrap.dedent(
        f"""
        suppressPackageStartupMessages({{library(dplyr); library(readr)}})
        {_extract_function("read_endpoint_performance")}

        path <- file.path(tempdir(), "upper.csv")
        write_csv(tibble(
          endpoint = "NEPC", mean_auc_t = 0.5,
          c_index = 0.5, integrated_brier = 0.2
        ), path)

        got <- read_endpoint_performance(
          path, "nepc", "mean_auc_t", "c_index", "integrated_brier"
        )
        cat(sprintf("%.4f\\n", got[["auc"]]))
        """
    )
    assert float(_run_r(script, tmp_path).strip()) == pytest.approx(0.5)


def test_missing_file_returns_na(tmp_path):
    """A missing metrics file degrades to NA rather than raising."""
    script = textwrap.dedent(
        f"""
        suppressPackageStartupMessages({{library(dplyr); library(readr)}})
        {_extract_function("read_endpoint_performance")}

        got <- read_endpoint_performance(
          file.path(tempdir(), "does_not_exist.csv"), "nepc", "a", "b", "c"
        )
        cat(all(is.na(got)), "\\n")
        """
    )
    assert _run_r(script, tmp_path).strip() == "TRUE"


def test_figure_roots_do_not_collide():
    """Platinum keeps the historical path; NEPC nests below it.

    The two endpoints share every plot stem, so a shared FIG_ROOT would have
    NEPC overwrite the platinum panels.
    """
    source = _pipeline_source()
    assert 'FIG_ROOT <- file.path(fig_root, toupper(COHORT))' in source, (
        "platinum must keep its un-suffixed figure root"
    )
    assert re.search(
        r'if \(!identical\(ENDPOINT_SUFFIX, ""\)\) \{\s*\n\s*'
        r"FIG_ROOT <- file\.path\(FIG_ROOT, ENDPOINT\)",
        source,
    ), "non-platinum endpoints must nest one level deeper"


def test_results_trees_are_endpoint_suffixed():
    """BASE and INPUTS_DIR must both carry the endpoint suffix.

    make_runs suffixes both trees, because the NEPC cohort is gated on
    t_nepc > 0 and is therefore a different patient set.
    """
    source = _pipeline_source()
    for prefix in ("local_runs_", "prediction_inputs_"):
        assert f'paste0("{prefix}", COHORT, ENDPOINT_SUFFIX)' in source, (
            f"{prefix} must be endpoint-suffixed"
        )


def test_pipeline_parses(tmp_path):
    """The pipeline must remain syntactically valid R."""
    script = f"invisible(parse({_r_literal(str(PIPELINE_R))})); cat('OK\\n')"
    assert _run_r(script, tmp_path).strip() == "OK"
