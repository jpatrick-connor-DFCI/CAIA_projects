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
    helpers (read_endpoint_performance defines `metric`) intact.
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
    assert re.search(r'avpc_nepc\s*=\s*"_avpc_nepc"', body), "avpc_nepc suffix must be _avpc_nepc"


def test_supported_endpoints_includes_avpc_nepc():
    """SUPPORTED_ENDPOINTS must list avpc_nepc alongside platinum/nepc."""
    source = _pipeline_source()
    block = re.search(r"SUPPORTED_ENDPOINTS <- c\((.*?)\)", source, re.S)
    assert block, "SUPPORTED_ENDPOINTS not found"
    body = block.group(1)
    assert '"avpc_nepc"' in body, "avpc_nepc must be a supported endpoint"


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
    ("endpoint", "expected_auc"), [("platinum", 0.71), ("nepc", 0.64), ("avpc_nepc", 0.58)]
)
def test_reads_the_requested_endpoints_row(tmp_path, endpoint, expected_auc):
    """A metrics CSV holding all three endpoints must yield the requested one."""
    script = textwrap.dedent(
        f"""
        suppressPackageStartupMessages({{library(dplyr); library(readr)}})
        {_extract_function("read_endpoint_performance")}

        path <- file.path(tempdir(), "metrics.csv")
        write_csv(tibble(
          endpoint = c("platinum", "nepc", "avpc_nepc"),
          test_mean_auc_t = c(0.71, 0.64, 0.58),
          test_c_index = c(0.69, 0.62, 0.57),
          test_integrated_brier = c(0.15, 0.19, 0.21)
        ), path)

        got <- read_endpoint_performance(path, {_r_literal(endpoint)})
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
          endpoint = "platinum", test_mean_auc_t = 0.9,
          test_c_index = 0.9, test_integrated_brier = 0.1
        ), path)

        got <- read_endpoint_performance(path, "nepc")
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
          endpoint = "NEPC", test_mean_auc_t = 0.5,
          test_c_index = 0.5, test_integrated_brier = 0.2
        ), path)

        got <- read_endpoint_performance(path, "nepc")
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
          file.path(tempdir(), "does_not_exist.csv"), "nepc"
        )
        cat(all(is.na(got)), "\\n")
        """
    )
    assert _run_r(script, tmp_path).strip() == "TRUE"


def test_figure_roots_do_not_collide():
    """Every endpoint nests one level below its cohort arm -- platinum included.

    All endpoints share every plot stem, so any two sharing a FIG_ROOT would
    overwrite each other's panels. Platinum used to keep the un-suffixed
    cohort root; it is now nested like the rest, so the figure path must be
    driven by ENDPOINT itself rather than by ENDPOINT_SUFFIX (which stays ""
    for platinum because it still names the un-suffixed *data* trees).
    """
    source = _pipeline_source()
    assert "FIG_ROOT <- file.path(fig_root, toupper(COHORT), ENDPOINT)" in source, (
        "every endpoint's figure root must nest under <cohort>/<endpoint>"
    )
    assert not re.search(
        r'if \(!identical\(ENDPOINT_SUFFIX, ""\)\) \{\s*\n\s*'
        r"FIG_ROOT <- file\.path\(FIG_ROOT, ENDPOINT\)",
        source,
    ), "platinum must no longer be special-cased out of the nesting"


def test_results_trees_are_endpoint_suffixed():
    """BASE and INPUTS_DIR must both carry the endpoint suffix.

    make_runs suffixes both trees, because the NEPC and AVPC_NEPC cohorts are
    each gated on their own t_* > 0 incident condition and are therefore
    different patient sets from platinum (and from each other).
    """
    source = _pipeline_source()
    for prefix in ("local_runs_", "prediction_inputs_"):
        assert f'paste0("{prefix}", COHORT, ENDPOINT_SUFFIX)' in source, (
            f"{prefix} must be endpoint-suffixed"
        )


def test_every_supported_endpoint_nests_uniformly():
    """avpc and avpc_nepc take the same nesting path as nepc and platinum.

    The FIG_ROOT expression is generic over ENDPOINT, so no endpoint-specific
    branch should exist (and none is needed) for any supported endpoint.
    """
    source = _pipeline_source()
    assert "FIG_ROOT <- file.path(fig_root, toupper(COHORT), ENDPOINT)" in source
    for endpoint in ("platinum", "nepc", "avpc", "avpc_nepc"):
        assert not re.search(
            rf'FIG_ROOT.*identical\(ENDPOINT, "{endpoint}"\)', source
        ), f"{endpoint} must not be special-cased in the figure root"


def test_pipeline_parses(tmp_path):
    """The pipeline must remain syntactically valid R."""
    script = f"invisible(parse({_r_literal(str(PIPELINE_R))})); cat('OK\\n')"
    assert _run_r(script, tmp_path).strip() == "OK"


# ---- ADT-intent supplemental section -------------------------------------
#
# The supplement reads CSVs written by adt_intent_comparison.py. Two things can
# silently break it: a filename drifting apart between the Python writer and the
# R reader, and the "adt_intent_" stem falling through figure_group() into a
# numbered figure group. Both are checked here because neither shows up as an
# error -- the first yields an empty section, the second files the supplement
# under Figure 1.


def _comparison_module_source() -> str:
    return (
        REPO_ROOT / "COMPASS" / "survival_analysis" / "adt_intent_comparison.py"
    ).read_text()


def test_supplement_reads_every_filename_the_module_writes():
    r_source = _pipeline_source()
    python_source = _comparison_module_source()

    filenames = re.findall(r'"(adt_intent_[a-z_]+\.csv)"', python_source)
    assert filenames, "the comparison module declares no output filenames"
    for filename in filenames:
        assert filename in r_source, (
            f"{filename} is written by adt_intent_comparison.py but never read "
            "by the R supplement"
        )


def test_supplement_stems_route_to_their_own_figure_group():
    source = _pipeline_source()
    assert 'if (startsWith(plot_stem, "adt_intent_")) return("supplement_adt_intent")' in source

    # Must be tested before the numbered-prefix branches, or an "adt_intent_"
    # stem would need only to also start with "figure1" to be misrouted. The
    # ordering is the guarantee, so assert on position rather than presence.
    supplement_at = source.index('startsWith(plot_stem, "adt_intent_")')
    figure1_at = source.index('startsWith(plot_stem, "figure1s")')
    assert supplement_at < figure1_at

    # supplement_adt_intent is not a numbered group, so it keeps the extra
    # per-stem directory level and must not be swept by the legacy cleanup.
    numbered = re.search(r"numbered_figure_groups <- c\((.*?)\)", source, re.S)
    assert numbered is not None
    assert "supplement_adt_intent" not in numbered.group(1)


def test_supplement_is_gated_to_the_adt_arm():
    # The intent strata are defined by ADT medication history; running the
    # section on the ARPI arm would read trees that are never built.
    source = _pipeline_source()
    supplement_at = source.index("Supplement -- localized-adjuvant vs metastatic")
    section = source[supplement_at:]
    assert "if (IS_ADT) {" in section


def test_supplement_never_refits_anything():
    section_source = _pipeline_source()
    supplement_at = section_source.index("Supplement -- localized-adjuvant vs metastatic")
    section = section_source[supplement_at:]
    for forbidden in ("coxph(", "survfit(", "glmnet("):
        assert forbidden not in section, (
            f"the supplement calls {forbidden}; it must only read the "
            "comparison CSVs"
        )
