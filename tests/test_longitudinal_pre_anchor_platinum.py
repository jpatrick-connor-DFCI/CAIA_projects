"""Longitudinal figure cohort and endpoint scoping regressions.

Descriptive longitudinal panels use the exact base-landmark cohort and are
endpoint-independent. Plotting GAMs are fit directly in R to the same
zero-anchored patient-bin trajectories.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_R = (
    REPO_ROOT / "COMPASS" / "survival_analysis" / "COMPASS_generate_figures_pipeline.R"
)


def test_descriptive_longitudinal_uses_exact_base_cohort():
    source = PIPELINE_R.read_text()
    assert "cohort_mrns <- unique(as.character(patient_df[[ID_COL]]))" in source
    assert "aggregated_landmark_mrns <- function" not in source
    assert "restricted to exact base-landmark cohort" in source


def test_descriptive_longitudinal_is_endpoint_independent_and_fresh():
    source = PIPELINE_R.read_text()
    assert "if (!EMIT_ENDPOINT_INDEPENDENT) {" in source
    assert "canonical_long_df <- load_canonical_longitudinal()" in source
    assert "Per-lab figures: removed %d legacy NEPC-labelled file(s)" in source
    assert source.count("force_overwrite = TRUE") >= 2


def test_every_lab_figure_family_is_platinum_only():
    source = PIPELINE_R.read_text()
    gate = source.index("if (!EMIT_ENDPOINT_INDEPENDENT) {", source.index("labs_output_root"))
    supplement = source.index("Supplement -- localized-adjuvant vs metastatic", gate)
    block = source[gate:supplement]
    assert "plot_km_androgen_quartile" in block
    assert "plot_androgen_dist_by_platinum" in block
    assert "load_canonical_longitudinal" in block
    assert "plot_group_gam_panel" in block
    assert 'retired_lab_leaf <- paste0("nepc__", cohort_leaf_slug(COHORT), ".png")' in source


def test_figure_strata_are_limited_to_binary_nepc():
    source = PIPELINE_R.read_text()
    assert 'FIGURE_LLM_STRATA <- LLM_STRATA["has_nepc"]' in source
    assert "for (scheme_name in names(FIGURE_LLM_STRATA))" in source
    assert "for (scheme_name in names(LLM_STRATA))" not in source
    assert 'save_fig(p_has_avpc_v3' not in source
    assert 'save_fig(pB_v3' not in source
    assert 'save_fig(pC_v3' not in source
    assert '"figure2v3_nepc_validation"' in source


def test_descriptive_longitudinal_titles_name_the_cohort():
    source = PIPELINE_R.read_text()
    assert "lab_group, COHORT_DISPLAY, ANCHOR_LABEL" in source
    assert "scheme_name, COHORT_DISPLAY, ANCHOR_LABEL" in source


def test_by_figure_is_the_only_output_layout():
    source = PIPELINE_R.read_text()
    assert 'file.path(FIG_ROOT, "by_figure"' in source
    assert 'legacy_cohort_roots <- file.path(fig_root, toupper(COHORT_ARMS), "by_cohort")' in source
    assert "unlink(legacy_cohort_root, recursive = TRUE, force = TRUE)" in source
    assert "rebuild_cohort_view" not in source
    assert "file.symlink" not in source


def test_longitudinal_bins_are_180_days_and_zero_anchored():
    source = PIPELINE_R.read_text()
    assert "BIN_WIDTH_DAYS <- 180" in source
    assert "anchored_bin_edges <- function" in source
    assert "right = FALSE" in source
    assert "patient_bin_trajectory" in source


def test_plotting_gams_are_fit_in_r_with_reml_tuning():
    source = PIPELINE_R.read_text()
    start = source.index("fit_plotting_gam <- function")
    end = source.index("Retired precomputed feature-extraction GAM figures", start)
    block = source[start:end]
    assert "mgcv::bam(" in block
    assert 'method = "fREML"' in block
    assert "select = TRUE" in block
    assert "patient_bin_trajectory" in block
    assert 'sprintf("gam_longitudinal_platinum_%s"' in block
    assert 'sprintf("gam_longitudinal_has_nepc_%s"' in block
