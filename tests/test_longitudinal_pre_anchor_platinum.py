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
FIGURES_RMD = REPO_ROOT / "COMPASS" / "survival_analysis" / "05_figures.Rmd"


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


def test_retained_lab_figure_families_are_platinum_only():
    source = PIPELINE_R.read_text()
    gate = source.index("if (!EMIT_ENDPOINT_INDEPENDENT) {", source.index("labs_output_root"))
    supplement = source.index("Supplement -- localized-adjuvant vs metastatic", gate)
    block = source[gate:supplement]
    assert "load_canonical_longitudinal" in block
    assert "plot_group_gam_panel" in block
    assert "Retired: per-lab quartile KM figures" in block
    assert "Retired: per-lab distribution figures" in block
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


def test_longitudinal_and_gam_titles_only_show_cohort_and_counts():
    source = PIPELINE_R.read_text()
    assert 'ttl <- sprintf("%s (n=%s)", COHORT_DISPLAY' in source
    assert 'ttl_s <- sprintf("%s (n=%s/%s labeled)", COHORT_DISPLAY' in source
    assert 'sprintf("%s (n=%s)", COHORT_DISPLAY' in source
    assert 'sprintf("%s (n=%s/%s labeled)", COHORT_DISPLAY' in source
    assert "R-fitted GAM by platinum status" not in source
    assert "group mean +/- 95%% CI vs. days" not in source


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
    assert "PRE_DAYS  <- 1 * 365.25" in source
    assert "POST_DAYS <- 5 * 365.25" in source
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
    assert 'sprintf("gam_longitudinal_platinum_%s%s"' in block
    assert 'sprintf("gam_longitudinal_has_nepc_%s%s"' in block


def test_figure_notebook_toggles_gam_fitting_off_without_removing_it():
    notebook = FIGURES_RMD.read_text()
    pipeline = PIPELINE_R.read_text()
    assert "PLOT_GAM_TRAJECTORIES <- FALSE" in notebook
    assert "plot_gam_trajectories = PLOT_GAM_TRAJECTORIES" in notebook
    assert "plot_gam_trajectories = TRUE" in pipeline
    assert "if (!plot_gam_trajectories)" in pipeline
    assert "mgcv::bam(" in pipeline


def test_longitudinal_and_gam_figures_also_emit_log_space_versions():
    source = PIPELINE_R.read_text()
    start = source.index("patient_bin_trajectory <- function")
    end = source.index("Retired precomputed feature-extraction GAM figures", start)
    block = source[start:end]
    assert "log_scale = FALSE" in block
    assert "mutate(LAB_VALUE = log1p(LAB_VALUE))" in block
    assert 'scale_suffix <- if (log_scale) "_log" else ""' in block
    assert 'sprintf("longitudinal_platinum_%s%s"' in block
    assert 'sprintf("longitudinal_%s_%s%s"' in block
    assert 'sprintf("gam_longitudinal_platinum_%s%s"' in block
    assert 'sprintf("gam_longitudinal_has_nepc_%s%s"' in block
    assert 'paste0("log1p(", lab_group, ")")' in block


def test_pre_adt_androgen_coverage_keeps_zero_measurement_patients():
    source = PIPELINE_R.read_text()
    start = source.index("Pre-ADT PSA/testosterone coverage diagnostics")
    end = source.index("for (lab_group in labs_present)", start)
    block = source[start:end]
    assert "crossing(" in block
    assert "coverage_strata" in block
    assert "LAB_GROUP = coverage_labs" in block
    assert "n_pre = coalesce(as.integer(n_pre), 0L)" in block
    assert "n_pre_180 = coalesce(as.integer(n_pre_180), 0L)" in block
    assert 'n_pre == 0 ~ "0"' in block
    assert "n_patients = n()" in block


def test_pre_adt_coverage_emits_two_complementary_figures():
    source = PIPELINE_R.read_text()
    assert 'return("androgen_pre_adt_coverage")' in source
    assert '"pre_adt_coverage_any_psa_testosterone"' in source
    assert '"pre_adt_coverage_by_bin_psa_testosterone"' in source
    assert "Retired: patient-level count-burden figure" in source
    assert "Within 180 days before ADT" in source
    assert "COVERAGE_PRE_DAYS <- 5 * 365.25" in source
    assert "pre_edges <- anchored_bin_edges(COVERAGE_PRE_DAYS, 0, BIN_WIDTH_DAYS)" in source


def test_requested_figure_families_are_retired_and_cleaned():
    source = PIPELINE_R.read_text()
    assert '"figure1s_analysis_sets", "km_llm", "distribution", "km_quartile"' in source
    assert '"s_multivariate_all_models", "figure4s_multivariate_all_models"' in source
    assert 'grepl("significance", basename(generated_dirs)' in source
    assert 'grepl("gleason|prs", basename(generated_dirs)' in source
    assert "Retired: Figure 1 analysis-set-size supplement" in source
    assert "Retired: supplemental all-model comparison" in source
    assert 'SG_ANALYSES  <- "sequencing"' in source
    assert "figure3_univariate_%s_significance_landmark%d" not in source
    assert "plot_volcano_panel_by_significance(sub" not in source


def test_output_artifact_names_drop_redundant_parent_tokens():
    source = PIPELINE_R.read_text()
    assert "artifact_name_for_stem <- function" in source
    assert 'figure3 = "^figure3_?"' in source
    assert 'longitudinal = "^(androgen_)?longitudinal_?"' in source
    assert 'gam_trajectory = "^gam_(trajectory|longitudinal)_?"' in source
    assert "lab_slug <- lab_stem_slug(lab)" in source
    assert "artifact_name_for_stem(plot_stem, group)" in source
    assert "remove_legacy_artifacts <- function" in source


def test_figure_notebook_defaults_to_two_cell_workers():
    source = FIGURES_RMD.read_text()
    assert 'Sys.getenv("COMPASS_RENDER_WORKERS", "2")' in source
    assert "render_grid <- tidyr::crossing(COHORT = COHORTS, ENDPOINT = ENDPOINTS)" in source
    assert "n_workers <- min(RENDER_WORKERS, nrow(render_grid))" in source
    assert "mc.cores = n_workers" in source


def test_figure_notebook_preloads_shared_data_before_forking():
    source = FIGURES_RMD.read_text()
    preload = source.index("invisible(cached_profile_patient_and_labs(")
    fork = source.index("parallel::mclapply(")
    assert preload < fork
    assert "invisible(cached_canonical_longitudinal(shared_longitudinal_csv))" in source
    assert "forking workers from warm cache" in source
    assert 'cell_started <- proc.time()[["elapsed"]]' in source


def test_png_is_default_and_pdf_is_opt_in():
    pipeline = PIPELINE_R.read_text()
    notebook = FIGURES_RMD.read_text()
    assert "save_pdf = FALSE" in pipeline
    assert "write_png <- overwrite || force_overwrite || !file.exists(png_out)" in pipeline
    assert "write_pdf <- save_pdf &&" in pipeline
    assert "removed %d stale PDF figure(s); PNG-only mode is active" in pipeline
    assert 'Sys.getenv("COMPASS_RENDER_PDF", "false")' in notebook
    assert "save_pdf = RENDER_PDF" in notebook
