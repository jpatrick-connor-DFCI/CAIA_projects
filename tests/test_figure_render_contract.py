"""Render-contract regressions for the COMPASS figure pipeline.

The longitudinal lab, plotting-GAM, coverage and ADT-intent figure families
were removed with the figure pruning to the ADT export; what remains is the
output layout, stem naming, PDF-only export and render reuse.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_R = (
    REPO_ROOT / "COMPASS" / "figures" / "COMPASS_generate_figures_pipeline.R"
)
FIGURES_SCRIPT = REPO_ROOT / "COMPASS" / "figures" / "run_figures.R"


def test_by_figure_is_the_only_output_layout():
    source = PIPELINE_R.read_text()
    assert 'file.path(FIG_ROOT, "by_figure"' in source
    assert "legacy_cohort_roots" not in source
    assert "rebuild_cohort_view" not in source
    assert "file.symlink" not in source


def test_output_artifact_names_drop_redundant_parent_tokens():
    source = PIPELINE_R.read_text()
    assert "artifact_name_for_stem <- function" in source
    assert 'figure3 = "^figure3_?"' in source
    assert "artifact_name_for_stem(plot_stem, group)" in source
    assert "remove_legacy_artifacts" not in source


def test_removed_figure_families_stay_removed():
    source = PIPELINE_R.read_text()
    for retired in ("plot_group_gam_panel", "patient_bin_trajectory", "coverage_strata",
                    "plot_adt_intent_supplement", "resolve_dynamic_deephit_metrics"):
        assert retired not in source


def test_pdf_is_the_only_export_format():
    pipeline = PIPELINE_R.read_text()
    script = FIGURES_SCRIPT.read_text()
    assert "ggsave(temporary, plot = plot" in pipeline
    assert "device = figure_pdf_device" in pipeline
    assert "agg_png" not in pipeline
    assert "figure_svg_device" not in pipeline
    assert "kept existing" not in pipeline
    assert 'RENDER_DPI <- if (RENDER_PROFILE == "publication") 600 else 200' in script


def test_render_reuses_completed_outputs_with_explicit_overwrite():
    pipeline = PIPELINE_R.read_text()
    script = FIGURES_SCRIPT.read_text()
    setup = pipeline[pipeline.index("COHORT_ARM_DIR <-"):pipeline.index("COHORT_LEAF <-")]
    save = pipeline[pipeline.index("save_fig <- function"):pipeline.index("COHORT_LABEL <-")]
    assert "list.files(" not in setup
    assert "list.dirs(" not in setup
    assert "dir.exists(" not in setup
    assert "figure_file_complete(pdf_out)" in save
    assert "remove_legacy" not in save
    assert "overwrite ||" in save
    assert 'RENDER_OVERWRITE <- env_flag("COMPASS_RENDER_OVERWRITE")' in script
    assert "render_overwrite = RENDER_OVERWRITE" in script
