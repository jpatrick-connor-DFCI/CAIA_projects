"""Build the ADT metastatic-filtering label and write its CSVs and figures.

This is the whole 08_adt_intent analysis as one callable: classify, cross-check
against stage / coded metastasis, render the figures, write the tables. The
notebook is a thin wrapper over `run()` so that the same outputs can be
regenerated from the command line without opening Jupyter.

Everything lands under a single directory, by default

    <FIG_ROOT>/ADT_METASTATIC_FILTERING/

with FIG_ROOT matching 05_figures.Rmd's
`/data/gusev/USERS/jpconnor/figures/CAIA/COMPASS`, overridable via the
COMPASS_FIG_ROOT environment variable or --fig-root.

Outputs
-------
CSV
  adt_intent_labels.csv          one row per patient: the label itself, plus
                                 every cross-reference column joined on
  summary_class_counts.csv       class sizes and eligible counts
  summary_survival.csv           survival by class -- the primary go/no-go
  summary_stage_nearest.csv      label vs. stage nearest ADT start
  summary_stage_max.csv          max stage before / after ADT start
  summary_met_burden.csv         label vs. metastatic organ-group burden
  summary_met_site_pattern.csv   per-site involvement by class
  summary_stage_contradictions.csv  excluded patients carrying stage IV
  summary_by_adt_start_year.csv  ARPI-era drift
  summary_gap_sensitivity.csv    gap-threshold sensitivity

PNG
  stage_metburden.png            stage mix, burden distribution, site pattern
  max_stage.png                  max-stage mix, stage IV rate, upstaging

A table whose inputs are absent is skipped, not written empty, and the run
reports which ones were skipped and why.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import polars as pl

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from COMPASS.data_preprocessing.classify_adt_intent import (  # noqa: E402
    GAP_THRESHOLD_DAYS,
    ID_COL,
    classify_adt_intent,
    summarize_intent,
)
from COMPASS.data_preprocessing.adt_intent_trajectories import (  # noqa: E402
    plot_max_stage_panel,
    plot_stage_metburden_panel,
)
from COMPASS.data_preprocessing.validate_adt_intent import (  # noqa: E402
    compute_first_metastasis_icd_date,
    compute_met_burden_at_adt,
    load_stage_max_around_adt,
    load_stage_nearest_adt,
    report_against_met_burden,
    report_against_metastasis_icd,
    report_against_stage,
    report_by_adt_start_year,
    report_gap_sensitivity,
    report_met_site_pattern,
    report_stage_contradictions,
    report_stage_max,
    report_survival,
)

# Matches FIG_ROOT in COMPASS/survival_analysis/05_figures.Rmd.
DEFAULT_FIG_ROOT = os.environ.get(
    "COMPASS_FIG_ROOT", "/data/gusev/USERS/jpconnor/figures/CAIA/COMPASS"
)
OUTPUT_SUBDIR = "ADT_METASTATIC_FILTERING"

DEFAULT_PROFILE_DATA_PATH = os.environ.get(
    "PROFILE_DATA_PATH", "/data/gusev/USERS/jpconnor/data/PROFILE_DATA/"
)
DEFAULT_STAGE_NOTE_LEVEL_PATH = (
    Path(DEFAULT_PROFILE_DATA_PATH) / "CANCER_ANNOTATIONS" / "CANCER_STAGE_NOTE_LEVEL.parquet"
)

FIGURE_DPI = 200


def resolve_out_dir(fig_root: str | Path | None = None) -> Path:
    """<fig_root>/ADT_METASTATIC_FILTERING, created if absent."""
    root = Path(fig_root) if fig_root is not None else Path(DEFAULT_FIG_ROOT)
    out = root / OUTPUT_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write(frame: pl.DataFrame | None, out_dir: Path, name: str, log: list) -> None:
    """Write one table, or record why it was skipped.

    An empty frame means the inputs for that table weren't available. Writing
    a header-only CSV would be indistinguishable from "measured, found
    nothing", so skip and say so instead.
    """
    if frame is None or frame.height == 0:
        log.append(f"  [skip] {name}.csv -- no data")
        return
    path = out_dir / f"{name}.csv"
    frame.write_csv(path)
    log.append(f"  wrote {name}.csv ({frame.height:,} rows)")


def build_labels(
    meds: pl.DataFrame,
    follow_up: pl.DataFrame | None = None,
    icds: pl.DataFrame | None = None,
    stage_note_level_path: str | Path | None = None,
    gap_threshold_days: int = GAP_THRESHOLD_DAYS,
) -> pl.DataFrame:
    """Classify, then join on every cross-reference whose input is present.

    Each join is independently optional: a missing ICD file costs the burden
    columns and nothing else, so a partial run still produces a usable label.
    """
    labelled = classify_adt_intent(
        meds, follow_up=follow_up, gap_threshold_days=gap_threshold_days
    )

    # classify_adt_intent consumes follow_up for FOLLOW_UP_END_DATE only and
    # does not carry DEATH through, so join it on explicitly. Without this the
    # survival report -- the primary go/no-go -- silently comes back empty.
    if follow_up is not None and "DEATH" in follow_up.columns:
        labelled = labelled.join(
            follow_up.select(
                pl.col(ID_COL)
                .cast(pl.Float64, strict=False)
                .cast(pl.Int64, strict=False)
                .alias(ID_COL),
                pl.col("DEATH").cast(pl.Int64, strict=False),
            ),
            on=ID_COL,
            how="left",
        )

    if icds is not None and icds.height:
        labelled = labelled.join(
            compute_first_metastasis_icd_date(icds), on=ID_COL, how="left"
        )
        labelled = labelled.join(
            compute_met_burden_at_adt(icds, labelled), on=ID_COL, how="left"
        )

    if stage_note_level_path is not None and Path(stage_note_level_path).exists():
        path = str(stage_note_level_path)
        labelled = labelled.join(
            load_stage_nearest_adt(path, labelled), on=ID_COL, how="left"
        )
        labelled = labelled.join(
            load_stage_max_around_adt(path, labelled), on=ID_COL, how="left"
        )

    return labelled


def write_tables(labelled: pl.DataFrame, out_dir: Path, meds=None, follow_up=None) -> list:
    """Write the label plus every summary table its columns support."""
    log: list = []

    path = out_dir / "adt_intent_labels.csv"
    labelled.write_csv(path)
    log.append(f"  wrote adt_intent_labels.csv ({labelled.height:,} rows)")

    counts = summarize_intent(labelled)
    _write(counts, out_dir, "summary_class_counts", log)
    _write(report_survival(labelled), out_dir, "summary_survival", log)

    if "FIRST_METASTASIS_ICD_DATE" in labelled.columns:
        _write(
            report_against_metastasis_icd(labelled),
            out_dir, "summary_metastasis_icd", log,
        )
    if "CANCER_STAGE" in labelled.columns:
        _write(report_against_stage(labelled), out_dir, "summary_stage_nearest", log)
        _write(
            report_stage_contradictions(labelled),
            out_dir, "summary_stage_contradictions", log,
        )
    if "MAX_STAGE_BEFORE_INT" in labelled.columns:
        _write(report_stage_max(labelled), out_dir, "summary_stage_max", log)
    if "N_MET_SITES" in labelled.columns:
        _write(report_against_met_burden(labelled), out_dir, "summary_met_burden", log)
        _write(
            report_met_site_pattern(labelled),
            out_dir, "summary_met_site_pattern", log,
        )

    _write(report_by_adt_start_year(labelled), out_dir, "summary_by_adt_start_year", log)
    if meds is not None:
        _write(
            report_gap_sensitivity(meds, follow_up),
            out_dir, "summary_gap_sensitivity", log,
        )
    return log


def write_figures(labelled: pl.DataFrame, out_dir: Path) -> list:
    """Render both panels. Panels with no coverage draw a placeholder."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log: list = []
    for name, fn in (
        ("stage_metburden", plot_stage_metburden_panel),
        ("max_stage", plot_max_stage_panel),
    ):
        fig, _ = fn(labelled)
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        log.append(f"  wrote {name}.png")
    return log


def run(
    meds: pl.DataFrame,
    follow_up: pl.DataFrame | None = None,
    icds: pl.DataFrame | None = None,
    stage_note_level_path: str | Path | None = DEFAULT_STAGE_NOTE_LEVEL_PATH,
    fig_root: str | Path | None = None,
    gap_threshold_days: int = GAP_THRESHOLD_DAYS,
    verbose: bool = True,
) -> tuple:
    """Classify, cross-reference, write every CSV and figure.

    Returns (labelled, out_dir). This is what the notebook calls.
    """
    out_dir = resolve_out_dir(fig_root)
    labelled = build_labels(
        meds,
        follow_up=follow_up,
        icds=icds,
        stage_note_level_path=stage_note_level_path,
        gap_threshold_days=gap_threshold_days,
    )
    log = write_tables(labelled, out_dir, meds=meds, follow_up=follow_up)
    log += write_figures(labelled, out_dir)

    if verbose:
        print(f"output dir: {out_dir}")
        print("\n".join(log))
        n_elig = (
            labelled.filter(pl.col("ELIGIBLE") == 1).height
            if "ELIGIBLE" in labelled.columns else labelled.height
        )
        print(f"\n{labelled.height:,} patients labelled | {n_elig:,} eligible")
    return labelled, out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the ADT metastatic-filtering label, tables and figures."
    )
    parser.add_argument("--medications-path", required=True)
    parser.add_argument("--patient-status-path", default=None)
    parser.add_argument("--icd-path", default=None)
    parser.add_argument(
        "--stage-note-level-path", default=str(DEFAULT_STAGE_NOTE_LEVEL_PATH)
    )
    parser.add_argument(
        "--fig-root", default=None,
        help=f"defaults to $COMPASS_FIG_ROOT or {DEFAULT_FIG_ROOT}",
    )
    parser.add_argument("--gap-threshold-days", type=int, default=GAP_THRESHOLD_DAYS)
    args = parser.parse_args()

    from data_preprocessing_common.oncdrs_sources import scan_source
    from COMPASS.data_preprocessing.compile_COMPASS_cohort_data import (
        parse_mixed_datetime_expr,
    )

    meds = scan_source(args.medications_path).collect()
    follow_up = None
    if args.patient_status_path:
        follow_up = scan_source(args.patient_status_path).collect().with_columns(
            parse_mixed_datetime_expr("FOLLOW_UP_END_DATE").alias("FOLLOW_UP_END_DATE")
        )
    icds = pl.read_csv(args.icd_path, infer_schema_length=0) if args.icd_path else None

    run(
        meds,
        follow_up=follow_up,
        icds=icds,
        stage_note_level_path=args.stage_note_level_path,
        fig_root=args.fig_root,
        gap_threshold_days=args.gap_threshold_days,
    )


if __name__ == "__main__":
    main()
