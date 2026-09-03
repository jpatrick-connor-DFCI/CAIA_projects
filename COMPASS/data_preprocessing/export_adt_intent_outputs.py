"""Build the ADT metastatic-filtering label and write its figures.

This is the whole 08_adt_intent analysis as one callable: classify, cross-check
against stage / coded metastasis, render the figures. The
notebook is a thin wrapper over `run()` so that the same outputs can be
regenerated from the command line without opening Jupyter.

Everything lands under a single directory, by default

    <FIG_ROOT>/ADT_METASTATIC_FILTERING/

with FIG_ROOT matching 05_figures.Rmd's
`/data/gusev/USERS/jpconnor/figures/CAIA/COMPASS`, overridable via the
COMPASS_FIG_ROOT environment variable or --fig-root.

Outputs
-------
PNG only. Nothing is written as CSV -- not the summaries, and not the label
itself. `run()` returns the labelled frame, so a caller that needs the
per-patient label takes it from the return value rather than from disk.

  stage_metburden.png            stage mix, burden distribution, site pattern
  max_stage.png                  max-stage mix, stage IV rate
  max_stage_llm.png              same max-stage panel, grouped by the LLM
                                 primary prostate-subtype label
  km_death.png                   overall survival, all ADT-exposed
  km_platinum.png / km_nepc.png / km_avpc.png
                                 cohort-only endpoints; need `longitudinal`
  lab_trajectories.png           testosterone and PSA around ADT start

Because the summary tables are gone, the numbers that qualify a figure travel
on the figure: each KM panel is annotated with per-class event counts, median
follow-up and log-rank p-values. Lab coverage, which is the diagnostic for a
silently-empty trajectory panel, goes to the run log.

A figure whose inputs are absent is skipped, and the run reports which ones
were skipped and why.
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
)
from COMPASS.data_preprocessing.adt_intent_trajectories import (  # noqa: E402
    CASTRATE_NG_DL,
    CASTRATE_STRICT_NG_DL,
    INTENT_COL,
    INTENT_COLORS,
    KM_ENDPOINTS,
    PSA_LAB_NAME,
    PSA_LOG_FLOOR,
    TESTOSTERONE_LAB_NAME,
    build_death_km_input,
    build_km_input,
    build_lab_trajectory,
    km_series_by_intent,
    load_longitudinal,
    logrank_by_intent,
    plot_lab_trajectory,
    plot_max_stage_panel,
    plot_stage_metburden_panel,
    summarize_km,
    summarize_trajectory_coverage,
)
from survival_common.plotting import overlay_km  # noqa: E402
from COMPASS.data_preprocessing.validate_adt_intent import (  # noqa: E402
    compute_first_metastasis_icd_date,
    compute_met_burden_at_adt,
    load_stage_max_around_adt,
    load_stage_nearest_adt,
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
DEFAULT_LLM_LABELS_PATH = Path(
    os.environ.get(
        "LLM_NEPC_CLASSIFIER_LABELS_PATH",
        "/data/gusev/USERS/jpconnor/data/LLM_annotations/LLM_NEPC_labels/"
        "LLM_NEPC_classifier_labels.tsv",
    )
)
LLM_LABEL_COL = "LLM_PRIMARY_LABEL"
LLM_LABEL_ORDER = ("conventional", "biomarker", "avpc", "nepc")

FIGURE_DPI = 200


def load_llm_primary_labels(path: str | Path) -> pl.DataFrame:
    """Load and normalize the classifier's mutually exclusive primary label.

    This mirrors the normalization used by the R figure pipeline: a raw
    ``biomarker`` label is retained only when the reported biomarker text names
    BRCA1/2, PTEN, TP53, or RB1; otherwise NEPC and then AVPC take precedence,
    with conventional as the final fallback.
    """
    path = Path(path)
    labels = pl.read_csv(path, separator="\t", infer_schema_length=0)
    required = {ID_COL, "primary_label", "has_nepc", "has_avpc"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    biomarker_candidates = (
        "biomarker_genes", "reported_biomarkers", "biomarkers_reported",
        "reported_biomarker", "biomarkers", "biomarker",
    )
    by_lower = {col.lower(): col for col in labels.columns}
    biomarker_col = next(
        (by_lower[name] for name in biomarker_candidates if name in by_lower), None
    )
    raw = pl.col("primary_label").cast(pl.Utf8).str.strip_chars().str.to_lowercase()
    labels = labels.with_columns(
        pl.col(ID_COL).cast(pl.Float64, strict=False).cast(pl.Int64, strict=False),
        pl.when(raw.is_in(["", "nan", "na", "null", "none"]))
        .then(None).otherwise(raw).alias(LLM_LABEL_COL),
        pl.col("has_nepc").cast(pl.Float64, strict=False).alias("_has_nepc"),
        pl.col("has_avpc").cast(pl.Float64, strict=False).alias("_has_avpc"),
    )

    has_raw_biomarker = labels.filter(pl.col(LLM_LABEL_COL) == "biomarker").height > 0
    if biomarker_col is None and has_raw_biomarker:
        raise ValueError(
            f"{path} contains primary_label='biomarker' but has no reported-biomarker column"
        )
    qualifying = (
        pl.col(biomarker_col)
        .cast(pl.Utf8, strict=False)
        .str.to_uppercase()
        .str.contains(r"(^|[^A-Z0-9])(BRCA1|BRCA2|PTEN|TP53|RB1)([^A-Z0-9]|$)")
        .fill_null(False)
        if biomarker_col is not None else pl.lit(False)
    )
    needs_fallback = (pl.col(LLM_LABEL_COL) == "biomarker") & ~qualifying
    labels = labels.with_columns(
        pl.when(needs_fallback & (pl.col("_has_nepc") == 1))
        .then(pl.lit("nepc"))
        .when(needs_fallback & (pl.col("_has_avpc") == 1))
        .then(pl.lit("avpc"))
        .when(needs_fallback)
        .then(pl.lit("conventional"))
        .otherwise(pl.col(LLM_LABEL_COL))
        .alias(LLM_LABEL_COL)
    ).filter(
        pl.col(ID_COL).is_not_null() & pl.col(LLM_LABEL_COL).is_in(LLM_LABEL_ORDER)
    )

    conflicts = (
        labels.group_by(ID_COL)
        .agg(pl.col(LLM_LABEL_COL).n_unique().alias("n_labels"))
        .filter(pl.col("n_labels") > 1)
    )
    if conflicts.height:
        raise ValueError(f"{path} has conflicting primary labels for {conflicts.height} MRNs")
    return labels.select(ID_COL, LLM_LABEL_COL).unique(subset=[ID_COL])


def resolve_out_dir(fig_root: str | Path | None = None) -> Path:
    """<fig_root>/ADT_METASTATIC_FILTERING, created if absent."""
    root = Path(fig_root) if fig_root is not None else Path(DEFAULT_FIG_ROOT)
    out = root / OUTPUT_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_labels(
    meds: pl.DataFrame,
    follow_up: pl.DataFrame | None = None,
    icds: pl.DataFrame | None = None,
    stage_note_level_path: str | Path | None = None,
    llm_labels_path: str | Path | None = None,
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

    if llm_labels_path is not None and Path(llm_labels_path).exists():
        labelled = labelled.join(
            load_llm_primary_labels(llm_labels_path), on=ID_COL, how="left"
        )

    return labelled


def write_time_to_event(
    labelled: pl.DataFrame,
    out_dir: Path,
    follow_up: pl.DataFrame | None = None,
    longitudinal: pl.DataFrame | None = None,
) -> list:
    """KM figures and their summary tables, one file per endpoint.

    Death is computable for every ADT-exposed patient from patient status
    alone. The other three endpoints (platinum, NEPC, AVPC) are defined only
    in the eligible survival cohort, so they need the Stage 2 longitudinal
    frame and are skipped without it.

    Event counts are annotated onto each panel rather than written alongside
    it: a curve built on a handful of events looks just as confident as one
    built on hundreds, and with no summary table the figure has to carry that
    information itself.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log: list = []

    def _km_figure(km_input, endpoint, title, ylabel):
        """One KM panel. Returns False if there was nothing to draw."""
        if km_input.height == 0:
            log.append(f"  [skip] km_{endpoint}.png -- no {endpoint} data")
            return False

        counts = summarize_km(km_input)

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        try:
            overlay_km(
                ax, km_series_by_intent(km_input), colors=INTENT_COLORS,
                title=title, xlabel="Days from first ADT", ylabel=ylabel,
            )
        except ModuleNotFoundError:
            plt.close(fig)
            log.append(f"  [skip] km_{endpoint}.png -- lifelines not installed")
            return False
        ax.grid(alpha=0.2)

        # Event counts, and the log-rank p when lifelines is available. With
        # no summary CSV these are the only record of how much data each curve
        # rests on, so they go onto the panel itself.
        lines = [
            f"{r[INTENT_COL]}: {r['n_events']}/{r['n_patients']} events "
            f"({r['pct_event']}%), median {r['median_duration_days']:.0f}d"
            for r in counts.iter_rows(named=True)
        ]
        lr = logrank_by_intent(km_input)
        lines += [
            f"log-rank {r['group_a']} vs {r['group_b']}: p={r['p_value']:.3g}"
            for r in lr.iter_rows(named=True)
        ]

        # The legend and this box both want a free corner. matplotlib's "best"
        # placement puts the legend bottom-left whenever the curves stay high
        # -- which is exactly where the annotation sits, and for time-to-
        # platinum and time-to-NEPC the curves nearly always stay high. Pin the
        # legend top-right and the annotation bottom-left so they can never
        # contend, and add headroom above 1.0 so a flat curve does not run
        # under the legend either.
        ax.set_ylim(0, 1.28)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.text(
            0.02, 0.02, "\n".join(lines), transform=ax.transAxes,
            fontsize=7.5, va="bottom", ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85),
        )

        fig.savefig(out_dir / f"km_{endpoint}.png", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        log.append(f"  wrote km_{endpoint}.png")
        return True

    # --- death: the full ADT-exposed population ---
    if follow_up is not None:
        death_km = build_death_km_input(labelled, follow_up)
        _km_figure(death_km, "death",
                   "Overall survival by ADT intent (all ADT-exposed)",
                   "Survival probability")
    else:
        log.append("  [skip] km_death.png -- no follow-up frame")

    # --- cohort-only endpoints ---
    if longitudinal is None:
        log.append("  [skip] km_platinum/nepc/avpc -- no longitudinal frame")
        return log

    for endpoint in [e for e in KM_ENDPOINTS if e != "death"]:
        km_input = build_km_input(longitudinal, labelled, endpoint)
        _km_figure(km_input, endpoint, KM_ENDPOINTS[endpoint][2],
                   "Event-free probability")
    return log


def write_lab_trajectories(
    labelled: pl.DataFrame,
    out_dir: Path,
    longitudinal: pl.DataFrame | None = None,
) -> list:
    """Testosterone and PSA trajectories around ADT start, by intent class."""
    if longitudinal is None:
        return ["  [skip] lab_trajectories.png -- no longitudinal frame"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log: list = []
    specs = [
        (TESTOSTERONE_LAB_NAME, "Testosterone (ng/dL)", False,
         (CASTRATE_NG_DL, CASTRATE_STRICT_NG_DL)),
        (PSA_LAB_NAME, "PSA (ng/mL, log scale)", True, ()),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, (lab, ylabel, log_scale, hlines) in zip(axes, specs):
        # Coverage is the diagnostic for a silently-empty panel (a lab-name
        # mismatch draws nothing and raises nothing), so with the CSV gone it
        # is logged instead.
        cov = summarize_trajectory_coverage(longitudinal, labelled, lab)
        if cov.height == 0:
            log.append(f"  [warn] {lab}: no coverage -- check the lab name")
        else:
            for r in cov.iter_rows(named=True):
                log.append(f"  {lab} coverage: " + ", ".join(
                    f"{k}={v}" for k, v in r.items()
                ))

        traj = build_lab_trajectory(longitudinal, labelled, lab)
        if log_scale and traj.height:
            # Zeros are real below-detection values, but log10(0) is not
            # plottable -- floor for the axis only, after the median.
            traj = traj.with_columns(
                *[pl.col(c).clip(lower_bound=PSA_LOG_FLOOR) for c in ("median", "q1", "q3")]
            )
        plot_lab_trajectory(ax, traj, title=f"{lab} around ADT start",
                            ylabel=ylabel, log_scale=log_scale, hlines=hlines)
    fig.tight_layout()
    fig.savefig(out_dir / "lab_trajectories.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.append("  wrote lab_trajectories.png")
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
    if LLM_LABEL_COL in labelled.columns:
        llm_labelled = labelled.filter(pl.col(LLM_LABEL_COL).is_not_null())
        if llm_labelled.height:
            fig, _ = plot_max_stage_panel(
                llm_labelled,
                figsize=(12, 4.6),
                class_col=LLM_LABEL_COL,
                class_order=LLM_LABEL_ORDER,
            )
            fig.suptitle("Max stage around ADT start, by LLM primary label", y=1.03)
            path = out_dir / "max_stage_llm.png"
            fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
            plt.close(fig)
            log.append("  wrote max_stage_llm.png")
    return log


def run(
    meds: pl.DataFrame,
    follow_up: pl.DataFrame | None = None,
    icds: pl.DataFrame | None = None,
    stage_note_level_path: str | Path | None = DEFAULT_STAGE_NOTE_LEVEL_PATH,
    llm_labels_path: str | Path | None = DEFAULT_LLM_LABELS_PATH,
    longitudinal: pl.DataFrame | None = None,
    fig_root: str | Path | None = None,
    gap_threshold_days: int = GAP_THRESHOLD_DAYS,
    verbose: bool = True,
) -> tuple:
    """Classify, cross-reference, write every figure.

    Returns (labelled, out_dir). The labelled frame is returned rather than
    written: nothing is persisted but PNGs, so a caller that needs the label
    itself takes it from the return value.
    """
    out_dir = resolve_out_dir(fig_root)
    labelled = build_labels(
        meds,
        follow_up=follow_up,
        icds=icds,
        stage_note_level_path=stage_note_level_path,
        llm_labels_path=llm_labels_path,
        gap_threshold_days=gap_threshold_days,
    )
    log = write_figures(labelled, out_dir)
    log += write_time_to_event(
        labelled, out_dir, follow_up=follow_up, longitudinal=longitudinal
    )
    log += write_lab_trajectories(labelled, out_dir, longitudinal=longitudinal)

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
        "--llm-labels-path", default=str(DEFAULT_LLM_LABELS_PATH),
        help=("LLM_NEPC_classifier_labels.tsv; when present, writes the "
              "alternate max_stage_llm.png panel"),
    )
    parser.add_argument(
        "--longitudinal-path", default=None,
        help=("Stage 2 longitudinal_prediction_data_adt.csv. Enables the "
              "platinum/NEPC/AVPC KMs and the lab trajectories; death KM does "
              "not need it."),
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

    longitudinal = None
    if args.longitudinal_path:
        # load_longitudinal cross-checks TREATMENT_ANCHOR_DATE against each
        # patient's ADT_FIRST_DATE and refuses the arpi-arm file, whose anchor
        # is first ARPI/taxane exposure -- same columns, wrong origin.
        labels_for_check = build_labels(
            meds, follow_up=follow_up, gap_threshold_days=args.gap_threshold_days
        )
        longitudinal = load_longitudinal(args.longitudinal_path, labels=labels_for_check)

    run(
        meds,
        follow_up=follow_up,
        icds=icds,
        stage_note_level_path=args.stage_note_level_path,
        llm_labels_path=args.llm_labels_path,
        longitudinal=longitudinal,
        fig_root=args.fig_root,
        gap_threshold_days=args.gap_threshold_days,
    )


if __name__ == "__main__":
    main()
