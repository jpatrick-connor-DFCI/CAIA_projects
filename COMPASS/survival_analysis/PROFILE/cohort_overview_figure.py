"""
Figure 1 -- PROFILE cohort overview.

Builds the manuscript's Figure 1 (cohort overview) from pipeline outputs:
  Panel A -- CONSORT attrition diagram, from the structured attrition counts
             `longitudinal_data_processing.py` and `build_prediction_inputs.py`
             now write alongside their other outputs (cohort_attrition.json,
             landmark_attrition.json) -- no log-scraping.
  Panel B -- Kaplan-Meier curves: platinum-free survival and overall survival
             (the latter is really "time to last contact"; see caveat below).
  Panel C -- Timing histograms: record span, diagnosis -> first treatment,
             time to platinum (event patients only).

Each panel is saved standalone (figure1a/b/c_*) AND assembled into one
compiled figure1_cohort_overview.{png,pdf}, per the manuscript's figure
convention. A companion Table 1 (baseline characteristics) is written as CSV
+ markdown -- a separate artifact, not a plotted panel.

Caveat carried through to the OS panel and Table 1: the source data has no
true date of death, so `t_death` == `t_last_contact` for every patient
(see longitudinal_data_processing.py). The DEATH *indicator* is real; the
*duration* is really "time to last contact" for dead and censored patients
alike.

Run e.g.:
    python cohort_overview_figure.py \
      --longitudinal-csv /data/.../longitudinal_prediction_data.csv \
      --inputs-dir /data/.../prediction_inputs \
      --output-dir /data/.../figures/CAIA/COMPASS/PROFILE
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

SCRIPT_DIR = Path(__file__).resolve().parent           # .../survival_analysis/PROFILE
SURVIVAL_PARENT = SCRIPT_DIR.parent                    # .../survival_analysis
for _p in (str(SURVIVAL_PARENT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from helpers.loaders import load_profile_longitudinal, record_span_days  # noqa: E402
from helpers.plotting import RCPARAMS, COHORT_COLORS, overlay_hist, overlay_km  # noqa: E402

DATA_ROOT = Path("/data/gusev/USERS/jpconnor/data")
NEPC_PROJ_PATH = DATA_ROOT / "CAIA" / "COMPASS"
RESULTS = NEPC_PROJ_PATH / "survival_analysis" / "PROFILE"
DEFAULT_LONGITUDINAL_CSV = NEPC_PROJ_PATH / "longitudinal_prediction_data.csv"
DEFAULT_INPUTS_DIR = RESULTS / "prediction_inputs"
DEFAULT_OUTPUT_DIR = Path("/data/gusev/USERS/jpconnor/figures/CAIA/COMPASS/PROFILE")

COHORT_LABEL = "PROFILE"
DEFAULT_FORMATS = ("png", "pdf")

# Mirrors the fixed dummy schema written by build_prediction_inputs.py's
# load_stage_dummies (Stage I is the all-zero / all-NaN-if-unknown reference).
STAGE_COLUMNS = ["CANCER_STAGE_II", "CANCER_STAGE_III", "CANCER_STAGE_IV"]
GLEASON_COL = "GLEASON_GROUP"


def _save(fig: plt.Figure, out_base: Path, formats: tuple[str, ...] = DEFAULT_FORMATS) -> list[Path]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        p = out_base.with_suffix(f".{ext}")
        fig.savefig(p)
        paths.append(p)
    return paths


def load_attrition(longitudinal_csv: Path, inputs_dir: Path) -> dict:
    cohort_path = longitudinal_csv.parent / "cohort_attrition.json"
    landmark_path = inputs_dir / "landmark_attrition.json"
    if not cohort_path.exists():
        raise FileNotFoundError(
            f"{cohort_path} not found -- re-run longitudinal_data_processing.py "
            "(it now writes cohort_attrition.json alongside its CSV outputs)."
        )
    if not landmark_path.exists():
        raise FileNotFoundError(
            f"{landmark_path} not found -- re-run build_prediction_inputs.py "
            "(it now writes landmark_attrition.json alongside its CSV outputs)."
        )
    return {**json.loads(cohort_path.read_text()), **json.loads(landmark_path.read_text())}


# --------------------------- Panel A: CONSORT ----------------------------

def render_consort_panel(ax, attrition: dict) -> None:
    steps = [
        ("Patients with usable labs/vitals", attrition["n_with_labs"]),
        ("+ first prostate (C61) diagnosis", attrition["n_after_diagnosis_join"]),
        ("+ in death/last-contact table", attrition["n_after_death_table_join"]),
        ("+ recorded first treatment", attrition["n_after_first_treatment_filter"]),
        (f"+ ≥{attrition['min_psa_count']} PSA records", attrition["n_after_psa_count_filter"]),
        ("− PARPi-exposed excluded", attrition["n_after_parpi_exclusion"]),
    ]
    for landmark_str, n in sorted(attrition["eligible_by_landmark"].items(), key=lambda kv: int(kv[0])):
        sign = "+" if int(landmark_str) > 0 else ""
        steps.append((f"Eligible at landmark {sign}{landmark_str}d", n))
    steps.append(("Common cohort (all landmarks)", attrition["n_common_across_landmarks"]))

    n_steps = len(steps)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_steps + 0.4)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title("Cohort attrition (CONSORT)", fontsize=12.5, weight="bold")

    box_w, box_h = 0.86, 0.62
    for i, (label, n) in enumerate(steps):
        y = i + (1 - box_h) / 2
        ax.add_patch(FancyBboxPatch(
            ((1 - box_w) / 2, y), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor="#eef1f5", edgecolor="#5d6d7e", linewidth=1.0,
        ))
        ax.text(0.5, y + box_h / 2, f"{label}\nn = {n:,}",
                ha="center", va="center", fontsize=9)
        if i < n_steps - 1:
            ax.annotate(
                "", xy=(0.5, i + 1 + (1 - box_h) / 2), xytext=(0.5, y + box_h),
                arrowprops=dict(arrowstyle="-|>", color="#5d6d7e", lw=1.2),
            )

    split_sizes = attrition["split_sizes"]
    footer = "  ·  ".join(f"{k}: n={v:,}" for k, v in split_sizes.items())
    ax.text(0.5, n_steps + 0.15, f"Train/valid/test split — {footer}",
            ha="center", va="top", fontsize=8.5, color="#5d6d7e", style="italic")


# ------------------------------ Panel B: KM ------------------------------

def render_km_panel(axes, patient_df: pd.DataFrame) -> None:
    ax_plat, ax_os = axes
    overlay_km(
        ax_plat, {COHORT_LABEL: (patient_df["t_platinum"], patient_df["PLATINUM"])},
        title="Platinum-free survival", xlabel="Days from first record",
        ylabel="Platinum-free probability",
    )
    overlay_km(
        ax_os, {COHORT_LABEL: (patient_df["t_death"], patient_df["DEATH"])},
        title="Overall survival", xlabel="Days from first record",
        ylabel="Survival probability",
    )
    ax_os.text(
        0.02, 0.03,
        "Caveat: no true death date in source data -- duration is\n"
        "time to last contact for every patient, not a true death time.",
        transform=ax_os.transAxes, fontsize=7.5, color="#c0392b", va="bottom",
    )


# --------------------------- Panel C: timing -----------------------------

def render_timing_panel(axes, patient_df: pd.DataFrame, labs_df: pd.DataFrame, id_col: str) -> None:
    ax_span, ax_dx2tx, ax_tplat = axes

    span = record_span_days(labs_df, id_col=id_col)
    overlay_hist(ax_span, {COHORT_LABEL: span}, bins=50)
    ax_span.set_xlabel("Record span (days)")
    ax_span.set_ylabel("Density")
    ax_span.set_title("Per-patient lab record span")

    dx2tx = pd.to_numeric(patient_df.get("t_dx_to_tx", pd.Series(dtype=float)), errors="coerce")
    overlay_hist(ax_dx2tx, {COHORT_LABEL: dx2tx}, bins=50)
    ax_dx2tx.set_xlabel("Days: diagnosis → first treatment")
    ax_dx2tx.set_ylabel("Density")
    ax_dx2tx.set_title("Diagnosis → first treatment")

    is_event = pd.to_numeric(patient_df["PLATINUM"], errors="coerce").fillna(0).astype(int).astype(bool)
    t_platinum_events = pd.to_numeric(patient_df.loc[is_event, "t_platinum"], errors="coerce")
    overlay_hist(ax_tplat, {COHORT_LABEL: t_platinum_events}, bins=40)
    ax_tplat.set_xlabel("Days to platinum, from first record (events only)")
    ax_tplat.set_ylabel("Density")
    ax_tplat.set_title("Time to platinum (event patients only)")


# ----------------------------- Table 1 -----------------------------------

def _mean_sd(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "n/a"
    return f"{s.mean():.1f} ± {s.std():.1f}"


def _median_iqr(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "n/a"
    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
    return f"{med:.1f} ({q1:.1f}–{q3:.1f})"


def _count_pct(mask: pd.Series, total: int) -> str:
    n = int(mask.sum())
    return f"{n:,} ({100 * n / total:.1f}%)" if total else "n/a"


def _reconstruct_stage(aggregated_df: pd.DataFrame) -> pd.Series | None:
    if not set(STAGE_COLUMNS).issubset(aggregated_df.columns):
        return None
    onehot = aggregated_df[STAGE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    unknown = onehot.isna().all(axis=1)
    any_hit = (onehot.fillna(0) > 0).any(axis=1)
    picked = onehot.fillna(0).idxmax(axis=1).str[len("CANCER_STAGE_"):]
    stage = picked.where(any_hit, "I")
    stage = stage.mask(unknown, "Unknown")
    return stage


def build_table1(patient_df: pd.DataFrame, aggregated_df: pd.DataFrame | None) -> pd.DataFrame:
    n = len(patient_df)
    rows = [("N", f"{n:,}")]

    age = patient_df.get("AGE_AT_TREATMENTSTART", pd.Series(dtype=float))
    rows.append(("Age at first treatment, mean ± SD", _mean_sd(age)))
    rows.append(("Age at first treatment, median (IQR)", _median_iqr(age)))

    rows.append(("Platinum exposure, n (%)",
                 _count_pct(pd.to_numeric(patient_df["PLATINUM"], errors="coerce").fillna(0).astype(bool), n)))
    rows.append(("Deaths (proxy: time to last contact), n (%)",
                 _count_pct(pd.to_numeric(patient_df["DEATH"], errors="coerce").fillna(0).astype(bool), n)))
    rows.append(("Median follow-up from first record, days (IQR)",
                 _median_iqr(patient_df.get("t_last_contact", pd.Series(dtype=float)))))

    if aggregated_df is None:
        rows.append(("Gleason grade group / cancer stage",
                     "not available (no --aggregated-csv with GLEASON_GROUP / CANCER_STAGE_* columns)"))
    else:
        n_agg = len(aggregated_df)
        if GLEASON_COL in aggregated_df.columns:
            gleason = pd.to_numeric(aggregated_df[GLEASON_COL], errors="coerce")
            n_known = int(gleason.notna().sum())
            rows.append(("Gleason grade group known, n (%)", _count_pct(gleason.notna(), n_agg)))
            for grp in range(1, 6):
                rows.append((f"  ISUP grade group {grp}, n (% of known)",
                             _count_pct(gleason.eq(grp), max(n_known, 1))))
        else:
            rows.append(("Gleason grade group", "not available in --aggregated-csv"))

        stage = _reconstruct_stage(aggregated_df)
        if stage is not None:
            for cat in ["I", "II", "III", "IV", "Unknown"]:
                rows.append((f"  Cancer stage {cat}, n (%)", _count_pct(stage.eq(cat), n_agg)))
        else:
            rows.append(("Cancer stage", "not available in --aggregated-csv"))

    return pd.DataFrame(rows, columns=["Characteristic", "Value"])


def _to_markdown_table(df: pd.DataFrame) -> str:
    """Minimal GFM table renderer (no `tabulate` dependency, which isn't in
    this repo's pinned dependency list -- see README)."""
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    )
    return "\n".join([header, sep, body])


def write_table1(table1: pd.DataFrame, out_base: Path) -> list[Path]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_base.with_suffix(".csv")
    md_path = out_base.with_suffix(".md")
    table1.to_csv(csv_path, index=False)
    md_path.write_text(_to_markdown_table(table1))
    return [csv_path, md_path]


# -------------------------------- main ------------------------------------

def run(
    *,
    longitudinal_csv: Path,
    inputs_dir: Path,
    aggregated_csv: Path | None,
    output_dir: Path,
    id_col: str,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
) -> None:
    plt.rcParams.update(RCPARAMS)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading attrition counts from {longitudinal_csv.parent} and {inputs_dir} ...")
    attrition = load_attrition(longitudinal_csv, inputs_dir)

    print(f"Loading {longitudinal_csv} ...")
    patient_df, labs_df = load_profile_longitudinal(longitudinal_csv, id_col=id_col)
    print(f"  patients={len(patient_df):,}  labs={len(labs_df):,}")

    aggregated_df = None
    if aggregated_csv is not None:
        if aggregated_csv.exists():
            aggregated_df = pd.read_csv(aggregated_csv, low_memory=False)
            print(f"Loaded {aggregated_csv} for Table 1 enrichment: {len(aggregated_df):,} rows")
        else:
            print(f"[warn] --aggregated-csv {aggregated_csv} not found; Table 1 will omit Gleason/stage")

    print("\nPanel A: CONSORT attrition ...")
    fig, ax = plt.subplots(figsize=(7.5, 0.62 * (len(attrition["eligible_by_landmark"]) + 8) + 1))
    render_consort_panel(ax, attrition)
    fig.tight_layout()
    for p in _save(fig, output_dir / "figure1a_consort", formats):
        print(f"  {p}")
    fig_consort = fig

    print("Panel B: KM curves ...")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    render_km_panel(axes, patient_df)
    fig.tight_layout()
    for p in _save(fig, output_dir / "figure1b_km", formats):
        print(f"  {p}")
    fig_km = fig

    print("Panel C: timing histograms ...")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    render_timing_panel(axes, patient_df, labs_df, id_col)
    fig.tight_layout()
    for p in _save(fig, output_dir / "figure1c_timing", formats):
        print(f"  {p}")
    fig_timing = fig

    print("Compiled Figure 1 ...")
    n_landmarks = len(attrition["eligible_by_landmark"])
    fig = plt.figure(figsize=(14, 6 + 0.62 * (n_landmarks + 8)))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.62 * (n_landmarks + 8), 1.0, 1.0])
    ax_consort = fig.add_subplot(gs[0, :])
    render_consort_panel(ax_consort, attrition)
    ax_plat = fig.add_subplot(gs[1, 0])
    ax_os = fig.add_subplot(gs[1, 1])
    render_km_panel((ax_plat, ax_os), patient_df)
    fig.add_subplot(gs[1, 2]).set_axis_off()
    ax_span = fig.add_subplot(gs[2, 0])
    ax_dx2tx = fig.add_subplot(gs[2, 1])
    ax_tplat = fig.add_subplot(gs[2, 2])
    render_timing_panel((ax_span, ax_dx2tx, ax_tplat), patient_df, labs_df, id_col)
    fig.suptitle("Figure 1 — PROFILE cohort overview", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    for p in _save(fig, output_dir / "figure1_cohort_overview", formats):
        print(f"  {p}")

    plt.close(fig_consort)
    plt.close(fig_km)
    plt.close(fig_timing)
    plt.close(fig)

    print("\nTable 1: baseline characteristics ...")
    table1 = build_table1(patient_df, aggregated_df)
    for p in write_table1(table1, output_dir / "table1_baseline_characteristics"):
        print(f"  {p}")

    print("\nDone.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--longitudinal-csv", type=Path, default=DEFAULT_LONGITUDINAL_CSV,
                   help="Path to longitudinal_prediction_data.csv (default: %(default)s).")
    p.add_argument("--inputs-dir", type=Path, default=DEFAULT_INPUTS_DIR,
                   help="prediction_inputs/ directory from build_prediction_inputs.py "
                        "(default: %(default)s).")
    p.add_argument("--aggregated-csv", type=Path, default=None,
                   help="Optional aggregated_landmark{D}.csv to enrich Table 1 with "
                        "GLEASON_GROUP / CANCER_STAGE_* (only present if the pipeline "
                        "run passed --gleason-file / --stage-file).")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Directory to write figures + Table 1 into (default: %(default)s).")
    p.add_argument("--id-col", default="DFCI_MRN")
    p.add_argument("--formats", nargs="+", default=list(DEFAULT_FORMATS),
                   choices=["png", "pdf", "svg"])
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(
        longitudinal_csv=args.longitudinal_csv,
        inputs_dir=args.inputs_dir,
        aggregated_csv=args.aggregated_csv,
        output_dir=args.output_dir,
        id_col=args.id_col,
        formats=tuple(args.formats),
    )
