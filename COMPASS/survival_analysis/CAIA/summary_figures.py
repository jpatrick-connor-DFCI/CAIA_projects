"""Pre-analysis summary-statistics figures for the PROFILE and CAIA cohorts.

Generates four families of figures into `--output-dir`:

  1. `time_distributions.{png,pdf}`     - record span, dx->tx, time-to-platinum
                                          histograms overlaid PROFILE vs CAIA
  2. `platinum_incidence.{png,pdf}`     - crude rate (Wilson CI) + KM curves
  3. `overall_survival_km.{png,pdf}`    - overall-survival KM overlay
  4. `labs_raw/<LAB>.{png,pdf}` and
     `labs_irnt/<LAB>.{png,pdf}`        - per-lab raw + IRNT histograms

Run from the cluster:

    python COMPASS/survival_analysis/CAIA/summary_figures.py \
      --profile-csv /data/.../longitudinal_prediction_data.csv \
      --caia-parquet /data/.../caia_compass_longitudinal.parquet \
      --output-dir /data/.../figures/CAIA/COMPASS/summary_stats
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent           # .../survival_analysis/CAIA
SURVIVAL_PARENT = SCRIPT_DIR.parent                    # .../survival_analysis
for _p in (str(SURVIVAL_PARENT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from helpers.helper import select_canonical_labs  # noqa: E402
from helpers.loaders import (  # noqa: E402
    load_caia_parquet,
    load_profile_longitudinal,
    record_span_days,
)
from helpers.plotting import (  # noqa: E402
    COHORT_COLORS,
    RCPARAMS,
    canonicalize_lab_name,
    irnt,
    overlay_hist,
    overlay_km,
    wilson_ci,
)

DEFAULT_FORMATS = ("png", "pdf")
DEFAULT_MIN_COVERAGE = 0.20


def _save(fig: plt.Figure, out_base: Path, formats: tuple[str, ...]) -> list[Path]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ext in formats:
        p = out_base.with_suffix(f".{ext}")
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def _coerce_days(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _optional_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return df[col] if present, else an empty float Series (so overlay plots
    degrade gracefully instead of raising on a missing optional column)."""
    if col in df.columns:
        return df[col]
    return pd.Series(dtype=float)


def _event_mask(df: pd.DataFrame, col: str = "PLATINUM") -> pd.Series:
    """Boolean event mask from a 0/1 column. Coerces to numeric and treats NaN as
    0 (not True) — `df.get(col, 0).astype(bool)` would turn NaN into True and
    mislabel censored patients as events."""
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).astype(bool)


def build_time_distributions_figure(
    profile_patient: pd.DataFrame,
    caia_patient: pd.DataFrame,
    profile_labs: pd.DataFrame,
    caia_labs: pd.DataFrame,
    *,
    profile_id_col: str = "DFCI_MRN",
    caia_id_col: str = "person_id",
) -> plt.Figure:
    """3-panel figure: record span, dx->tx, time-to-platinum overlays."""
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 9.5))

    p_span = record_span_days(profile_labs, id_col=profile_id_col)
    c_span = record_span_days(caia_labs, id_col=caia_id_col)
    overlay_hist(
        axes[0],
        {"PROFILE": p_span, "CAIA": c_span},
        bins=50,
    )
    axes[0].set_xlabel("Record span (days, max LAB_DATE - min LAB_DATE)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Per-patient lab record span")

    p_dx2tx = _coerce_days(_optional_series(profile_patient, "t_dx_to_tx"))
    c_dx2tx = _coerce_days(_optional_series(caia_patient, "t_dx_to_tx"))
    overlay_hist(
        axes[1],
        {"PROFILE": p_dx2tx, "CAIA": c_dx2tx},
        bins=50,
    )
    axes[1].set_xlabel("Days from diagnosis to first treatment")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Diagnosis -> first treatment")

    p_tplat_evt = _coerce_days(
        profile_patient.loc[_event_mask(profile_patient, "PLATINUM"), "t_platinum"]
    )
    c_tplat_evt = _coerce_days(
        caia_patient.loc[_event_mask(caia_patient, "PLATINUM"), "t_platinum"]
    )
    overlay_hist(
        axes[2],
        {"PROFILE": p_tplat_evt, "CAIA": c_tplat_evt},
        bins=40,
    )
    axes[2].set_xlabel("Time to platinum exposure (days; events only)")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Time to platinum (event patients only)")

    fig.tight_layout()
    return fig


def build_platinum_incidence_figure(
    profile_patient: pd.DataFrame,
    caia_patient: pd.DataFrame,
) -> plt.Figure:
    """2-panel: crude bar with Wilson CI + KM overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    crude = []
    for label, df in (("PROFILE", profile_patient), ("CAIA", caia_patient)):
        evt = pd.to_numeric(df.get("PLATINUM", pd.Series(dtype=float)), errors="coerce")
        evt = evt.dropna().astype(int)
        k = int(evt.sum())
        n = int(len(evt))
        p, lo, hi = wilson_ci(k, n)
        crude.append((label, p, lo, hi, k, n))

    ax = axes[0]
    xs = np.arange(len(crude))
    heights = [c[1] for c in crude]
    yerr = np.array([[c[1] - c[2] for c in crude], [c[3] - c[1] for c in crude]])
    bars = ax.bar(xs, heights, yerr=yerr, capsize=6,
                  color=[COHORT_COLORS[c[0]] for c in crude],
                  edgecolor="white")
    for x, c, h in zip(xs, crude, heights):
        ax.text(x, h + 0.005,
                f"{c[1]*100:.1f}%\n({c[4]}/{c[5]:,})",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([c[0] for c in crude])
    ax.set_ylabel("Crude platinum incidence")
    ax.set_title("Platinum incidence (95% Wilson CI)")
    ax.set_ylim(0, max(0.1, max(c[3] for c in crude) * 1.4))

    surv_by = {
        "PROFILE": (_coerce_days(profile_patient["t_platinum"]),
                    pd.to_numeric(profile_patient["PLATINUM"], errors="coerce")),
        "CAIA":    (_coerce_days(caia_patient["t_platinum"]),
                    pd.to_numeric(caia_patient["PLATINUM"], errors="coerce")),
    }
    overlay_km(axes[1], surv_by,
               title="Time to platinum (Kaplan-Meier)",
               xlabel="Days from first treatment",
               ylabel="Platinum-free survival")
    fig.tight_layout()
    return fig


def build_overall_survival_figure(
    profile_patient: pd.DataFrame,
    caia_patient: pd.DataFrame,
) -> plt.Figure:
    """Single panel: overall-survival KM overlay."""
    fig, ax = plt.subplots(figsize=(7, 4.8))
    surv_by = {
        "PROFILE": (_coerce_days(profile_patient["t_death"]),
                    pd.to_numeric(profile_patient["DEATH"], errors="coerce")),
        "CAIA":    (_coerce_days(caia_patient["t_death"]),
                    pd.to_numeric(caia_patient["DEATH"], errors="coerce")),
    }
    overlay_km(ax, surv_by,
               title="Overall survival (Kaplan-Meier)",
               xlabel="Days from first treatment",
               ylabel="Survival probability")
    fig.tight_layout()
    return fig


def _canonical_lab_union(
    profile_labs: pd.DataFrame,
    caia_labs: pd.DataFrame,
    *,
    profile_id_col: str,
    caia_id_col: str,
    min_coverage: float,
) -> tuple[set[str], dict[str, dict[str, pd.Series]]]:
    """Pick canonical labs per cohort (coverage threshold) and return the union
    keyed by the canonical (short) lab name. Per-lab value series are gathered
    once, keyed by canonical name so downstream plotting can pull both cohorts'
    values cheaply.
    """
    p_ids = profile_labs[profile_id_col].dropna().unique()
    c_ids = caia_labs[caia_id_col].dropna().unique()
    p_keep = set(select_canonical_labs(
        profile_labs,
        mrns=list(p_ids),
        min_coverage=min_coverage,
        id_col=profile_id_col,
        min_distinct_values=2,
    ))
    c_keep = set(select_canonical_labs(
        caia_labs,
        mrns=list(c_ids),
        min_coverage=min_coverage,
        id_col=caia_id_col,
        min_distinct_values=2,
    ))

    values: dict[str, dict[str, pd.Series]] = {}
    for raw_name in p_keep:
        canon = canonicalize_lab_name(raw_name)
        vals = profile_labs.loc[profile_labs["LAB_NAME"] == raw_name, "LAB_VALUE"]
        values.setdefault(canon, {})["PROFILE"] = pd.to_numeric(vals, errors="coerce").dropna()
    for raw_name in c_keep:
        canon = canonicalize_lab_name(raw_name)
        vals = caia_labs.loc[caia_labs["LAB_NAME"] == raw_name, "LAB_VALUE"]
        values.setdefault(canon, {})["CAIA"] = pd.to_numeric(vals, errors="coerce").dropna()
    return set(values), values


def write_per_lab_figures(
    values_by_lab: dict[str, dict[str, pd.Series]],
    out_dir: Path,
    formats: tuple[str, ...],
) -> int:
    raw_dir = out_dir / "labs_raw"
    irnt_dir = out_dir / "labs_irnt"
    raw_dir.mkdir(parents=True, exist_ok=True)
    irnt_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for lab_canon in sorted(values_by_lab):
        cohort_to_values = values_by_lab[lab_canon]
        safe = lab_canon.replace("/", "_").replace(" ", "_")

        fig, ax = plt.subplots(figsize=(7, 4))
        overlay_hist(ax, cohort_to_values, bins=60)
        ax.set_xlabel(f"{lab_canon} (raw)")
        ax.set_ylabel("Density")
        ax.set_title(f"{lab_canon} - raw value distribution")
        fig.tight_layout()
        _save(fig, raw_dir / safe, formats)

        irnt_vals = {k: irnt(v).dropna() for k, v in cohort_to_values.items()}
        fig, ax = plt.subplots(figsize=(7, 4))
        overlay_hist(ax, irnt_vals, bins=60)
        ax.set_xlabel(f"{lab_canon} (IRNT)")
        ax.set_ylabel("Density")
        ax.set_title(f"{lab_canon} - inverse rank-normal transform")
        ax.set_xlim(-4.5, 4.5)
        fig.tight_layout()
        _save(fig, irnt_dir / safe, formats)
        n += 1
    return n


def run(
    *,
    profile_csv: Path,
    caia_parquet: Path,
    output_dir: Path,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    caia_id_col: str = "person_id",
    profile_id_col: str = "DFCI_MRN",
) -> None:
    plt.rcParams.update(RCPARAMS)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading PROFILE: {profile_csv}")
    profile_patient, profile_labs = load_profile_longitudinal(profile_csv, id_col=profile_id_col)
    print(f"  patients={len(profile_patient):,}  labs={len(profile_labs):,}")

    print(f"Loading CAIA:    {caia_parquet}")
    caia_patient, caia_labs = load_caia_parquet(caia_parquet, id_col=caia_id_col)
    print(f"  patients={len(caia_patient):,}  labs={len(caia_labs):,}")

    print("\nFigure 1: time distributions ...")
    fig = build_time_distributions_figure(
        profile_patient, caia_patient, profile_labs, caia_labs,
        profile_id_col=profile_id_col, caia_id_col=caia_id_col,
    )
    paths = _save(fig, output_dir / "time_distributions", formats)
    for p in paths: print(f"  {p}")

    print("Figure 2: platinum incidence ...")
    fig = build_platinum_incidence_figure(profile_patient, caia_patient)
    paths = _save(fig, output_dir / "platinum_incidence", formats)
    for p in paths: print(f"  {p}")

    print("Figure 3: overall survival KM ...")
    fig = build_overall_survival_figure(profile_patient, caia_patient)
    paths = _save(fig, output_dir / "overall_survival_km", formats)
    for p in paths: print(f"  {p}")

    print(f"\nPer-lab figures (min_coverage={min_coverage}) ...")
    lab_union, values = _canonical_lab_union(
        profile_labs, caia_labs,
        profile_id_col=profile_id_col, caia_id_col=caia_id_col,
        min_coverage=min_coverage,
    )
    print(f"  canonical labs (union): {len(lab_union)}")
    n_written = write_per_lab_figures(values, output_dir, formats)
    print(f"  wrote {n_written} raw + IRNT panels into {output_dir}/labs_raw and labs_irnt")

    print("\nDone.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile-csv", required=True, type=Path,
                   help="Path to PROFILE longitudinal_prediction_data.csv.")
    p.add_argument("--caia-parquet", required=True, type=Path,
                   help="Path to CAIA caia_compass_longitudinal.parquet.")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Directory to write figures into (created if missing).")
    p.add_argument("--formats", nargs="+", default=list(DEFAULT_FORMATS),
                   choices=["png", "pdf", "svg"],
                   help="Image formats to write (default: png pdf).")
    p.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE,
                   help="Coverage threshold for per-cohort canonical-lab selection (default 0.20).")
    p.add_argument("--profile-id-col", default="DFCI_MRN")
    p.add_argument("--caia-id-col", default="person_id")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(
        profile_csv=args.profile_csv,
        caia_parquet=args.caia_parquet,
        output_dir=args.output_dir,
        formats=tuple(args.formats),
        min_coverage=args.min_coverage,
        profile_id_col=args.profile_id_col,
        caia_id_col=args.caia_id_col,
    )
