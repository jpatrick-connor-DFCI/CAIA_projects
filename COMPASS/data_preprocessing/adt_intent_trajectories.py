"""
Script Name: adt_intent_trajectories.py

Description:
Longitudinal lab trajectories and Kaplan-Meier curves stratified by the
`ADT_INTENT` label from `classify_adt_intent.py`.

Two views of the same three groups (METASTATIC / LOCALIZED_ADJUVANT /
INDETERMINATE):

  1. Lab trajectories -- testosterone and PSA from one year before the ADT
     anchor to five years after, summarized as a median with an IQR band in
     30-day bins.
  2. Kaplan-Meier curves -- death, plus the three modelled endpoints
     (platinum, NEPC, AVPC).

Why testosterone and PSA
------------------------
Both are held-out signals: `classify_adt_intent` sees only MEDICATIONS, so
neither lab can be a self-fulfilling readout of the label. They test different
things.

Testosterone tests whether the *exposure* the label claims actually happened.
Both intent groups should castrate (< 50 ng/dL) within about three months of
the anchor, because both really do receive ADT. The groups separate on the
*right* side of the plot instead: the adjuvant group's testosterone should
recover toward the eugonadal range once a protocol-limited course ends,
typically 6-24 months out, while the metastatic group stays suppressed
indefinitely. A metastatic-labelled group that recovers testosterone, or an
adjuvant-labelled group that never does, means the duration signal is not
measuring what it is supposed to.

PSA tests the *disease*. The adjuvant pattern is a deep nadir that stays down;
the metastatic pattern is a nadir followed by a rise as castration resistance
emerges. This is the same signal `validate_adt_intent.compute_psa_nadir_features`
scores per patient, shown here as a trajectory instead of a summary.

The pre-anchor year matters as much as the post-anchor five. Untreated
metastatic disease usually presents with a substantially higher PSA than
localized disease does, so a separation that is already visible *before* ADT
starts is real prognostic signal. If the two groups instead look identical
before the anchor and only diverge after it, the label may be tracking
treatment duration alone rather than the underlying disease.

Time axis
---------
`t_lab` in the Stage 2 longitudinal output is already days from
`TREATMENT_ANCHOR_DATE` -- negative before the anchor, positive after -- so it
is used directly with no re-derivation. For the `adt` arm that anchor is the
first ADT exposure, which is exactly the origin this module's window is defined
around. Passing an `arpi`-arm file would silently shift every trajectory to a
different origin, so `load_longitudinal` checks the anchor date against the
label file's own `ADT_FIRST_DATE` and refuses on mismatch.

Population
----------
The death KM runs on every ADT-exposed patient, because follow-up comes from
the patient-status table and is available for all of them. The three endpoint
KMs run only on the eligible survival cohort, because `PLATINUM`/`NEPC`/`AVPC`
are defined there and nowhere else. That split is deliberate: comparing the
death curve across the two populations shows whether the cohort exclusions
themselves select on survival.

Usage:
    python adt_intent_trajectories.py \
        --labels adt_intent/adt_intent_labels.csv \
        --longitudinal longitudinal_prediction_data_adt.csv \
        --out-dir adt_intent/

Notes:
    - PSA is plotted on a log10 axis. It spans several orders of magnitude, so
      a linear axis is dominated by a handful of very high values and the
      adjuvant group is squashed flat against zero.
    - Below-detection PSA and testosterone are genuinely imputed to 0.0
      upstream in `dfci_labs`, so zeros are real measurements. They are kept in
      the median but floored for the log axis only (see PSA_LOG_FLOOR).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import polars as pl

# Run as a script (`python adt_intent_trajectories.py`), only this file's own
# directory is on sys.path, so `survival_common` is not importable. Notebooks
# add the repo root themselves and are unaffected either way.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

ID_COL = "DFCI_MRN"

INTENT_COL = "ADT_INTENT"
INTENT_ORDER = ["METASTATIC", "LOCALIZED_ADJUVANT", "INDETERMINATE"]

# Colours are fixed per intent class so every figure in the set is read the
# same way. Red = metastatic, blue = adjuvant, grey = indeterminate.
INTENT_COLORS = {
    "METASTATIC": "#c1272d",
    "LOCALIZED_ADJUVANT": "#0b6ba8",
    "INDETERMINATE": "#8a8a8a",
}

# Window around the ADT anchor, in days. One year before, five years after.
WINDOW_PRE_DAYS = -365
WINDOW_POST_DAYS = 365 * 5
# Trajectory bin width. 30 days is roughly the tightest spacing at which
# routine oncology labs are actually drawn, so a narrower bin mostly adds
# empty bins and noise rather than resolution.
BIN_DAYS = 30
# A bin with fewer than this many patients is dropped rather than plotted: a
# median over 2-3 patients is not a summary, and the late-follow-up tail
# thins out fast.
MIN_PATIENTS_PER_BIN = 5

# Canonical (OMOP-collapsed) lab names from dfci_labs. This is the narrow,
# unit-standardized LAB_NAME set that drives prediction features -- NOT the
# broad raw-TEST_TYPE_CD PSA set used for the cohort PSA-count gate, which
# mixes free/complexed/ultrasensitive assays that are not comparable on one
# axis.
PSA_LAB_NAME = "PSA"
TESTOSTERONE_LAB_NAME = "Testosterone"

# Castrate threshold. The conventional cutoff is 50 ng/dL; 20 ng/dL is the
# stricter modern target. Both are drawn as reference lines.
CASTRATE_NG_DL = 50.0
CASTRATE_STRICT_NG_DL = 20.0

# Zeros are real (below-detection, imputed upstream), but log10(0) is not
# plottable. Floor for the log axis only -- the median is computed on the
# true values first.
PSA_LOG_FLOOR = 0.01

# Endpoint -> (duration column, event column, human title). Mirrors
# COMPASS.survival_analysis.cox_aggregated.ENDPOINTS, which is the authority
# for these pairings; kept as a literal here so this module does not import
# the modelling stack just to plot.
KM_ENDPOINTS = {
    "death": ("t_death", "DEATH", "Overall survival"),
    "platinum": ("t_platinum", "PLATINUM", "Time to platinum exposure"),
    "nepc": ("t_nepc", "NEPC", "Time to NEPC"),
    "avpc": ("t_avpc", "AVPC", "Time to AVPC"),
}

# Endpoints defined only in the eligible survival cohort. Death is computable
# for every ADT-exposed patient from patient status, so it is not in this set.
COHORT_ONLY_ENDPOINTS = ("platinum", "nepc", "avpc")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_labels(path) -> pl.DataFrame:
    """Load the per-patient intent labels written by the 08 notebook."""
    labels = pl.read_csv(path, infer_schema_length=0)
    missing = {ID_COL, INTENT_COL} - set(labels.columns)
    if missing:
        raise ValueError(
            f"{path} is missing expected columns: {sorted(missing)}. "
            "Run the classify cell of 08_adt_intent.ipynb first."
        )
    casts = [
        pl.col(ID_COL).cast(pl.Float64, strict=False).cast(pl.Int64, strict=False),
        pl.col(INTENT_COL).cast(pl.Utf8),
    ]
    for col, dtype in (
        ("ELIGIBLE", pl.Int8),
        ("ADT_SPAN_DAYS", pl.Float64),
        ("DEATH", pl.Int64),
    ):
        if col in labels.columns:
            casts.append(pl.col(col).cast(pl.Float64, strict=False).cast(dtype, strict=False))
    if "ADT_FIRST_DATE" in labels.columns:
        casts.append(pl.col("ADT_FIRST_DATE").str.to_datetime(strict=False))
    return labels.with_columns(casts).drop_nulls(ID_COL)


def load_longitudinal(path, labels: pl.DataFrame | None = None) -> pl.DataFrame:
    """Load the Stage 2 longitudinal prediction frame.

    `t_lab` is already days from TREATMENT_ANCHOR_DATE, so no re-derivation
    happens here. When `labels` carries ADT_FIRST_DATE, the anchor is checked
    against it -- an arpi-arm file anchors on first ARPI/taxane exposure
    instead of first ADT, which would shift every trajectory to a different
    origin without changing any column name.
    """
    lf = pl.read_csv(path, infer_schema_length=0)
    missing = {ID_COL, "t_lab", "LAB_NAME", "LAB_VALUE"} - set(lf.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {sorted(missing)}")

    casts = [
        pl.col(ID_COL).cast(pl.Float64, strict=False).cast(pl.Int64, strict=False),
        pl.col("t_lab").cast(pl.Float64, strict=False),
        pl.col("LAB_VALUE").cast(pl.Float64, strict=False),
        pl.col("LAB_NAME").cast(pl.Utf8),
    ]
    for dur, evt, _ in KM_ENDPOINTS.values():
        if dur in lf.columns:
            casts.append(pl.col(dur).cast(pl.Float64, strict=False))
        if evt in lf.columns:
            casts.append(pl.col(evt).cast(pl.Float64, strict=False).cast(pl.Int64, strict=False))
    if "TREATMENT_ANCHOR_DATE" in lf.columns:
        casts.append(pl.col("TREATMENT_ANCHOR_DATE").str.to_datetime(strict=False))
    lf = lf.with_columns(casts).drop_nulls(ID_COL)

    if labels is not None:
        _check_anchor_is_adt(lf, labels)
    return lf


def _check_anchor_is_adt(lf: pl.DataFrame, labels: pl.DataFrame) -> None:
    """Refuse a longitudinal file anchored on something other than first ADT.

    Both arms write the same column names, so a mis-passed arpi file is
    otherwise silent: every trajectory would just be measured from a later
    origin. Compared on shared patients rather than assumed.
    """
    if "TREATMENT_ANCHOR_DATE" not in lf.columns or "ADT_FIRST_DATE" not in labels.columns:
        print("[anchor] skipped -- no TREATMENT_ANCHOR_DATE / ADT_FIRST_DATE to compare")
        return

    anchors = (
        lf.select(ID_COL, "TREATMENT_ANCHOR_DATE")
        .drop_nulls()
        .unique(subset=[ID_COL])
        .join(labels.select(ID_COL, "ADT_FIRST_DATE").drop_nulls(), on=ID_COL, how="inner")
    )
    if anchors.height == 0:
        print("[anchor] skipped -- no overlapping patients to compare")
        return

    delta = anchors.select(
        (pl.col("TREATMENT_ANCHOR_DATE") - pl.col("ADT_FIRST_DATE")).dt.total_days().abs()
    ).to_series()
    n_mismatch = int((delta > 1).sum())
    pct = 100.0 * n_mismatch / anchors.height
    if pct > 5.0:
        raise ValueError(
            f"{n_mismatch:,}/{anchors.height:,} ({pct:.1f}%) of patients have a "
            "TREATMENT_ANCHOR_DATE more than a day from their first ADT date. "
            "This looks like the arpi-arm longitudinal file, whose anchor is "
            "first ARPI/taxane exposure -- pass the adt-arm file instead, or "
            "every trajectory will be measured from the wrong origin."
        )
    print(f"[anchor] OK -- anchor matches first ADT for {anchors.height - n_mismatch:,}/{anchors.height:,} patients")


def attach_intent(frame: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Inner-join the intent label onto a patient-keyed frame."""
    keep = [ID_COL, INTENT_COL] + [c for c in ("ELIGIBLE",) if c in labels.columns]
    return frame.join(labels.select(keep), on=ID_COL, how="inner")


# ---------------------------------------------------------------------------
# Lab trajectories
# ---------------------------------------------------------------------------

def build_lab_trajectory(
    longitudinal: pl.DataFrame,
    labels: pl.DataFrame,
    lab_name: str,
    *,
    pre_days: int = WINDOW_PRE_DAYS,
    post_days: int = WINDOW_POST_DAYS,
    bin_days: int = BIN_DAYS,
    min_patients_per_bin: int = MIN_PATIENTS_PER_BIN,
) -> pl.DataFrame:
    """Median / IQR of one lab per intent class per time bin.

    Returns one row per (intent, bin) with the bin midpoint in days, the
    median, Q1, Q3, the number of contributing patients, and the number of
    measurements.

    A patient contributing several draws to one bin would otherwise weight
    that bin by how sick they were -- heavily-monitored metastatic patients
    get drawn far more often than adjuvant ones -- so values are averaged
    within (patient, bin) first and the median is taken across patients.
    """
    window = longitudinal.filter(
        (pl.col("LAB_NAME") == lab_name)
        & pl.col("LAB_VALUE").is_not_null()
        & pl.col("t_lab").is_not_null()
        & (pl.col("t_lab") >= pre_days)
        & (pl.col("t_lab") <= post_days)
    )
    window = attach_intent(window, labels)
    if window.height == 0:
        return pl.DataFrame(
            schema={
                INTENT_COL: pl.Utf8, "bin_start": pl.Int64, "bin_mid": pl.Float64,
                "median": pl.Float64, "q1": pl.Float64, "q3": pl.Float64,
                "n_patients": pl.UInt32, "n_measurements": pl.UInt32,
            }
        )

    # Floor-divide so bins tile cleanly across zero: [-30,0) and [0,30) stay
    # distinct, keeping the pre/post anchor boundary sharp.
    binned = window.with_columns(
        ((pl.col("t_lab") // bin_days) * bin_days).cast(pl.Int64).alias("bin_start")
    )

    per_patient = binned.group_by([INTENT_COL, "bin_start", ID_COL]).agg(
        pl.col("LAB_VALUE").mean().alias("value"),
        pl.len().alias("n_draws"),
    )

    traj = (
        per_patient.group_by([INTENT_COL, "bin_start"])
        .agg(
            pl.col("value").median().alias("median"),
            pl.col("value").quantile(0.25).alias("q1"),
            pl.col("value").quantile(0.75).alias("q3"),
            pl.len().alias("n_patients"),
            pl.col("n_draws").sum().cast(pl.UInt32).alias("n_measurements"),
        )
        .filter(pl.col("n_patients") >= min_patients_per_bin)
        .with_columns((pl.col("bin_start") + bin_days / 2.0).alias("bin_mid"))
        .sort([INTENT_COL, "bin_start"])
    )
    return traj


def summarize_trajectory_coverage(
    longitudinal: pl.DataFrame, labels: pl.DataFrame, lab_name: str
) -> pl.DataFrame:
    """Per-intent count of patients with any in-window measurement of `lab_name`.

    Read this before the plot. Testosterone is ordered far less consistently
    than PSA, so a flat or absent curve is usually missing data rather than a
    biological finding, and this table is what tells the two apart.
    """
    total = labels.group_by(INTENT_COL).agg(pl.len().alias("n_labelled"))
    window = longitudinal.filter(
        (pl.col("LAB_NAME") == lab_name)
        & pl.col("LAB_VALUE").is_not_null()
        & (pl.col("t_lab") >= WINDOW_PRE_DAYS)
        & (pl.col("t_lab") <= WINDOW_POST_DAYS)
    )
    window = attach_intent(window, labels)
    if window.height == 0:
        return total.with_columns(
            pl.lit(0).cast(pl.UInt32).alias("n_with_lab"),
            pl.lit(0.0).alias("pct_with_lab"),
            pl.lit(None, dtype=pl.Float64).alias("median_draws_per_patient"),
        )

    have = window.group_by([INTENT_COL, ID_COL]).agg(pl.len().alias("n_draws"))
    have = have.group_by(INTENT_COL).agg(
        pl.len().alias("n_with_lab"),
        pl.col("n_draws").median().alias("median_draws_per_patient"),
    )
    return (
        total.join(have, on=INTENT_COL, how="left")
        .with_columns(
            pl.col("n_with_lab").fill_null(0).cast(pl.UInt32),
            (pl.col("n_with_lab").fill_null(0) / pl.col("n_labelled") * 100).round(1).alias("pct_with_lab"),
        )
        .sort("n_labelled", descending=True)
    )


def plot_lab_trajectory(
    ax,
    traj: pl.DataFrame,
    *,
    title: str,
    ylabel: str,
    log_scale: bool = False,
    hlines: tuple[float, ...] = (),
    intent_order: list[str] | None = None,
) -> None:
    """Draw median lines with IQR bands, one per intent class, onto `ax`."""
    order = intent_order or INTENT_ORDER
    for intent in order:
        sub = traj.filter(pl.col(INTENT_COL) == intent).sort("bin_mid")
        if sub.height == 0:
            continue
        x = sub["bin_mid"].to_list()
        med = sub["median"].to_list()
        q1 = sub["q1"].to_list()
        q3 = sub["q3"].to_list()
        n_pts = int(sub["n_patients"].max())
        color = INTENT_COLORS.get(intent, "#444444")
        ax.plot(x, med, color=color, lw=2, label=f"{intent} (max n={n_pts:,})", zorder=3)
        ax.fill_between(x, q1, q3, color=color, alpha=0.15, lw=0, zorder=2)

    # The anchor itself -- everything left of this is pre-ADT.
    ax.axvline(0, color="black", ls="--", lw=1, alpha=0.7, zorder=1)
    for y in hlines:
        ax.axhline(y, color="#555555", ls=":", lw=1, alpha=0.8, zorder=1)

    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Days from first ADT (0 = ADT start)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.2)


# ---------------------------------------------------------------------------
# Kaplan-Meier
# ---------------------------------------------------------------------------

def build_km_input(
    frame: pl.DataFrame,
    labels: pl.DataFrame,
    endpoint: str,
) -> pl.DataFrame:
    """One row per patient with the duration/event pair for `endpoint`.

    The longitudinal frame is one row per lab draw, so it is collapsed to one
    row per patient first -- the endpoint columns are patient-level constants
    repeated down every row of that patient's block.
    """
    if endpoint not in KM_ENDPOINTS:
        raise ValueError(f"Unknown endpoint {endpoint!r} (expected one of {sorted(KM_ENDPOINTS)})")
    dur_col, evt_col, _ = KM_ENDPOINTS[endpoint]
    missing = {dur_col, evt_col} - set(frame.columns)
    if missing:
        return pl.DataFrame(
            schema={ID_COL: pl.Int64, INTENT_COL: pl.Utf8,
                    "duration": pl.Float64, "event": pl.Int64}
        )

    per_patient = (
        frame.select(ID_COL, dur_col, evt_col)
        .drop_nulls(ID_COL)
        .group_by(ID_COL)
        .agg(pl.col(dur_col).max().alias("duration"), pl.col(evt_col).max().alias("event"))
    )
    out = attach_intent(per_patient, labels)
    return out.filter(
        pl.col("duration").is_not_null()
        & pl.col("event").is_not_null()
        & (pl.col("duration") >= 0)
    ).select(ID_COL, INTENT_COL, "duration", "event")


def build_death_km_input(labels: pl.DataFrame, follow_up: pl.DataFrame) -> pl.DataFrame:
    """Death KM for the full ADT-exposed population.

    Duration is days from first ADT to death or last contact, computed from
    the patient-status follow-up date rather than the survival cohort, so it
    covers patients the cohort excluded.
    """
    needed = {ID_COL, "FOLLOW_UP_END_DATE", "DEATH"}
    missing = needed - set(follow_up.columns)
    if missing:
        raise ValueError(f"follow_up is missing expected columns: {sorted(missing)}")
    if "ADT_FIRST_DATE" not in labels.columns:
        raise ValueError("labels is missing ADT_FIRST_DATE -- cannot compute time from ADT start")

    joined = labels.select(ID_COL, INTENT_COL, "ADT_FIRST_DATE").join(
        follow_up.select(ID_COL, "FOLLOW_UP_END_DATE", "DEATH"), on=ID_COL, how="inner"
    )
    return (
        joined.with_columns(
            (pl.col("FOLLOW_UP_END_DATE") - pl.col("ADT_FIRST_DATE"))
            .dt.total_days()
            .cast(pl.Float64)
            .alias("duration"),
            pl.col("DEATH").cast(pl.Int64).alias("event"),
        )
        .filter(
            pl.col("duration").is_not_null()
            & pl.col("event").is_not_null()
            & (pl.col("duration") >= 0)
        )
        .select(ID_COL, INTENT_COL, "duration", "event")
    )


def km_series_by_intent(km_input: pl.DataFrame, intent_order: list[str] | None = None) -> dict:
    """Convert a KM input frame into the {label: (durations, events)} mapping
    that `survival_common.plotting.overlay_km` expects.

    pandas is imported here rather than at module top so the trajectory half
    of this module still works in an environment without it -- overlay_km
    needs pandas Series, but nothing above does.
    """
    import pandas as pd

    order = intent_order or INTENT_ORDER
    out: dict[str, tuple] = {}
    for intent in order:
        sub = km_input.filter(pl.col(INTENT_COL) == intent)
        if sub.height == 0:
            continue
        out[intent] = (
            pd.Series(sub["duration"].to_list(), dtype="float64"),
            pd.Series(sub["event"].to_list(), dtype="float64"),
        )
    return out


def summarize_km(km_input: pl.DataFrame) -> pl.DataFrame:
    """Event counts and median follow-up per intent class.

    Printed alongside every KM: a curve built on a handful of events looks
    just as confident as one built on hundreds, and only this table shows the
    difference.
    """
    if km_input.height == 0:
        return pl.DataFrame(
            schema={INTENT_COL: pl.Utf8, "n_patients": pl.UInt32, "n_events": pl.Int64,
                    "pct_event": pl.Float64, "median_duration_days": pl.Float64}
        )
    return (
        km_input.group_by(INTENT_COL)
        .agg(
            pl.len().alias("n_patients"),
            pl.col("event").sum().alias("n_events"),
            (pl.col("event").mean() * 100).round(1).alias("pct_event"),
            pl.col("duration").median().round(0).alias("median_duration_days"),
        )
        .sort("n_patients", descending=True)
    )


def logrank_by_intent(km_input: pl.DataFrame) -> pl.DataFrame:
    """Pairwise log-rank p-values between intent classes.

    Descriptive only -- these groups are defined by a derived label, not
    randomized, so a small p-value says the label separates the outcome, not
    that intent caused it.
    """
    try:
        from lifelines.statistics import logrank_test
    except ModuleNotFoundError:
        return pl.DataFrame(
            schema={"group_a": pl.Utf8, "group_b": pl.Utf8,
                    "n_a": pl.Int64, "n_b": pl.Int64, "p_value": pl.Float64}
        )

    present = [i for i in INTENT_ORDER if km_input.filter(pl.col(INTENT_COL) == i).height > 0]
    rows = []
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            sa = km_input.filter(pl.col(INTENT_COL) == a)
            sb = km_input.filter(pl.col(INTENT_COL) == b)
            res = logrank_test(
                sa["duration"].to_list(), sb["duration"].to_list(),
                event_observed_A=sa["event"].to_list(),
                event_observed_B=sb["event"].to_list(),
            )
            rows.append({
                "group_a": a, "group_b": b,
                "n_a": sa.height, "n_b": sb.height,
                "p_value": float(res.p_value),
            })
    if not rows:
        return pl.DataFrame(
            schema={"group_a": pl.Utf8, "group_b": pl.Utf8,
                    "n_a": pl.Int64, "n_b": pl.Int64, "p_value": pl.Float64}
        )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lab trajectories and KM curves stratified by ADT_INTENT."
    )
    parser.add_argument("--labels", required=True,
                        help="adt_intent_labels.csv from 08_adt_intent.ipynb")
    parser.add_argument("--longitudinal", required=True,
                        help="Stage 2 longitudinal_prediction_data_adt.csv (adt arm)")
    parser.add_argument("--out-dir", required=True, help="directory for figures and tables")
    parser.add_argument("--bin-days", type=int, default=BIN_DAYS)
    parser.add_argument("--min-patients-per-bin", type=int, default=MIN_PATIENTS_PER_BIN)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    labels = load_labels(args.labels)
    longitudinal = load_longitudinal(args.longitudinal, labels=labels)
    print(f"labels: {labels.height:,} patients | longitudinal rows: {longitudinal.height:,}")

    # --- trajectories ---
    specs = [
        (TESTOSTERONE_LAB_NAME, "Testosterone (ng/dL)", False,
         (CASTRATE_NG_DL, CASTRATE_STRICT_NG_DL)),
        (PSA_LAB_NAME, "PSA (ng/mL, log scale)", True, ()),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, (lab, ylabel, log_scale, hlines) in zip(axes, specs):
        print(f"\n[{lab}] coverage:")
        print(summarize_trajectory_coverage(longitudinal, labels, lab))
        traj = build_lab_trajectory(
            longitudinal, labels, lab,
            bin_days=args.bin_days,
            min_patients_per_bin=args.min_patients_per_bin,
        )
        traj.write_csv(out_dir / f"adt_intent_trajectory_{lab.lower()}.csv")
        if log_scale and traj.height:
            traj = traj.with_columns(
                *[pl.col(c).clip(lower_bound=PSA_LOG_FLOOR) for c in ("median", "q1", "q3")]
            )
        plot_lab_trajectory(
            ax, traj, title=f"{lab} around ADT start", ylabel=ylabel,
            log_scale=log_scale, hlines=hlines,
        )
    fig.tight_layout()
    fig.savefig(out_dir / "adt_intent_lab_trajectories.png", dpi=150)
    plt.close(fig)
    print(f"\nwrote {out_dir / 'adt_intent_lab_trajectories.png'}")

    # --- KM curves ---
    from survival_common.plotting import overlay_km

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, endpoint in zip(axes.ravel(), KM_ENDPOINTS):
        km_input = build_km_input(longitudinal, labels, endpoint)
        _, _, title = KM_ENDPOINTS[endpoint]
        print(f"\n[{endpoint}]")
        print(summarize_km(km_input))
        if km_input.height == 0:
            ax.set_title(f"{title} -- no data")
            ax.axis("off")
            continue
        km_input.write_csv(out_dir / f"adt_intent_km_{endpoint}.csv")
        overlay_km(
            ax, km_series_by_intent(km_input), colors=INTENT_COLORS,
            title=title, xlabel="Days from first ADT",
            ylabel="Event-free probability" if endpoint != "death" else "Survival probability",
        )
    fig.tight_layout()
    fig.savefig(out_dir / "adt_intent_km_curves.png", dpi=150)
    plt.close(fig)
    print(f"\nwrote {out_dir / 'adt_intent_km_curves.png'}")


if __name__ == "__main__":
    main()
