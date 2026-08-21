"""
Script Name: adt_intent_trajectories.py

Description:
Longitudinal lab trajectories and Kaplan-Meier curves stratified by the
`ADT_INTENT` label from `classify_adt_intent.py`.

Two views of the same two groups (METASTATIC / LOCALIZED_ADJUVANT):

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
# The label is binary. INDETERMINATE is kept in the order and palette only so
# that previously written label files still plot; current runs never emit it.
INTENT_ORDER = ["METASTATIC", "LOCALIZED_ADJUVANT", "INDETERMINATE"]

# Colours are fixed per intent class so every figure in the set is read the
# same way. Red = metastatic, blue = adjuvant, grey = legacy indeterminate.
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

# ---------------------------------------------------------------------------
# Stage / metastatic-burden distribution plots
# ---------------------------------------------------------------------------
# These compare the medication-only label against two external references
# (see validate_adt_intent.load_stage_nearest_adt / compute_met_burden_at_adt,
# both anchored on ADT_FIRST_DATE). Every panel is drawn as a WITHIN-CLASS
# percentage rather than a raw count: the two classes differ in size by design
# -- most of the cohort is retained -- so raw bars would show cohort
# composition rather than a distributional difference.

STAGE_PLOT_ORDER = ["I", "II", "III", "IV"]
# Sequential blue: stage is ordinal, so the palette should read as a ramp
# rather than as four unrelated categories.
STAGE_COLORS = {
    "I": "#c6dbef",
    "II": "#9ecae1",
    "III": "#4292c6",
    "IV": "#08519c",
}


def _class_order_present(df: pl.DataFrame, col: str = INTENT_COL) -> list[str]:
    """INTENT_ORDER restricted to classes actually present, order preserved."""
    present = set(df[col].unique().to_list())
    return [c for c in INTENT_ORDER if c in present]


def stage_distribution_by_intent(labelled: pl.DataFrame) -> pl.DataFrame:
    """Within-class stage distribution, as counts and percentages.

    Restricted to patients with a stage observation in the matching window;
    `n_in_class` is that covered denominator, not the full class size.
    """
    if "CANCER_STAGE" not in labelled.columns:
        return pl.DataFrame()
    covered = labelled.filter(pl.col("CANCER_STAGE").is_not_null())
    if covered.height == 0:
        return pl.DataFrame()
    counts = covered.group_by([INTENT_COL, "CANCER_STAGE"]).agg(
        pl.len().alias("n_patients")
    )
    totals = covered.group_by(INTENT_COL).agg(pl.len().alias("n_in_class"))
    return (
        counts.join(totals, on=INTENT_COL, how="left")
        .with_columns(
            (pl.col("n_patients") / pl.col("n_in_class") * 100).round(1).alias("pct")
        )
        .sort([INTENT_COL, "CANCER_STAGE"])
    )


def plot_stage_distribution(ax, labelled: pl.DataFrame) -> None:
    """Stacked stage composition per class, as % within class."""
    dist = stage_distribution_by_intent(labelled)
    if dist.height == 0:
        ax.text(0.5, 0.5, "no stage coverage", ha="center", va="center",
                transform=ax.transAxes, color="#888888")
        ax.set_axis_off()
        return

    classes = _class_order_present(dist)
    bottoms = {c: 0.0 for c in classes}
    for stage in STAGE_PLOT_ORDER:
        heights = []
        for cls in classes:
            row = dist.filter(
                (pl.col(INTENT_COL) == cls) & (pl.col("CANCER_STAGE") == stage)
            )
            heights.append(float(row["pct"][0]) if row.height else 0.0)
        ax.bar(
            classes, heights,
            bottom=[bottoms[c] for c in classes],
            color=STAGE_COLORS[stage], edgecolor="white", linewidth=0.8,
            label=f"stage {stage}", zorder=3,
        )
        for cls, h in zip(classes, heights):
            if h >= 6:  # only label a slice tall enough to hold the text
                ax.text(cls, bottoms[cls] + h / 2, f"{h:.0f}%",
                        ha="center", va="center", fontsize=8,
                        color="white" if stage in ("III", "IV") else "#222222",
                        zorder=4)
            bottoms[cls] += h

    denoms = {
        r[INTENT_COL]: r["n_in_class"]
        for r in dist.unique(subset=[INTENT_COL]).iter_rows(named=True)
    }
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([f"{c}\n(n={denoms.get(c, 0):,})" for c in classes], fontsize=9)
    ax.set_ylabel("% of class with a staged observation")
    ax.set_ylim(0, 100)
    ax.set_title("Stage nearest ADT start, by medication-derived label")
    ax.legend(fontsize=8, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)


def met_burden_distribution_by_intent(
    labelled: pl.DataFrame, max_sites: int = 4
) -> pl.DataFrame:
    """Within-class distribution of N_MET_SITES, top-coded at `max_sites`."""
    if "N_MET_SITES" not in labelled.columns:
        return pl.DataFrame()
    covered = labelled.filter(pl.col("N_MET_SITES").is_not_null())
    if covered.height == 0:
        return pl.DataFrame()
    binned = covered.with_columns(
        pl.min_horizontal(pl.col("N_MET_SITES"), pl.lit(max_sites)).alias("_n_sites")
    )
    counts = binned.group_by([INTENT_COL, "_n_sites"]).agg(pl.len().alias("n_patients"))
    totals = binned.group_by(INTENT_COL).agg(pl.len().alias("n_in_class"))
    return (
        counts.join(totals, on=INTENT_COL, how="left")
        .with_columns(
            (pl.col("n_patients") / pl.col("n_in_class") * 100).round(1).alias("pct")
        )
        .rename({"_n_sites": "n_met_sites"})
        .sort([INTENT_COL, "n_met_sites"])
    )


def plot_met_burden_distribution(ax, labelled: pl.DataFrame, max_sites: int = 4) -> None:
    """Grouped bars: % of each class at 0, 1, ... `max_sites`+ organ groups."""
    dist = met_burden_distribution_by_intent(labelled, max_sites=max_sites)
    if dist.height == 0:
        ax.text(0.5, 0.5, "no met-burden coverage", ha="center", va="center",
                transform=ax.transAxes, color="#888888")
        ax.set_axis_off()
        return

    classes = _class_order_present(dist)
    levels = list(range(max_sites + 1))
    width = 0.8 / max(len(classes), 1)
    for i, cls in enumerate(classes):
        sub = dist.filter(pl.col(INTENT_COL) == cls)
        lookup = {int(r["n_met_sites"]): float(r["pct"]) for r in sub.iter_rows(named=True)}
        heights = [lookup.get(l, 0.0) for l in levels]
        offsets = [l - 0.4 + width * (i + 0.5) for l in levels]
        n_cls = int(sub["n_in_class"][0])
        ax.bar(offsets, heights, width=width,
               color=INTENT_COLORS.get(cls, "#444444"), alpha=0.85,
               label=f"{cls} (n={n_cls:,})", zorder=3)

    ax.set_xticks(levels)
    ax.set_xticklabels([str(l) if l < max_sites else f"{max_sites}+" for l in levels])
    ax.set_xlabel("distinct metastatic organ groups coded by ADT start")
    ax.set_ylabel("% of class")
    ax.set_title("Metastatic burden at ADT start, by label")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)


def plot_met_site_pattern(ax, labelled: pl.DataFrame) -> None:
    """Per-organ-group involvement rate, as horizontal grouped bars.

    Sorted by the metastatic class's rate so the dominant sites read top-down.
    Bone-predominant spread is the expected prostate pattern; a liver/brain
    signal skews toward aggressive-variant biology.
    """
    site_cols = [c for c in labelled.columns if c.startswith("MET_SITE_")]
    covered = (
        labelled.filter(pl.col("N_MET_SITES").is_not_null())
        if "N_MET_SITES" in labelled.columns
        else labelled
    )
    if not site_cols or covered.height == 0:
        ax.text(0.5, 0.5, "no met-site coverage", ha="center", va="center",
                transform=ax.transAxes, color="#888888")
        ax.set_axis_off()
        return

    classes = _class_order_present(covered)
    rates = {}
    for cls in classes:
        sub = covered.filter(pl.col(INTENT_COL) == cls)
        rates[cls] = {
            c[len("MET_SITE_"):]: (
                float(sub[c].cast(pl.Float64).mean() * 100) if sub.height else 0.0
            )
            for c in site_cols
        }

    lead = classes[0] if classes else None
    sites = sorted(
        (c[len("MET_SITE_"):] for c in site_cols),
        key=lambda s: rates[lead][s] if lead else 0.0,
    )
    y = list(range(len(sites)))
    height = 0.8 / max(len(classes), 1)
    for i, cls in enumerate(classes):
        offs = [v - 0.4 + height * (i + 0.5) for v in y]
        ax.barh(offs, [rates[cls][s] for s in sites], height=height,
                color=INTENT_COLORS.get(cls, "#444444"), alpha=0.85,
                label=cls, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(sites)
    ax.set_xlabel("% of class with the site coded by ADT start")
    ax.set_title("Metastatic site pattern at ADT start")
    # Bars run from the left, so the legend sits clear of them on the right at
    # mid-height rather than overlapping the shortest (bottom) site.
    ax.legend(fontsize=8, frameon=False, loc="center right")
    ax.set_xlim(0, max(105 * 0.01, ax.get_xlim()[1] * 1.18))
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)


def max_stage_distribution_by_intent(labelled: pl.DataFrame) -> pl.DataFrame:
    """Within-class max-stage composition on each side of ADT start.

    Long form: one row per (side, class, stage). `n_in_class` is the covered
    denominator for *that side* -- a patient with no pre-ADT staging note is
    absent from the "before" denominator rather than counted as stage I, so
    the two sides generally have different denominators.
    """
    if "MAX_STAGE_BEFORE" not in labelled.columns:
        return pl.DataFrame()

    frames = []
    for side, col in (("before", "MAX_STAGE_BEFORE"), ("after", "MAX_STAGE_AFTER")):
        if col not in labelled.columns:
            continue
        covered = labelled.filter(pl.col(col).is_not_null())
        if covered.height == 0:
            continue
        counts = covered.group_by([INTENT_COL, col]).agg(pl.len().alias("n_patients"))
        totals = covered.group_by(INTENT_COL).agg(pl.len().alias("n_in_class"))
        frames.append(
            counts.join(totals, on=INTENT_COL, how="left")
            .rename({col: "MAX_STAGE"})
            .with_columns(
                (pl.col("n_patients") / pl.col("n_in_class") * 100).round(1).alias("pct"),
                pl.lit(side).alias("side"),
            )
            .select("side", INTENT_COL, "MAX_STAGE", "n_patients", "n_in_class", "pct")
        )
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames).sort(["side", INTENT_COL, "MAX_STAGE"])


def plot_max_stage_distribution(ax, labelled: pl.DataFrame) -> None:
    """Stacked max-stage composition, before vs after ADT start, per class.

    Bars are grouped as (class, side) pairs so the pre/post shift within a
    class is read vertically, side by side.
    """
    dist = max_stage_distribution_by_intent(labelled)
    if dist.height == 0:
        ax.text(0.5, 0.5, "no max-stage coverage", ha="center", va="center",
                transform=ax.transAxes, color="#888888")
        ax.set_axis_off()
        return

    classes = _class_order_present(dist)
    sides = [s for s in ("before", "after") if s in set(dist["side"].to_list())]
    # One slot per (class, side); a small gap between classes keeps the pairing
    # visually obvious without a second axis.
    positions, labels_x, keys = [], [], []
    x = 0.0
    for cls in classes:
        for side in sides:
            positions.append(x)
            keys.append((cls, side))
            labels_x.append(side)
            x += 1.0
        x += 0.6

    bottoms = {k: 0.0 for k in keys}
    for stage in STAGE_PLOT_ORDER:
        heights = []
        for cls, side in keys:
            row = dist.filter(
                (pl.col(INTENT_COL) == cls)
                & (pl.col("side") == side)
                & (pl.col("MAX_STAGE") == stage)
            )
            heights.append(float(row["pct"][0]) if row.height else 0.0)
        ax.bar(
            positions, heights, width=0.82,
            bottom=[bottoms[k] for k in keys],
            color=STAGE_COLORS[stage], edgecolor="white", linewidth=0.8,
            label=f"stage {stage}", zorder=3,
        )
        for pos, k, h in zip(positions, keys, heights):
            if h >= 6:
                ax.text(pos, bottoms[k] + h / 2, f"{h:.0f}%",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if stage in ("III", "IV") else "#222222",
                        zorder=4)
            bottoms[k] += h

    denoms = {
        (r[INTENT_COL], r["side"]): r["n_in_class"]
        for r in dist.unique(subset=[INTENT_COL, "side"]).iter_rows(named=True)
    }
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{side}\nn={denoms.get(k, 0):,}" for k, side in zip(keys, labels_x)],
        fontsize=7.5,
    )
    # Class name centred under its pair, via a secondary tick axis -- keeping it
    # inside the axes means tight_layout can still size the figure.
    sec = ax.secondary_xaxis("bottom")
    sec.set_xticks(
        [
            sum(p for p, k in zip(positions, keys) if k[0] == cls)
            / max(1, sum(1 for k in keys if k[0] == cls))
            for cls in classes
        ]
    )
    sec.set_xticklabels(classes, fontsize=9)
    sec.tick_params(length=0, pad=22)
    sec.spines["bottom"].set_visible(False)

    ax.set_ylabel("% of side-covered patients in class")
    ax.set_ylim(0, 100)
    ax.set_title("Max stage before vs after ADT start")
    ax.legend(fontsize=8, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)


def plot_max_stage_iv_rate(ax, labelled: pl.DataFrame) -> None:
    """Stage IV rate before vs after ADT start, per class, with coverage.

    The single number the label most needs to be right about: a
    LOCALIZED_ADJUVANT class carrying a high pre-ADT stage IV rate is the
    label failing.
    """
    if "IS_MAX_STAGE_IV_BEFORE" not in labelled.columns:
        ax.text(0.5, 0.5, "no max-stage coverage", ha="center", va="center",
                transform=ax.transAxes, color="#888888")
        ax.set_axis_off()
        return

    classes = _class_order_present(labelled)
    sides = [
        ("before", "IS_MAX_STAGE_IV_BEFORE", "#4a6fa5"),
        ("after", "IS_MAX_STAGE_IV_AFTER", "#b5651d"),
    ]
    width = 0.36
    xs = list(range(len(classes)))
    any_data = False
    for i, (side, col, colour) in enumerate(sides):
        if col not in labelled.columns:
            continue
        rates, ns = [], []
        for cls in classes:
            covered = labelled.filter(
                (pl.col(INTENT_COL) == cls) & pl.col(col).is_not_null()
            )
            ns.append(covered.height)
            rates.append(
                float(covered[col].mean()) * 100 if covered.height else 0.0
            )
        if any(n for n in ns):
            any_data = True
        offset = (i - (len(sides) - 1) / 2) * width
        ax.bar([x + offset for x in xs], rates, width=width,
               color=colour, label=side, zorder=3)
        for x, r, n in zip(xs, rates, ns):
            if not n:
                # No coverage on this side: say so, rather than drawing a
                # flat bar that would read as a measured 0%.
                ax.text(x + offset, 1.5, "no\ncov.", ha="center", va="bottom",
                        fontsize=7, color="#888888", zorder=4)
                continue
            if r == 0:
                # A real, measured zero needs a visible stub or it looks like
                # missing data.
                ax.plot([x + offset - width / 2, x + offset + width / 2],
                        [0.6, 0.6], color=colour, linewidth=2.5, zorder=4)
            ax.text(x + offset, r + 1.5, f"{r:.0f}%\nn={n:,}",
                    ha="center", va="bottom", fontsize=7.5, zorder=4)

    if not any_data:
        ax.text(0.5, 0.5, "no max-stage coverage", ha="center", va="center",
                transform=ax.transAxes, color="#888888")
        ax.set_axis_off()
        return

    ax.set_xticks(xs)
    ax.set_xticklabels(classes, fontsize=9)
    ax.set_ylabel("% stage IV (of side-covered)")
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("Stage IV rate, before vs after ADT start")
    ax.legend(fontsize=8, frameon=False, title="side", title_fontsize=8)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)


def plot_stage_upstaging(ax, labelled: pl.DataFrame) -> None:
    """Share upstaged after ADT start, among patients covered on both sides.

    Denominator is deliberately narrow: upstaging is only observable for a
    patient with a staging note on each side, so patients missing either side
    are excluded rather than counted as not-upstaged.
    """
    if "STAGE_UPSTAGED_AFTER_ADT" not in labelled.columns:
        ax.text(0.5, 0.5, "no max-stage coverage", ha="center", va="center",
                transform=ax.transAxes, color="#888888")
        ax.set_axis_off()
        return

    classes = _class_order_present(labelled)
    rates, ns = [], []
    for cls in classes:
        covered = labelled.filter(
            (pl.col(INTENT_COL) == cls)
            & pl.col("STAGE_UPSTAGED_AFTER_ADT").is_not_null()
        )
        ns.append(covered.height)
        rates.append(
            float(covered["STAGE_UPSTAGED_AFTER_ADT"].mean()) * 100
            if covered.height else 0.0
        )

    if not any(ns):
        ax.text(0.5, 0.5, "no patient staged on both sides",
                ha="center", va="center", transform=ax.transAxes, color="#888888")
        ax.set_axis_off()
        return

    ax.bar(range(len(classes)), rates, width=0.55,
           color=[INTENT_COLORS.get(c, "#777777") for c in classes], zorder=3)
    for x, r, n in zip(range(len(classes)), rates, ns):
        ax.text(x, r + 1.5, f"{r:.0f}%\nn={n:,}", ha="center", va="bottom",
                fontsize=8, zorder=4)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=9)
    ax.set_ylabel("% upstaged (of both-sides-covered)")
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("Upstaged after ADT start")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)


def plot_max_stage_panel(labelled: pl.DataFrame, figsize=(15, 4.6)):
    """Three-panel figure: max-stage mix, stage IV rate, upstaging.

    Returns (fig, axes). Caller saves; nothing is written here.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    plot_max_stage_distribution(axes[0], labelled)
    plot_max_stage_iv_rate(axes[1], labelled)
    plot_stage_upstaging(axes[2], labelled)
    fig.tight_layout()
    return fig, axes


def plot_stage_metburden_panel(labelled: pl.DataFrame, figsize=(15, 4.6)):
    """Three-panel figure: stage mix, burden distribution, site pattern.

    Returns (fig, axes). Caller saves; nothing is written here.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    plot_stage_distribution(axes[0], labelled)
    plot_met_burden_distribution(axes[1], labelled)
    plot_met_site_pattern(axes[2], labelled)
    fig.tight_layout()
    return fig, axes

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
