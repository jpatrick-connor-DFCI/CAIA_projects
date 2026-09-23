"""Score federated CAIA model bundles on local long-format data, on the cluster.

:mod:`federated_inference` scores a frame that already has the model's
covariates in it.  This module produces that frame.  It takes the long-format
COMPASS-style table a site actually holds -- one row per lab measurement, with
the ADT anchor and platinum dates -- rebuilds the landmark feature matrix the
federated run trained on, scores every model in a bundle, and (when the local
rows carry an outcome) reports how well the federated model discriminates here.

Why this is a separate module: the feature matrix must be built by the SAME
rules as the federated fit, and those rules are not in the bundle.  The bundle
carries the pooled *transform* (impute means, centers, scales) but not the
*derivation* -- how a lab measurement becomes ``Hemoglobin__mean``, which
measurements are pre-landmark, how durations are shifted onto the landmark
clock.  Those are reimplemented here from
``caia-project-compass/rhino_scripts/*/preprocessing.py``, which is the
authority; the docstrings name the function each rule comes from so the two can
be diffed when the federated side changes.

The rules that matter, and that are easy to get silently wrong:

* **Pre-landmark** means ``days_relative_to_anchor < landmark``, strictly.  At
  landmark 0 the labs drawn the day ADT started are NOT available.
* **Feature names** are ``{sanitized_lab_name}__{stat}`` with ``[^\\w] -> _``,
  over stats ``mean/min/max/last/delta/n_observations``.  ``delta`` is
  ``last - first`` and is NaN with fewer than two observations.
* **Patients with no pre-landmark labs are kept**, all-NaN, because that is what
  makes the bundle's ``__missing`` indicators informative.
* **Durations** are shifted by the landmark, non-positive durations are dropped,
  and follow-up is administratively censored at 3650 days -- an event past the
  horizon becomes a censored observation AT the horizon.
* **The pooled preprocessing is never refit locally.**  That happens in
  :mod:`federated_inference`; this module only assembles its input.

CLI -- score one bundle at every landmark it contains::

    python -m survival_common.federated_scoring \\
        --bundle xgboost_federated_model_adt.json \\
        --data compass_long.parquet \\
        --output-dir risk_scores/

With an evaluation against the local outcome, and both cohort arms::

    python -m survival_common.federated_scoring \\
        --bundle cox_federated_elasticnet_model_adt.json \\
        --data compass_long.parquet \\
        --exclusion pre_anchor_castrate \\
        --evaluate \\
        --output-dir risk_scores/
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from survival_common.federated_inference import (
    BundleError,
    load_bundle,
    predict_risk,
    predict_survival,
)

# --------------------------------------------------------------------------- #
# Column names and constants, mirroring rhino_scripts/*/io_utils.py
# --------------------------------------------------------------------------- #
ID_COL = "person_id"
AGE_COL = "age_at_anchor"
AGE_AT_DIAGNOSIS_COL = "age_at_diagnosis"
LAB_NAME_COL = "lab_name"
LAB_VALUE_COL = "lab_value"

DIAGNOSIS_DATE_COL = "diagnosis_date"
ANCHOR_DATE_COL = "adt_start_date_post_diagnosis"
PLATINUM_DATE_COL = "platinum_start_date"
LAST_FOLLOWUP_DATE_COL = "last_followup_date"
MEASUREMENT_DATE_COL = "measurement_date"

ANCHOR_DAYS_COL = "days_relative_to_anchor"
DIAGNOSIS_TO_ANCHOR_COL = "days_diagnosis_to_anchor"

EVENT_COL = "event_platinum"
DURATION_COL = "time_to_platinum_or_censor"

DATE_COLS = [
    DIAGNOSIS_DATE_COL,
    ANCHOR_DATE_COL,
    PLATINUM_DATE_COL,
    LAST_FOLLOWUP_DATE_COL,
    MEASUREMENT_DATE_COL,
]

DAYS_PER_YEAR = 365.25
MIN_DELTA_OBS = 2
DEFAULT_MAX_FOLLOWUP_DAYS = 3650

# preprocessing.py: CASTRATE_TESTOSTERONE_NG_DL / DEFAULT_CASTRATE_LAB_NAME
CASTRATE_TESTOSTERONE_NG_DL = 50.0
DEFAULT_CASTRATE_LAB_NAME = "Testosterone [Mass/volume] in Serum or Plasma"

LAB_STATS = ("mean", "min", "max", "last", "delta", "n_observations")


# --------------------------------------------------------------------------- #
# Timing columns (port of io_utils.derive_analysis_columns)
# --------------------------------------------------------------------------- #
def derive_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ADT-anchored timing columns and age at anchor.

    Port of ``io_utils.derive_analysis_columns``.  Returns the frame unchanged
    when the derived columns are already present, so a caller may pass either a
    raw extract or one that has been through the federated pipeline.

    ``event_platinum`` is recomputed from ``platinum_start_date.notna()`` rather
    than trusted from the input, so the indicator can never disagree with the
    date that defines the duration.
    """
    derived = {ANCHOR_DAYS_COL, DIAGNOSIS_TO_ANCHOR_COL, AGE_COL, DURATION_COL}
    if derived <= set(df.columns):
        return df

    df = df.copy()
    for column in DATE_COLS:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    def _days(end: str, start: str) -> pd.Series:
        return (df[end] - df[start]).dt.days

    if {MEASUREMENT_DATE_COL, ANCHOR_DATE_COL} <= set(df.columns):
        df[ANCHOR_DAYS_COL] = _days(MEASUREMENT_DATE_COL, ANCHOR_DATE_COL)

    if {ANCHOR_DATE_COL, DIAGNOSIS_DATE_COL} <= set(df.columns):
        df[DIAGNOSIS_TO_ANCHOR_COL] = _days(ANCHOR_DATE_COL, DIAGNOSIS_DATE_COL)
        if AGE_AT_DIAGNOSIS_COL in df.columns:
            df[AGE_COL] = (
                pd.to_numeric(df[AGE_AT_DIAGNOSIS_COL], errors="coerce")
                + df[DIAGNOSIS_TO_ANCHOR_COL] / DAYS_PER_YEAR
            )

    if {PLATINUM_DATE_COL, LAST_FOLLOWUP_DATE_COL, ANCHOR_DATE_COL} <= set(df.columns):
        platinum_days = _days(PLATINUM_DATE_COL, ANCHOR_DATE_COL)
        censor_days = _days(LAST_FOLLOWUP_DATE_COL, ANCHOR_DATE_COL)
        df[DURATION_COL] = platinum_days.where(
            df[PLATINUM_DATE_COL].notna(), censor_days
        )
        df[EVENT_COL] = df[PLATINUM_DATE_COL].notna().astype(int)

    return df


# --------------------------------------------------------------------------- #
# Cohort exclusion (port of preprocessing.apply_exclusion)
# --------------------------------------------------------------------------- #
def apply_exclusion(
    df: pd.DataFrame,
    kind: str = "none",
    *,
    castrate_lab_name: str = DEFAULT_CASTRATE_LAB_NAME,
    threshold_ng_dl: float = CASTRATE_TESTOSTERONE_NG_DL,
) -> tuple[pd.DataFrame, dict]:
    """Apply a cohort exclusion to the long frame, before landmarking.

    Port of ``preprocessing.apply_exclusion``.  ``pre_anchor_castrate`` drops
    every patient with a testosterone below ``threshold_ng_dl`` recorded
    STRICTLY before the ADT anchor: such a patient was androgen-deprived
    elsewhere first, so their recorded ADT start is a transfer-of-care artifact.
    A same-day result may already reflect the first dose, hence the strict ``<``.

    Match a bundle's cohort arm here: a bundle trained on the no-precastrate
    cohort should be scored on the same exclusion, or the comparison drifts.
    """
    df = derive_analysis_columns(df)
    stats = {
        "exclusion": kind,
        "n_patients_before": int(df[ID_COL].nunique()) if ID_COL in df.columns else 0,
        "n_patients_with_pre_anchor_testosterone": 0,
        "n_patients_excluded": 0,
        "n_patients_after": 0,
    }

    if kind in ("none", "", None):
        stats["n_patients_after"] = stats["n_patients_before"]
        return df, stats

    if kind != "pre_anchor_castrate":
        raise ValueError(
            f"apply_exclusion: unknown exclusion {kind!r}; "
            "expected 'none' or 'pre_anchor_castrate'."
        )

    required = {LAB_NAME_COL, LAB_VALUE_COL, ANCHOR_DAYS_COL}
    if not required <= set(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise ValueError(f"apply_exclusion missing columns: {missing}")

    values = pd.to_numeric(df[LAB_VALUE_COL], errors="coerce")
    anchor_days = pd.to_numeric(df[ANCHOR_DAYS_COL], errors="coerce")
    # Matched on the RAW lab name, before any sanitization -- same as federated.
    is_testosterone = (
        df[LAB_NAME_COL].astype(str).str.strip().eq(str(castrate_lab_name).strip())
        & values.notna()
        & anchor_days.notna()
    )

    pre_anchor = is_testosterone & anchor_days.lt(0)
    stats["n_patients_with_pre_anchor_testosterone"] = int(
        df.loc[pre_anchor, ID_COL].nunique()
    )

    castrate_ids = set(df.loc[pre_anchor & values.lt(float(threshold_ng_dl)), ID_COL])
    stats["n_patients_excluded"] = len(castrate_ids)

    out = df[~df[ID_COL].isin(castrate_ids)].copy()
    stats["n_patients_after"] = int(out[ID_COL].nunique())
    return out, stats


# --------------------------------------------------------------------------- #
# Feature matrix (port of preprocessing.py's build_feature_matrix chain)
# --------------------------------------------------------------------------- #
def extract_static_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One row per patient: age at anchor and the platinum outcome.

    Port of ``preprocessing.extract_static_frame``.  Patients without a usable
    ADT anchor or without age are dropped -- they have no time origin, so no
    landmark can be defined for them.
    """
    df = derive_analysis_columns(df)
    cols = [c for c in (ID_COL, AGE_COL, EVENT_COL, DURATION_COL) if c in df.columns]
    static = df[cols].drop_duplicates(subset=[ID_COL]).reset_index(drop=True)

    if ANCHOR_DATE_COL in df.columns:
        anchored = df.loc[df[ANCHOR_DATE_COL].notna(), ID_COL].unique()
        static = static[static[ID_COL].isin(anchored)]
    if AGE_COL in static.columns:
        static = static[static[AGE_COL].notna()]

    return static.reset_index(drop=True)


def filter_pre_landmark(df: pd.DataFrame, landmark_day: int) -> pd.DataFrame:
    """Keep measurements STRICTLY before the landmark, no lookback bound.

    Port of ``preprocessing.filter_pre_landmark``.  The strict inequality is the
    leakage guard: at landmark 0, labs drawn the day ADT started are excluded.
    """
    df = derive_analysis_columns(df)
    anchor_days = pd.to_numeric(df[ANCHOR_DAYS_COL], errors="coerce")
    return df[anchor_days.notna() & anchor_days.lt(float(landmark_day))].copy()


def aggregate_lab_features(pre_df: pd.DataFrame) -> pd.DataFrame:
    """Per (person_id, lab_name): mean, min, max, last, delta, n_observations.

    Port of ``preprocessing.aggregate_lab_features``.  Sorted by anchor_days
    before aggregating so ``last`` is chronological rather than row-order;
    ``delta = last - first``, NaN below ``MIN_DELTA_OBS`` observations.
    """
    pre_df = pre_df.copy()
    pre_df[LAB_VALUE_COL] = pd.to_numeric(pre_df[LAB_VALUE_COL], errors="coerce")
    pre_df[LAB_NAME_COL] = pre_df[LAB_NAME_COL].astype(str).str.strip()
    pre_df = pre_df.dropna(subset=[ID_COL, LAB_NAME_COL, LAB_VALUE_COL, ANCHOR_DAYS_COL])

    if pre_df.empty:
        return pd.DataFrame(columns=[ID_COL, LAB_NAME_COL, *LAB_STATS])

    pre_sorted = pre_df.sort_values([ID_COL, LAB_NAME_COL, ANCHOR_DAYS_COL])
    agg = (
        pre_sorted.groupby([ID_COL, LAB_NAME_COL], sort=False)[LAB_VALUE_COL]
        .agg(
            mean="mean", min="min", max="max",
            _first="first", last="last", n_observations="count",
        )
        .reset_index()
    )
    agg["delta"] = np.where(
        agg["n_observations"] >= MIN_DELTA_OBS, agg["last"] - agg["_first"], np.nan
    )
    return agg.drop(columns=["_first"])


def make_outcome(
    static_df: pd.DataFrame,
    landmark_day: int,
    *,
    max_followup_days: int | None = DEFAULT_MAX_FOLLOWUP_DAYS,
) -> tuple[pd.DataFrame, dict]:
    """Shift durations onto the landmark clock and administratively censor.

    Port of ``preprocessing.make_outcome``.  Patients whose shifted duration is
    not strictly positive are dropped -- an event at or before the landmark is
    not predictable from it -- and an event past ``max_followup_days`` becomes a
    censored observation AT the horizon.

    When the frame carries no outcome columns (scoring patients whose outcome is
    unknown, which is the point of inference) the shift is skipped and every
    patient is kept.
    """
    out = static_df.copy()
    stats = {
        "landmark_days": int(landmark_day),
        "n_patients_before": int(len(out)),
        "n_dropped_non_positive_duration": 0,
        "n_admin_censored": 0,
        "n_patients": 0,
        "n_events": 0,
        "has_outcome": False,
    }

    if DURATION_COL not in out.columns or EVENT_COL not in out.columns:
        stats["n_patients"] = int(len(out))
        return out.reset_index(drop=True), stats

    stats["has_outcome"] = True
    out[DURATION_COL] = (
        pd.to_numeric(out[DURATION_COL], errors="coerce") - float(landmark_day)
    )
    out[EVENT_COL] = pd.to_numeric(out[EVENT_COL], errors="coerce").fillna(0).astype(int)

    valid = out[DURATION_COL].notna() & out[DURATION_COL].gt(0)
    stats["n_dropped_non_positive_duration"] = int((~valid).sum())
    out = out[valid].copy()

    if max_followup_days is not None:
        horizon = float(max_followup_days)
        past = out[EVENT_COL].eq(1) & out[DURATION_COL].gt(horizon)
        stats["n_admin_censored"] = int(past.sum())
        out.loc[past, EVENT_COL] = 0
        out[DURATION_COL] = out[DURATION_COL].clip(upper=horizon)

    stats["n_patients"] = int(len(out))
    stats["n_events"] = int(out[EVENT_COL].sum())
    return out.reset_index(drop=True), stats


def build_wide_frame(agg: pd.DataFrame, outcome_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot aggregated lab stats wide and LEFT JOIN onto the outcome rows.

    Port of ``preprocessing.build_wide_frame``.  Columns are
    ``{sanitized_lab_name}__{stat}``.  The join is left from the outcome side, so
    a patient with no pre-landmark labs is kept with all-NaN features -- which
    is what makes the bundle's ``__missing`` indicators informative.
    """
    if agg.empty:
        return outcome_df.copy().reset_index(drop=True)

    stat_cols = [c for c in agg.columns if c not in (ID_COL, LAB_NAME_COL)]
    melted = agg.melt(
        id_vars=[ID_COL, LAB_NAME_COL], value_vars=stat_cols,
        var_name="stat", value_name="value",
    )
    melted["feature"] = (
        melted[LAB_NAME_COL].str.replace(r"[^\w]", "_", regex=True)
        + "__" + melted["stat"]
    )

    wide_labs = melted.pivot(index=ID_COL, columns="feature", values="value").reset_index()
    wide_labs.columns.name = None
    wide_labs = wide_labs.sort_index(axis=1)

    return outcome_df.merge(wide_labs, on=ID_COL, how="left").reset_index(drop=True)


def build_landmark_frame(
    long_df: pd.DataFrame,
    landmark_day: int,
    *,
    max_followup_days: int | None = DEFAULT_MAX_FOLLOWUP_DAYS,
) -> tuple[pd.DataFrame, dict]:
    """Long-format rows -> the wide frame a bundle can be scored on.

    Chains the federated pipeline in its own order: static frame, pre-landmark
    filter, lab aggregation, outcome construction, wide pivot.  Returns the
    frame and the per-landmark stats, which are worth logging -- a large
    ``n_dropped_non_positive_duration`` means the landmark is past most of the
    local cohort's follow-up.
    """
    long_df = derive_analysis_columns(long_df)
    static = extract_static_frame(long_df)
    outcome, stats = make_outcome(
        static, landmark_day, max_followup_days=max_followup_days
    )
    pre = filter_pre_landmark(long_df, landmark_day)
    pre = pre[pre[ID_COL].isin(outcome[ID_COL])]
    agg = aggregate_lab_features(pre)
    wide = build_wide_frame(agg, outcome)
    stats["n_labs"] = int(agg[LAB_NAME_COL].nunique()) if not agg.empty else 0
    stats["n_feature_cols"] = int(
        sum(1 for c in wide.columns if "__" in c)
    )
    return wide, stats


# --------------------------------------------------------------------------- #
# Coverage diagnostics
# --------------------------------------------------------------------------- #
def covariate_coverage(wide: pd.DataFrame, model: dict) -> pd.DataFrame:
    """Per-covariate local availability for one model.

    The single most useful diagnostic before trusting a transported score.  A
    covariate absent from the local frame is imputed to the pooled train mean,
    which is silent and legitimate for a few features but means something is
    wrong with the extract when it is most of them -- typically a lab
    dictionary that does not match the training site's.
    """
    pre = model["preprocessing"]
    cols = list(pre["covariate_cols"])
    source = dict(pre.get("source_cols") or {})
    missing_cols = set(pre.get("missing_indicator_cols") or [])

    rows = []
    for col in cols:
        base = col[: -len("__missing")] if col in missing_cols else col
        src = source.get(base, base)
        present = src in wide.columns
        n_obs = (
            int(pd.to_numeric(wide[src], errors="coerce").notna().sum())
            if present else 0
        )
        rows.append({
            "covariate": col,
            "source_col": src,
            "is_missing_indicator": col in missing_cols,
            "in_local_frame": present,
            "n_observed": n_obs,
            "frac_observed": (n_obs / len(wide)) if len(wide) else 0.0,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_risk(
    risk: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
) -> dict:
    """Local discrimination for a transported risk score.

    Harrell's C via lifelines' ``concordance_index``, which needs no training
    distribution and so is the honest metric for a model fitted elsewhere.
    Returns NaN rather than raising when the local rows cannot support it (no
    events, or a degenerate score).
    """
    risk = np.asarray(risk, dtype=float)
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=float)

    ok = np.isfinite(risk) & np.isfinite(duration) & np.isfinite(event)
    out = {
        "n": int(ok.sum()),
        "n_events": int(event[ok].sum()) if ok.any() else 0,
        "c_index": float("nan"),
    }
    if out["n"] < 2 or out["n_events"] < 1:
        return out

    try:
        from lifelines.utils import concordance_index
        # Higher risk must mean shorter survival, hence the negation.
        out["c_index"] = float(
            concordance_index(duration[ok], -risk[ok], event_observed=event[ok])
        )
    except Exception as exc:  # pragma: no cover - depends on local environment
        out["c_index_error"] = str(exc)
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def score_bundle(
    long_df: pd.DataFrame,
    bundle: dict,
    *,
    configs: tuple[str, ...] = ("both", "baseline"),
    landmarks: tuple[int, ...] | None = None,
    horizons_days: tuple[float, ...] = (),
    evaluate: bool = False,
    max_followup_days: int | None = DEFAULT_MAX_FOLLOWUP_DAYS,
    id_col: str = ID_COL,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score every model in a bundle, building each landmark's frame once.

    Returns ``(scores, metrics, coverage)``:

    * ``scores``    one row per patient per model, with the risk score
    * ``metrics``   one row per model: cohort counts and local C-index
    * ``coverage``  one row per model per covariate (the diagnostic above)

    The frame is rebuilt per landmark, not per model, because the landmark --
    not the model -- determines which measurements are visible.
    """
    wanted = [
        m for m in bundle["models"]
        if str(m.get("config")) in configs
        and (landmarks is None or int(m["landmark_days"]) in landmarks)
    ]
    if not wanted:
        available = sorted(
            {(int(m["landmark_days"]), str(m.get("config"))) for m in bundle["models"]}
        )
        raise BundleError(
            f"no model matched configs={list(configs)} landmarks={landmarks}; "
            f"bundle has {available}"
        )

    family = bundle.get("model_family")
    label = bundle.get("analysis_label")
    score_rows, metric_rows, coverage_rows = [], [], []

    for lm in sorted({int(m["landmark_days"]) for m in wanted}):
        wide, stats = build_landmark_frame(
            long_df, lm, max_followup_days=max_followup_days
        )
        if verbose:
            print(
                f"[federated_scoring] landmark {lm}d: {stats['n_patients']:,} patients, "
                f"{stats['n_events']:,} events, {stats['n_feature_cols']:,} feature cols "
                f"(dropped {stats['n_dropped_non_positive_duration']:,} non-positive "
                f"duration, admin-censored {stats['n_admin_censored']:,})"
            )
        if wide.empty:
            print(f"[federated_scoring] landmark {lm}d: no patients, skipped")
            continue

        for model in [m for m in wanted if int(m["landmark_days"]) == lm]:
            config = str(model.get("config"))
            risk = predict_risk(wide, bundle, model, age_col=AGE_COL)

            block = pd.DataFrame({
                "landmark_days": lm,
                "config": config,
                "model_family": family,
                "analysis_label": label,
                "risk_score": risk,
            })
            if id_col in wide.columns:
                block.insert(0, id_col, wide[id_col].to_numpy())
            if stats["has_outcome"]:
                block[DURATION_COL] = wide[DURATION_COL].to_numpy()
                block[EVENT_COL] = wide[EVENT_COL].to_numpy()

            if horizons_days and model.get("baseline_cumhaz"):
                surv = predict_survival(risk, model, horizons_days)
                block = pd.concat([block, surv.set_index(block.index)], axis=1)
            elif horizons_days and verbose:
                print(
                    f"[federated_scoring]   landmark {lm}d config={config}: no "
                    "baseline hazard in the bundle, survival probabilities skipped"
                )

            score_rows.append(block)

            cov = covariate_coverage(wide, model)
            cov.insert(0, "config", config)
            cov.insert(0, "landmark_days", lm)
            coverage_rows.append(cov)

            row = {
                "landmark_days": lm,
                "config": config,
                "model_family": family,
                "analysis_label": label,
                "n_patients": stats["n_patients"],
                "n_events": stats["n_events"],
                "n_covariates": len(model["preprocessing"]["covariate_cols"]),
                "n_covariates_absent_locally": int((~cov["in_local_frame"]).sum()),
                "risk_mean": float(np.nanmean(risk)) if len(risk) else float("nan"),
                "risk_sd": float(np.nanstd(risk)) if len(risk) else float("nan"),
            }
            if evaluate and stats["has_outcome"]:
                row.update(evaluate_risk(
                    risk, wide[DURATION_COL].to_numpy(), wide[EVENT_COL].to_numpy()
                ))
            metric_rows.append(row)

            if verbose:
                c = row.get("c_index")
                msg = f"[federated_scoring]   landmark {lm}d config={config}"
                if c is not None and np.isfinite(c):
                    msg += f": C-index {c:.4f}"
                if row["n_covariates_absent_locally"]:
                    msg += (
                        f"  [{row['n_covariates_absent_locally']}/"
                        f"{row['n_covariates']} covariates absent locally]"
                    )
                print(msg)

    scores = (
        pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()
    )
    metrics = pd.DataFrame(metric_rows)
    coverage = (
        pd.concat(coverage_rows, ignore_index=True) if coverage_rows else pd.DataFrame()
    )
    return scores, metrics, coverage


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def read_table(path: str) -> pd.DataFrame:
    """Read a local table, choosing the reader from the extension."""
    if str(path).endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    if str(path).endswith((".tsv", ".tsv.gz")):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Score federated CAIA model bundles on local long-format data."
    )
    ap.add_argument("--bundle", required=True, nargs="+",
                    help="one or more model bundle JSONs (xgboost and/or elastic-net)")
    ap.add_argument("--data", required=True,
                    help="long-format table (.csv/.tsv/.parquet), one row per measurement")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--exclusion", default="none",
                    choices=["none", "pre_anchor_castrate"],
                    help="cohort arm; match the bundle's analysis_label")
    ap.add_argument("--landmarks", default="",
                    help="comma-separated days; default is every landmark in the bundle")
    ap.add_argument("--configs", default="both,baseline",
                    help="comma-separated: both (full model) and/or baseline (age-only)")
    ap.add_argument("--horizons", default="",
                    help="comma-separated days for S(t|x); elastic-net bundles only")
    ap.add_argument("--max-followup-days", type=int, default=DEFAULT_MAX_FOLLOWUP_DAYS)
    ap.add_argument("--evaluate", action="store_true",
                    help="compute the local C-index (requires outcome columns)")
    ap.add_argument("--id-col", default=ID_COL)
    args = ap.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    long_df = read_table(args.data)
    print(f"[federated_scoring] read {len(long_df):,} rows from {args.data}")
    if args.id_col not in long_df.columns:
        print(
            f"[federated_scoring] ERROR: identifier column {args.id_col!r} is not in "
            f"the input frame; columns are {list(long_df.columns)[:20]}",
            file=sys.stderr,
        )
        return 2

    long_df, excl = apply_exclusion(long_df, args.exclusion)
    print(
        f"[federated_scoring] exclusion={excl['exclusion']}: "
        f"{excl['n_patients_before']:,} -> {excl['n_patients_after']:,} patients "
        f"({excl['n_patients_excluded']:,} excluded)"
    )

    landmarks = (
        tuple(int(x) for x in args.landmarks.split(",") if x.strip())
        if args.landmarks.strip() else None
    )
    configs = tuple(c.strip() for c in args.configs.split(",") if c.strip())
    horizons = tuple(float(h) for h in args.horizons.split(",") if h.strip())

    all_scores, all_metrics, all_coverage = [], [], []
    for path in args.bundle:
        bundle = load_bundle(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        print(
            f"\n[federated_scoring] {stem}: family={bundle.get('model_family')} "
            f"label={bundle.get('analysis_label')} "
            f"({len(bundle['models'])} fitted model(s))"
        )
        if bundle.get("analysis_label") and args.exclusion != "none" \
                and "noprecastrate" not in str(bundle["analysis_label"]):
            print(
                f"[federated_scoring]   WARNING: bundle label "
                f"{bundle['analysis_label']!r} looks like the full ADT cohort but "
                f"--exclusion={args.exclusion} was given; the cohorts will not match"
            )

        scores, metrics, coverage = score_bundle(
            long_df, bundle,
            configs=configs, landmarks=landmarks, horizons_days=horizons,
            evaluate=args.evaluate, max_followup_days=args.max_followup_days,
            id_col=args.id_col,
        )
        for frame in (scores, metrics, coverage):
            if not frame.empty:
                frame.insert(0, "bundle", stem)
        all_scores.append(scores)
        all_metrics.append(metrics)
        all_coverage.append(coverage)

        if not scores.empty:
            out = os.path.join(args.output_dir, f"{stem}_risk_scores.csv")
            scores.to_csv(out, index=False)
            print(f"[federated_scoring] wrote {len(scores):,} rows -> {out}")

    def _write(frames, name):
        frames = [f for f in frames if not f.empty]
        if not frames:
            return
        df = pd.concat(frames, ignore_index=True)
        out = os.path.join(args.output_dir, name)
        df.to_csv(out, index=False)
        print(f"[federated_scoring] wrote {len(df):,} rows -> {out}")

    _write(all_metrics, "federated_inference_metrics.csv")
    _write(all_coverage, "federated_inference_coverage.csv")

    with open(os.path.join(args.output_dir, "federated_inference_run.json"), "w") as fh:
        json.dump({
            "data": args.data,
            "bundles": list(args.bundle),
            "exclusion": args.exclusion,
            "exclusion_stats": excl,
            "landmarks": list(landmarks) if landmarks else None,
            "configs": list(configs),
            "horizons_days": list(horizons),
            "max_followup_days": args.max_followup_days,
            "evaluated": bool(args.evaluate),
        }, fh, indent=2)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
