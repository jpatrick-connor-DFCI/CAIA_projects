"""Incremental value of accruing lab history for fixed-horizon platinum risk.

Answers the clinically motivating question behind the dynamic arm: *given this
patient's labs today, what is their 6-month risk, and does that estimate
sharpen as new labs arrive?*

Consumes the dynamic arm's ``dynamic_deephit_dyn_predictions_{config}.csv``
directly. **Nothing is retrained and nothing is re-inferred** -- one trained
dynamic model already emits a prediction at every observation time, and this
script is pure post-processing of that frame. It therefore needs neither torch
nor a GPU.

Two analyses, both reported:

**5a. Sequential landmarks** (``incremental_risk_by_landmark.csv``). Slice the
predictions at a grid of prediction times and score a fixed horizon (default 6
months) at each. Rising discrimination = accruing history sharpens the estimate.

**5b. Held-back-window ablation** (``incremental_risk_ablation.csv``). At each
prediction time ``t``, compare the model's own prediction from step ``t`` with
its prediction from step ``t - delta``, scored against *the same outcome and the
same patients*. This isolates the marginal value of the NEWEST labs from the
value of history length in general.

5b needs no separate inference pass because the GRU is causal: the prediction
the model already emitted at step ``t - delta`` is, by construction, its
estimate from history truncated at ``t - delta``. That causality is asserted
empirically in ``tests/test_dynamic_no_future_leakage.py``, not assumed here.

**Lead with 5b.** In 5a the risk set shrinks as ``t`` grows and is increasingly
selected for patients who have *not* yet received platinum, so a rising curve
is partly a changing-population artifact and not a clean "the model improves"
claim. 5b holds the patient set and the outcome fixed at each ``t`` and varies
only the input window, which is the comparison that supports a causal reading.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
SURVIVAL_DIR = THIS_DIR.parent
SURVIVAL_PARENT = SURVIVAL_DIR.parent
REPO_ROOT = SURVIVAL_PARENT.parent
for _p in (str(REPO_ROOT), str(SURVIVAL_PARENT), str(SURVIVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from survival_common.metrics_schema import DEFAULT_COHORT  # noqa: E402

# Deliberately NOT order_canonical_first / the canonical metrics block.
# Invariant #9 governs `*_metrics_*.csv`: one model's held-out performance, one
# row per cause. These files are a different artifact -- a PAIRED comparison
# (auc_full_history vs auc_held_back) indexed by prediction time -- and they are
# named incremental_risk_*.csv to stay outside that namespace. Stamping them
# with NaN train-side twins to satisfy a schema they do not belong to would
# make them look like model-comparison rows they are not. They do carry the
# identity columns, which is the part a reader actually needs to join on.

# Matched to deephit_engine's by-time gate: a prediction time is reported only
# if its risk set can support a metric. See MIN_RISK_SET_FOR_BY_TIME there for
# why underpowered rows are emitted with a note rather than dropped.
MIN_RISK_SET = 25
MIN_EVENTS = 5
DEFAULT_HORIZON_DAYS = 182  # ~6 months
DEFAULT_DELTA_DAYS = 90
DEFAULT_GRID_DAYS = (0, 90, 180, 270, 360)


def _concordance(duration, event, risk):
    """Harrell C-index with risk oriented so higher = sooner event."""
    from lifelines.utils import concordance_index

    valid = np.isfinite(duration) & (duration > 0) & np.isfinite(risk)
    if valid.sum() == 0 or event[valid].sum() == 0:
        return float("nan")
    try:
        return float(concordance_index(duration[valid], -risk[valid], event[valid]))
    except ZeroDivisionError:
        # lifelines raises rather than returning NaN when no pair is orderable
        # (e.g. every retained row shares one duration, which happens readily at
        # a late prediction time once the risk set has thinned). A missing
        # C-index must not take the whole run down: the caller's underpowered
        # gate is what should speak for such a row.
        return float("nan")


def _binary_auc(label, risk):
    """AUC for the binary 'event within the horizon' outcome.

    Deliberately NOT the IPCW time-dependent AUC: within one prediction time the
    outcome here is already binarized at a single fixed horizon, and patients
    censored before that horizon are excluded (see `_horizon_outcome`), so there
    is no censoring left inside the comparison for IPCW to reweight. Using the
    plain AUC keeps 5a and 5b on the same footing and makes the two arms of 5b
    differ by input window alone.
    """
    from sklearn.metrics import roc_auc_score

    valid = np.isfinite(risk) & np.isfinite(label)
    if valid.sum() < 2 or len(np.unique(label[valid])) < 2:
        return float("nan")
    return float(roc_auc_score(label[valid], risk[valid]))


def _brier(label, risk):
    valid = np.isfinite(risk) & np.isfinite(label)
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean((risk[valid] - label[valid]) ** 2))


def _risk_column(pred: pd.DataFrame, event_idx: int, horizon_bin: int) -> str:
    """The CIF column at `horizon_bin`, or the total-risk fallback.

    The fallback is reported in the output's `risk_column` so a reader can tell
    whether the number is a genuine fixed-horizon risk or the time-constant
    total; conflating the two would silently change what is being measured.
    """
    col = f"event_{event_idx}_risk_h{int(horizon_bin)}"
    if col in pred.columns:
        return col
    return f"event_{event_idx}_risk_total"


def _horizon_outcome(block: pd.DataFrame, *, event_idx: int, horizon_bin: int):
    """Binarize each row's residual outcome at `horizon_bin`.

    Returns ``(label, keep)``. ``label`` is 1 iff the cause occurred within the
    horizon. ``keep`` excludes rows censored (or lost to a competing event)
    *before* the horizon, whose horizon outcome is genuinely unknown -- counting
    them as non-events would bias the risk downward, and it would bias it
    differently at different prediction times, which is exactly the artifact
    these analyses are meant to avoid.
    """
    duration = block["duration"].to_numpy(dtype=float)
    label_code = block["label"].to_numpy(dtype=int)
    had_event = (label_code == event_idx) & (duration <= horizon_bin)
    # Known-outcome rows: the event happened inside the window, or the patient
    # was followed (event-free or otherwise) at least to the window's end.
    known = had_event | (duration >= horizon_bin)
    return had_event.astype(int), known


def _score_block(block, *, event_idx, horizon_bin, risk_col):
    label, keep = _horizon_outcome(block, event_idx=event_idx, horizon_bin=horizon_bin)
    risk = block[risk_col].to_numpy(dtype=float)
    label_k, risk_k = label[keep], risk[keep]
    duration = block["duration"].to_numpy(dtype=float)[keep]
    event_any = (block["label"].to_numpy(dtype=int) == event_idx)[keep].astype(int)
    return {
        "n_scored": int(keep.sum()),
        "n_events": int(label_k.sum()),
        "auc": _binary_auc(label_k, risk_k),
        "brier": _brier(label_k, risk_k),
        "c_index": _concordance(duration, event_any, risk_k),
        "mean_risk": float(np.mean(risk_k)) if keep.sum() else float("nan"),
    }


def _underpowered(n_scored, n_events, *, min_risk_set, min_events):
    if n_scored < min_risk_set or n_events < min_events:
        return (
            f"underpowered: n_scored={n_scored} (min {min_risk_set}), "
            f"n_events={n_events} (min {min_events})"
        )
    return ""


def by_landmark(
    pred: pd.DataFrame,
    *,
    event_idx: int,
    event_name: str,
    horizon_bin: int,
    grid_bins: list[int],
    time_col: str = "TIME",
    min_risk_set: int = MIN_RISK_SET,
    min_events: int = MIN_EVENTS,
    time_unit_days: int = 7,
) -> pd.DataFrame:
    """5a: fixed-horizon performance at each prediction time on the grid."""
    risk_col = _risk_column(pred, event_idx, horizon_bin)
    rows = []
    available = np.sort(pred[time_col].astype(float).unique())
    for target in grid_bins:
        # Snap to the nearest materialized prediction time; the grid is in days
        # and need not land exactly on an observation bin.
        if len(available) == 0:
            break
        nearest = float(available[np.argmin(np.abs(available - float(target)))])
        block = pred.loc[np.isclose(pred[time_col].astype(float), nearest)]
        base = {
            "endpoint": event_name,
            "prediction_time": nearest,
            "prediction_day": nearest * float(time_unit_days),
            "requested_prediction_day": float(target) * float(time_unit_days),
            "horizon_bin": int(horizon_bin),
            "horizon_days": int(horizon_bin * time_unit_days),
            "risk_column": risk_col,
            "n_at_risk": int(len(block)),
        }
        if block.empty:
            rows.append({**base, "note": "no rows at this prediction time"})
            continue
        scored = _score_block(
            block, event_idx=event_idx, horizon_bin=horizon_bin, risk_col=risk_col
        )
        note = _underpowered(
            scored["n_scored"], scored["n_events"],
            min_risk_set=min_risk_set, min_events=min_events,
        )
        if note:
            scored = {**scored, "auc": np.nan, "brier": np.nan, "c_index": np.nan}
        rows.append({**base, **scored, "note": note})
    out = pd.DataFrame(rows)
    # A snapped grid can land twice on the same bin when the grid is finer than
    # the observation spacing; keep the first so the series has one row per time.
    if not out.empty:
        out = out.drop_duplicates(subset=["endpoint", "prediction_time"], keep="first")
    return out.reset_index(drop=True)


def ablation(
    pred: pd.DataFrame,
    *,
    event_idx: int,
    event_name: str,
    horizon_bin: int,
    delta_bins: int,
    id_col: str,
    time_col: str = "TIME",
    grid_bins: list[int] | None = None,
    min_risk_set: int = MIN_RISK_SET,
    min_events: int = MIN_EVENTS,
    time_unit_days: int = 7,
) -> pd.DataFrame:
    """5b: full history at ``t`` vs. history truncated at ``t - delta``.

    Both arms are scored on the SAME patients and the SAME outcome -- the one
    defined at ``t`` -- so the only thing that differs is how much history the
    model had seen. That is what makes the delta interpretable.

    The stale arm reuses the model's own prediction from step ``t - delta``,
    which is its estimate from history up to that step (the GRU is causal). Its
    risk is read at a horizon extended by ``delta`` so both arms target the same
    absolute point in time: a 6-month-ahead question asked ``delta`` earlier is a
    ``6 months + delta`` question.
    """
    rows = []
    fresh_risk_col = _risk_column(pred, event_idx, horizon_bin)
    # Same calendar target from `delta` earlier => a longer residual horizon.
    stale_horizon_bin = int(horizon_bin + delta_bins)
    stale_risk_col = _risk_column(pred, event_idx, stale_horizon_bin)

    pred = pred.copy()
    pred["__id"] = pred[id_col].map(str)
    pred["__t"] = pred[time_col].astype(float)
    by_id_time = pred.set_index(["__id", "__t"])

    available = np.sort(pred["__t"].unique())
    targets = (
        [float(t) for t in available]
        if grid_bins is None
        else [
            float(available[np.argmin(np.abs(available - float(g)))])
            for g in grid_bins
            if len(available)
        ]
    )
    for t in sorted(set(targets)):
        fresh = pred.loc[np.isclose(pred["__t"], t)]
        if fresh.empty:
            continue
        stale_t = t - float(delta_bins)
        # Patients must have a materialized prediction at BOTH steps. A patient
        # with no observation at t-delta is excluded rather than back-filled;
        # substituting a different step would silently vary the window.
        keys = [(i, stale_t) for i in fresh["__id"]]
        has_stale = np.array(
            [k in by_id_time.index for k in keys], dtype=bool
        )
        base = {
            "endpoint": event_name,
            "prediction_time": t,
            "prediction_day": t * float(time_unit_days),
            "delta_bins": int(delta_bins),
            "delta_days": int(delta_bins * time_unit_days),
            "horizon_bin": int(horizon_bin),
            "horizon_days": int(horizon_bin * time_unit_days),
            "fresh_risk_column": fresh_risk_col,
            "stale_risk_column": stale_risk_col,
            "n_at_risk": int(len(fresh)),
            "n_paired": int(has_stale.sum()),
        }
        if has_stale.sum() == 0:
            rows.append({**base, "note": f"no patient has a step at t-delta={stale_t}"})
            continue

        paired = fresh.loc[has_stale]
        stale_rows = by_id_time.loc[[(i, stale_t) for i in paired["__id"]]]

        # The outcome is the one defined at t, for both arms.
        label, keep = _horizon_outcome(
            paired, event_idx=event_idx, horizon_bin=horizon_bin
        )
        fresh_risk = paired[fresh_risk_col].to_numpy(dtype=float)[keep]
        stale_risk = stale_rows[stale_risk_col].to_numpy(dtype=float)[keep]
        label_k = label[keep]
        n_scored, n_events = int(keep.sum()), int(label_k.sum())

        note = _underpowered(
            n_scored, n_events, min_risk_set=min_risk_set, min_events=min_events
        )
        if note:
            rows.append({
                **base, "n_scored": n_scored, "n_events": n_events,
                "auc_full_history": np.nan, "auc_held_back": np.nan,
                "auc_gain": np.nan, "brier_full_history": np.nan,
                "brier_held_back": np.nan, "brier_gain": np.nan, "note": note,
            })
            continue

        auc_fresh = _binary_auc(label_k, fresh_risk)
        auc_stale = _binary_auc(label_k, stale_risk)
        brier_fresh = _brier(label_k, fresh_risk)
        brier_stale = _brier(label_k, stale_risk)
        rows.append({
            **base,
            "n_scored": n_scored,
            "n_events": n_events,
            "auc_full_history": auc_fresh,
            "auc_held_back": auc_stale,
            # Positive => the newest `delta` of labs added discrimination.
            "auc_gain": auc_fresh - auc_stale,
            "brier_full_history": brier_fresh,
            "brier_held_back": brier_stale,
            # Positive => the newest labs improved calibration (Brier fell).
            "brier_gain": brier_stale - brier_fresh,
            "mean_risk_full_history": float(np.mean(fresh_risk)),
            "mean_risk_held_back": float(np.mean(stale_risk)),
            "note": "",
        })
    return pd.DataFrame(rows).reset_index(drop=True)


# Identity columns, front-loaded. Same spellings as the canonical identity
# block so these files join cleanly against the metrics CSVs.
_IDENTITY = ["model", "cohort", "endpoint", "landmark_days", "config"]


def _stamp(df: pd.DataFrame, *, args) -> pd.DataFrame:
    """Front-load the identity columns so these files self-describe."""
    if df.empty:
        return df
    out = df.copy()
    out["model"] = "dynamic-deephit-dyn"
    out["cohort"] = str(args.cohort) if args.cohort else DEFAULT_COHORT
    out["landmark_days"] = int(args.landmark_day)
    out["config"] = args.config
    rest = [c for c in out.columns if c not in set(_IDENTITY)]
    return out.loc[:, _IDENTITY + rest]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Incremental value of accruing lab history, from the dynamic arm's "
            "per-timepoint predictions. Post-processing only: no refit, no torch."
        )
    )
    p.add_argument(
        "--predictions",
        type=Path,
        help=(
            "Path to dynamic_deephit_dyn_predictions_{config}.csv. Defaults to "
            "the conventional name inside --output-dir."
        ),
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config", default="platinum")
    p.add_argument("--landmark-day", type=int, default=0)
    p.add_argument("--cohort", default="all")
    p.add_argument(
        "--endpoint", default="platinum",
        help="Cause of interest. Maps to event index 1 unless --event-index is given.",
    )
    p.add_argument(
        "--event-index", type=int, default=1,
        help="1-based cause index in the prediction frame's event_{k}_* columns.",
    )
    p.add_argument("--id-col", default="DFCI_MRN")
    p.add_argument("--time-col", default="TIME")
    p.add_argument("--time-unit-days", type=int, default=7)
    p.add_argument(
        "--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS,
        help="Fixed prediction horizon (default ~6 months).",
    )
    p.add_argument(
        "--delta-days", type=int, default=DEFAULT_DELTA_DAYS,
        help="Held-back window for the 5b ablation.",
    )
    p.add_argument(
        "--grid-days", type=int, nargs="*", default=list(DEFAULT_GRID_DAYS),
        help="Prediction times to report, in days from the landmark.",
    )
    p.add_argument("--min-risk-set", type=int, default=MIN_RISK_SET)
    p.add_argument("--min-events", type=int, default=MIN_EVENTS)
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    return p


def main(args) -> None:
    output_dir = Path(args.output_dir)
    pred_path = args.predictions or (
        output_dir / f"dynamic_deephit_dyn_predictions_{args.config}.csv"
    )
    by_landmark_path = output_dir / "incremental_risk_by_landmark.csv"
    ablation_path = output_dir / "incremental_risk_ablation.csv"
    if not args.overwrite and by_landmark_path.exists() and ablation_path.exists():
        print(f"[skip] {by_landmark_path} exists (pass --overwrite to recompute)")
        return

    if not Path(pred_path).exists():
        raise FileNotFoundError(
            f"Missing {pred_path}. Run dynamic_deephit.py --dynamic first "
            "(and build inputs with --longitudinal-full-followup)."
        )
    pred = pd.read_csv(pred_path)
    if args.time_col not in pred.columns:
        raise ValueError(
            f"{pred_path} has no {args.time_col!r} column, so it is a landmark "
            "prediction frame, not a dynamic one. These analyses need the "
            "per-timepoint predictions written by --dynamic."
        )

    unit = int(args.time_unit_days)
    horizon_bin = max(1, int(round(args.horizon_days / unit)))
    delta_bins = max(1, int(round(args.delta_days / unit)))
    grid_bins = [int(round(d / unit)) for d in args.grid_days]

    print(
        f"Loaded {len(pred):,} prediction rows across "
        f"{pred[args.time_col].nunique()} prediction times from {pred_path}"
    )
    print(
        f"horizon={args.horizon_days}d (bin {horizon_bin}), "
        f"delta={args.delta_days}d (bin {delta_bins}), "
        f"grid={list(args.grid_days)}d"
    )

    common = dict(
        event_idx=int(args.event_index),
        event_name=args.endpoint,
        horizon_bin=horizon_bin,
        time_col=args.time_col,
        min_risk_set=int(args.min_risk_set),
        min_events=int(args.min_events),
        time_unit_days=unit,
    )
    landmark_df = by_landmark(pred, grid_bins=grid_bins, **common)
    ablation_df = ablation(
        pred, delta_bins=delta_bins, id_col=args.id_col, grid_bins=grid_bins, **common
    )

    landmark_df = _stamp(landmark_df, args=args)
    ablation_df = _stamp(ablation_df, args=args)

    output_dir.mkdir(parents=True, exist_ok=True)
    landmark_df.to_csv(by_landmark_path, index=False)
    ablation_df.to_csv(ablation_path, index=False)

    manifest = {
        "predictions": str(pred_path),
        "config": args.config,
        "landmark_day": int(args.landmark_day),
        "cohort": args.cohort,
        "endpoint": args.endpoint,
        "horizon_days": int(args.horizon_days),
        "horizon_bin": horizon_bin,
        "delta_days": int(args.delta_days),
        "delta_bins": delta_bins,
        "grid_days": [int(d) for d in args.grid_days],
        "time_unit_days": unit,
        "n_prediction_rows": int(len(pred)),
        "n_prediction_times": int(pred[args.time_col].nunique()),
        "min_risk_set": int(args.min_risk_set),
        "min_events": int(args.min_events),
        "interpretation": (
            "Lead with the ablation. by_landmark's risk set shrinks and is "
            "selected for patients still event-free, so a rising curve there is "
            "partly a changing-population artifact. The ablation holds patients "
            "and outcome fixed at each prediction time and varies only the "
            "input window."
        ),
    }
    manifest_path = output_dir / "incremental_risk_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("\nSaved:")
    for path in (by_landmark_path, ablation_path, manifest_path):
        print(f"  {path}")

    if not ablation_df.empty:
        usable = ablation_df.loc[ablation_df["note"].eq("")]
        if not usable.empty:
            print("\n5b held-back-window ablation (lead with this):")
            cols = ["prediction_day", "n_scored", "n_events",
                    "auc_held_back", "auc_full_history", "auc_gain"]
            print(usable[cols].to_string(index=False))
        else:
            print("\n[warn] every ablation row was underpowered; nothing to report.")


if __name__ == "__main__":
    main(build_parser().parse_args())
