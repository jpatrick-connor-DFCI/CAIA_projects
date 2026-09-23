"""Transport federated platinum models to local NEPC / AVPC endpoints.

The federated bundles (caia-project-compass/rhino_scripts) are trained on time
to platinum only -- NEPC and AVPC are not annotated at the federated sites.
DFCI has fine-grained LLM-derived AVPC/NEPC criteria timelines, so this module
asks: does the platinum-trained risk score also rank patients by time to NEPC
and time to AVPC on the full local ADT cohort?

Pipeline:

1. Read the Stage-2 ADT long frame (``longitudinal_prediction_data_adt.csv``)
   and map each local canonical ``LAB_NAME`` to the OMOP concept name the
   federated features are named after (``OMOP_to_DFCI_lab_ids.csv``).
2. For each endpoint, write that endpoint's event date into the federated
   chain's event slot (``platinum_start_date``) and build every landmark frame
   with :mod:`federated_scoring`, which is parity-tested against rhino_scripts.
   Reusing that chain unchanged means the features, the strict pre-landmark
   filter, the incident gate (events at/before the landmark are dropped) and
   the 3650-day administrative censoring are identical across endpoints.
3. Score every bundle model with the pooled preprocessing (never refit).
4. Evaluate: Harrell's C with a patient bootstrap CI, the paired C-index gain
   of the full model over the age-only baseline, Uno's C, cumulative/dynamic
   AUC(t), and a Cox slope of the endpoint on the standardized risk score.

Death is not a competing event here: NEPC/AVPC follow-up ends at last contact,
which is the death date for patients who died.  Platinum is scored alongside as
the reference (the model's own endpoint at a new site), which is what makes the
NEPC/AVPC numbers interpretable.

CLI::

    python -m survival_common.federated_endpoint_transfer \\
        --bundle xgboost_federated_model_adt.json \\
                 cox_federated_elasticnet_model_adt.json \\
        --data /data/gusev/USERS/jpconnor/data/CAIA/COMPASS/longitudinal_prediction_data_adt.csv \\
        --output-dir federated_transfer_adt/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from survival_common import federated_scoring as fs
from survival_common.federated_inference import load_bundle, predict_risk

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LAB_MAPPING = (
    REPO_ROOT / "data_preprocessing_common" / "resources" / "lab_mappings"
    / "OMOP_to_DFCI_lab_ids.csv"
)
DEFAULT_DATA = Path(
    "/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/longitudinal_prediction_data_adt.csv"
)

LOCAL_ID_COL = "DFCI_MRN"
LOCAL_COLUMNS = {
    LOCAL_ID_COL: fs.ID_COL,
    "AGE_AT_TREATMENTSTART": fs.AGE_COL,
    "DIAGNOSIS_DATE": fs.DIAGNOSIS_DATE_COL,
    "TREATMENT_ANCHOR_DATE": fs.ANCHOR_DATE_COL,
    "LAST_CONTACT_DATE": fs.LAST_FOLLOWUP_DATE_COL,
    "LAB_DATE": fs.MEASUREMENT_DATE_COL,
    "LAB_NAME": fs.LAB_NAME_COL,
    "LAB_VALUE": fs.LAB_VALUE_COL,
}

# (event indicator, event date) in the local long frame.  PLATINUM_DATE is
# back-filled with LAST_CONTACT_DATE for non-events upstream, so every date is
# masked by its indicator before use.
ENDPOINT_SOURCES = {
    "platinum": ("PLATINUM", "PLATINUM_DATE"),
    "nepc": ("NEPC", "NEPC_DATE"),
    "avpc": ("AVPC", "AVPC_DATE"),
}
DEFAULT_ENDPOINTS = ("nepc", "avpc", "platinum")
DEFAULT_HORIZONS_DAYS = (365, 730, 1095, 1825)

EVENT_COL = "event"
DURATION_COL = "duration_days"

# Pooled-vs-local mean ratio outside this band flags a likely unit mismatch.
UNIT_RATIO_BAND = (1 / 3, 3.0)
UNIT_CHECK_STATS = ("mean", "min", "max", "last")


# --------------------------------------------------------------------------- #
# Local frame -> federated long schema
# --------------------------------------------------------------------------- #
def load_lab_mapping(path=DEFAULT_LAB_MAPPING) -> dict[str, str]:
    """Local canonical lab name -> OMOP concept name used by the federated side."""
    m = pd.read_csv(path)
    m = m[m["mapping_status"].eq("mapped")]
    return dict(zip(
        m["collapsed_measurement"].astype(str).str.strip(),
        m["omop_measurement_name"].astype(str).str.strip(),
    ))


def read_local_frame(path, endpoints) -> pd.DataFrame:
    """Read only the columns the transfer needs from the (large) Stage-2 CSV."""
    wanted = set(LOCAL_COLUMNS)
    for ep in endpoints:
        wanted.update(ENDPOINT_SOURCES[ep])
    path = str(path)
    if path.endswith((".parquet", ".pq")):
        df = pd.read_parquet(path)
        df = df[[c for c in df.columns if c in wanted]]
    else:
        df = pd.read_csv(path, usecols=lambda c: c in wanted, low_memory=False)
    missing = sorted(wanted - set(df.columns))
    if missing:
        raise ValueError(
            f"{path} is missing {missing}. NEPC/AVPC columns are only written when "
            "Stage 1 ran with the AVPC/NEPC timeline labels mounted."
        )
    return df


def to_federated_long(
    local: pd.DataFrame,
    endpoint: str,
    lab_mapping: dict[str, str],
) -> tuple[pd.DataFrame, dict]:
    """Rename to the COMPASS long schema, with ``endpoint`` in the event slot.

    ``age_at_anchor`` is passed directly (local AGE is age at the ADT anchor);
    ``age_at_diagnosis`` is deliberately absent so the federated derivation
    does not recompute it.  Labs without an OMOP mapping cannot match any
    federated feature and are dropped.
    """
    event_col, date_col = ENDPOINT_SOURCES[endpoint]
    df = local.rename(columns=LOCAL_COLUMNS)

    event = pd.to_numeric(local[event_col], errors="coerce").fillna(0).eq(1)
    event_date = pd.to_datetime(local[date_col], errors="coerce", format="mixed")
    df[fs.PLATINUM_DATE_COL] = event_date.where(event)

    for col in (fs.DIAGNOSIS_DATE_COL, fs.ANCHOR_DATE_COL,
                fs.LAST_FOLLOWUP_DATE_COL, fs.MEASUREMENT_DATE_COL):
        df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")

    raw_names = df[fs.LAB_NAME_COL].astype(str).str.strip()
    mapped = raw_names.map(lab_mapping)
    stats = {
        "endpoint": endpoint,
        "n_rows": int(len(df)),
        "n_rows_unmapped_lab": int(mapped.isna().sum()),
        "unmapped_labs": sorted(raw_names[mapped.isna()].unique().tolist())[:50],
        "n_patients": int(df[fs.ID_COL].nunique()),
        "n_patients_event": int(df.loc[event.to_numpy(), fs.ID_COL].nunique()),
        "n_patients_event_undated": int(
            df.loc[(event & event_date.isna()).to_numpy(), fs.ID_COL].nunique()
        ),
    }
    df[fs.LAB_NAME_COL] = mapped
    # Unmapped lab rows are blanked, not dropped: dropping could remove a
    # patient whose only rows are unmapped, which would change the cohort.
    df.loc[mapped.isna(), fs.LAB_VALUE_COL] = np.nan

    keep = [fs.ID_COL, fs.AGE_COL, fs.DIAGNOSIS_DATE_COL, fs.ANCHOR_DATE_COL,
            fs.PLATINUM_DATE_COL, fs.LAST_FOLLOWUP_DATE_COL,
            fs.MEASUREMENT_DATE_COL, fs.LAB_NAME_COL, fs.LAB_VALUE_COL]
    return fs.derive_analysis_columns(df[keep]), stats


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def unit_check(wide: pd.DataFrame, model: dict) -> pd.DataFrame:
    """Compare local raw feature means with the bundle's pooled impute means.

    The federated site converts units in its own OMOP pipeline and DFCI in
    ``dfci_labs.py``; a mismatch (mg/dL vs mmol/L, say) is invisible to the
    model and shifts every score.  A mean off by more than ~3x is flagged.
    """
    pre = model["preprocessing"]
    impute = dict(pre.get("impute_means") or {})
    source = dict(pre.get("source_cols") or {})
    rows = []
    for col in pre["covariate_cols"]:
        if col.endswith("__missing") or col.rsplit("__", 1)[-1] not in UNIT_CHECK_STATS:
            continue
        pooled = impute.get(col)
        src = source.get(col, col)
        if pooled is None or src not in wide.columns:
            continue
        local = float(pd.to_numeric(wide[src], errors="coerce").mean())
        ratio = local / pooled if pooled not in (0, 0.0) else np.nan
        flag = bool(
            np.isfinite(ratio)
            and not (UNIT_RATIO_BAND[0] <= abs(ratio) <= UNIT_RATIO_BAND[1])
        )
        rows.append({"covariate": col, "local_mean": local,
                     "pooled_mean": float(pooled), "mean_ratio": ratio,
                     "unit_flag": flag})
    return pd.DataFrame(
        rows, columns=["covariate", "local_mean", "pooled_mean", "mean_ratio", "unit_flag"]
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def harrell_c(duration, event, risk) -> float:
    from lifelines.utils import concordance_index

    event = np.asarray(event)
    if event.sum() < 1 or len(event) < 2:
        return float("nan")
    return float(concordance_index(duration, -np.asarray(risk), event_observed=event))


def bootstrap_c(duration, event, risks: dict[str, np.ndarray], n_boot: int, seed: int):
    """Patient-bootstrap Harrell's C for each score, resampling rows jointly.

    Joint resampling makes the paired differences between scores valid.
    Returns ``{name: array of n_boot C values}``.
    """
    rng = np.random.default_rng(seed)
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=int)
    n = len(duration)
    out = {k: np.full(n_boot, np.nan) for k in risks}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if event[idx].sum() < 1:
            continue
        for k, r in risks.items():
            out[k][b] = harrell_c(duration[idx], event[idx], np.asarray(r)[idx])
    return out


def _ci(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 10:
        return float("nan"), float("nan")
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def cox_slope(duration, event, risk) -> dict:
    """Hazard ratio of the endpoint per SD of the transported risk score."""
    from lifelines import CoxPHFitter

    risk = np.asarray(risk, dtype=float)
    sd = risk.std()
    out = {"hr_per_sd": np.nan, "hr_per_sd_lo": np.nan,
           "hr_per_sd_hi": np.nan, "hr_per_sd_p": np.nan}
    if np.asarray(event).sum() < 2 or not sd > 0:
        return out
    df = pd.DataFrame({"T": duration, "E": event, "z": (risk - risk.mean()) / sd})
    try:
        cph = CoxPHFitter().fit(df, "T", "E")
    except Exception:
        return out
    s = cph.summary.loc["z"]
    out.update({
        "hr_per_sd": float(s["exp(coef)"]),
        "hr_per_sd_lo": float(s["exp(coef) lower 95%"]),
        "hr_per_sd_hi": float(s["exp(coef) upper 95%"]),
        "hr_per_sd_p": float(s["p"]),
    })
    return out


def ipcw_metrics(duration, event, risk, horizons_days) -> tuple[dict, pd.DataFrame]:
    """Uno's C and cumulative/dynamic AUC(t), censoring estimated on this cohort.

    External validation has no training cohort to borrow a censoring
    distribution from, so the local cohort supplies it.  Horizons with no event
    before them, or at/after the last follow-up, are inestimable and skipped.
    """
    empty = pd.DataFrame(columns=["horizon_days", "auc", "n_cases"])
    try:
        from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc
        from sksurv.util import Surv
    except ModuleNotFoundError:
        return {"uno_c": np.nan, "mean_auc": np.nan, "note": "sksurv not installed"}, empty

    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event).astype(bool)
    risk = np.asarray(risk, dtype=float)
    y = Surv.from_arrays(event=event, time=duration)

    ok = [h for h in horizons_days
          if (event & (duration <= h)).any() and h < duration.max()]
    summary = {"uno_c": np.nan, "mean_auc": np.nan, "note": ""}
    if not ok:
        summary["note"] = "no estimable horizon"
        return summary, empty

    try:
        summary["uno_c"] = float(concordance_index_ipcw(y, y, risk, tau=max(ok))[0])
    except Exception as exc:
        summary["note"] += f"uno_c: {exc}; "
    try:
        auc, mean_auc = cumulative_dynamic_auc(y, y, risk, np.asarray(ok, dtype=float))
        summary["mean_auc"] = float(mean_auc)
        auc_df = pd.DataFrame({
            "horizon_days": ok,
            "auc": auc,
            "n_cases": [int((event & (duration <= h)).sum()) for h in ok],
        })
    except Exception as exc:
        summary["note"] += f"auc: {exc}; "
        auc_df = empty
    return summary, auc_df


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def evaluate_transfer(
    local: pd.DataFrame,
    bundles: dict[str, dict],
    *,
    endpoints=DEFAULT_ENDPOINTS,
    lab_mapping: dict[str, str],
    exclusion: str = "none",
    landmarks: tuple[int, ...] | None = None,
    horizons_days=DEFAULT_HORIZONS_DAYS,
    n_bootstrap: int = 200,
    seed: int = 0,
    max_followup_days: int | None = fs.DEFAULT_MAX_FOLLOWUP_DAYS,
    verbose: bool = True,
) -> dict[str, pd.DataFrame | list]:
    """Score every bundle model on every endpoint and evaluate it.

    Each (endpoint, landmark) frame is built once and shared by every bundle
    and config, so the full and baseline models are compared on the same
    patients and their bootstrap is paired.
    """
    all_landmarks = sorted({
        int(m["landmark_days"]) for b in bundles.values() for m in b["models"]
    })
    if landmarks is not None:
        all_landmarks = [lm for lm in all_landmarks if lm in landmarks]

    scores, metrics, auc_rows, coverage, units, prep_stats = [], [], [], [], [], []

    for endpoint in endpoints:
        long_df, stats = to_federated_long(local, endpoint, lab_mapping)
        long_df, excl = fs.apply_exclusion(long_df, exclusion)
        stats["exclusion"] = excl
        prep_stats.append(stats)
        if verbose:
            print(
                f"\n[transfer] endpoint={endpoint}: {stats['n_patients']:,} patients, "
                f"{stats['n_patients_event']:,} with an event "
                f"({stats['n_patients_event_undated']:,} undated -> censored); "
                f"exclusion={exclusion} kept {excl['n_patients_after']:,}"
            )

        for lm in all_landmarks:
            wide, lm_stats = fs.build_landmark_frame(
                long_df, lm, max_followup_days=max_followup_days
            )
            if wide.empty:
                continue
            duration = wide[fs.DURATION_COL].to_numpy(dtype=float)
            event = wide[fs.EVENT_COL].to_numpy(dtype=int)
            if verbose:
                print(
                    f"[transfer]   landmark {lm}d: {len(wide):,} at risk, "
                    f"{int(event.sum()):,} incident events "
                    f"(dropped {lm_stats['n_dropped_non_positive_duration']:,} with "
                    f"event/censoring at or before the landmark)"
                )

            risks: dict[tuple[str, str], np.ndarray] = {}
            for stem, bundle in bundles.items():
                for model in bundle["models"]:
                    if int(model["landmark_days"]) != lm:
                        continue
                    config = str(model.get("config"))
                    risk = predict_risk(wide, bundle, model, age_col=fs.AGE_COL)
                    risks[(stem, config)] = risk

                    scores.append(pd.DataFrame({
                        "endpoint": endpoint, "bundle": stem,
                        "model_family": bundle.get("model_family"),
                        "landmark_days": lm, "config": config,
                        fs.ID_COL: wide[fs.ID_COL].to_numpy(),
                        "risk_score": risk,
                        DURATION_COL: duration, EVENT_COL: event,
                    }))
                    if endpoint == endpoints[0]:
                        cov = fs.covariate_coverage(wide, model)
                        cov.insert(0, "config", config)
                        cov.insert(0, "landmark_days", lm)
                        cov.insert(0, "bundle", stem)
                        coverage.append(cov)
                        uc = unit_check(wide, model)
                        uc.insert(0, "config", config)
                        uc.insert(0, "landmark_days", lm)
                        uc.insert(0, "bundle", stem)
                        units.append(uc)

            boot = bootstrap_c(
                duration, event,
                {f"{s}|{c}": r for (s, c), r in risks.items()},
                n_bootstrap, seed,
            ) if n_bootstrap > 0 else {}

            for (stem, config), risk in risks.items():
                bundle = bundles[stem]
                row = {
                    "endpoint": endpoint, "bundle": stem,
                    "model_family": bundle.get("model_family"),
                    "analysis_label": bundle.get("analysis_label"),
                    "trained_endpoint": bundle.get("endpoint", "platinum"),
                    "landmark_days": lm, "config": config,
                    "n_patients": int(len(wide)), "n_events": int(event.sum()),
                    "n_admin_censored": int(lm_stats["n_admin_censored"]),
                    "median_followup_days": float(np.median(duration)),
                    "c_index": harrell_c(duration, event, risk),
                }
                key = f"{stem}|{config}"
                if key in boot:
                    row["c_index_lo"], row["c_index_hi"] = _ci(boot[key])
                base_key = f"{stem}|baseline"
                if config != "baseline" and (stem, "baseline") in risks:
                    row["c_index_baseline"] = harrell_c(
                        duration, event, risks[(stem, "baseline")]
                    )
                    row["delta_c_vs_baseline"] = row["c_index"] - row["c_index_baseline"]
                    if key in boot and base_key in boot:
                        diff = boot[key] - boot[base_key]
                        row["delta_c_lo"], row["delta_c_hi"] = _ci(diff)
                row.update(cox_slope(duration, event, risk))
                ipcw, auc_df = ipcw_metrics(duration, event, risk, horizons_days)
                row.update(ipcw)
                metrics.append(row)
                if not auc_df.empty:
                    auc_df.insert(0, "config", config)
                    auc_df.insert(0, "landmark_days", lm)
                    auc_df.insert(0, "bundle", stem)
                    auc_df.insert(0, "endpoint", endpoint)
                    auc_rows.append(auc_df)

                if verbose:
                    ci = (
                        f" [{row['c_index_lo']:.3f}, {row['c_index_hi']:.3f}]"
                        if np.isfinite(row.get("c_index_lo", np.nan)) else ""
                    )
                    delta = row.get("delta_c_vs_baseline")
                    extra = (
                        f", vs age-only {delta:+.3f}" if delta is not None
                        and np.isfinite(delta) else ""
                    )
                    print(
                        f"[transfer]     {stem} {config}: C={row['c_index']:.3f}{ci}"
                        f"{extra}, HR/SD={row['hr_per_sd']:.2f}"
                    )

    def _cat(frames):
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return {
        "scores": _cat(scores),
        "metrics": pd.DataFrame(metrics),
        "auc_t": _cat(auc_rows),
        "coverage": _cat(coverage),
        "unit_check": _cat(units),
        "prep_stats": prep_stats,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate federated platinum models on local NEPC/AVPC endpoints."
    )
    ap.add_argument("--bundle", required=True, nargs="+", help="model bundle JSON(s)")
    ap.add_argument("--data", default=str(DEFAULT_DATA),
                    help="Stage-2 ADT long frame (longitudinal_prediction_data_adt.csv)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--endpoints", nargs="+", default=list(DEFAULT_ENDPOINTS),
                    choices=sorted(ENDPOINT_SOURCES))
    ap.add_argument("--lab-mapping", default=str(DEFAULT_LAB_MAPPING))
    ap.add_argument("--exclusion", default="none",
                    choices=["none", "pre_anchor_castrate"],
                    help="'none' is the full ADT cohort; match the bundle's analysis_label")
    ap.add_argument("--landmarks", default="", help="comma-separated; default all in bundles")
    ap.add_argument("--horizons", default=",".join(map(str, DEFAULT_HORIZONS_DAYS)),
                    help="AUC(t) horizons, days after the landmark")
    ap.add_argument("--n-bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-followup-days", type=int, default=fs.DEFAULT_MAX_FOLLOWUP_DAYS)
    args = ap.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    bundles = {}
    for path in args.bundle:
        bundle = load_bundle(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        label = str(bundle.get("analysis_label") or "")
        if ("noprecastrate" in label) != (args.exclusion == "pre_anchor_castrate"):
            print(
                f"[transfer] WARNING: {stem} has analysis_label={label!r} but "
                f"--exclusion={args.exclusion}; the cohort arms do not match"
            )
        bundles[stem] = bundle

    local = read_local_frame(args.data, args.endpoints)
    print(f"[transfer] read {len(local):,} rows, "
          f"{local[LOCAL_ID_COL].nunique():,} patients from {args.data}")

    landmarks = (
        tuple(int(x) for x in args.landmarks.split(",") if x.strip())
        if args.landmarks.strip() else None
    )
    horizons = tuple(float(h) for h in args.horizons.split(",") if h.strip())

    res = evaluate_transfer(
        local, bundles,
        endpoints=tuple(args.endpoints),
        lab_mapping=load_lab_mapping(args.lab_mapping),
        exclusion=args.exclusion, landmarks=landmarks, horizons_days=horizons,
        n_bootstrap=args.n_bootstrap, seed=args.seed,
        max_followup_days=args.max_followup_days,
    )

    for name in ("scores", "metrics", "auc_t", "coverage", "unit_check"):
        frame = res[name]
        if frame.empty:
            continue
        out = os.path.join(args.output_dir, f"transfer_{name}.csv")
        frame.to_csv(out, index=False)
        print(f"[transfer] wrote {len(frame):,} rows -> {out}")

    flagged = res["unit_check"]
    if not flagged.empty and flagged["unit_flag"].any():
        bad = flagged.loc[flagged["unit_flag"], "covariate"].unique()
        print(
            f"[transfer] WARNING: {len(bad)} covariate(s) differ from the pooled "
            f"mean by >3x, likely a unit mismatch: {', '.join(sorted(bad)[:10])}"
        )

    with open(os.path.join(args.output_dir, "transfer_run.json"), "w") as fh:
        json.dump({
            "data": args.data,
            "bundles": {s: {"path": p, "model_family": b.get("model_family"),
                            "analysis_label": b.get("analysis_label"),
                            "xgboost_version": b.get("xgboost_version")}
                        for p, (s, b) in zip(args.bundle, bundles.items())},
            "endpoints": args.endpoints,
            "exclusion": args.exclusion,
            "landmarks": list(landmarks) if landmarks else None,
            "horizons_days": list(horizons),
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "max_followup_days": args.max_followup_days,
            "prep_stats": res["prep_stats"],
        }, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
