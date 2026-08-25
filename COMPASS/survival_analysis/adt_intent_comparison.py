"""Localized-adjuvant vs. metastatic ADT-intent result comparison.

Read-only: reads the Stage-3 input trees and the model output trees that
``compass_pipeline`` builds for the ``localized`` and ``metastatic`` cohorts and
writes comparison tables. It never rebuilds a cohort or refits a model.

This was notebook 10, where every table existed only in kernel memory and the
concordance panel was rendered inline and never saved. Writing CSVs is the
point of the extraction: the R figure pipeline's supplemental section consumes
these files.

Three interpretation constraints apply to everything here:

1. ``ADT_INTENT`` is a retrospective medication-history stratum, not a
   metastatic-status label available prospectively at the landmark.
2. The two cohorts are disjoint patients with independently derived splits and
   feature sets. Differences can reflect cohort size, event count, case mix, or
   biology.
3. "Significant in one cohort but not the other" is not evidence of a different
   effect. The univariate table therefore carries an approximate Wald
   heterogeneity test on the difference of the two log hazard ratios, which is
   not a substitute for a pooled model with an explicit feature x intent
   interaction term.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Same bootstrap as compass_pipeline: the notebooks import this module with
# only COMPASS/survival_analysis on sys.path, so the project root has to be
# added before the absolute package imports below resolve.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from COMPASS.survival_analysis import compass_pipeline as cp  # noqa: E402
from COMPASS.survival_analysis import cox_aggregated as _ca  # noqa: E402

ID_COL = "DFCI_MRN"

# The two retrospective cohorts this module contrasts. Ordering is fixed:
# every delta below is metastatic minus localized.
LOCALIZED = "localized"
METASTATIC = "metastatic"
COMPARED_COHORTS = (LOCALIZED, METASTATIC)

DEFAULT_ENDPOINTS = tuple(_ca.ENDPOINTS)
DEFAULT_LANDMARKS = (0, 90, 180)

# Filenames the R supplemental section reads. Kept as module constants so the
# R side and the tests name the same strings.
COHORT_COUNTS_FILENAME = "adt_intent_cohort_counts.csv"
OVERLAP_FILENAME = "adt_intent_cohort_overlap.csv"
ASSOCIATION_FILENAME = "adt_intent_univariate_heterogeneity.csv"
ASSOCIATION_SUMMARY_FILENAME = "adt_intent_univariate_summary.csv"
PERFORMANCE_FILENAME = "adt_intent_performance_delta.csv"

SPARSE_EVENT_THRESHOLD = 50


def comparison_dir(data_root: Path | None = None) -> Path:
    """Where the comparison CSVs land."""
    root = cp._PROFILE_OUTPUT_ROOT if data_root is None else Path(data_root)
    return root / "survival_analysis" / "adt_intent_comparison"


def _tree(cohort: str, endpoint: str, kind: str, data_root: Path | None = None) -> Path:
    """Resolve one cohort/endpoint tree.

    Delegates the label and suffix rules to compass_pipeline rather than
    re-deriving them, which is what let notebook 10 drift from the pipeline.
    """
    root = cp._PROFILE_OUTPUT_ROOT if data_root is None else Path(data_root)
    label = f"adt{cp.COHORT_SPECS[cohort]['label_suffix']}"
    prefix = "prediction_inputs" if kind == "inputs" else "local_runs"
    return root / "survival_analysis" / f"{prefix}_{label}{cp.endpoint_suffix(endpoint)}"


def bh_adjust(values) -> pd.Series:
    """Benjamini-Hochberg step-up adjustment, NaN-preserving."""
    p = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    if valid.empty:
        return out
    adjusted = valid * len(valid) / np.arange(1, len(valid) + 1)
    adjusted = np.minimum.accumulate(adjusted.iloc[::-1])[::-1].clip(upper=1.0)
    out.loc[valid.index] = adjusted
    return out


def collect_cohort_counts(
    *,
    endpoints=DEFAULT_ENDPOINTS,
    landmarks=DEFAULT_LANDMARKS,
    data_root: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Per-landmark modelled cohort sizes and event counts.

    These are the cohorts that survive endpoint eligibility, the PSA/PARPi
    filters, and lab availability -- not the Stage-1 cohort. Returns the table
    and the patient-ID sets, which `collect_overlap` needs.
    """
    rows = []
    cohort_ids: dict[tuple[str, str, int], set[int]] = {}
    for endpoint in endpoints:
        spec = _ca.ENDPOINTS[endpoint]
        event_col = spec["event_col"]
        duration_col = spec["duration_col"]
        for cohort in COMPARED_COHORTS:
            for landmark in landmarks:
                base = {"endpoint": endpoint, "cohort": cohort, "landmark_days": landmark}
                path = (
                    _tree(cohort, endpoint, "inputs", data_root)
                    / f"aggregated_landmark{landmark}.csv"
                )
                if not path.exists():
                    rows.append({**base, "status": "missing"})
                    continue
                frame = pd.read_csv(path, low_memory=False)
                if event_col not in frame.columns:
                    rows.append({**base, "status": f"no {event_col}"})
                    continue
                events = pd.to_numeric(frame[event_col], errors="coerce").fillna(0)
                durations = pd.to_numeric(frame.get(duration_col), errors="coerce")
                cohort_ids[(cohort, endpoint, landmark)] = set(
                    pd.to_numeric(frame[ID_COL], errors="coerce").dropna().astype(int)
                )
                rows.append({
                    **base,
                    "n_patients": len(frame),
                    "n_events": int(events.eq(1).sum()),
                    "event_rate_pct": 100 * events.eq(1).mean(),
                    "median_event_days": durations.loc[events.eq(1)].median(),
                    "median_followup_days": durations.median(),
                    "status": "ok",
                })
    counts = pd.DataFrame(rows)
    if not counts.empty and "n_events" in counts.columns:
        # Surfaced as a column rather than a print so the R side can annotate
        # underpowered panels instead of the warning dying with the kernel.
        counts["sparse_events"] = (
            counts["status"].eq("ok")
            & pd.to_numeric(counts["n_events"], errors="coerce").lt(SPARSE_EVENT_THRESHOLD)
        )
    return counts, cohort_ids


def collect_overlap(
    cohort_ids: dict,
    *,
    endpoints=DEFAULT_ENDPOINTS,
    landmarks=DEFAULT_LANDMARKS,
) -> pd.DataFrame:
    """Cohort disjointness check. The two intent strata must not share patients."""
    rows = []
    for endpoint in endpoints:
        for landmark in landmarks:
            loc = cohort_ids.get((LOCALIZED, endpoint, landmark))
            met = cohort_ids.get((METASTATIC, endpoint, landmark))
            if loc is None or met is None:
                continue
            rows.append({
                "endpoint": endpoint,
                "landmark_days": landmark,
                "n_localized": len(loc),
                "n_metastatic": len(met),
                "n_overlap": len(loc & met),
            })
    return pd.DataFrame(rows)


def _load_univariate(endpoints, landmarks, data_root) -> pd.DataFrame:
    frames = []
    for endpoint in endpoints:
        for cohort in COMPARED_COHORTS:
            for landmark in landmarks:
                path = (
                    _tree(cohort, endpoint, "results", data_root)
                    / "cox"
                    / f"landmark_{landmark}"
                    / "both"
                    / "cox_agg_univariate_nobs_adjusted.csv"
                )
                if not path.exists():
                    continue
                frame = pd.read_csv(path, low_memory=False)
                frame = frame.loc[
                    frame["endpoint"].astype(str).str.lower().eq(endpoint)
                ].copy()
                frame["endpoint"] = endpoint
                frame["cohort"] = cohort
                frame["landmark_days"] = landmark
                frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compare_univariate(
    *,
    endpoints=DEFAULT_ENDPOINTS,
    landmarks=DEFAULT_LANDMARKS,
    data_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join the two cohorts' univariate effects and test them for heterogeneity.

    Standard errors are reconstructed from the reported 95% CIs, so the test is
    an approximate between-cohort Wald comparison of the two log-HRs.
    """
    univariate = _load_univariate(endpoints, landmarks, data_root)
    if univariate.empty:
        return pd.DataFrame(), pd.DataFrame()

    keys = ["endpoint", "landmark_days", "feature"]
    value_cols = [
        "lab_name", "feature_stat", "n_patients_used", "n_events_used",
        "coef_feature", "hazard_ratio_per_sd", "ci_lower", "ci_upper",
        "p_value", "q_value", "note",
    ]
    sides = {}
    for cohort in COMPARED_COHORTS:
        side = univariate.loc[univariate["cohort"].eq(cohort)].copy()
        side = side[keys + [c for c in value_cols if c in side.columns]]
        sides[cohort] = side.drop_duplicates(keys)

    comparison = sides[LOCALIZED].merge(
        sides[METASTATIC],
        on=keys,
        how="outer",
        suffixes=(f"_{LOCALIZED}", f"_{METASTATIC}"),
        indicator="merge_side",
    )
    for col in ("coef_feature", "hazard_ratio_per_sd", "ci_lower", "ci_upper",
                "p_value", "q_value"):
        for cohort in COMPARED_COHORTS:
            name = f"{col}_{cohort}"
            if name in comparison:
                comparison[name] = pd.to_numeric(comparison[name], errors="coerce")

    comparison["delta_log_hr_met_minus_loc"] = (
        comparison[f"coef_feature_{METASTATIC}"] - comparison[f"coef_feature_{LOCALIZED}"]
    )
    comparison["hr_ratio_met_vs_loc"] = np.exp(comparison["delta_log_hr_met_minus_loc"])
    comparison["same_direction"] = (
        np.sign(comparison[f"coef_feature_{METASTATIC}"])
        == np.sign(comparison[f"coef_feature_{LOCALIZED}"])
    )
    for cohort in COMPARED_COHORTS:
        lo = comparison[f"ci_lower_{cohort}"]
        hi = comparison[f"ci_upper_{cohort}"]
        comparison[f"se_log_hr_{cohort}"] = (np.log(hi) - np.log(lo)) / (2 * 1.96)
    se_delta = np.sqrt(
        comparison[f"se_log_hr_{LOCALIZED}"] ** 2
        + comparison[f"se_log_hr_{METASTATIC}"] ** 2
    )
    comparison["z_heterogeneity"] = comparison["delta_log_hr_met_minus_loc"] / se_delta
    comparison["p_heterogeneity"] = comparison["z_heterogeneity"].abs().map(
        lambda z: math.erfc(z / math.sqrt(2)) if pd.notna(z) else np.nan
    )
    comparison["q_heterogeneity"] = comparison.groupby(
        ["endpoint", "landmark_days"], group_keys=False
    )["p_heterogeneity"].transform(bh_adjust)
    for cohort in COMPARED_COHORTS:
        comparison[f"fdr_{cohort}"] = comparison[f"q_value_{cohort}"].lt(0.05)

    summary_rows = []
    for (endpoint, landmark), group in comparison.groupby(["endpoint", "landmark_days"]):
        both = group.loc[group["merge_side"].eq("both")]
        spearman = np.nan
        if len(both) > 1:
            spearman = both[
                [f"coef_feature_{LOCALIZED}", f"coef_feature_{METASTATIC}"]
            ].corr(method="spearman").iloc[0, 1]
        summary_rows.append({
            "endpoint": endpoint,
            "landmark_days": landmark,
            "n_features_both": len(both),
            "n_same_direction": int(both["same_direction"].fillna(False).sum()),
            "spearman_log_hr": spearman,
            f"n_fdr_{LOCALIZED}": int(group[f"fdr_{LOCALIZED}"].sum()),
            f"n_fdr_{METASTATIC}": int(group[f"fdr_{METASTATIC}"].sum()),
            "n_heterogeneity_fdr": int(group["q_heterogeneity"].lt(0.05).sum()),
        })
    return comparison.sort_values("p_heterogeneity"), pd.DataFrame(summary_rows)


def compare_performance(
    *,
    endpoints=DEFAULT_ENDPOINTS,
    landmarks=DEFAULT_LANDMARKS,
    data_root: Path | None = None,
) -> pd.DataFrame:
    """Held-out performance deltas, metastatic minus localized.

    Positive deltas favor metastatic for C-index and AUC(t); negative deltas
    favor metastatic for integrated Brier score. Test cohorts differ between
    the two, so these are descriptive contrasts, not paired tests.
    """
    frames = []
    for endpoint in endpoints:
        for cohort in COMPARED_COHORTS:
            results_dir = _tree(cohort, endpoint, "results", data_root)
            if not results_dir.exists():
                continue
            run = {
                "label": f"adt{cp.COHORT_SPECS[cohort]['label_suffix']}",
                "output_dir": results_dir,
                "landmarks": list(landmarks),
                "endpoint": endpoint,
            }
            frame = cp.summarize_outputs(run)
            frame["cohort"] = cohort
            frames.append(frame)
    if not frames:
        return pd.DataFrame()

    performance = pd.concat(frames, ignore_index=True)
    ok = performance.loc[performance["status"].eq("ok")].copy()
    if ok.empty:
        return pd.DataFrame()

    index = ["endpoint", "landmark", "model", "config"]
    # summarize_outputs' own display spellings, which are its frame's schema
    # and deliberately shorter than the on-disk canonical metrics columns.
    metrics = ["n_test", "n_test_events", "c_index", "mean_auc_t", "integrated_brier"]
    pivot = ok.pivot_table(index=index, columns="cohort", values=metrics, aggfunc="first")
    pivot.columns = [f"{metric}_{cohort}" for metric, cohort in pivot.columns]
    pivot = pivot.reset_index()
    for metric in ("c_index", "mean_auc_t", "integrated_brier"):
        loc, met = f"{metric}_{LOCALIZED}", f"{metric}_{METASTATIC}"
        if loc in pivot and met in pivot:
            pivot[f"delta_{metric}_met_minus_loc"] = pivot[met] - pivot[loc]
    return pivot


def build_comparison(
    *,
    endpoints=DEFAULT_ENDPOINTS,
    landmarks=DEFAULT_LANDMARKS,
    data_root: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Run every comparison and write the CSVs the R pipeline reads."""
    out_dir = comparison_dir(data_root) if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts, cohort_ids = collect_cohort_counts(
        endpoints=endpoints, landmarks=landmarks, data_root=data_root
    )
    overlap = collect_overlap(cohort_ids, endpoints=endpoints, landmarks=landmarks)
    if not overlap.empty and not overlap["n_overlap"].eq(0).all():
        offending = overlap.loc[overlap["n_overlap"].gt(0)]
        raise AssertionError(
            "ADT-intent cohorts overlap; the strata must be disjoint:\n"
            f"{offending.to_string(index=False)}"
        )
    association, association_summary = compare_univariate(
        endpoints=endpoints, landmarks=landmarks, data_root=data_root
    )
    performance = compare_performance(
        endpoints=endpoints, landmarks=landmarks, data_root=data_root
    )

    written: dict[str, Path] = {}
    for key, frame, filename in (
        ("cohort_counts", counts, COHORT_COUNTS_FILENAME),
        ("overlap", overlap, OVERLAP_FILENAME),
        ("association", association, ASSOCIATION_FILENAME),
        ("association_summary", association_summary, ASSOCIATION_SUMMARY_FILENAME),
        ("performance", performance, PERFORMANCE_FILENAME),
    ):
        path = out_dir / filename
        # Empty frames are still written: the R side distinguishes "no
        # comparable runs" from "this stage never ran" by the file's presence.
        frame.to_csv(path, index=False)
        written[key] = path
        print(f"  {key}: {len(frame)} rows -> {path}")

    sparse = counts.loc[counts.get("sparse_events", False) == True] if not counts.empty else counts
    if not sparse.empty:
        print(f"\n  {len(sparse)} cohort/endpoint/landmark cell(s) under "
              f"{SPARSE_EVENT_THRESHOLD} events; treat those estimates as underpowered:")
        for _, row in sparse.iterrows():
            print(f"    {row['cohort']} / {row['endpoint']} @ {row['landmark_days']}d: "
                  f"{int(row['n_events'])} events")

    missing = counts.loc[counts["status"].ne("ok")] if not counts.empty else counts
    if not missing.empty:
        print(f"\n  {len(missing)} cohort/endpoint/landmark cell(s) had no usable input:")
        for _, row in missing.iterrows():
            print(f"    {row['cohort']} / {row['endpoint']} @ {row['landmark_days']}d: {row['status']}")

    return written


if __name__ == "__main__":
    build_comparison()
