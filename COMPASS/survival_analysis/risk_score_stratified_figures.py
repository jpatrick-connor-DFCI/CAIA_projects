"""Stratified time-to-event figures comparing the model risk score to known clinical features.

Reads the held-out per-patient risk scores written by the multivariable runners
(``*_patient_risks.csv``) and joins them to clinical stratifiers -- Gleason
score, overall stage, and the TP53/PTEN/RB1 somatic trio -- to answer one
question: does the model's risk score separate time-to-platinum better than the
features a clinician already has?

Outputs, per endpoint and landmark:

  risk_stratified_km_<endpoint>_landmark<N>.png
      A KM panel grid. Panel 1 is the risk score split at its held-out median;
      the rest are the clinical stratifiers on the same axes and time scale.
  risk_stratified_discrimination_<endpoint>_landmark<N>.csv
      Held-out C-index for the risk score and for each clinical feature alone,
      on the patients where that feature is observed, plus a paired
      restricted-to-overlap comparison so the numbers are read on equal cohorts.

Every stratifier is evaluated on the held-out block only; nothing here refits a
model, so this script cannot leak training information into a reported number.

Stage is deliberately NOT combined with Gleason into an NCCN risk group. NCCN
groups are defined on clinical T stage (T1c/T2a/T2b-c/T3a), which this dataset
does not carry -- DERIVED_STAGE_MERGED is overall stage I-IV only. A composite
built from overall stage would carry the NCCN name without its definition, so
stage and Gleason are reported as separate stratifiers instead.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

SURVIVAL_DIR = Path(__file__).resolve().parent
SURVIVAL_PARENT = SURVIVAL_DIR.parent
REPO_ROOT = SURVIVAL_PARENT.parent
for _p in (str(REPO_ROOT), str(SURVIVAL_PARENT), str(SURVIVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd


# The trio this analysis is about. PTEN and RB1 are canonically lost by deletion
# rather than point mutation, so the loader accepts any variant class present in
# the input and records which classes actually backed each gene's indicator
# (see GENE_ALTERATION_CLASSES / the gene_sources column of the manifest).
TRIO_GENES = ("TP53", "PTEN", "RB1")
GENE_ALTERATION_CLASSES = ("SNV", "DEL", "AMP", "SV")
GENE_COLUMN_RE = re.compile(
    r"^(?P<gene>[A-Za-z0-9.\-]+)_(?P<klass>SNV|DEL|AMP|SV)$"
)

GLEASON_FEATURE = "GLEASON_SCORE"
STAGE_COLUMN = "DERIVED_STAGE_MERGED"

# Gleason grade grouping. 6 or less is ISUP 1, 7 is ISUP 2-3 (the 3+4 vs 4+3
# split needs the primary/secondary pattern, which the total score does not
# carry), 8 is ISUP 4, and 9-10 is ISUP 5. Collapsed to three bins so the KM
# panels keep usable group sizes in an ADT-anchored cohort.
GLEASON_BINS = ((0, 6, "Gleason <=6"), (7, 7, "Gleason 7"), (8, 10, "Gleason 8-10"))

RISK_HIGH = "High risk (above median)"
RISK_LOW = "Low risk (at or below median)"

# Shared with survival_common.plotting.COHORT_COLORS where labels overlap; these
# are the labels this script introduces.
STRATUM_COLORS = {
    RISK_HIGH: "#b2182b",
    RISK_LOW: "#2166ac",
    "Gleason <=6": "#92c5de",
    "Gleason 7": "#f4a582",
    "Gleason 8-10": "#b2182b",
    "Stage I-II": "#92c5de",
    "Stage III": "#f4a582",
    "Stage IV": "#b2182b",
    "Altered": "#b2182b",
    "Wild-type": "#2166ac",
    "0 altered": "#2166ac",
    "1 altered": "#f4a582",
    "2+ altered": "#b2182b",
    "Neither": "#2166ac",
    "TP53 only": "#f4a582",
    "RB1 only": "#92c5de",
    "TP53+RB1": "#b2182b",
}


@dataclass(frozen=True)
class Stratifier:
    """A column of group labels to split the held-out cohort by.

    ``build`` returns a Series of labels aligned to the frame it is given, with
    NaN for patients the stratifier cannot classify (missing Gleason, no
    sequencing, and so on). Those patients drop out of that panel only -- never
    out of the risk-score panel -- so each panel reports the largest cohort its
    own feature supports and the manifest records the resulting n.
    """

    key: str
    title: str
    build: Callable[[pd.DataFrame], pd.Series]
    order: tuple[str, ...] = ()


def _read_csv(path: Path, what: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {what} at {path}")
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise ValueError(f"{what} at {path} is empty")
    return frame


def load_patient_risks(
    path: Path,
    *,
    endpoint: str,
    landmark_day: int | None,
    id_col: str,
    dataset: str = "test",
) -> pd.DataFrame:
    """Load held-out risk scores for one endpoint/landmark.

    The runners write one row per held-out patient per endpoint per landmark.
    Two schemes can appear in the same file and they are NOT interchangeable:

      dataset == "test"    the held-out test block, scored by the final model
                           refit on all of train/val.
      dataset == "cv_oof"  every patient in the cohort, each scored by the one
                           outer CV fold that excluded them (written only when
                           the run passed --out-of-fold-risks).

    Exactly one is selected, never both. Pooling them would enter every test
    patient twice under two different models, and would also silently defeat
    the duplicate check below. `dataset` picks which.
    """
    risks = _read_csv(path, "patient risk scores")
    required = {id_col, "endpoint", "risk_score", "duration_days", "event"}
    missing = required - set(risks.columns)
    if missing:
        raise ValueError(
            f"Patient risk file {path} is missing required columns: "
            f"{sorted(missing)}. Expected the schema written by "
            f"cox_runners/multivariate_analysis patient_risks output."
        )
    if "dataset" in risks.columns:
        available = sorted(risks["dataset"].astype(str).unique())
        risks = risks.loc[risks["dataset"].astype(str) == dataset]
        if risks.empty:
            raise ValueError(
                f"{path} has no dataset=={dataset!r} rows; it contains "
                f"{available}. Out-of-fold scores are written only when the "
                f"multivariable run passed --out-of-fold-risks."
            )
    elif dataset != "test":
        raise ValueError(
            f"{path} has no 'dataset' column, so it predates out-of-fold "
            f"scoring and cannot satisfy dataset=={dataset!r}."
        )
    risks = risks.loc[risks["endpoint"].astype(str) == endpoint]
    if landmark_day is not None and "landmark_days" in risks.columns:
        risks = risks.loc[
            pd.to_numeric(risks["landmark_days"], errors="coerce") == landmark_day
        ]
    if risks.empty:
        raise ValueError(
            f"No dataset=={dataset!r} rows in {path} for endpoint={endpoint!r}"
            + (f", landmark_days={landmark_day}" if landmark_day is not None else "")
        )
    duplicated = risks[id_col].duplicated().sum()
    if duplicated:
        raise ValueError(
            f"{path} has {duplicated} duplicate {id_col} rows for "
            f"endpoint={endpoint!r}; expected one held-out score per patient."
        )
    out = risks[[id_col, "risk_score", "duration_days", "event"]].copy()
    out["risk_score"] = pd.to_numeric(out["risk_score"], errors="coerce")
    out["duration_days"] = pd.to_numeric(out["duration_days"], errors="coerce")
    out["event"] = pd.to_numeric(out["event"], errors="coerce")
    out = out.dropna(subset=["risk_score", "duration_days", "event"])
    out[id_col] = out[id_col].astype(str)
    return out.reset_index(drop=True)


def load_clinical_features(
    path: Path | None, *, id_col: str, ids: Sequence[str]
) -> pd.DataFrame:
    """Load the clinical/genomic stratifier columns for the held-out patients.

    Accepts the aggregated landmark table (which already carries Gleason and the
    somatic indicators when built with --feature-set somatic-gleason) or any
    frame keyed on the id column. Only the recognised stratifier columns are
    kept; unknown columns are ignored rather than guessed at.
    """
    if path is None:
        return pd.DataFrame({id_col: list(ids)})
    frame = _read_csv(path, "clinical features")
    if id_col not in frame.columns:
        raise ValueError(
            f"Clinical feature file {path} has no {id_col} column; cannot join "
            f"to the held-out risk scores."
        )
    frame = frame.copy()
    frame[id_col] = frame[id_col].astype(str)
    keep = [id_col]
    if GLEASON_FEATURE in frame.columns:
        keep.append(GLEASON_FEATURE)
    if STAGE_COLUMN in frame.columns:
        keep.append(STAGE_COLUMN)
    keep.extend(
        c for c in frame.columns
        if (m := GENE_COLUMN_RE.match(str(c))) and m.group("gene").upper() in TRIO_GENES
    )
    frame = frame[keep].drop_duplicates(subset=[id_col])
    return frame


def gene_alteration_indicator(frame: pd.DataFrame, gene: str) -> tuple[pd.Series, list[str]]:
    """Collapse every variant class for `gene` into one altered/not indicator.

    Returns the indicator and the column names that backed it. A patient is
    altered if any class is 1; they are wild-type only if every present class is
    observed and 0. If the gene has no columns at all the indicator is all-NaN,
    which drops its panel rather than silently reporting everyone wild-type --
    the failure mode that would make an unsequenced cohort look uniformly
    low-risk.
    """
    cols = [
        c for c in frame.columns
        if (m := GENE_COLUMN_RE.match(str(c))) and m.group("gene").upper() == gene.upper()
    ]
    if not cols:
        return pd.Series(np.nan, index=frame.index, dtype=float), []
    numeric = frame[cols].apply(pd.to_numeric, errors="coerce")
    any_altered = (numeric == 1).any(axis=1)
    all_missing = numeric.isna().all(axis=1)
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    out.loc[~all_missing] = 0.0
    out.loc[any_altered] = 1.0
    return out, cols


def _gleason_group(frame: pd.DataFrame) -> pd.Series:
    if GLEASON_FEATURE not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=object)
    score = pd.to_numeric(frame[GLEASON_FEATURE], errors="coerce")
    out = pd.Series(np.nan, index=frame.index, dtype=object)
    for lo, hi, label in GLEASON_BINS:
        out.loc[score.between(lo, hi)] = label
    return out


def _stage_group(frame: pd.DataFrame) -> pd.Series:
    """Collapse overall stage I-IV into three bins.

    Mirrors the numeral/roman handling in prepare_figure_data.py so
    the two agree on what a stage value means.
    """
    if STAGE_COLUMN not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=object)
    text = frame[STAGE_COLUMN].astype(str).str.strip().str.upper()
    mapping = {
        "1": 1, "I": 1, "1.0": 1,
        "2": 2, "II": 2, "2.0": 2,
        "3": 3, "III": 3, "3.0": 3,
        "4": 4, "IV": 4, "4.0": 4,
    }
    stage = text.map(mapping)
    out = pd.Series(np.nan, index=frame.index, dtype=object)
    out.loc[stage.isin([1, 2])] = "Stage I-II"
    out.loc[stage == 3] = "Stage III"
    out.loc[stage == 4] = "Stage IV"
    return out


def _single_gene(gene: str) -> Callable[[pd.DataFrame], pd.Series]:
    def build(frame: pd.DataFrame) -> pd.Series:
        indicator, _ = gene_alteration_indicator(frame, gene)
        out = pd.Series(np.nan, index=frame.index, dtype=object)
        out.loc[indicator == 1] = "Altered"
        out.loc[indicator == 0] = "Wild-type"
        return out
    return build


def _trio_burden(frame: pd.DataFrame) -> pd.Series:
    """0 / 1 / 2+ altered across the trio.

    Requires all three genes observed for a patient; a patient missing one gene
    cannot be placed in a burden bin without assuming that gene is wild-type.
    """
    indicators = [gene_alteration_indicator(frame, g)[0] for g in TRIO_GENES]
    stacked = pd.concat(indicators, axis=1)
    complete = stacked.notna().all(axis=1)
    count = stacked.sum(axis=1)
    out = pd.Series(np.nan, index=frame.index, dtype=object)
    out.loc[complete & (count == 0)] = "0 altered"
    out.loc[complete & (count == 1)] = "1 altered"
    out.loc[complete & (count >= 2)] = "2+ altered"
    return out


def _tp53_rb1_pair(frame: pd.DataFrame) -> pd.Series:
    """The NEPC-associated TP53/RB1 co-alteration, as four mutually exclusive groups."""
    tp53, _ = gene_alteration_indicator(frame, "TP53")
    rb1, _ = gene_alteration_indicator(frame, "RB1")
    complete = tp53.notna() & rb1.notna()
    out = pd.Series(np.nan, index=frame.index, dtype=object)
    out.loc[complete & (tp53 == 0) & (rb1 == 0)] = "Neither"
    out.loc[complete & (tp53 == 1) & (rb1 == 0)] = "TP53 only"
    out.loc[complete & (tp53 == 0) & (rb1 == 1)] = "RB1 only"
    out.loc[complete & (tp53 == 1) & (rb1 == 1)] = "TP53+RB1"
    return out


def _trio_combinations(frame: pd.DataFrame) -> pd.Series:
    """All eight combinations of the trio, labelled by which genes are altered."""
    indicators = {g: gene_alteration_indicator(frame, g)[0] for g in TRIO_GENES}
    stacked = pd.concat(indicators.values(), axis=1)
    complete = stacked.notna().all(axis=1)
    out = pd.Series(np.nan, index=frame.index, dtype=object)
    for idx in frame.index[complete]:
        altered = [g for g in TRIO_GENES if indicators[g].loc[idx] == 1]
        out.loc[idx] = "+".join(altered) if altered else "None altered"
    return out


def risk_group(frame: pd.DataFrame, *, cutpoint: float) -> pd.Series:
    """Split held-out risk scores at `cutpoint` (the held-out median by default).

    Ties at the cutpoint go to the low group, so the split is reproducible when
    many patients share a score.
    """
    score = pd.to_numeric(frame["risk_score"], errors="coerce")
    out = pd.Series(np.nan, index=frame.index, dtype=object)
    out.loc[score > cutpoint] = RISK_HIGH
    out.loc[score <= cutpoint] = RISK_LOW
    return out


def build_stratifiers(frame: pd.DataFrame, *, cutpoint: float) -> list[Stratifier]:
    """Assemble the panel list, dropping stratifiers with no usable data."""
    candidates = [
        Stratifier(
            "risk_score", "Model risk score (held-out)",
            lambda f: risk_group(f, cutpoint=cutpoint),
            (RISK_HIGH, RISK_LOW),
        ),
        Stratifier("gleason", "Gleason score", _gleason_group,
                   tuple(label for _, _, label in GLEASON_BINS)),
        Stratifier("stage", "Overall stage", _stage_group,
                   ("Stage I-II", "Stage III", "Stage IV")),
        *[
            Stratifier(f"{g.lower()}", f"{g} alteration", _single_gene(g),
                       ("Altered", "Wild-type"))
            for g in TRIO_GENES
        ],
        Stratifier("trio_burden", "TP53/PTEN/RB1 burden", _trio_burden,
                   ("0 altered", "1 altered", "2+ altered")),
        Stratifier("tp53_rb1", "TP53/RB1 co-alteration", _tp53_rb1_pair,
                   ("Neither", "TP53 only", "RB1 only", "TP53+RB1")),
        Stratifier("trio_combinations", "TP53/PTEN/RB1 combinations", _trio_combinations),
    ]
    usable = []
    for strat in candidates:
        labels = strat.build(frame)
        if labels.notna().sum() == 0 or labels.dropna().nunique() < 2:
            continue
        usable.append(strat)
    return usable


def logrank_p(durations: pd.Series, events: pd.Series, groups: pd.Series) -> float:
    """Multivariate log-rank p across the groups, or NaN if lifelines is absent."""
    try:
        from lifelines.statistics import multivariate_logrank_test
    except ModuleNotFoundError:
        return float("nan")
    mask = durations.notna() & events.notna() & groups.notna()
    if mask.sum() == 0 or groups.loc[mask].nunique() < 2:
        return float("nan")
    result = multivariate_logrank_test(
        durations.loc[mask].to_numpy(dtype=float),
        groups.loc[mask].to_numpy(),
        events.loc[mask].to_numpy(dtype=float),
    )
    return float(result.p_value)


def concordance(
    durations: pd.Series, events: pd.Series, scores: pd.Series
) -> float:
    """Harrell C-index of `scores` as a risk (higher = sooner event).

    lifelines' concordance_index treats its predictor as a *predicted survival
    time*, so the risk score is negated to score it as a hazard.
    """
    try:
        from lifelines.utils import concordance_index
    except ModuleNotFoundError:
        return float("nan")
    mask = durations.notna() & events.notna() & scores.notna()
    if mask.sum() < 2 or events.loc[mask].sum() == 0:
        return float("nan")
    return float(
        concordance_index(
            durations.loc[mask].to_numpy(dtype=float),
            -scores.loc[mask].to_numpy(dtype=float),
            events.loc[mask].to_numpy(dtype=float),
        )
    )


def _ordinal_scores(labels: pd.Series, order: Sequence[str]) -> pd.Series:
    """Map ordered group labels to ranks so a categorical feature can be scored.

    Only meaningful for stratifiers whose order is genuinely ordinal; the
    unordered ones (trio combinations) are skipped by the caller.
    """
    ranks = {label: i for i, label in enumerate(order)}
    return labels.map(ranks).astype(float)


def discrimination_table(
    frame: pd.DataFrame, stratifiers: Sequence[Stratifier]
) -> pd.DataFrame:
    """Held-out C-index for the risk score vs each clinical feature.

    Each clinical feature is scored twice: on every patient where it is observed
    ("feature_cohort"), and with the continuous risk score restricted to that
    same subset ("risk_score_same_cohort"). The paired column is the honest
    comparison -- a feature observed in a small, sicker subset is not comparable
    to a risk score measured on everyone.
    """
    rows = []
    durations, events = frame["duration_days"], frame["event"]
    for strat in stratifiers:
        if strat.key == "risk_score":
            continue
        labels = strat.build(frame)
        if not strat.order:
            # Unordered grouping: no defensible scalar ranking, so report
            # separation via log-rank only.
            rows.append({
                "feature": strat.key,
                "title": strat.title,
                "n_observed": int(labels.notna().sum()),
                "n_events_observed": int(events.loc[labels.notna()].sum()),
                "feature_c_index": float("nan"),
                "risk_score_same_cohort_c_index": concordance(
                    durations.loc[labels.notna()],
                    events.loc[labels.notna()],
                    frame["risk_score"].loc[labels.notna()],
                ),
                "logrank_p": logrank_p(durations, events, labels),
                "note": "unordered groups; C-index not defined for this feature",
            })
            continue
        observed = labels.notna()
        feature_scores = _ordinal_scores(labels, strat.order)
        rows.append({
            "feature": strat.key,
            "title": strat.title,
            "n_observed": int(observed.sum()),
            "n_events_observed": int(events.loc[observed].sum()),
            "feature_c_index": concordance(
                durations.loc[observed], events.loc[observed],
                feature_scores.loc[observed],
            ),
            "risk_score_same_cohort_c_index": concordance(
                durations.loc[observed], events.loc[observed],
                frame["risk_score"].loc[observed],
            ),
            "logrank_p": logrank_p(durations, events, labels),
            "note": "",
        })
    table = pd.DataFrame(rows)
    if not table.empty:
        table.insert(0, "risk_score_full_cohort_c_index",
                     concordance(durations, events, frame["risk_score"]))
        table.insert(0, "n_held_out", len(frame))
    return table


def plot_panels(
    frame: pd.DataFrame,
    stratifiers: Sequence[Stratifier],
    *,
    title: str,
    xlabel: str,
    max_days: float | None,
) -> "object":
    """Draw one KM panel per stratifier on a shared grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from survival_common.plotting import RCPARAMS, overlay_km

    n = len(stratifiers)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    with plt.rc_context(RCPARAMS):
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False
        )
        flat = [ax for row in axes for ax in row]
        for ax, strat in zip(flat, stratifiers):
            labels = strat.build(frame)
            order = strat.order or tuple(
                sorted(labels.dropna().unique(), key=str)
            )
            series = {}
            for label in order:
                mask = labels == label
                if mask.sum() == 0:
                    continue
                series[label] = (
                    frame.loc[mask, "duration_days"], frame.loc[mask, "event"]
                )
            if not series:
                ax.set_visible(False)
                continue
            overlay_km(
                ax, series, colors=STRATUM_COLORS, title=strat.title,
                xlabel=xlabel, ylabel="Platinum-free probability",
                # Beyond four groups the overlapping bands hide the curves; the
                # per-group n in the legend still conveys the uncertainty.
                ci_show=len(series) <= 4,
            )
            p = logrank_p(frame["duration_days"], frame["event"], labels)
            if np.isfinite(p):
                ax.text(
                    0.03, 0.06,
                    f"log-rank p = {p:.3g}" if p >= 1e-4 else "log-rank p < 1e-4",
                    transform=ax.transAxes, fontsize=9,
                )
            if max_days:
                ax.set_xlim(0, max_days)
            ax.legend(fontsize=8, loc="upper right")
        for ax in flat[n:]:
            ax.set_visible(False)
        fig.suptitle(title, fontsize=13)
        fig.tight_layout()
    return fig


def gene_source_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """Record which variant-class columns backed each gene's indicator.

    Written alongside the figures so a reader can tell whether, say, PTEN loss
    was captured as a deletion or only as an SNV -- which materially changes how
    a null PTEN panel should be read.
    """
    rows = []
    for gene in TRIO_GENES:
        indicator, cols = gene_alteration_indicator(frame, gene)
        rows.append({
            "gene": gene,
            "gene_sources": ",".join(cols) if cols else "",
            "n_observed": int(indicator.notna().sum()),
            "n_altered": int((indicator == 1).sum()),
            "prevalence": (
                float((indicator == 1).mean()) if indicator.notna().any() else float("nan")
            ),
        })
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    risks = load_patient_risks(
        Path(args.patient_risks), endpoint=args.endpoint,
        landmark_day=args.landmark_days, id_col=args.id_col,
        dataset=getattr(args, "dataset", "test"),
    )
    clinical = load_clinical_features(
        Path(args.clinical_features) if args.clinical_features else None,
        id_col=args.id_col, ids=risks[args.id_col],
    )
    frame = risks.merge(clinical, on=args.id_col, how="left")
    dataset = getattr(args, "dataset", "test")
    scheme = (
        "held-out test block (final model)"
        if dataset == "test"
        else "full cohort, out-of-fold (nested CV)"
    )
    print(
        f"Scoring scheme: {scheme}\n"
        f"Patients: {len(frame):,} "
        f"({int(frame['event'].sum()):,} events) for endpoint={args.endpoint!r}"
        + (f", landmark +{args.landmark_days}d" if args.landmark_days is not None else "")
    )

    # The median is taken within this dataset, never across both: fold models
    # and the final model can sit on different score scales, so a shared
    # cutpoint would move patients across the split for reasons unrelated to
    # their risk.
    cutpoint = float(frame["risk_score"].median())
    print(f"Risk-score split at the {dataset} median: {cutpoint:.4f}")

    stratifiers = build_stratifiers(frame, cutpoint=cutpoint)
    available = {s.key for s in stratifiers}
    skipped = [
        k for k in ("gleason", "stage", "tp53", "pten", "rb1",
                    "trio_burden", "tp53_rb1", "trio_combinations")
        if k not in available
    ]
    if skipped:
        print(
            "  [note] no usable data (absent, all-missing, or single-valued) for: "
            + ", ".join(skipped)
        )
    for strat in stratifiers:
        labels = strat.build(frame)
        counts = labels.value_counts().to_dict()
        print(f"  {strat.key}: n={int(labels.notna().sum()):,} {counts}")

    suffix = (
        f"_{args.endpoint}"
        + (f"_landmark{args.landmark_days}" if args.landmark_days is not None else "")
    )

    table = discrimination_table(frame, stratifiers)
    if not table.empty:
        table_path = output_dir / f"risk_stratified_discrimination{suffix}.csv"
        table.to_csv(table_path, index=False)
        print(f"\nSaved {table_path.name}")
        print(table.to_string(index=False))

    manifest_path = output_dir / f"risk_stratified_gene_sources{suffix}.csv"
    gene_source_manifest(frame).to_csv(manifest_path, index=False)
    print(f"Saved {manifest_path.name}")

    if args.no_plot:
        return
    fig = plot_panels(
        frame, stratifiers,
        title=(
            f"Time to platinum by model risk score vs clinical features "
            f"({args.endpoint}"
            + (f", landmark +{args.landmark_days}d" if args.landmark_days is not None else "")
            + f", held-out n={len(frame):,})"
        ),
        xlabel="Days from treatment anchor",
        max_days=args.max_days,
    )
    for ext in args.formats:
        path = output_dir / f"risk_stratified_km{suffix}.{ext}"
        fig.savefig(path)
        print(f"Saved {path.name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "KM panels and held-out discrimination comparing the multivariable "
            "risk score against Gleason, stage, and TP53/PTEN/RB1."
        )
    )
    parser.add_argument(
        "--patient-risks", required=True,
        help="A *_patient_risks.csv written by the multivariable runner.",
    )
    parser.add_argument(
        "--clinical-features",
        help=(
            "Frame keyed on the id column carrying GLEASON_SCORE, "
            "DERIVED_STAGE_MERGED, and/or {GENE}_{SNV,DEL,AMP,SV} columns "
            "(e.g. an aggregated_landmark*.csv built with --feature-set "
            "somatic-gleason). Omitted, only the risk-score panel is drawn."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--endpoint", default="platinum")
    parser.add_argument(
        "--landmark-days", type=int, default=None,
        help="Restrict to one landmark when the risk file carries several.",
    )
    parser.add_argument("--id-col", default="DFCI_MRN")
    parser.add_argument(
        "--dataset", default="test", choices=["test", "cv_oof"],
        help=(
            "Which held-out scoring scheme to draw. 'test' is the held-out "
            "test block scored by the final model (the default). 'cv_oof' is "
            "the full cohort, each patient scored by the outer CV fold that "
            "excluded them, and requires a run with --out-of-fold-risks. The "
            "two come from different models and are never pooled; render them "
            "to separate --output-dir paths."
        ),
    )
    parser.add_argument(
        "--max-days", type=float, default=None,
        help="Truncate the KM x-axis, e.g. 1825 for five years.",
    )
    parser.add_argument("--formats", nargs="+", default=["png"])
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
