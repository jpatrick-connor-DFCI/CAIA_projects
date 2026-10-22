"""Build sequencing- and Gleason-indexed univariate Cox inputs.

The baseline ADT cohort and train/valid/test assignments are inherited from
the standard COMPASS landmark-0 inputs. Three distinct analyses are built:

* ``sequencing_snv`` selects the sample-collection date closest to ADT start
  and predicts from that collection date to platinum (or last contact). Every
  gene is tested as its SNV indicator only, with no burden adjustment.
* ``sequencing_any`` uses the same index date and cohort but tests every gene
  as one any-alteration indicator (SNV, DEL, AMP, or SV), and carries the
  per-patient count of altered genes across the tested set
  (``SOMATIC_N_GENES_ALTERED``) as an adjustment covariate, never a feature.
* ``gleason`` selects the Gleason score date closest to ADT start and predicts
  from that score date to platinum (or last contact).

Closest means minimum absolute calendar distance, so an observation may fall
before or after ADT. An exact-distance tie prefers the earlier date. Patients
whose selected index date is on/after platinum or last contact are excluded;
the builder never substitutes a different observation based on the outcome.

Additionally, ``gleason_available_case`` preserves the ADT time origin and
original split while retaining the patients with Gleason data; it supports the
matched multivariable Gleason-versus-labs sensitivity.

``gleason_somatic_available_case`` (the Gleason/somatic/labs clinical
combinations) is anchored on the sequencing date instead: the specimen closest
to ADT start defines time 0 for every patient, landmarks are sequencing + D,
lab summaries use only labs drawn before sequencing + D, and outcomes are
measured from sequencing + D. The original split is inherited. Its clinical
stratifiers (Gleason, TP53/PTEN/RB1, stage) are written on the same clock.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SURVIVAL_DIR = PROJECT_DIR / "survival_analysis"
REPO_ROOT = PROJECT_DIR.parent
for _path in (REPO_ROOT, PROJECT_DIR, SURVIVAL_DIR, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import cox_aggregated as ca  # noqa: E402
from build_genomic_inputs import GENE_VARIANT_RE  # noqa: E402
from build_prediction_inputs import (  # noqa: E402
    BUILD_MANIFEST_FILENAME,
    SPLIT_ASSIGNMENTS_FILENAME,
    aggregated_filename,
    build_aggregated_table,
    compute_horizon_grid,
    pre_treatment_lab_filename,
)
import survival_common.cohort as survival_cohort  # noqa: E402
from survival_common.cohort import (  # noqa: E402
    build_landmark_merged,
    build_pre_treatment_lab_long,
)

PROFILE_DATA_ROOT = Path(
    os.environ.get("PROFILE_DATA_PATH", "/data/gusev/USERS/jpconnor/data/PROFILE_DATA/")
)
LLM_ANNOTATIONS_ROOT = Path(
    os.environ.get(
        "LLM_ANNOTATIONS_DATA_PATH", "/data/gusev/USERS/jpconnor/data/LLM_annotations/"
    )
)
CLINICAL_EMBEDDINGS_ROOT = Path(
    os.environ.get(
        "CTEP_DATA_PATH",
        "/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/",
    )
)
DEFAULT_SOMATIC_PATH = PROFILE_DATA_ROOT / "SOMATIC_WIDE_BY_SAMPLE.parquet"
DEFAULT_SOMATIC_MANIFEST_PATH = PROFILE_DATA_ROOT / "SOMATIC_FEATURE_MANIFEST.parquet"
DEFAULT_GLEASON_PATH = LLM_ANNOTATIONS_ROOT / "LLM_gleason_timeline/gleason_timeline.parquet"

FEATURE_MANIFEST_FILENAME = "somatic_gleason_features.csv"
GLEASON_FEATURE = "GLEASON_SCORE"
SOMATIC_AVAILABLE_DATE = "_somatic_available_date"
GLEASON_AVAILABLE_DATE = "_gleason_available_date"
SOMATIC_TESTED_FEATURE = "HAS_SOMATIC_TEST"  # legacy compatibility constant
SEQUENCING_DATE = "SEQUENCING_DATE"
INDEX_DATE = "INDEX_DATE"
INDEX_TO_ADT_DAYS = "INDEX_TO_ADT_DAYS"
INDEX_ANALYSES = ("gleason", "sequencing_snv", "sequencing_any")
# Somatic feature encodings accepted by load_somatic_features. "hybrid" (SNV
# for every gene except the TP53/PTEN/RB1 any-alteration trio) feeds the
# multivariable available-case arms; "snv" and "any" feed the two
# sequencing-indexed univariate screens.
SOMATIC_VARIANT_MODES = ("hybrid", "snv", "any")
SEQUENCING_ANALYSIS_VARIANT_MODES = {"sequencing_snv": "snv", "sequencing_any": "any"}
# Per-patient count of altered genes across the tested set in the
# any-alteration screen. An adjustment covariate in every per-gene Cox fit
# (cox_aggregated.somatic_burden_covariate_columns), never a tested feature,
# so it is deliberately absent from the feature manifest.
SOMATIC_BURDEN_COVARIATE = "SOMATIC_N_GENES_ALTERED"
# Treatment-anchored, complete-availability cohort used to compare baseline
# labs with baseline Gleason.  This is deliberately distinct from
# INDEX_ANALYSES: those arms rebase follow-up to an observation date and
# therefore are not directly comparable with the lab model.
AVAILABLE_CASE_SENSITIVITIES = ("gleason_available_case",)
# Sequencing-anchored Gleason + somatic + labs cohort: every arm (labs alone
# included) is fit on the same patients, on a clock that starts at sequencing.
GLEASON_SOMATIC_AVAILABLE_CASE = "gleason_somatic_available_case"
# Days from ADT start to the selected sequencing date (negative = before ADT).
# Every ADT-relative t_* duration is shifted by this to re-clock on sequencing.
SEQUENCING_TO_ADT_DAYS = "SEQUENCING_TO_ADT_DAYS"
# Plan §4a: clinical-stratifier inputs for risk_score_stratified_figures.py.
# These are stratifiers only, never model features, so the trio columns are
# read from the UNFILTERED somatic matrix (every alteration class, one column
# per class) rather than the collapsed <GENE>_ANY model features.
CLINICAL_STRATIFIERS_DIRNAME = "clinical_stratifiers"
CLINICAL_STRATIFIERS_TRIO_GENES = ("TP53", "PTEN", "RB1")
CLINICAL_STRATIFIERS_ALTERATION_CLASSES = ("SNV", "DEL", "AMP", "SV")
# Somatic model features for these genes collapse every alteration class into
# one <GENE>_ANY indicator; every other gene stays SNV-only (README invariant #8).
ANY_ALTERATION_GENES = CLINICAL_STRATIFIERS_TRIO_GENES
ANY_ALTERATION_SUFFIX = "ANY"
STAGE_COLUMN = "DERIVED_STAGE_MERGED"
DEFAULT_REGEX_STAGE_PATH = Path(
    os.environ.get(
        "COMPASS_REGEX_STAGE_PATH",
        str(PROFILE_DATA_ROOT / "CANCER_ANNOTATIONS" / "CANCER_STAGE_NOTE_LEVEL.parquet"),
    )
)


def _delimited_separator(path: Path) -> str | None:
    """Return the field separator for a text table, or None if not one."""
    name = path.name.lower()
    if name.endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz")):
        return "\t"
    if name.endswith((".csv", ".csv.gz")):
        return ","
    return None


def _read_table(path: Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=columns)
    separator = _delimited_separator(path)
    if separator is not None:
        return pd.read_csv(path, sep=separator, usecols=columns)
    raise ValueError(
        f"Unsupported input format for {path}; expected Parquet, CSV, or TSV."
    )


def _drop_tz(dates: pd.Series) -> pd.Series:
    """Strip tz-awareness from a datetime Series, leaving naive ones as-is.

    Source date columns in this pipeline are parsed independently (some via
    plain `pd.to_datetime`, some inherited tz-aware from an upstream parquet),
    so two columns being compared can disagree on tz-awareness even though
    both represent plain calendar dates. pandas raises on a naive-vs-aware
    comparison rather than assuming UTC, so normalize immediately before use.
    """
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        return dates.dt.tz_localize(None)
    return dates


def _normalize_mrn(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    if ca.ID_COL not in frame.columns:
        raise ValueError(f"{source} is missing required column {ca.ID_COL!r}.")
    out = frame.copy()
    out[ca.ID_COL] = pd.to_numeric(out[ca.ID_COL], errors="coerce")
    out = out.loc[out[ca.ID_COL].notna()].copy()
    out[ca.ID_COL] = out[ca.ID_COL].astype(int)
    return out


def load_somatic_features(
    somatic_path: Path,
    manifest_path: Path,
    *,
    variant_mode: str = "hybrid",
    somatic_table: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load per-sample somatic indicators under one feature encoding.

    ``variant_mode`` selects the encoding (SOMATIC_VARIANT_MODES):

    * ``hybrid`` -- <GENE>_SNV for every gene except TP53/PTEN/RB1, which are
      tested as one any-alteration indicator per gene (<GENE>_ANY: SNV, DEL,
      AMP, or SV). PTEN and RB1 are canonically lost by deletion, so an
      SNV-only call misses most of their carriers.
    * ``snv`` -- <GENE>_SNV for every gene, the trio included; a gene the
      panel calls only by copy number or SV drops out.
    * ``any`` -- one <GENE>_ANY indicator per gene, collapsing every variant
      class the manifest declares for it.

    ``somatic_table`` lets a caller that needs several encodings read the wide
    matrix once; it is copied, never mutated.
    """
    if variant_mode not in SOMATIC_VARIANT_MODES:
        raise ValueError(
            f"Unknown somatic variant_mode {variant_mode!r}; expected one of "
            f"{SOMATIC_VARIANT_MODES}."
        )
    manifest = _read_table(manifest_path)
    if "FEATURE" not in manifest.columns:
        raise ValueError(f"Somatic manifest {manifest_path} is missing 'FEATURE'.")
    declared = manifest["FEATURE"].dropna().astype(str).drop_duplicates().tolist()
    if not declared:
        raise ValueError(f"Somatic manifest {manifest_path} contains no features.")

    # The manifest enumerates every variant class the panel calls, so
    # copy-number and structural features the encoding does not use would
    # otherwise enter this indexed-analysis path unconditionally -- unlike
    # build_genomic_inputs.py, this builder has no --variant-types gate.
    features = []
    any_alteration_sources: dict[str, list[str]] = {}
    dropped = []
    for feature in declared:
        match = GENE_VARIANT_RE.match(feature)
        if match is None:
            dropped.append(feature)
            continue
        gene, variant = match.group(1).upper(), match.group(2)
        collapse = variant_mode == "any" or (
            variant_mode == "hybrid" and gene in ANY_ALTERATION_GENES
        )
        if collapse:
            any_feature = f"{gene}_{ANY_ALTERATION_SUFFIX}"
            if any_feature not in any_alteration_sources:
                any_alteration_sources[any_feature] = []
                features.append(any_feature)
            any_alteration_sources[any_feature].append(feature)
        elif variant == "SNV":
            features.append(feature)
        else:
            dropped.append(feature)
    if not features:
        raise ValueError(
            f"Somatic manifest {manifest_path} declares {len(declared)} features "
            f"but none are testable under variant_mode={variant_mode!r}; the "
            "somatic arm has nothing to test."
        )
    if dropped:
        print(
            f"[somatic-features:{variant_mode}] dropped {len(dropped)} somatic "
            f"features of {len(declared)} declared; kept {len(features)}. "
            f"First dropped: {dropped[:5]}"
        )
    if variant_mode == "any":
        print(
            f"[somatic-features:any] {len(any_alteration_sources)} genes collapsed "
            f"from {sum(map(len, any_alteration_sources.values()))} variant columns"
        )
    else:
        for any_feature, sources in any_alteration_sources.items():
            print(f"[somatic-features:{variant_mode}] {any_feature} = any of {sources}")

    somatic = _normalize_mrn(
        _read_table(somatic_path) if somatic_table is None else somatic_table,
        source=str(somatic_path),
    )
    collision = [f for f in any_alteration_sources if f in somatic.columns]
    if collision:
        raise ValueError(
            f"Somatic matrix {somatic_path} already has columns {collision}; "
            "refusing to overwrite them with the derived any-alteration indicators."
        )
    source_columns = [
        column for column in features if column not in any_alteration_sources
    ] + [column for sources in any_alteration_sources.values() for column in sources]
    missing = [column for column in source_columns if column not in somatic.columns]
    if missing:
        raise ValueError(
            f"Somatic matrix is missing {len(missing)} manifest features; "
            f"first values: {missing[:10]}"
        )
    if "SAMPLE_COLLECTION_DT" not in somatic.columns:
        raise ValueError(
            f"Somatic matrix {somatic_path} is missing 'SAMPLE_COLLECTION_DT', "
            "which is required as the sequencing prediction index date."
        )
    somatic[SEQUENCING_DATE] = pd.to_datetime(
        somatic["SAMPLE_COLLECTION_DT"], errors="coerce"
    )
    # Retain the old internal alias for callers that import it, but its meaning
    # is now explicitly the specimen collection/index date rather than report
    # availability.
    somatic[SOMATIC_AVAILABLE_DATE] = somatic[SEQUENCING_DATE]
    for column in source_columns:
        somatic[column] = pd.to_numeric(somatic[column], errors="coerce")
    # Altered if any class is called; wild-type only if every observed class is
    # 0; missing only when no class was observed (max skips NaN).
    for any_feature, sources in any_alteration_sources.items():
        somatic[any_feature] = somatic[sources].max(axis=1)
    return somatic[
        [ca.ID_COL, SEQUENCING_DATE, SOMATIC_AVAILABLE_DATE, *features]
    ], features


def load_gleason(gleason_path: Path) -> pd.DataFrame:
    gleason = _normalize_mrn(_read_table(gleason_path), source=str(gleason_path))
    required = {"gleason_date", "gleason_total"}
    missing = required - set(gleason.columns)
    if missing:
        raise ValueError(f"Gleason timeline is missing columns: {sorted(missing)}")
    gleason["gleason_date"] = pd.to_datetime(gleason["gleason_date"], errors="coerce")
    if "source_note_date" in gleason.columns:
        source_date = pd.to_datetime(gleason["source_note_date"], errors="coerce")
    else:
        source_date = gleason["gleason_date"]
    # A retrospectively mentioned historical score is not available until the
    # source note itself exists. Taking the later date prevents that future
    # note from leaking the historical score into an earlier landmark.
    gleason[GLEASON_AVAILABLE_DATE] = pd.concat(
        [gleason["gleason_date"], source_date], axis=1
    ).max(axis=1)
    gleason[GLEASON_FEATURE] = pd.to_numeric(gleason["gleason_total"], errors="coerce")
    return gleason[
        [ca.ID_COL, "gleason_date", GLEASON_AVAILABLE_DATE, GLEASON_FEATURE]
    ]


def load_treatment_anchors(data_path: Path) -> pd.Series:
    """Recover the patient treatment dates dropped from aggregated inputs."""
    anchors = _normalize_mrn(
        _read_table(data_path, columns=[ca.ID_COL, "TREATMENT_ANCHOR_DATE"]),
        source=str(data_path),
    )
    if "TREATMENT_ANCHOR_DATE" not in anchors.columns:
        raise ValueError(f"{data_path} is missing 'TREATMENT_ANCHOR_DATE'.")
    anchors["TREATMENT_ANCHOR_DATE"] = pd.to_datetime(
        anchors["TREATMENT_ANCHOR_DATE"], errors="coerce"
    )
    non_null = anchors.dropna(subset=["TREATMENT_ANCHOR_DATE"])
    distinct = non_null.groupby(ca.ID_COL)["TREATMENT_ANCHOR_DATE"].nunique()
    conflicts = distinct.index[distinct.gt(1)]
    if len(conflicts):
        raise ValueError(
            f"Longitudinal input has conflicting treatment anchors for {len(conflicts)} "
            f"MRNs; first values: {conflicts.tolist()[:10]}"
        )
    return non_null.groupby(ca.ID_COL)["TREATMENT_ANCHOR_DATE"].first()


def latest_available_by_landmark(
    frame: pd.DataFrame,
    cutoffs: pd.DataFrame,
    *,
    date_col: str,
    value_cols: list[str],
    order_col: str | None = None,
    combine_latest_ties_with_max: bool = False,
) -> pd.DataFrame:
    """Return the latest eligible dated row per patient in ``cutoffs``."""
    candidates = frame.merge(cutoffs, on=ca.ID_COL, how="inner", validate="many_to_one")
    candidates = candidates.loc[
        candidates[date_col].notna()
        & candidates["_landmark_date"].notna()
        & candidates[date_col].le(candidates["_landmark_date"])
    ].copy()
    if candidates.empty:
        return pd.DataFrame(index=pd.Index([], name=ca.ID_COL), columns=value_cols)
    ordering_date = order_col or date_col
    candidates = candidates.sort_values(
        [ca.ID_COL, ordering_date, date_col], kind="mergesort"
    )
    if combine_latest_ties_with_max:
        latest_date = candidates.groupby(ca.ID_COL)[ordering_date].transform("max")
        tied = candidates.loc[candidates[ordering_date].eq(latest_date)]
        return tied.groupby(ca.ID_COL, sort=False)[value_cols].max()
    return candidates.groupby(ca.ID_COL, sort=False).tail(1).set_index(ca.ID_COL)[value_cols]


def closest_observation_to_adt(
    frame: pd.DataFrame,
    treatment_anchors: pd.Series,
    *,
    date_col: str,
    value_cols: list[str],
    combine_date_ties_with_max: bool = False,
) -> pd.DataFrame:
    """Select each patient's observation closest in absolute time to ADT start.

    An equal-distance tie is resolved toward the earlier date. Multiple rows on
    the selected date can optionally be collapsed with a per-feature maximum,
    which preserves alterations split across same-day sequencing groups.
    """
    anchor_frame = treatment_anchors.rename("_adt_start_date").reset_index()
    candidates = frame.merge(
        anchor_frame, on=ca.ID_COL, how="inner", validate="many_to_one"
    )
    candidates = candidates.loc[
        candidates[date_col].notna() & candidates["_adt_start_date"].notna()
    ].copy()
    output_cols = [INDEX_DATE, INDEX_TO_ADT_DAYS, *value_cols]
    if candidates.empty:
        return pd.DataFrame(index=pd.Index([], name=ca.ID_COL), columns=output_cols)

    candidates[INDEX_TO_ADT_DAYS] = (
        candidates[date_col] - candidates["_adt_start_date"]
    ).dt.days
    candidates["_absolute_distance_days"] = candidates[INDEX_TO_ADT_DAYS].abs()
    candidates = candidates.sort_values(
        [ca.ID_COL, "_absolute_distance_days", date_col], kind="mergesort"
    )
    best_distance = candidates.groupby(ca.ID_COL)[
        "_absolute_distance_days"
    ].transform("min")
    nearest = candidates.loc[
        candidates["_absolute_distance_days"].eq(best_distance)
    ].copy()
    # If observations are equally distant on opposite sides of ADT, use the
    # earlier one. This decision is outcome-independent.
    preferred_date = nearest.groupby(ca.ID_COL)[date_col].transform("min")
    nearest = nearest.loc[nearest[date_col].eq(preferred_date)].copy()
    nearest[INDEX_DATE] = nearest[date_col]

    if combine_date_ties_with_max:
        aggregations = {
            INDEX_DATE: "first",
            INDEX_TO_ADT_DAYS: "first",
            **{feature: "max" for feature in value_cols},
        }
        return nearest.groupby(ca.ID_COL, sort=False).agg(aggregations)[output_cols]
    return (
        nearest.groupby(ca.ID_COL, sort=False)
        .head(1)
        .set_index(ca.ID_COL)[output_cols]
    )


def closest_available_by_landmark(
    frame: pd.DataFrame,
    cutoffs: pd.DataFrame,
    *,
    date_col: str,
    available_date_col: str,
    value_cols: list[str],
    combine_date_ties_with_max: bool = False,
) -> pd.DataFrame:
    """Select the closest documented value to each landmark without leakage.

    Closeness is based on the clinical observation date, while eligibility is
    based on the date the datum was available. Exact-distance ties prefer the
    earlier observation; same-date somatic rows can be unioned with a maximum.
    """
    candidates = frame.merge(cutoffs, on=ca.ID_COL, how="inner", validate="many_to_one")
    candidates = candidates.loc[
        candidates[date_col].notna()
        & candidates[available_date_col].notna()
        & candidates["_landmark_date"].notna()
        & candidates[available_date_col].le(candidates["_landmark_date"])
    ].copy()
    if candidates.empty:
        return pd.DataFrame(index=pd.Index([], name=ca.ID_COL), columns=value_cols)
    candidates["_absolute_distance_days"] = (
        candidates[date_col] - candidates["_landmark_date"]
    ).dt.days.abs()
    candidates = candidates.sort_values(
        [ca.ID_COL, "_absolute_distance_days", date_col], kind="mergesort"
    )
    best_distance = candidates.groupby(ca.ID_COL)["_absolute_distance_days"].transform("min")
    nearest = candidates.loc[candidates["_absolute_distance_days"].eq(best_distance)].copy()
    preferred_date = nearest.groupby(ca.ID_COL)[date_col].transform("min")
    nearest = nearest.loc[nearest[date_col].eq(preferred_date)]
    if combine_date_ties_with_max:
        return nearest.groupby(ca.ID_COL, sort=False)[value_cols].max()
    return nearest.groupby(ca.ID_COL, sort=False).head(1).set_index(ca.ID_COL)[value_cols]


def _rebase_endpoint_from_index(
    base: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    feature_cols: list[str],
    endpoint: str,
) -> pd.DataFrame:
    """Join selected features and rebase survival from their index date."""
    metadata = [ca.ID_COL]
    metadata += [column for column in base.columns if column in ca.outcome_columns()]
    out = base[metadata].set_index(ca.ID_COL).join(selected, how="inner")

    endpoint_cfg = {
        "platinum": ("PLATINUM", "PLATINUM_DATE", "t_platinum"),
        "nepc": ("NEPC", "NEPC_DATE", "t_nepc"),
        "avpc": ("AVPC", "AVPC_DATE", "t_avpc"),
    }
    if endpoint not in endpoint_cfg:
        raise ValueError(f"Unknown endpoint {endpoint!r}; expected one of {sorted(endpoint_cfg)}")
    event_col, event_date_col, duration_col = endpoint_cfg[endpoint]
    required = {event_col, event_date_col, "LAST_CONTACT_DATE", INDEX_DATE}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(
            "Index-date survival rebasing requires columns: "
            f"{sorted(required)}; missing {sorted(missing)}."
        )
    out[INDEX_DATE] = pd.to_datetime(out[INDEX_DATE], errors="coerce")
    out[event_date_col] = pd.to_datetime(out[event_date_col], errors="coerce")
    out["LAST_CONTACT_DATE"] = pd.to_datetime(
        out["LAST_CONTACT_DATE"], errors="coerce"
    )
    event = pd.to_numeric(out[event_col], errors="coerce").fillna(0).eq(1)
    followup_end = out["LAST_CONTACT_DATE"].where(~event, out[event_date_col])
    out[duration_col] = (followup_end - out[INDEX_DATE]).dt.days.astype(float)
    out[event_col] = event.astype(int)

    valid = out[INDEX_DATE].notna() & followup_end.notna() & out[duration_col].gt(0)
    out = out.loc[valid].copy()
    for feature in feature_cols:
        out[feature] = pd.to_numeric(out[feature], errors="coerce")
    return out.rename_axis(ca.ID_COL).reset_index()


def _materialize_absolute_followup_dates(
    base: pd.DataFrame, anchors: pd.Series
) -> pd.DataFrame:
    """Restore absolute dates dropped from aggregated landmark inputs.

    Every modeled endpoint duration COMPASS emits -- ``t_platinum``,
    ``t_nepc``, ``t_avpc``, ``t_last_contact`` -- is days from ADT start.
    Therefore adding one back to the ADT anchor exactly reconstructs the
    absolute event/censoring date needed for observation-date rebasing.

    This map must carry an entry for every endpoint in
    ``_rebase_endpoint_from_index``: ``build_prediction_inputs`` strips all raw
    date columns from the aggregated table, so an endpoint missing here fails
    there with a missing-column error rather than being silently mis-rebased.

    For a censored patient the endpoint duration falls back to
    ``t_last_contact``, so the reconstructed event date is that patient's last
    contact rather than a real event. That is safe because the caller reads
    the event date only where the event indicator is 1.
    """
    out = base.copy()
    anchor_by_row = out[ca.ID_COL].map(anchors)
    duration_by_date = {
        "PLATINUM_DATE": "t_platinum",
        "NEPC_DATE": "t_nepc",
        "AVPC_DATE": "t_avpc",
        "LAST_CONTACT_DATE": "t_last_contact",
    }
    for date_col, duration_col in duration_by_date.items():
        if date_col not in out.columns and duration_col not in out.columns:
            continue
        existing = (
            pd.to_datetime(out[date_col], errors="coerce")
            if date_col in out.columns
            else pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
        )
        if existing.notna().all():
            out[date_col] = existing
            continue
        if duration_col not in out.columns:
            raise ValueError(
                f"Cannot reconstruct {date_col}: baseline inputs are missing "
                f"ADT-relative duration {duration_col!r}."
            )
        reconstructed = anchor_by_row + pd.to_timedelta(
            pd.to_numeric(out[duration_col], errors="coerce"), unit="D"
        )
        out[date_col] = existing.fillna(reconstructed)
    return out


def build_indexed_feature_sets(
    base: pd.DataFrame,
    sequencing_inputs: dict[str, tuple[pd.DataFrame, list[str]]],
    gleason: pd.DataFrame,
    *,
    treatment_anchors: pd.Series | None = None,
    endpoint: str = "platinum",
) -> dict[str, pd.DataFrame]:
    """Build the Gleason-indexed cohort and one sequencing-indexed cohort per encoding.

    ``sequencing_inputs`` maps each sequencing analysis name (a key of
    SEQUENCING_ANALYSIS_VARIANT_MODES) to its ``load_somatic_features`` output.
    Every encoding selects the same specimen, so the sequencing cohorts share
    patients and index dates and differ only in their feature columns. The
    any-alteration cohort also carries SOMATIC_BURDEN_COVARIATE.
    """
    unknown = set(sequencing_inputs) - set(SEQUENCING_ANALYSIS_VARIANT_MODES)
    if unknown:
        raise ValueError(
            f"Unknown sequencing analyses {sorted(unknown)}; expected keys of "
            f"{sorted(SEQUENCING_ANALYSIS_VARIANT_MODES)}."
        )
    base = _normalize_mrn(base, source="landmark +0 base inputs")
    if "TREATMENT_ANCHOR_DATE" in base.columns:
        anchors = pd.Series(
            pd.to_datetime(base["TREATMENT_ANCHOR_DATE"], errors="coerce").values,
            index=pd.Index(base[ca.ID_COL], name=ca.ID_COL),
        )
    elif treatment_anchors is None:
        raise ValueError(
            "Base inputs omit 'TREATMENT_ANCHOR_DATE' and no treatment-anchor "
            "mapping was supplied."
        )
    else:
        anchors = treatment_anchors.reindex(base[ca.ID_COL])
        anchors.index = pd.Index(base[ca.ID_COL], name=ca.ID_COL)
    if anchors.isna().any():
        missing_mrns = anchors.index[anchors.isna()].tolist()
        raise ValueError(
            f"Missing ADT anchors for {len(missing_mrns)} baseline-cohort MRNs; "
            f"first values: {missing_mrns[:10]}"
        )
    base = _materialize_absolute_followup_dates(base, anchors)

    outputs: dict[str, pd.DataFrame] = {}
    for analysis, (somatic, somatic_features) in sequencing_inputs.items():
        selected_somatic = closest_observation_to_adt(
            somatic,
            anchors,
            date_col=SEQUENCING_DATE,
            value_cols=somatic_features,
            combine_date_ties_with_max=True,
        )
        sequencing = _rebase_endpoint_from_index(
            base, selected_somatic, feature_cols=somatic_features, endpoint=endpoint
        )
        for feature in somatic_features:
            sequencing[feature] = sequencing[feature].fillna(0)
        if SEQUENCING_ANALYSIS_VARIANT_MODES[analysis] == "any":
            # Counted across every tested gene (after the wild-type fill), so
            # each per-gene fit is adjusted for the same overall burden.
            sequencing[SOMATIC_BURDEN_COVARIATE] = (
                sequencing[somatic_features].sum(axis=1).astype(int)
            )
        outputs[analysis] = sequencing

    selected_gleason = closest_observation_to_adt(
        gleason,
        anchors,
        date_col="gleason_date",
        value_cols=[GLEASON_FEATURE],
    )
    outputs["gleason"] = _rebase_endpoint_from_index(
        base, selected_gleason, feature_cols=[GLEASON_FEATURE], endpoint=endpoint
    )
    return outputs


def build_available_case_sensitivity(
    base: pd.DataFrame,
    somatic: pd.DataFrame,
    somatic_features: list[str],
    gleason: pd.DataFrame,
    *,
    treatment_anchors: pd.Series,
    landmark_day: int = 0,
    analysis: str = "gleason",
) -> pd.DataFrame:
    """Return one source-specific, ADT-landmarked sensitivity cohort.

    Each source-specific cohort has its own matched labs comparator. A datum
    must be available by the requested ADT-relative landmark. Gleason selects
    the score date closest to that landmark (among available values); somatic
    selects the latest available specimen. Outcome, split assignment, and lab
    summaries remain exactly those from the standard input at that landmark.

    The Gleason ∩ somatic cohort is not built here: anything using somatic
    data is anchored on the sequencing date (build_sequencing_anchored_case).
    """
    base = _normalize_mrn(base, source="landmark +0 base inputs")
    cutoff_dates = treatment_anchors + pd.to_timedelta(int(landmark_day), unit="D")
    cutoffs = pd.DataFrame(
        {ca.ID_COL: cutoff_dates.index, "_landmark_date": cutoff_dates.values}
    )
    if analysis == "gleason":
        features = closest_available_by_landmark(
            gleason,
            cutoffs,
            date_col="gleason_date",
            available_date_col=GLEASON_AVAILABLE_DATE,
            value_cols=[GLEASON_FEATURE],
        ).dropna(subset=[GLEASON_FEATURE])
    elif analysis == "somatic":
        features = latest_available_by_landmark(
            somatic,
            cutoffs,
            date_col=SOMATIC_AVAILABLE_DATE,
            order_col=SEQUENCING_DATE,
            value_cols=somatic_features,
            combine_latest_ties_with_max=True,
        )
    else:
        raise ValueError(
            "analysis must be 'gleason' or 'somatic'; the Gleason + somatic cohort "
            "is sequencing-anchored (build_sequencing_anchored_case)."
        )
    out = base.set_index(ca.ID_COL).join(features, how="inner")
    if out.empty:
        raise ValueError(
            f"No patients have {analysis} data available by landmark +{landmark_day}d; "
            "cannot build the matched sensitivity inputs."
        )
    return out.rename_axis(ca.ID_COL).reset_index()


def _cutoff_frame(origin_dates: pd.Series, landmark_day: int) -> pd.DataFrame:
    """MRN -> ``_landmark_date`` (origin + landmark days) for the selectors."""
    cutoff_dates = origin_dates + pd.to_timedelta(int(landmark_day), unit="D")
    return pd.DataFrame(
        {ca.ID_COL: cutoff_dates.index, "_landmark_date": cutoff_dates.values}
    )


def select_sequencing_specimen(
    somatic: pd.DataFrame,
    somatic_features: list[str],
    treatment_anchors: pd.Series,
) -> pd.DataFrame:
    """Pick each patient's sequencing specimen: the one closest to ADT start.

    This is the same specimen the sequencing-indexed screens use (earlier date
    wins an exact-distance tie; same-date testing groups are unioned with a
    maximum). Returns one row per MRN with SEQUENCING_DATE,
    SEQUENCING_TO_ADT_DAYS and the somatic features.
    """
    selected = closest_observation_to_adt(
        somatic,
        treatment_anchors,
        date_col=SEQUENCING_DATE,
        value_cols=somatic_features,
        combine_date_ties_with_max=True,
    )
    selected = selected.rename(
        columns={INDEX_DATE: SEQUENCING_DATE, INDEX_TO_ADT_DAYS: SEQUENCING_TO_ADT_DAYS}
    )
    selected[SEQUENCING_DATE] = pd.to_datetime(selected[SEQUENCING_DATE])
    selected[SEQUENCING_TO_ADT_DAYS] = selected[SEQUENCING_TO_ADT_DAYS].astype(float)
    return selected.rename_axis(ca.ID_COL)


def reclock_to_sequencing(
    longitudinal: pd.DataFrame, sequencing_to_adt_days: pd.Series
) -> pd.DataFrame:
    """Shift an ADT-anchored longitudinal frame onto a sequencing-date clock.

    Every ``t_*`` column in the COMPASS longitudinal CSV (``t_lab`` and every
    outcome duration) is days from TREATMENT_ANCHOR_DATE, so subtracting the
    per-patient sequencing-minus-ADT offset makes time 0 the sequencing date.
    Age moves to age at sequencing. Patients without a specimen are dropped.
    Re-clocking (rather than passing a sequencing anchor column to
    make_outcome_df) keeps patients sequenced before ADT, whose offset is
    negative and would fail that function's ``anchor >= 0`` gate.
    """
    out = longitudinal.loc[
        longitudinal[ca.ID_COL].isin(sequencing_to_adt_days.index)
    ].copy()
    shift = out[ca.ID_COL].map(sequencing_to_adt_days).astype(float)
    for column in [col for col in out.columns if col.startswith("t_")]:
        out[column] = pd.to_numeric(out[column], errors="coerce") - shift
    age_col = survival_cohort.AGE_COL
    if age_col in out.columns:
        out[age_col] = pd.to_numeric(out[age_col], errors="coerce") + shift / 365.25
    return out


def build_sequencing_anchored_case(
    reclocked: pd.DataFrame,
    specimen: pd.DataFrame,
    somatic_features: list[str],
    gleason: pd.DataFrame,
    *,
    split: pd.Series,
    landmark_day: int,
    max_followup_days: float | None,
    endpoint: str = "platinum",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the Gleason + somatic + labs cohort at sequencing + ``landmark_day``.

    ``reclocked`` is the longitudinal frame from reclock_to_sequencing, so the
    standard COMPASS landmark builder (anchor_col=None) gives outcomes from
    sequencing + D, lab summaries from labs before sequencing + D, and the
    base run's administrative censoring. Somatic features are the selected
    specimen's (available by construction, since it defines time 0); Gleason
    is the score closest to sequencing + D among those documented by then.
    Only MRNs in the base split are kept, so the base PSA/PARPi cohort filters
    and the held-out test assignment both carry over.

    Returns (aggregated table with DFCI_MRN as a column, pre-landmark lab long).
    """
    _, _, merged = build_landmark_merged(
        reclocked,
        landmark_offset_days=landmark_day,
        anchor_col=None,
        require_first_treatment=False,
        max_followup_days=max_followup_days,
        endpoint=endpoint,
    )
    merged = merged.loc[merged.index.isin(split.index)]
    sequencing_dates = specimen[SEQUENCING_DATE]
    gleason_values = closest_available_by_landmark(
        gleason,
        _cutoff_frame(sequencing_dates, landmark_day),
        date_col="gleason_date",
        available_date_col=GLEASON_AVAILABLE_DATE,
        value_cols=[GLEASON_FEATURE],
    ).dropna(subset=[GLEASON_FEATURE])
    features = gleason_values.join(specimen[somatic_features], how="inner")
    aggregated = build_aggregated_table(merged, split=split).join(features, how="inner")
    if aggregated.empty:
        raise ValueError(
            f"No patients have Gleason and sequencing data at sequencing +{landmark_day}d; "
            "cannot build the sequencing-anchored inputs."
        )
    labs = build_pre_treatment_lab_long(
        reclocked,
        cohort_index=aggregated.index,
        landmark_offset_days=landmark_day,
        anchor_col=None,
    )
    return aggregated.rename_axis(ca.ID_COL).reset_index(), labs


def _available_case_manifest(
    base_manifest: dict,
    available_by_landmark: dict[int, pd.DataFrame],
    *,
    prediction_time_origin: str = "ADT start",
) -> dict:
    """Clone the base manifest with horizons estimable in the available cases."""
    manifest = dict(base_manifest)
    time_unit_days = int(manifest["auc_time_unit_days"])
    quantiles = tuple(manifest["auc_quantiles"])
    admin_days = int(manifest["auc_max_time_units"]) * time_unit_days
    horizons_by_landmark: dict[str, dict[str, list[int]]] = {}
    n_patients_by_landmark: dict[str, int] = {}
    split_sizes_by_landmark: dict[str, dict[str, int]] = {}
    for landmark_day, available in available_by_landmark.items():
        train_val = available.loc[available["split"].isin(["train", "valid"])]
        horizons: dict[str, list[int]] = {}
        for endpoint, cfg in ca.ENDPOINTS.items():
            if not {cfg["duration_col"], cfg["event_col"]}.issubset(train_val.columns):
                continue
            # The available-case cohorts are small subsets of the base cohort
            # (patients with Gleason/somatic data by the landmark), so an
            # endpoint that has events in the full cohort can have zero here
            # -- compute_horizon_grid requires at least one event that is
            # still "observed" after IT'S OWN admin-censoring cap is applied
            # (events past the cap are recoded to censored there). Replicate
            # that recoding here so this guard agrees with what the call
            # below will actually see, rather than just checking event==1 and
            # duration>0, which can pass while every one of those events gets
            # zeroed out by the cap inside compute_horizon_grid. Skip the
            # endpoint rather than failing the whole manifest; it just isn't
            # estimable in this available-case subset.
            event = pd.to_numeric(train_val[cfg["event_col"]], errors="coerce")
            duration = pd.to_numeric(train_val[cfg["duration_col"]], errors="coerce")
            censored_event = event.where(duration <= admin_days, 0)
            if not ((censored_event == 1) & duration.notna() & (duration > 0)).any():
                continue
            horizons[endpoint] = [
                int(value)
                for value in compute_horizon_grid(
                    train_val,
                    duration_col=cfg["duration_col"],
                    event_col=cfg["event_col"],
                    quantiles=quantiles,
                    time_unit_days=time_unit_days,
                    admin_censor_days=admin_days,
                )
            ]
        landmark_key = str(int(landmark_day))
        horizons_by_landmark[landmark_key] = horizons
        n_patients_by_landmark[landmark_key] = int(len(available))
        n_by_split = available["split"].value_counts().to_dict()
        split_sizes_by_landmark[landmark_key] = {
            key: int(n_by_split.get(key, 0)) for key in ("train", "valid", "test")
        }
    manifest.update(
        {
            "feature_set": "source_specific_available_case_sensitivity",
            "prediction_time_origin": prediction_time_origin,
            "landmark_days": sorted(available_by_landmark),
            "n_patients": int(n_patients_by_landmark.get("0", 0)),
            "n_patients_by_landmark": n_patients_by_landmark,
            "split_sizes_by_landmark": split_sizes_by_landmark,
            "auc_horizons_by_landmark": horizons_by_landmark,
        }
    )
    return manifest


def load_stage(stage_path: Path) -> pd.DataFrame:
    """Load the note-level regex stage table used by prepare_figure_data.py.

    Reads the same parquet as `METASTATIC_SOURCES["stage"]` in
    06_prep_figure_data.ipynb (columns DFCI_MRN, EVENT_DATE,
    DERIVED_STAGE_MERGED). Read directly rather than imported from
    prepare_figure_data.py, since both files live in this repo -- the
    no-cross-repo-imports rule is only about the separate CTEP repo.
    """
    stage = _normalize_mrn(
        _read_table(stage_path, columns=[ca.ID_COL, "EVENT_DATE", STAGE_COLUMN]),
        source=str(stage_path),
    )
    stage["EVENT_DATE"] = pd.to_datetime(stage["EVENT_DATE"], errors="coerce")
    stage = stage.dropna(subset=["EVENT_DATE", STAGE_COLUMN])
    return stage[[ca.ID_COL, "EVENT_DATE", STAGE_COLUMN]]


def build_clinical_stratifiers(
    base: pd.DataFrame,
    somatic_unfiltered: pd.DataFrame,
    trio_columns: list[str],
    gleason: pd.DataFrame,
    stage: pd.DataFrame,
    *,
    treatment_anchors: pd.Series,
    landmark_day: int = 0,
) -> pd.DataFrame:
    """Build one row per patient of clinical-only stratifier columns.

    These columns (Gleason, the TP53/PTEN/RB1 trio, and overall stage) are
    used only to stratify held-out risk-score figures (Plan §4) -- never as
    model features -- so the trio is taken from the unfiltered somatic wide
    table (one column per alteration class), not the collapsed <GENE>_ANY
    model features.

    ``treatment_anchors`` is the per-patient time origin the landmark is
    measured from: ADT start for the full cohort, or the sequencing date for
    the sequencing-anchored cohort (build_sequencing_clinical_stratifiers).
    Pass no ``trio_columns`` to omit the trio.

    Every column is independently left-joined onto the full landmark cohort,
    so a patient missing one clinical stratifier still gets the others rather
    than being dropped entirely.
    """
    base = _normalize_mrn(base, source="landmark base inputs")
    cutoffs = _cutoff_frame(treatment_anchors, landmark_day)

    gleason_values = closest_available_by_landmark(
        gleason,
        cutoffs,
        date_col="gleason_date",
        available_date_col=GLEASON_AVAILABLE_DATE,
        value_cols=[GLEASON_FEATURE],
    ).dropna(subset=[GLEASON_FEATURE])

    trio_values = latest_available_by_landmark(
        somatic_unfiltered,
        cutoffs,
        date_col=SOMATIC_AVAILABLE_DATE,
        order_col=SEQUENCING_DATE,
        value_cols=trio_columns,
        combine_latest_ties_with_max=True,
    ) if trio_columns else pd.DataFrame(
        index=pd.Index([], name=ca.ID_COL), columns=trio_columns
    )

    stage_candidates = stage.merge(cutoffs, on=ca.ID_COL, how="inner")
    # EVENT_DATE (from the regex stage parquet) and _landmark_date (derived
    # from TREATMENT_ANCHOR_DATE) are not guaranteed to agree on tz-awareness
    # -- pandas raises rather than comparing naive vs tz-aware datetimes. Both
    # represent calendar dates with no meaningful intraday/tz semantics, so
    # drop tz on whichever side has it immediately before comparing.
    event_date = _drop_tz(stage_candidates["EVENT_DATE"])
    landmark_date = _drop_tz(stage_candidates["_landmark_date"])
    stage_candidates = stage_candidates.loc[
        event_date.notna()
        & landmark_date.notna()
        & event_date.le(landmark_date)
    ]
    if stage_candidates.empty:
        stage_values = pd.DataFrame(
            index=pd.Index([], name=ca.ID_COL), columns=[STAGE_COLUMN]
        )
    else:
        stage_candidates = stage_candidates.sort_values(
            [ca.ID_COL, "EVENT_DATE"], kind="mergesort"
        )
        stage_values = (
            stage_candidates.groupby(ca.ID_COL, sort=False)
            .tail(1)
            .set_index(ca.ID_COL)[[STAGE_COLUMN]]
        )

    out = base[[ca.ID_COL]].set_index(ca.ID_COL)
    out = out.join(gleason_values, how="left")
    out = out.join(trio_values, how="left")
    out = out.join(stage_values, how="left")
    return out.rename_axis(ca.ID_COL).reset_index()


def sequencing_clinical_stratifiers_filename(landmark_day: int) -> str:
    return f"clinical_stratifiers_sequencing_landmark{int(landmark_day)}.csv"


def build_sequencing_clinical_stratifiers(
    base: pd.DataFrame,
    somatic_unfiltered: pd.DataFrame,
    trio_columns: list[str],
    gleason: pd.DataFrame,
    stage: pd.DataFrame,
    *,
    specimen: pd.DataFrame,
    landmark_day: int = 0,
) -> pd.DataFrame:
    """Clinical stratifiers for the sequencing-anchored cohort at sequencing + D.

    The trio comes from the same specimen the cohort is anchored on (rows of
    the unfiltered matrix on that patient's selected SEQUENCING_DATE), so a
    later specimen collected before sequencing + D can never replace it.
    Gleason and stage are those available by sequencing + D.
    """
    selected_dates = specimen[[SEQUENCING_DATE]].rename_axis(ca.ID_COL).reset_index()
    specimen_rows = somatic_unfiltered.merge(
        selected_dates, on=[ca.ID_COL, SEQUENCING_DATE], how="inner"
    )
    return build_clinical_stratifiers(
        base,
        specimen_rows,
        trio_columns,
        gleason,
        stage,
        treatment_anchors=specimen[SEQUENCING_DATE],
        landmark_day=landmark_day,
    )


def main(args: argparse.Namespace) -> None:
    base_inputs_dir = Path(args.base_inputs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_manifest_path = base_inputs_dir / BUILD_MANIFEST_FILENAME
    if not base_manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {base_manifest_path}. Build the standard prediction inputs first."
        )
    base_manifest = json.loads(base_manifest_path.read_text())
    longitudinal_value = base_manifest.get("data")
    if not longitudinal_value:
        raise ValueError(f"{base_manifest_path} has no longitudinal 'data' path.")
    longitudinal_path = Path(longitudinal_value)
    treatment_anchors = load_treatment_anchors(longitudinal_path)
    print(
        f"Recovered treatment anchors for {len(treatment_anchors):,} patients "
        f"from {longitudinal_path}."
    )
    if args.landmark_days and [int(value) for value in args.landmark_days] != [0]:
        raise ValueError(
            "The sequencing/Gleason index-date arm is baseline-only; "
            "--landmark-days must be exactly 0."
        )

    # Read the wide somatic matrix once and derive every encoding from it: the
    # hybrid features feed the multivariable available-case arms below, the
    # snv/any encodings feed the two sequencing-indexed univariate screens.
    somatic_table = _read_table(Path(args.somatic_path))
    somatic, somatic_features = load_somatic_features(
        Path(args.somatic_path),
        Path(args.somatic_manifest_path),
        somatic_table=somatic_table,
    )
    sequencing_inputs = {
        analysis: load_somatic_features(
            Path(args.somatic_path),
            Path(args.somatic_manifest_path),
            variant_mode=variant_mode,
            somatic_table=somatic_table,
        )
        for analysis, variant_mode in SEQUENCING_ANALYSIS_VARIANT_MODES.items()
    }
    del somatic_table
    gleason = load_gleason(Path(args.gleason_path))
    print(
        f"Loaded {len(somatic):,} somatic testing groups with {len(somatic_features):,} "
        "hybrid features ("
        + ", ".join(
            f"{analysis}: {len(features):,}"
            for analysis, (_frame, features) in sequencing_inputs.items()
        )
        + f"); {gleason[ca.ID_COL].nunique():,} patients have Gleason timeline rows."
    )

    base_path = base_inputs_dir / aggregated_filename(0)
    if not base_path.exists():
        raise FileNotFoundError(f"Missing base landmark input: {base_path}")
    source_labs = base_inputs_dir / pre_treatment_lab_filename(0)
    if not source_labs.exists():
        raise FileNotFoundError(f"Missing base pre-landmark lab input: {source_labs}")
    feature_sets = build_indexed_feature_sets(
        pd.read_csv(base_path, low_memory=False),
        sequencing_inputs,
        gleason,
        treatment_anchors=treatment_anchors,
        endpoint=args.endpoint,
    )

    # SOMATIC_BURDEN_COVARIATE is deliberately absent from every feature
    # manifest: it is an adjustment covariate, never a tested feature.
    feature_rows_by_analysis = {
        analysis: [
            {
                "feature": feature,
                "feature_kind": "somatic_binary",
                "source": str(args.somatic_path),
            }
            for feature in features
        ]
        for analysis, (_frame, features) in sequencing_inputs.items()
    }
    feature_rows_by_analysis["gleason"] = [
        {
            "feature": GLEASON_FEATURE,
            "feature_kind": "gleason_continuous",
            "source": str(args.gleason_path),
        }
    ]

    cohort_sizes: dict[str, int] = {}
    for analysis in INDEX_ANALYSES:
        analysis_dir = output_dir / analysis
        analysis_dir.mkdir(parents=True, exist_ok=True)
        built = feature_sets[analysis]
        output_path = analysis_dir / aggregated_filename(0)
        built.to_csv(output_path, index=False)
        cohort_sizes[analysis] = len(built)
        pd.DataFrame(feature_rows_by_analysis[analysis]).to_csv(
            analysis_dir / FEATURE_MANIFEST_FILENAME, index=False
        )
        pd.DataFrame(columns=[ca.ID_COL, "LAB_NAME"]).to_csv(
            analysis_dir / pre_treatment_lab_filename(0), index=False
        )

        index_source = "sequencing" if analysis in SEQUENCING_ANALYSIS_VARIANT_MODES else analysis
        variant_mode = SEQUENCING_ANALYSIS_VARIANT_MODES.get(analysis)
        manifest = dict(base_manifest)
        manifest.update(
            {
                "feature_set": "somatic_gleason",
                "index_analysis": analysis,
                "base_inputs_dir": str(base_inputs_dir),
                "treatment_anchor_source": str(longitudinal_path),
                "somatic_path": str(args.somatic_path),
                "somatic_manifest_path": str(args.somatic_manifest_path),
                "gleason_path": str(args.gleason_path),
                "landmark_days": [0],
                "n_patients_by_landmark": {"0": len(built)},
                "n_patients": len(built),
                "prediction_time_origin": INDEX_DATE,
                "index_selection": "minimum absolute days from ADT start; earlier date wins exact tie",
                "outcome": f"days from selected index date to {args.endpoint} or last contact",
                "endpoint_description": (
                    f"Time from the selected {index_source} index date to {args.endpoint}"
                ),
                "endpoint": args.endpoint,
                "patients_with_nonpositive_followup_excluded": True,
                "somatic_variant_mode": variant_mode,
                "burden_covariate": (
                    SOMATIC_BURDEN_COVARIATE if variant_mode == "any" else None
                ),
            }
        )
        (analysis_dir / BUILD_MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2)
        )
        print(
            f"{analysis}: {len(built):,} patients indexed from {INDEX_DATE} -> {output_path}"
        )

    # Unlike the three observation-indexed univariate analyses above, the
    # Gleason sensitivity preserves the standard ADT outcome clock and split
    # assignment, with its own labs comparator, so Gleason-vs-labs never
    # conflates a feature comparison with a cohort shift.
    sensitivity_landmarks = [int(value) for value in base_manifest["landmark_days"]]
    sensitivity_cases: dict[str, dict[int, pd.DataFrame]] = {}
    gleason_feature_rows = [
        {"feature": GLEASON_FEATURE, "feature_kind": "gleason_continuous", "source": str(args.gleason_path)}
    ]
    somatic_feature_rows = [
        {"feature": feature, "feature_kind": "somatic_binary", "source": str(args.somatic_path)}
        for feature in somatic_features
    ]
    for analysis, sensitivity_name in zip(("gleason",), AVAILABLE_CASE_SENSITIVITIES):
        sensitivity_dir = output_dir / sensitivity_name
        sensitivity_dir.mkdir(parents=True, exist_ok=True)
        available_cases: dict[int, pd.DataFrame] = {}
        for landmark_day in sensitivity_landmarks:
            landmark_base_path = base_inputs_dir / aggregated_filename(landmark_day)
            landmark_labs_path = base_inputs_dir / pre_treatment_lab_filename(landmark_day)
            if not landmark_base_path.exists() or not landmark_labs_path.exists():
                raise FileNotFoundError(
                    "Available-case sensitivity requires standard aggregate and lab inputs at "
                    f"landmark +{landmark_day}d; missing {landmark_base_path} or {landmark_labs_path}."
                )
            available = build_available_case_sensitivity(
                pd.read_csv(landmark_base_path, low_memory=False),
                somatic,
                somatic_features,
                gleason,
                treatment_anchors=treatment_anchors,
                landmark_day=landmark_day,
                analysis=analysis,
            )
            available_cases[landmark_day] = available
            available.to_csv(sensitivity_dir / aggregated_filename(landmark_day), index=False)
            available_mrns = set(available[ca.ID_COL])
            pd.read_csv(landmark_labs_path, low_memory=False).loc[
                lambda frame: frame[ca.ID_COL].isin(available_mrns)
            ].to_csv(sensitivity_dir / pre_treatment_lab_filename(landmark_day), index=False)
            print(
                f"{sensitivity_name} +{landmark_day}d: {len(available):,} patients "
                f"with {analysis} data available by the landmark"
            )
        pd.DataFrame(gleason_feature_rows).to_csv(
            sensitivity_dir / FEATURE_MANIFEST_FILENAME, index=False
        )
        sensitivity_manifest = _available_case_manifest(base_manifest, available_cases)
        sensitivity_manifest.update(
            {
                "base_inputs_dir": str(base_inputs_dir),
                "sensitivity_analysis": analysis,
                "somatic_path": str(args.somatic_path),
                "somatic_manifest_path": str(args.somatic_manifest_path),
                "gleason_path": str(args.gleason_path),
            }
        )
        (sensitivity_dir / BUILD_MANIFEST_FILENAME).write_text(
            json.dumps(sensitivity_manifest, indent=2)
        )
        sensitivity_cases[analysis] = available_cases

    # Gleason + somatic + labs: anchored on the sequencing date. Every arm of
    # the clinical combinations (labs alone included) shares this cohort, so
    # the comparison is on one clock and one patient set.
    split_path = base_inputs_dir / SPLIT_ASSIGNMENTS_FILENAME
    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing {split_path}; the sequencing-anchored cohort inherits the base split."
        )
    if str(base_manifest.get("split_mode", "shared")) != "shared":
        raise ValueError(
            "The sequencing-anchored cohort needs one split across landmarks; rebuild "
            "the base inputs with --split-mode shared."
        )
    split = (
        _normalize_mrn(pd.read_csv(split_path), source=str(split_path))
        .set_index(ca.ID_COL)["split"]
        .astype(str)
    )
    specimen = select_sequencing_specimen(somatic, somatic_features, treatment_anchors)
    specimen = specimen.loc[specimen.index.isin(split.index)]
    for feature in somatic_features:
        specimen[feature] = pd.to_numeric(specimen[feature], errors="coerce")
    offsets = specimen[SEQUENCING_TO_ADT_DAYS]
    print(
        f"Sequencing anchor: {len(specimen):,} base-cohort patients with a specimen; "
        f"sequencing minus ADT start median {offsets.median():.0f}d "
        f"(IQR {offsets.quantile(0.25):.0f} to {offsets.quantile(0.75):.0f}d), "
        f"{int(offsets.lt(0).sum()):,} sequenced before ADT."
    )
    longitudinal = _normalize_mrn(
        pd.read_csv(longitudinal_path, low_memory=False), source=str(longitudinal_path)
    )
    reclocked = reclock_to_sequencing(longitudinal, offsets)
    del longitudinal
    max_followup_days = base_manifest.get("max_followup_days")
    sequencing_dir = output_dir / GLEASON_SOMATIC_AVAILABLE_CASE
    sequencing_dir.mkdir(parents=True, exist_ok=True)
    sequencing_cases: dict[int, pd.DataFrame] = {}
    for landmark_day in sensitivity_landmarks:
        print(f"\n##### SEQUENCING-ANCHORED COHORT: SEQUENCING +{landmark_day} DAYS #####")
        aggregated, labs = build_sequencing_anchored_case(
            reclocked,
            specimen,
            somatic_features,
            gleason,
            split=split,
            landmark_day=landmark_day,
            max_followup_days=max_followup_days,
            endpoint=args.endpoint,
        )
        sequencing_cases[landmark_day] = aggregated
        aggregated.to_csv(sequencing_dir / aggregated_filename(landmark_day), index=False)
        labs.to_csv(sequencing_dir / pre_treatment_lab_filename(landmark_day), index=False)
        print(
            f"{GLEASON_SOMATIC_AVAILABLE_CASE} sequencing +{landmark_day}d: "
            f"{len(aggregated):,} patients with Gleason and sequencing data"
        )
    pd.DataFrame(gleason_feature_rows + somatic_feature_rows).to_csv(
        sequencing_dir / FEATURE_MANIFEST_FILENAME, index=False
    )
    sequencing_manifest = _available_case_manifest(
        base_manifest, sequencing_cases, prediction_time_origin="sequencing date"
    )
    sequencing_manifest.update(
        {
            "feature_set": "sequencing_anchored_gleason_somatic",
            "base_inputs_dir": str(base_inputs_dir),
            "sensitivity_analysis": "gleason_somatic",
            "time_origin_column": SEQUENCING_DATE,
            "specimen_selection": (
                "sample collection date closest to ADT start; earlier date wins an "
                "exact tie; same-date testing groups unioned with max"
            ),
            "landmark_definition": "sequencing date + landmark days",
            "lab_window": "labs drawn before sequencing date + landmark days",
            "outcome": f"days from sequencing date + landmark to {args.endpoint} or last contact",
            "endpoint": args.endpoint,
            "n_sequenced_before_adt": int(offsets.lt(0).sum()),
            "somatic_path": str(args.somatic_path),
            "somatic_manifest_path": str(args.somatic_manifest_path),
            "gleason_path": str(args.gleason_path),
        }
    )
    (sequencing_dir / BUILD_MANIFEST_FILENAME).write_text(
        json.dumps(sequencing_manifest, indent=2)
    )

    # Plan §4a: clinical stratifier inputs for risk_score_stratified_figures.py.
    # Read the trio columns from the UNFILTERED somatic wide table (every
    # alteration class, one column per class), never the collapsed <GENE>_ANY
    # model features -- these are stratifiers only, never model features.
    stratifiers_dir = output_dir / CLINICAL_STRATIFIERS_DIRNAME
    stratifiers_dir.mkdir(parents=True, exist_ok=True)
    raw_somatic = _normalize_mrn(
        _read_table(Path(args.somatic_path)), source=str(args.somatic_path)
    )
    raw_somatic[SEQUENCING_DATE] = pd.to_datetime(
        raw_somatic["SAMPLE_COLLECTION_DT"], errors="coerce"
    )
    raw_somatic[SOMATIC_AVAILABLE_DATE] = raw_somatic[SEQUENCING_DATE]
    trio_columns = [
        f"{gene}_{alteration}"
        for gene in CLINICAL_STRATIFIERS_TRIO_GENES
        for alteration in CLINICAL_STRATIFIERS_ALTERATION_CLASSES
        if f"{gene}_{alteration}" in raw_somatic.columns
    ]
    for column in trio_columns:
        raw_somatic[column] = pd.to_numeric(raw_somatic[column], errors="coerce")
    missing_trio = [
        f"{gene}_{alteration}"
        for gene in CLINICAL_STRATIFIERS_TRIO_GENES
        for alteration in CLINICAL_STRATIFIERS_ALTERATION_CLASSES
        if f"{gene}_{alteration}" not in raw_somatic.columns
    ]
    if missing_trio:
        print(
            f"[clinical-stratifiers] {len(missing_trio)} trio columns absent from "
            f"{args.somatic_path}; skipping: {missing_trio}"
        )

    stage_path = Path(args.regex_stage_path)
    if stage_path.exists():
        stage = load_stage(stage_path)
    else:
        print(f"[clinical-stratifiers] stage source not found at {stage_path}; skipping stage.")
        stage = pd.DataFrame(columns=[ca.ID_COL, "EVENT_DATE", STAGE_COLUMN])

    for landmark_day in sensitivity_landmarks:
        # Full ADT cohort: Gleason and stage only. The trio is somatic, so it is
        # stratified only on the sequencing-anchored cohort below.
        landmark_base_path = base_inputs_dir / aggregated_filename(landmark_day)
        landmark_base = pd.read_csv(landmark_base_path, low_memory=False)
        clinical = build_clinical_stratifiers(
            landmark_base,
            raw_somatic,
            [],
            gleason,
            stage,
            treatment_anchors=treatment_anchors,
            landmark_day=landmark_day,
        )
        clinical_path = stratifiers_dir / f"clinical_stratifiers_landmark{landmark_day}.csv"
        clinical.to_csv(clinical_path, index=False)
        print(
            f"clinical_stratifiers +{landmark_day}d: {len(clinical):,} patients -> {clinical_path}"
        )
        sequencing_clinical = build_sequencing_clinical_stratifiers(
            sequencing_cases[landmark_day],
            raw_somatic,
            trio_columns,
            gleason,
            stage,
            specimen=specimen,
            landmark_day=landmark_day,
        )
        sequencing_clinical_path = stratifiers_dir / sequencing_clinical_stratifiers_filename(
            landmark_day
        )
        sequencing_clinical.to_csv(sequencing_clinical_path, index=False)
        print(
            f"clinical_stratifiers sequencing +{landmark_day}d: "
            f"{len(sequencing_clinical):,} patients -> {sequencing_clinical_path}"
        )

    root_manifest = {
        "feature_set": "somatic_gleason_indexed",
        "analyses": list(INDEX_ANALYSES),
        "cohort_sizes": cohort_sizes,
        "available_case_sensitivities": {
            "gleason": {
                "path": AVAILABLE_CASE_SENSITIVITIES[0],
                "prediction_time_origin": "ADT start",
                "n_patients_by_landmark": {
                    str(landmark_day): int(len(available))
                    for landmark_day, available in sensitivity_cases["gleason"].items()
                },
                "comparison": "labs versus gleason, same available cases at each landmark",
            },
            "gleason_somatic": {
                "path": GLEASON_SOMATIC_AVAILABLE_CASE,
                "prediction_time_origin": "sequencing date",
                "n_patients_by_landmark": {
                    str(landmark_day): int(len(available))
                    for landmark_day, available in sequencing_cases.items()
                },
                "comparison": (
                    "labs versus gleason/somatic combinations, same patients at each "
                    "sequencing landmark"
                ),
            },
        },
    }
    (output_dir / BUILD_MANIFEST_FILENAME).write_text(
        json.dumps(root_manifest, indent=2)
    )
    print(f"Wrote indexed-analysis manifest: {output_dir / BUILD_MANIFEST_FILENAME}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-inputs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", choices=sorted(ca.ENDPOINTS), default="platinum")
    parser.add_argument("--somatic-path", type=Path, default=DEFAULT_SOMATIC_PATH)
    parser.add_argument(
        "--somatic-manifest-path", type=Path, default=DEFAULT_SOMATIC_MANIFEST_PATH
    )
    parser.add_argument("--gleason-path", type=Path, default=DEFAULT_GLEASON_PATH)
    parser.add_argument(
        "--regex-stage-path",
        type=Path,
        default=DEFAULT_REGEX_STAGE_PATH,
        help=(
            "Note-level regex stage parquet (DFCI_MRN, EVENT_DATE, "
            "DERIVED_STAGE_MERGED), same source as prepare_figure_data.py's "
            "metastatic_sources['stage']. Used for the Plan §4a clinical "
            "stratifier's stage column."
        ),
    )
    parser.add_argument("--landmark-days", nargs="+", type=int, default=None)
    main(parser.parse_args())
