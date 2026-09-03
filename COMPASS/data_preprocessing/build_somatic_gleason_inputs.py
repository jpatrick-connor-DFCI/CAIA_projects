"""Build sequencing-, Gleason-, and PRS-indexed univariate Cox inputs.

The baseline ADT cohort and train/valid/test assignments are inherited from
the standard COMPASS landmark-0 inputs. Three distinct analyses are built:

* ``sequencing`` selects the sample-collection date closest to ADT start and
  predicts from that collection date to platinum (or last contact).
* ``gleason`` selects the Gleason score date closest to ADT start and predicts
  from that score date to platinum (or last contact).
* ``prs`` uses ADT start as time zero and predicts from ADT start to platinum
  (or last contact), because germline scores have no observation-date clock.

Closest means minimum absolute calendar distance, so an observation may fall
before or after ADT. An exact-distance tie prefers the earlier date. Patients
whose selected index date is on/after platinum or last contact are excluded;
the builder never substitutes a different observation based on the outcome.

The PRS matrix is bridged from sample ID to MRN and duplicate sample rows are
averaged per patient, as in the prior implementation.
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
    aggregated_filename,
    pre_treatment_lab_filename,
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
PROFILE_CLINICAL_ROOT = Path(
    os.environ.get("PROFILE_CLINICAL_PATH", "/data/gusev/PROFILE/CLINICAL/")
)
DEFAULT_SOMATIC_PATH = PROFILE_DATA_ROOT / "SOMATIC_WIDE_BY_SAMPLE.parquet"
DEFAULT_SOMATIC_MANIFEST_PATH = PROFILE_DATA_ROOT / "SOMATIC_FEATURE_MANIFEST.parquet"
DEFAULT_GLEASON_PATH = LLM_ANNOTATIONS_ROOT / "LLM_gleason_timeline/gleason_timeline.parquet"
# Raw PGS catalog matrix, keyed on IID (== cbio_sample_id), not DFCI_MRN.
DEFAULT_PRS_PATH = Path(
    os.environ.get(
        "PRS_MATRIX_FILE",
        "/data/gusev/USERS/mjsaleh/PRS_PGScatalog/pgs_matrix_with_avg.tsv",
    )
)
DEFAULT_IDMAP_PATH = PROFILE_CLINICAL_ROOT / "PROFILE_2024_idmap.csv"

FEATURE_MANIFEST_FILENAME = "somatic_gleason_features.csv"
GLEASON_FEATURE = "GLEASON_SCORE"
SOMATIC_AVAILABLE_DATE = "_somatic_available_date"
GLEASON_AVAILABLE_DATE = "_gleason_available_date"
SOMATIC_TESTED_FEATURE = "HAS_SOMATIC_TEST"  # legacy compatibility constant
SEQUENCING_DATE = "SEQUENCING_DATE"
INDEX_DATE = "INDEX_DATE"
INDEX_TO_ADT_DAYS = "INDEX_TO_ADT_DAYS"
INDEX_ANALYSES = ("gleason", "sequencing", "prs")
PRS_SAMPLE_ID_COL = "cbio_sample_id"

# Exact PGS IDs from the user-supplied complete_germline_data_df column list.
# Matching on the terminal PGS ID is less brittle than repeating the long
# human-readable prefix, while still selecting exactly those columns because
# each PGS ID occurs once in the matrix. Original column names are preserved in
# model outputs and in the feature manifest.
PRS_PGS_IDS_OF_INTEREST = (
    "PGS000030", "PGS000044", "PGS000049", "PGS000067", "PGS000084",
    "PGS000086", "PGS000160", "PGS000198", "PGS000321", "PGS000322",
    "PGS000323", "PGS000333", "PGS000342", "PGS000348",
    *(f"PGS{value:06d}" for value in range(565, 593)),
    *(f"PGS{value:06d}" for value in range(595, 605)),
    "PGS000662", "PGS000696", "PGS000714", "PGS000719", "PGS000733",
    "PGS000741", "PGS000742", "PGS000751", "PGS000795", "PGS000796",
    "PGS000878", "PGS000881", "PGS000940", "PGS000997", "PGS001015",
    "PGS001164", "PGS001291", "PGS001292", "PGS001338", "PGS001516",
    "PGS001805", "PGS001806", "PGS001865", "PGS001914", "PGS001988",
    "PGS002016", "PGS002076", "PGS002130", "PGS002205", "PGS002240",
    "PGS002241", "PGS002268", "PGS002747", "PGS002791", "PGS002792",
    "PGS002793", "PGS002796", "PGS002797", "PGS002798", "PGS002799",
    "PGS003331", "PGS003378", "PGS003379", "PGS003383", "PGS003415",
    "PGS003418", "PGS003419", "PGS003460", "PGS003507", "PGS003559",
    "PGS003743", "PGS003765", "PGS003766", "PGS003985", "PGS004001",
    "PGS004027", "PGS004042", "PGS004055", "PGS004071", "PGS004085",
    "PGS004099", "PGS004109", "PGS004125", "PGS004139", "PGS004155",
    "PGS004251", "PGS004320", "PGS004474", "PGS004475", "PGS004500",
    "PGS004544", "PGS004545", "PGS004570", "PGS004581", "PGS004599",
    "PGS004601", "PGS004694", "PGS004815", "PGS004816", "PGS004872",
    "PGS005107",
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


def _table_columns(path: Path) -> list[str]:
    """Read a table schema without materializing the potentially wide PRS matrix."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    separator = _delimited_separator(path)
    if separator is not None:
        return pd.read_csv(path, sep=separator, nrows=0).columns.tolist()
    if path.suffix.lower() in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pyarrow is required to inspect a Parquet PRS matrix."
            ) from exc
        return pq.ParquetFile(path).schema.names
    raise ValueError(
        f"Unsupported input format for {path}; expected Parquet, CSV, or TSV."
    )


def _normalize_mrn(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    if ca.ID_COL not in frame.columns:
        raise ValueError(f"{source} is missing required column {ca.ID_COL!r}.")
    out = frame.copy()
    out[ca.ID_COL] = pd.to_numeric(out[ca.ID_COL], errors="coerce")
    out = out.loc[out[ca.ID_COL].notna()].copy()
    out[ca.ID_COL] = out[ca.ID_COL].astype(int)
    return out


def load_somatic_features(
    somatic_path: Path, manifest_path: Path
) -> tuple[pd.DataFrame, list[str]]:
    manifest = _read_table(manifest_path)
    if "FEATURE" not in manifest.columns:
        raise ValueError(f"Somatic manifest {manifest_path} is missing 'FEATURE'.")
    declared = manifest["FEATURE"].dropna().astype(str).drop_duplicates().tolist()
    if not declared:
        raise ValueError(f"Somatic manifest {manifest_path} contains no features.")

    # The genomic arm is standardized to SNV-only. The manifest enumerates every
    # variant class the panel calls, so copy-number and structural features
    # would otherwise enter this indexed-analysis path unconditionally -- unlike
    # build_genomic_inputs.py, this builder has no --variant-types gate.
    features = []
    dropped = []
    for feature in declared:
        match = GENE_VARIANT_RE.match(feature)
        (features if match and match.group(2) == "SNV" else dropped).append(feature)
    if not features:
        raise ValueError(
            f"Somatic manifest {manifest_path} declares {len(declared)} features "
            "but none are <GENE>_SNV columns; the SNV-only genomic arm has "
            "nothing to test."
        )
    if dropped:
        print(
            f"[snv-only] dropped {len(dropped)} non-SNV somatic features of "
            f"{len(declared)} declared; kept {len(features)}. "
            f"First dropped: {dropped[:5]}"
        )

    somatic = _normalize_mrn(_read_table(somatic_path), source=str(somatic_path))
    missing = [column for column in features if column not in somatic.columns]
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
    for feature in features:
        somatic[feature] = pd.to_numeric(somatic[feature], errors="coerce")
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


def _prs_id_in_column(column: str, pgs_id: str) -> bool:
    normalized = str(column).upper().replace("-", "_").replace(".", "_")
    return normalized == pgs_id or normalized.endswith(f"_{pgs_id}")


def _prs_trait_group(column: str) -> str:
    normalized = str(column).lower()
    if "testosterone" in normalized:
        return "TESTOSTERONE"
    if "prostate_specific_antigen" in normalized or "psa_pgs" in normalized:
        return "PSA"
    if "psoriatic_arthritis" in normalized:
        return "PSORIATIC_ARTHRITIS"
    if "testis" in normalized or "testicular" in normalized:
        return "TESTICULAR"
    if "prostate" in normalized or "prca" in normalized or "pca" in normalized:
        return "PROSTATE"
    return "OTHER"


def load_prs_idmap(idmap_path: Path) -> pd.DataFrame:
    """Load the cbio_sample_id -> DFCI_MRN bridge for the raw PGS matrix.

    The PGS matrix is keyed on a sample identifier that carries no MRN, so this
    map is the only route to a patient. Rows lacking either key are unusable
    and dropped.
    """
    idmap = _read_table(idmap_path, columns=[ca.ID_COL, PRS_SAMPLE_ID_COL])
    missing = {ca.ID_COL, PRS_SAMPLE_ID_COL} - set(idmap.columns)
    if missing:
        raise ValueError(f"Idmap {idmap_path} is missing columns: {sorted(missing)}")
    idmap = _normalize_mrn(idmap, source=str(idmap_path))
    idmap[PRS_SAMPLE_ID_COL] = idmap[PRS_SAMPLE_ID_COL].astype(str).str.strip()
    idmap = idmap.loc[idmap[PRS_SAMPLE_ID_COL].ne("")]
    # One sample belongs to exactly one patient; a sample mapped to several MRNs
    # would fan rows out in the join below and silently duplicate scores.
    conflicting = idmap.groupby(PRS_SAMPLE_ID_COL)[ca.ID_COL].nunique()
    conflicts = conflicting.index[conflicting.gt(1)]
    if len(conflicts):
        raise ValueError(
            f"Idmap {idmap_path} maps {len(conflicts)} sample id(s) to more than one "
            f"{ca.ID_COL}; first values: {conflicts.tolist()[:10]}"
        )
    return idmap.drop_duplicates([PRS_SAMPLE_ID_COL, ca.ID_COL])


def load_biomarker_prs(
    prs_path: Path,
    *,
    require_all: bool = False,
    idmap_path: Path | None = None,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Load the supplied PGS allowlist and collapse sample rows by patient.

    The raw PGS catalog matrix is keyed on ``IID`` (a cbio sample id), so it is
    bridged to ``DFCI_MRN`` through ``idmap_path``; patients absent from the
    idmap have no reachable germline data. A matrix already keyed on
    ``DFCI_MRN`` is accepted as-is and needs no bridge.

    One patient may have several genotyped samples. PRSs are continuous
    germline estimates, so repeated sample rows are reduced to their per-score
    arithmetic mean. This is a no-op for identical duplicates and avoids
    choosing an arbitrary sample when technical estimates differ slightly.
    """
    header = _table_columns(prs_path)
    keyed_by_mrn = ca.ID_COL in header
    sample_key = next(
        (column for column in ("IID", PRS_SAMPLE_ID_COL) if column in header), None
    )
    if not keyed_by_mrn and sample_key is None:
        raise ValueError(
            f"PRS input {prs_path} has neither {ca.ID_COL!r} nor a sample key "
            f"('IID'/{PRS_SAMPLE_ID_COL!r}); cannot identify patients."
        )

    selected: list[tuple[str, str]] = []
    missing_ids = []
    for pgs_id in PRS_PGS_IDS_OF_INTEREST:
        matches = [column for column in header if _prs_id_in_column(column, pgs_id)]
        if len(matches) > 1:
            raise ValueError(
                f"PRS input has multiple columns matching {pgs_id}: {matches}. "
                "Provide a matrix with one score column per PGS ID."
            )
        if matches:
            selected.append((matches[0], pgs_id))
        else:
            missing_ids.append(pgs_id)
    if not selected:
        raise ValueError(
            f"PRS input {prs_path} has none of the {len(PRS_PGS_IDS_OF_INTEREST)} "
            "allowlisted PGS IDs."
        )
    if missing_ids:
        if require_all:
            raise ValueError(
                f"PRS input {prs_path} is missing {len(missing_ids)} allowlisted PGS IDs: "
                f"{missing_ids}"
            )
        print(
            f"WARNING: {len(missing_ids)} of {len(PRS_PGS_IDS_OF_INTEREST)} allowlisted "
            f"PGS IDs are absent from {prs_path} and will be omitted; "
            f"first values: {missing_ids[:10]}"
        )

    raw_columns = [source for source, _ in selected]
    if keyed_by_mrn:
        prs = _normalize_mrn(
            _read_table(prs_path, columns=[ca.ID_COL, *raw_columns]),
            source=str(prs_path),
        )
    else:
        if idmap_path is None:
            raise ValueError(
                f"PRS input {prs_path} is keyed on {sample_key!r} and needs an idmap "
                "to reach DFCI_MRN, but no idmap path was supplied."
            )
        idmap = load_prs_idmap(idmap_path)
        prs = _read_table(prs_path, columns=[sample_key, *raw_columns])
        prs = prs.rename(columns={sample_key: PRS_SAMPLE_ID_COL})
        prs[PRS_SAMPLE_ID_COL] = prs[PRS_SAMPLE_ID_COL].astype(str).str.strip()
        n_samples = prs[PRS_SAMPLE_ID_COL].nunique()
        # Inner join: an unmapped sample cannot be attributed to a patient, and
        # an unmapped patient has no germline data to attribute.
        prs = prs.merge(idmap, on=PRS_SAMPLE_ID_COL, how="inner", validate="many_to_one")
        print(
            f"PRS idmap join: {n_samples:,} genotyped samples -> "
            f"{prs[ca.ID_COL].nunique():,} patients via {idmap_path}."
        )
        if prs.empty:
            raise ValueError(
                f"PRS idmap join produced no rows; {prs_path} and {idmap_path} share "
                "no sample identifiers."
            )
        prs = prs.drop(columns=[PRS_SAMPLE_ID_COL])
    manifest_rows = []
    for source, pgs_id in selected:
        manifest_rows.append(
            {
                "feature": source,
                "feature_kind": "prs_continuous",
                "source": str(prs_path),
                "pgs_id": pgs_id,
                "reported_trait_group": _prs_trait_group(source),
                "source_column": source,
            }
        )
    feature_cols = [row["feature"] for row in manifest_rows]
    numeric_scores = prs[feature_cols].apply(pd.to_numeric, errors="coerce")
    # Reassemble once so assigning 133 converted columns cannot leave a highly
    # fragmented frame before the duplicate-MRN groupby.
    prs = pd.concat(
        [prs[[ca.ID_COL]].reset_index(drop=True), numeric_scores.reset_index(drop=True)],
        axis=1,
    )

    if prs[ca.ID_COL].duplicated().any():
        row_counts = prs.groupby(ca.ID_COL).size()
        duplicate_mrns = row_counts.index[row_counts.gt(1)]
        conflicts = (
            prs.loc[prs[ca.ID_COL].isin(duplicate_mrns)]
            .groupby(ca.ID_COL)[feature_cols]
            .nunique(dropna=True)
            .gt(1)
            .any(axis=1)
        )
        print(
            f"PRS duplicate resolution: averaging sample rows for "
            f"{len(duplicate_mrns):,} MRNs; {int(conflicts.sum()):,} have at least "
            "one discordant score."
        )
        prs = prs.groupby(ca.ID_COL, as_index=False)[feature_cols].mean()
    return prs, manifest_rows


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
    somatic: pd.DataFrame,
    somatic_features: list[str],
    gleason: pd.DataFrame,
    *,
    prs: pd.DataFrame | None = None,
    prs_features: list[str] | None = None,
    treatment_anchors: pd.Series | None = None,
    endpoint: str = "platinum",
) -> dict[str, pd.DataFrame]:
    """Build Gleason-, sequencing-, and ADT-indexed PRS cohorts."""
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

    selected_somatic = closest_observation_to_adt(
        somatic,
        anchors,
        date_col=SEQUENCING_DATE,
        value_cols=somatic_features,
        combine_date_ties_with_max=True,
    )
    selected_gleason = closest_observation_to_adt(
        gleason,
        anchors,
        date_col="gleason_date",
        value_cols=[GLEASON_FEATURE],
    )
    sequencing = _rebase_endpoint_from_index(
        base, selected_somatic, feature_cols=somatic_features, endpoint=endpoint
    )
    for feature in somatic_features:
        sequencing[feature] = sequencing[feature].fillna(0)
    gleason_out = _rebase_endpoint_from_index(
        base, selected_gleason, feature_cols=[GLEASON_FEATURE], endpoint=endpoint
    )

    prs_features = list(prs_features or [])
    metadata = [ca.ID_COL]
    metadata += [column for column in base.columns if column in ca.outcome_columns()]
    prs_out = base[metadata].set_index(ca.ID_COL)
    prs_out[INDEX_DATE] = anchors
    prs_out[INDEX_TO_ADT_DAYS] = 0
    if prs is not None and prs_features:
        prs_out = prs_out.join(prs.set_index(ca.ID_COL)[prs_features], how="left")
    prs_out = prs_out.rename_axis(ca.ID_COL).reset_index()

    return {"gleason": gleason_out, "sequencing": sequencing, "prs": prs_out}


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

    somatic, somatic_features = load_somatic_features(
        Path(args.somatic_path), Path(args.somatic_manifest_path)
    )
    gleason = load_gleason(Path(args.gleason_path))
    prs, prs_manifest_rows = load_biomarker_prs(
        Path(args.prs_path),
        require_all=args.require_all_prs,
        idmap_path=Path(args.idmap_path),
    )
    prs_features = [row["feature"] for row in prs_manifest_rows]
    print(
        f"Loaded {len(somatic):,} somatic testing groups with {len(somatic_features):,} features; "
        f"{gleason[ca.ID_COL].nunique():,} patients have Gleason timeline rows; "
        f"{prs[ca.ID_COL].nunique():,} patients have {len(prs_features)} selected PRSs."
    )

    base_path = base_inputs_dir / aggregated_filename(0)
    if not base_path.exists():
        raise FileNotFoundError(f"Missing base landmark input: {base_path}")
    source_labs = base_inputs_dir / pre_treatment_lab_filename(0)
    if not source_labs.exists():
        raise FileNotFoundError(f"Missing base pre-landmark lab input: {source_labs}")
    feature_sets = build_indexed_feature_sets(
        pd.read_csv(base_path, low_memory=False),
        somatic,
        somatic_features,
        gleason,
        prs=prs,
        prs_features=prs_features,
        treatment_anchors=treatment_anchors,
        endpoint=args.endpoint,
    )

    feature_rows_by_analysis = {
        "sequencing": [
            {
                "feature": feature,
                "feature_kind": "somatic_binary",
                "source": str(args.somatic_path),
            }
            for feature in somatic_features
        ],
        "gleason": [
            {
                "feature": GLEASON_FEATURE,
                "feature_kind": "gleason_continuous",
                "source": str(args.gleason_path),
            }
        ],
        "prs": prs_manifest_rows,
    }

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
                "prediction_time_origin": (
                    "ADT start" if analysis == "prs" else INDEX_DATE
                ),
                "index_selection": (
                    "ADT start"
                    if analysis == "prs"
                    else "minimum absolute days from ADT start; earlier date wins exact tie"
                ),
                "outcome": (
                    f"days from ADT start to {args.endpoint} or last contact"
                    if analysis == "prs"
                    else f"days from selected index date to {args.endpoint} or last contact"
                ),
                "endpoint_description": (
                    f"Time from ADT start to {args.endpoint}"
                    if analysis == "prs"
                    else f"Time from the selected {analysis} index date to {args.endpoint}"
                ),
                "endpoint": args.endpoint,
                "patients_with_nonpositive_followup_excluded": True,
                "prs_included": analysis == "prs",
            }
        )
        (analysis_dir / BUILD_MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2)
        )
        print(
            f"{analysis}: {len(built):,} patients indexed from {INDEX_DATE} -> {output_path}"
        )

    root_manifest = {
        "feature_set": "somatic_gleason_indexed",
        "analyses": list(INDEX_ANALYSES),
        "cohort_sizes": cohort_sizes,
        "prs_included": True,
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
        "--prs-path",
        type=Path,
        default=DEFAULT_PRS_PATH,
        help="Raw PGS matrix keyed on IID, or a matrix already keyed on DFCI_MRN.",
    )
    parser.add_argument(
        "--idmap-path",
        type=Path,
        default=DEFAULT_IDMAP_PATH,
        help="cbio_sample_id to DFCI_MRN bridge for a sample-keyed PGS matrix.",
    )
    parser.add_argument(
        "--require-all-prs",
        action="store_true",
        help="Fail if any allowlisted PGS ID is absent from the matrix.",
    )
    parser.add_argument("--landmark-days", nargs="+", type=int, default=None)
    main(parser.parse_args())
