"""Build a separate somatic-feature + Gleason + selected-PRS univariate arm.

The risk sets, outcomes, and train/valid/test assignments are inherited from
the standard COMPASS landmark inputs.  Features come from the two upstream
artifacts named in the project workflow:

* PROFILE_data_processing/SOMATIC_WIDE_BY_SAMPLE.parquet
* LLM_clinical_annotations/LLM_gleason_timeline/gleason_timeline.parquet
* the raw PGS catalog matrix (pgs_matrix_with_avg.tsv), bridged to DFCI_MRN
  through PROFILE_2024_idmap.csv

For each treatment landmark, the most recent somatic testing group whose
result was available by the landmark and the most recent dated Gleason score
at or before the landmark are selected.

PRS is read from the raw PGS matrix rather than from the clinical text
embedding project's ``complete_germline_data_df.csv.gz``.  That derived file
is gated on a pan-cancer, note-filtered, treatment-anchored cohort built for
a different project, so joining COMPASS MRNs against it returned mostly
nulls.  The raw matrix is keyed on ``IID`` (a cbio sample id carrying no
MRN), so the idmap join remains structurally required; patients absent from
the idmap have no reachable germline data and stay missing.  Patients with
several genotyped samples are collapsed to a per-score arithmetic mean.

A patient without an eligible somatic test is treated as wild type (0) for
every somatic feature, and carries ``HAS_SOMATIC_TEST=0`` so models can still
separate an untested patient from a tested all-negative one.
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
# Marks whether a patient had any somatic result available by the landmark, so
# an untested patient stays separable from a tested all-wild-type one even
# though both carry 0 for every alteration indicator.
SOMATIC_TESTED_FEATURE = "HAS_SOMATIC_TEST"
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
    features = manifest["FEATURE"].dropna().astype(str).drop_duplicates().tolist()
    if not features:
        raise ValueError(f"Somatic manifest {manifest_path} contains no features.")

    date_cols = ["REPORT_DT", "SAMPLE_COLLECTION_DT", "TEST_ORDER_DT"]
    somatic = _normalize_mrn(_read_table(somatic_path), source=str(somatic_path))
    missing = [column for column in features if column not in somatic.columns]
    if missing:
        raise ValueError(
            f"Somatic matrix is missing {len(missing)} manifest features; "
            f"first values: {missing[:10]}"
        )
    present_dates = [column for column in date_cols if column in somatic.columns]
    if not present_dates:
        raise ValueError(
            f"Somatic matrix {somatic_path} has none of the availability-date columns {date_cols}."
        )
    for column in present_dates:
        somatic[column] = pd.to_datetime(somatic[column], errors="coerce")

    # REPORT_DT is the conservative availability time. Fall back only when it
    # is absent, then prefer order date over collection date.
    somatic[SOMATIC_AVAILABLE_DATE] = pd.NaT
    for column in ("REPORT_DT", "TEST_ORDER_DT", "SAMPLE_COLLECTION_DT"):
        if column in somatic.columns:
            somatic[SOMATIC_AVAILABLE_DATE] = somatic[SOMATIC_AVAILABLE_DATE].fillna(
                somatic[column]
            )
    for feature in features:
        somatic[feature] = pd.to_numeric(somatic[feature], errors="coerce")
    return somatic[[ca.ID_COL, SOMATIC_AVAILABLE_DATE, *features]], features


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


def build_landmark_features(
    base: pd.DataFrame,
    somatic: pd.DataFrame,
    somatic_features: list[str],
    gleason: pd.DataFrame,
    landmark_day: int,
    prs: pd.DataFrame | None = None,
    prs_features: list[str] | None = None,
    treatment_anchors: pd.Series | None = None,
) -> pd.DataFrame:
    base = _normalize_mrn(base, source=f"landmark +{landmark_day} base inputs")
    if "TREATMENT_ANCHOR_DATE" in base.columns:
        anchor = pd.to_datetime(base["TREATMENT_ANCHOR_DATE"], errors="coerce")
    elif treatment_anchors is not None:
        anchor = base[ca.ID_COL].map(treatment_anchors)
    else:
        raise ValueError(
            "Base inputs omit 'TREATMENT_ANCHOR_DATE' and no treatment-anchor "
            "mapping was supplied."
        )
    if anchor.isna().any():
        missing_mrns = base.loc[anchor.isna(), ca.ID_COL].tolist()
        raise ValueError(
            f"Missing treatment anchors for {len(missing_mrns)} landmark-cohort MRNs; "
            f"first values: {missing_mrns[:10]}"
        )
    cutoffs = pd.DataFrame(
        {
            ca.ID_COL: base[ca.ID_COL].values,
            "_landmark_date": anchor + pd.to_timedelta(int(landmark_day), unit="D"),
        }
    )

    latest_somatic = latest_available_by_landmark(
        somatic,
        cutoffs,
        date_col=SOMATIC_AVAILABLE_DATE,
        value_cols=somatic_features,
        combine_latest_ties_with_max=True,
    )
    latest_gleason = latest_available_by_landmark(
        gleason,
        cutoffs,
        date_col=GLEASON_AVAILABLE_DATE,
        value_cols=[GLEASON_FEATURE],
        order_col="gleason_date",
    )

    metadata = [ca.ID_COL]
    metadata += [column for column in base.columns if column in ca.outcome_columns()]
    out = base[metadata].set_index(ca.ID_COL)
    out = out.join(latest_somatic, how="left").join(latest_gleason, how="left")

    # Patients with no somatic result available by this landmark are treated as
    # wild type. HAS_SOMATIC_TEST records who was actually tested, so an
    # untested patient stays distinguishable from a tested all-negative one.
    out[SOMATIC_TESTED_FEATURE] = (
        out.index.isin(latest_somatic.index).astype(int)
        if len(latest_somatic.index)
        else 0
    )
    for feature in somatic_features:
        # to_numeric first: an all-missing column arrives as object dtype, and
        # a bare fillna would leave it object rather than numeric.
        out[feature] = pd.to_numeric(out[feature], errors="coerce").fillna(0)

    if prs is not None and prs_features:
        out = out.join(prs.set_index(ca.ID_COL)[prs_features], how="left")
    return out.rename_axis(ca.ID_COL).reset_index()


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
    requested_landmarks = (
        [int(value) for value in args.landmark_days]
        if args.landmark_days
        else [int(value) for value in base_manifest["landmark_days"]]
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

    feature_rows = [
        {"feature": feature, "feature_kind": "somatic_binary", "source": str(args.somatic_path)}
        for feature in somatic_features
    ]
    feature_rows.append(
        {
            "feature": SOMATIC_TESTED_FEATURE,
            "feature_kind": "somatic_tested_indicator",
            "source": str(args.somatic_path),
        }
    )
    feature_rows.append(
        {
            "feature": GLEASON_FEATURE,
            "feature_kind": "gleason_continuous",
            "source": str(args.gleason_path),
        }
    )
    feature_rows.extend(prs_manifest_rows)
    pd.DataFrame(feature_rows).to_csv(output_dir / FEATURE_MANIFEST_FILENAME, index=False)

    cohort_sizes: dict[str, int] = {}
    for landmark_day in requested_landmarks:
        base_path = base_inputs_dir / aggregated_filename(landmark_day)
        if not base_path.exists():
            raise FileNotFoundError(f"Missing base landmark input: {base_path}")
        built = build_landmark_features(
            pd.read_csv(base_path, low_memory=False),
            somatic,
            somatic_features,
            gleason,
            landmark_day,
            prs=prs,
            prs_features=prs_features,
            treatment_anchors=treatment_anchors,
        )
        output_path = output_dir / aggregated_filename(landmark_day)
        built.to_csv(output_path, index=False)
        cohort_sizes[str(landmark_day)] = len(built)
        # Alteration columns are wild-type filled, so the tested indicator -- not
        # notna() -- is what counts genuine somatic coverage here.
        n_somatic = int(built[SOMATIC_TESTED_FEATURE].sum())
        n_gleason = int(built[GLEASON_FEATURE].notna().sum())
        n_prs = int(built[prs_features].notna().any(axis=1).sum())
        print(
            f"Landmark +{landmark_day}d: {len(built):,} patients; "
            f"somatic tested={n_somatic:,} (untested filled wild type), "
            f"Gleason={n_gleason:,}, PRS={n_prs:,} -> {output_path}"
        )

        # The shared loader requires this companion filename, but this arm has
        # no lab candidates. Use a header-only table instead of duplicating the
        # potentially very large standard lab-long file.
        source_labs = base_inputs_dir / pre_treatment_lab_filename(landmark_day)
        if not source_labs.exists():
            raise FileNotFoundError(f"Missing base pre-landmark lab input: {source_labs}")
        pd.DataFrame(columns=[ca.ID_COL, "LAB_NAME"]).to_csv(
            output_dir / source_labs.name, index=False
        )

    manifest = dict(base_manifest)
    manifest.update(
        {
            "feature_set": "somatic_gleason",
            "base_inputs_dir": str(base_inputs_dir),
            "treatment_anchor_source": str(longitudinal_path),
            "somatic_path": str(args.somatic_path),
            "somatic_manifest_path": str(args.somatic_manifest_path),
            "gleason_path": str(args.gleason_path),
            "prs_path": str(args.prs_path),
            "prs_idmap_path": str(args.idmap_path),
            "prs_source": "raw PGS catalog matrix bridged via PROFILE idmap",
            "landmark_days": requested_landmarks,
            "n_patients_by_landmark": cohort_sizes,
            "n_somatic_features": len(somatic_features),
            "somatic_tested_feature": SOMATIC_TESTED_FEATURE,
            "somatic_missing_policy": (
                "untested patients filled wild type (0); HAS_SOMATIC_TEST records "
                "whether any result was available by the landmark"
            ),
            "gleason_feature": GLEASON_FEATURE,
            "prs_features": prs_features,
            "n_prs_features_requested": len(PRS_PGS_IDS_OF_INTEREST),
            "prs_duplicate_resolution": "arithmetic mean per PGS column within DFCI_MRN",
            "somatic_selection": "latest result available on or before landmark",
            "gleason_selection": "latest dated score on or before landmark",
        }
    )
    (output_dir / BUILD_MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {output_dir / BUILD_MANIFEST_FILENAME}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-inputs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--somatic-path", type=Path, default=DEFAULT_SOMATIC_PATH)
    parser.add_argument(
        "--somatic-manifest-path", type=Path, default=DEFAULT_SOMATIC_MANIFEST_PATH
    )
    parser.add_argument("--gleason-path", type=Path, default=DEFAULT_GLEASON_PATH)
    parser.add_argument(
        "--prs-path",
        type=Path,
        default=DEFAULT_PRS_PATH,
        help="Raw PGS catalog matrix (keyed on IID) or a matrix already keyed on DFCI_MRN.",
    )
    parser.add_argument(
        "--idmap-path",
        type=Path,
        default=DEFAULT_IDMAP_PATH,
        help="cbio_sample_id -> DFCI_MRN bridge; required when the PRS matrix is sample-keyed.",
    )
    parser.add_argument(
        "--require-all-prs",
        action="store_true",
        help=(
            "Fail if any allowlisted PGS ID is absent from the matrix. Off by "
            "default so a single missing score degrades instead of aborting."
        ),
    )
    parser.add_argument("--landmark-days", nargs="+", type=int, default=None)
    main(parser.parse_args())
