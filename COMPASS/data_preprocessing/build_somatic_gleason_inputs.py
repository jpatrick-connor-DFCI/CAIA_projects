"""Build a separate somatic-feature + Gleason + biomarker-PRS univariate arm.

The risk sets, outcomes, and train/valid/test assignments are inherited from
the standard COMPASS landmark inputs.  Features come from the two upstream
artifacts named in the project workflow:

* PROFILE_data_processing/SOMATIC_WIDE_BY_SAMPLE.parquet
* LLM_clinical_annotations/LLM_gleason_timeline/gleason_timeline.parquet
* clinical_text_embedding_project/complete_germline_data_df.csv.gz

For each treatment landmark, the most recent somatic testing group whose
result was available by the landmark and the most recent dated Gleason score
at or before the landmark are selected.  A patient without an eligible
somatic test remains missing for every somatic feature; this deliberately
does not equate "not tested" with wild type.
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
DEFAULT_SOMATIC_PATH = PROFILE_DATA_ROOT / "SOMATIC_WIDE_BY_SAMPLE.parquet"
DEFAULT_SOMATIC_MANIFEST_PATH = PROFILE_DATA_ROOT / "SOMATIC_FEATURE_MANIFEST.parquet"
DEFAULT_GLEASON_PATH = LLM_ANNOTATIONS_ROOT / "LLM_gleason_timeline/gleason_timeline.parquet"
DEFAULT_PRS_PATH = (
    CLINICAL_EMBEDDINGS_ROOT
    / "clinical_and_genomic_features/complete_germline_data_df.csv.gz"
)

FEATURE_MANIFEST_FILENAME = "somatic_gleason_features.csv"
GLEASON_FEATURE = "GLEASON_SCORE"
SOMATIC_AVAILABLE_DATE = "_somatic_available_date"
GLEASON_AVAILABLE_DATE = "_gleason_available_date"

# PGS Catalog scores mapped specifically to PSA level or testosterone level.
# The COMPASS cohort is male, so female-specific testosterone scores are
# intentionally excluded. The loader intersects this allowlist with the
# locally generated matrix, allowing older matrix snapshots to omit newer IDs.
# Trait sources: OBA_2050200 (PSA amount) and EFO_0004908 (testosterone).
PRS_TRAITS = {
    "PSA": (
        "PGS003378",
        "PGS003379",
        "PGS005098",
        "PGS005099",
        "PGS005100",
        "PGS005101",
        "PGS005107",
    ),
    "TESTOSTERONE": (
        "PGS000321",  # combined-sex serum testosterone
        "PGS000323",  # male-specific serum testosterone
        "PGS000696",  # testosterone concentration
        "PGS001988",  # male-specific log testosterone
        "PGS002205",  # male-specific log testosterone
        "PGS003559",  # male-specific log testosterone
    ),
}


def _read_table(path: Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=columns)
    if path.suffix.lower() == ".csv" or path.name.lower().endswith(".csv.gz"):
        return pd.read_csv(path, usecols=columns)
    raise ValueError(f"Unsupported input format for {path}; expected Parquet or CSV.")


def _table_columns(path: Path) -> list[str]:
    """Read a table schema without materializing the potentially wide PRS matrix."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".csv" or path.name.lower().endswith(".csv.gz"):
        return pd.read_csv(path, nrows=0).columns.tolist()
    if path.suffix.lower() in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pyarrow is required to inspect a Parquet PRS matrix."
            ) from exc
        return pq.ParquetFile(path).schema.names
    raise ValueError(f"Unsupported input format for {path}; expected Parquet or CSV.")


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


def _prs_id_in_column(column: str, pgs_id: str) -> bool:
    normalized = str(column).upper().replace("-", "_").replace(".", "_")
    return normalized == pgs_id or normalized.startswith(f"{pgs_id}_")


def load_biomarker_prs(prs_path: Path) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Load only PSA/testosterone PGS columns and collapse identical MRN duplicates."""
    header = _table_columns(prs_path)
    if ca.ID_COL not in header:
        raise ValueError(f"PRS input {prs_path} is missing required column {ca.ID_COL!r}.")

    selected: list[tuple[str, str, str]] = []
    for trait, pgs_ids in PRS_TRAITS.items():
        for pgs_id in pgs_ids:
            matches = [column for column in header if _prs_id_in_column(column, pgs_id)]
            if len(matches) > 1:
                raise ValueError(
                    f"PRS input has multiple columns matching {pgs_id}: {matches}. "
                    "Provide a matrix with one score column per PGS ID."
                )
            if matches:
                selected.append((matches[0], trait, pgs_id))
    if not selected:
        expected = [pgs_id for values in PRS_TRAITS.values() for pgs_id in values]
        raise ValueError(
            f"PRS input {prs_path} has none of the PSA/testosterone scores: {expected}"
        )

    raw_columns = [source for source, _, _ in selected]
    prs = _normalize_mrn(
        _read_table(prs_path, columns=[ca.ID_COL, *raw_columns]), source=str(prs_path)
    )
    manifest_rows = []
    rename = {}
    for source, trait, pgs_id in selected:
        feature = f"PRS_{trait}_{pgs_id}"
        rename[source] = feature
        manifest_rows.append(
            {
                "feature": feature,
                "feature_kind": "prs_continuous",
                "source": str(prs_path),
                "pgs_id": pgs_id,
                "reported_trait_group": trait,
                "source_column": source,
            }
        )
    prs = prs.rename(columns=rename)
    feature_cols = [row["feature"] for row in manifest_rows]
    for feature in feature_cols:
        prs[feature] = pd.to_numeric(prs[feature], errors="coerce")

    if prs[ca.ID_COL].duplicated().any():
        conflicts = prs.groupby(ca.ID_COL)[feature_cols].nunique(dropna=True).gt(1).any(axis=1)
        if conflicts.any():
            bad = conflicts.index[conflicts].tolist()[:10]
            raise ValueError(f"PRS input has conflicting duplicate rows for MRNs: {bad}")
        prs = prs.groupby(ca.ID_COL, as_index=False)[feature_cols].first()
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
) -> pd.DataFrame:
    base = _normalize_mrn(base, source=f"landmark +{landmark_day} base inputs")
    if "TREATMENT_ANCHOR_DATE" not in base.columns:
        raise ValueError("Base inputs are missing 'TREATMENT_ANCHOR_DATE'.")
    anchor = pd.to_datetime(base["TREATMENT_ANCHOR_DATE"], errors="coerce")
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
    requested_landmarks = (
        [int(value) for value in args.landmark_days]
        if args.landmark_days
        else [int(value) for value in base_manifest["landmark_days"]]
    )

    somatic, somatic_features = load_somatic_features(
        Path(args.somatic_path), Path(args.somatic_manifest_path)
    )
    gleason = load_gleason(Path(args.gleason_path))
    prs, prs_manifest_rows = load_biomarker_prs(Path(args.prs_path))
    prs_features = [row["feature"] for row in prs_manifest_rows]
    print(
        f"Loaded {len(somatic):,} somatic testing groups with {len(somatic_features):,} features; "
        f"{gleason[ca.ID_COL].nunique():,} patients have Gleason timeline rows; "
        f"{prs[ca.ID_COL].nunique():,} patients have {len(prs_features)} PSA/testosterone PRSs."
    )

    feature_rows = [
        {"feature": feature, "feature_kind": "somatic_binary", "source": str(args.somatic_path)}
        for feature in somatic_features
    ]
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
        )
        output_path = output_dir / aggregated_filename(landmark_day)
        built.to_csv(output_path, index=False)
        cohort_sizes[str(landmark_day)] = len(built)
        n_somatic = int(built[somatic_features].notna().any(axis=1).sum())
        n_gleason = int(built[GLEASON_FEATURE].notna().sum())
        n_prs = int(built[prs_features].notna().any(axis=1).sum())
        print(
            f"Landmark +{landmark_day}d: {len(built):,} patients; "
            f"somatic={n_somatic:,}, Gleason={n_gleason:,}, PRS={n_prs:,} -> {output_path}"
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
            "somatic_path": str(args.somatic_path),
            "somatic_manifest_path": str(args.somatic_manifest_path),
            "gleason_path": str(args.gleason_path),
            "prs_path": str(args.prs_path),
            "landmark_days": requested_landmarks,
            "n_patients_by_landmark": cohort_sizes,
            "n_somatic_features": len(somatic_features),
            "gleason_feature": GLEASON_FEATURE,
            "prs_features": prs_features,
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
    parser.add_argument("--prs-path", type=Path, default=DEFAULT_PRS_PATH)
    parser.add_argument("--landmark-days", nargs="+", type=int, default=None)
    main(parser.parse_args())
