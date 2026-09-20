"""Build clinical-note-embedding prediction inputs for the COMPASS platinum arm.

Produces, per landmark, the matched three-arm inputs that
``03c_multivariate_text.ipynb`` fits:

* ``text/``      -- pooled note embeddings only (plus age, added downstream)
* ``labs/``      -- the standard lab summaries, refit on the *same* patients
* ``labs_text/`` -- both

The ``labs`` arm is rebuilt here rather than read from the standard landmark
inputs because the text cohort is a strict subset: the embedding project pools
only patients with all three pre-anchor note modalities (Clinician, Imaging,
Pathology). Comparing a text model on that subset against the published
``03_multivariate.ipynb`` numbers, which are fit on the full ADT cohort, would
confound the feature comparison with a cohort shift. This mirrors the
matched-comparator design already used by ``build_somatic_gleason_inputs.py``'s
``*_available_case`` outputs.

Time zero and the outcome clock are COMPASS's throughout: ADT start, with
``t_platinum``/``PLATINUM`` carried through untouched. Only the note window is
new. Notes are pooled from the embedding project's
``generate_survival_embedding_df`` with ``survival_df=None`` -- deliberately, so
that helper's own ``tt_*`` landmark shifting never touches COMPASS's outcome
columns, which ``build_prediction_inputs.py`` has already landmarked.

Leakage: note times are made ADT-relative here, then the pooling helper filters
to ``NOTE_TIME_REL_ADT < landmark_day`` and asserts the re-centered maximum is
<= 0. A note written on or after the landmark cannot enter the features.

Usage:

    python build_text_embedding_inputs.py \\
        --base-inputs-dir /path/to/prediction_inputs_adt \\
        --output-dir /path/to/prediction_inputs_adt/text_embedding
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
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
    compute_horizon_grid,
    pre_treatment_lab_filename,
)
from build_somatic_gleason_inputs import (  # noqa: E402
    _normalize_mrn,
    _read_table,
    load_treatment_anchors,
)

# The embedding project is a separate checkout, not an installed package. Its
# pooling helpers are the single implementation of the note window and the
# time-decay weighting, so they are imported rather than reimplemented.
#
# The default is the project's OWN declared cluster location, copied from the
# PROJECT_ROOT every one of its slurm/*.sh launchers defaults to. It is not
# derived from this repo's directory layout: COMPASS lives under code/CAIA/ on
# the cluster but under a differently-nested checkout locally, so any path
# computed by walking up from __file__ is right in one place and wrong in the
# other. A fixed cluster path plus CTEP_REPO_PATH is right in both.
CTEP_CLUSTER_REPO = Path(
    "/data/gusev/USERS/jpconnor/code/clinical_text_embedding_project"
)

CLINICAL_EMBEDDINGS_REPO = Path(
    os.environ.get("CTEP_REPO_PATH") or CTEP_CLUSTER_REPO
)

FEATURE_MANIFEST_FILENAME = "text_embedding_features.csv"
FEATURE_SETS = ("text", "labs", "labs_text")
NOTE_TYPES = ("Clinician", "Imaging", "Pathology")
# Matches every pooling call in the embedding project
# (generate_embedding_prediction_datasets.py, generate_mortality_trajectories.py,
# ICI_generate_embeddings.py): time-decayed mean, decay 0.01/day. Kept identical
# so this arm's features are the manuscript models' features.
POOL_STRATEGY = "time_decay_mean"
DECAY_PARAM = 0.01
NOTE_TIME_COL = "NOTE_TIME_REL_ADT"
ANCHOR = "adt"


def _import_embedding_helpers():
    """Import the sibling embedding project's pooling helpers.

    Imported lazily and by path so that ``--help`` and the argument parser work
    without the sibling checkout present, and so the failure names the env var
    to set rather than surfacing as a bare ImportError.
    """
    if not (CLINICAL_EMBEDDINGS_REPO / "anchors.py").exists():
        raise FileNotFoundError(
            f"Clinical text embedding project not found at {CLINICAL_EMBEDDINGS_REPO}.\n"
            f"        Default is the project's cluster root ({CTEP_CLUSTER_REPO}).\n"
            "        Set CTEP_REPO_PATH to the repo root (the directory containing "
            "anchors.py and survival/) to point elsewhere."
        )
    if str(CLINICAL_EMBEDDINGS_REPO) not in sys.path:
        sys.path.insert(0, str(CLINICAL_EMBEDDINGS_REPO))
    try:
        from anchors import ensure_anchor, note_time_col  # noqa: E402
        from config import NOTES_PATH  # noqa: E402
        from survival.preprocessing import (  # noqa: E402
            generate_survival_embedding_df,
        )
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            f"Failed to import the embedding project from {CLINICAL_EMBEDDINGS_REPO}: {exc}. "
            "Confirm CTEP_REPO_PATH points at the repo root and its dependencies "
            "(polars, zstandard) are installed in this environment."
        ) from exc
    # Fail loudly here if the anchor registry and this module disagree on the
    # column name, rather than silently pooling against a column of NaNs.
    ensure_anchor(ANCHOR)
    registry_col = note_time_col(ANCHOR)
    if registry_col != NOTE_TIME_COL:
        raise ValueError(
            f"anchors.py defines note_time_col({ANCHOR!r}) = {registry_col!r}, but this "
            f"builder computes {NOTE_TIME_COL!r}. Reconcile the two."
        )
    return generate_survival_embedding_df, NOTES_PATH


def load_notes_and_embeddings(notes_path: str | Path):
    """Load the knitted note metadata and the embedding matrix.

    Mirrors ``generate_embedding_prediction_datasets._load_shared_inputs``'s
    reader for these two artifacts.
    """
    import io

    import polars as pl
    import zstandard as zstd

    notes_path = Path(notes_path)
    embeddings_file = notes_path / "full_clinical_notes_embeddings_as_array.npy.zst"
    metadata_file = notes_path / "full_clinical_notes_embeddings_metadata.parquet"
    for path in (embeddings_file, metadata_file):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing knitted embedding artifact: {path}. Run the embedding "
                "project's 1_data/03_prediction_datasets.ipynb first."
            )
    with open(embeddings_file, "rb") as handle:
        embeddings = np.load(io.BytesIO(zstd.decompress(handle.read())))
    embeddings = embeddings.astype(np.float32)
    notes_meta = pl.read_parquet(metadata_file)
    return notes_meta, embeddings


def attach_adt_relative_note_times(
    notes_meta,
    treatment_anchors: pd.Series,
):
    """Add ``NOTE_TIME_REL_ADT`` (days from ADT start) to the note metadata.

    Notes belonging to patients outside the COMPASS ADT cohort are dropped:
    they have no anchor, so no ADT-relative time exists for them.
    """
    import polars as pl

    if "NOTE_DATETIME" not in notes_meta.columns:
        raise ValueError("Note metadata is missing 'NOTE_DATETIME'.")
    if "DFCI_MRN" not in notes_meta.columns:
        raise ValueError("Note metadata is missing 'DFCI_MRN'.")

    anchor_frame = pl.DataFrame(
        {
            "DFCI_MRN": treatment_anchors.index.to_numpy(),
            "_adt_start_date": pd.to_datetime(treatment_anchors.values),
        }
    ).with_columns(
        pl.col("DFCI_MRN").cast(pl.Int64, strict=False),
        pl.col("_adt_start_date").cast(pl.Datetime("us")),
    )

    out = notes_meta.with_columns(
        pl.col("DFCI_MRN").cast(pl.Int64, strict=False)
    ).join(anchor_frame, on="DFCI_MRN", how="inner")

    out = out.with_columns(
        pl.col("NOTE_DATETIME").cast(pl.Datetime("us"), strict=False)
    )
    out = out.with_columns(
        (pl.col("NOTE_DATETIME") - pl.col("_adt_start_date"))
        .dt.total_days()
        .cast(pl.Float64)
        .alias(NOTE_TIME_COL)
    ).drop("_adt_start_date")

    out = out.filter(pl.col(NOTE_TIME_COL).is_not_null())
    if out.is_empty():
        raise ValueError(
            "No notes could be dated relative to ADT start. Check that the note "
            "metadata's DFCI_MRN and NOTE_DATETIME align with the COMPASS cohort."
        )
    return out


def pool_embeddings_for_landmark(
    notes_meta,
    embeddings: np.ndarray,
    landmark_day: int,
    *,
    generate_survival_embedding_df,
) -> pd.DataFrame:
    """Pool pre-landmark notes into one row per patient.

    ``survival_df=None`` is deliberate: COMPASS owns the outcome clock, and the
    helper shifts ``tt_*`` columns by the landmark when handed a survival frame.
    ``continuous_window=False`` matches the embedding project's own prediction
    datasets, which do not require an unbroken note history.
    """
    from shared.polars_utils import filter_finite_rows

    pooled = generate_survival_embedding_df(
        notes_meta=notes_meta,
        survival_df=None,
        embedding_array=embeddings,
        note_types=list(NOTE_TYPES),
        note_timing_col=NOTE_TIME_COL,
        max_note_window=int(landmark_day),
        continuous_window=False,
        pool_fx={note_type: POOL_STRATEGY for note_type in NOTE_TYPES},
        decay_param=DECAY_PARAM,
    )
    n_any_text = len(pooled)
    feature_cols = [c for c in pooled.columns if c != "DFCI_MRN"]
    # Complete-case across all three modalities, as in the embedding project:
    # a patient missing one note type pools to NaN for that block.
    pooled = filter_finite_rows(pooled.drop_nulls("DFCI_MRN"), feature_cols)
    print(
        f"  landmark +{landmark_day}d: complete-case text cohort "
        f"{len(pooled)}/{n_any_text} patients "
        "(Clinician + Imaging + Pathology)"
    )
    if pooled.is_empty():
        raise ValueError(
            f"No patients have complete pre-landmark embeddings at +{landmark_day}d."
        )
    out = pooled.to_pandas()
    return _normalize_mrn(out, source=f"pooled embeddings (+{landmark_day}d)")


def embedding_feature_columns(pooled: pd.DataFrame) -> list[str]:
    """The pooled feature columns, in stable order.

    Covers both the ``<TYPE>_EMBEDDING_<i>`` dimensions and the two
    ``PERCENT_<TYPE>_NOTES_PRE_2015`` note-vintage covariates the pooling
    helper emits for Imaging and Pathology.
    """
    return [c for c in pooled.columns if c != ca.ID_COL]


def build_landmark_feature_sets(
    base: pd.DataFrame,
    pooled: pd.DataFrame,
    *,
    landmark_day: int,
) -> dict[str, pd.DataFrame]:
    """Return the three matched arms for one landmark.

    All three share one patient set -- the intersection of the standard landmark
    cohort and the complete-case text cohort -- so the arms differ only in which
    columns they carry, never in who is in them.
    """
    base = _normalize_mrn(base, source=f"landmark +{landmark_day}d base inputs")
    text_cols = embedding_feature_columns(pooled)

    overlapping = sorted(set(base.columns) & set(text_cols))
    if overlapping:
        raise ValueError(
            f"Embedding feature names collide with {len(overlapping)} existing "
            f"column(s) in the landmark +{landmark_day}d inputs; first values: "
            f"{overlapping[:10]}"
        )

    matched = base.merge(pooled, on=ca.ID_COL, how="inner", validate="one_to_one")
    if matched.empty:
        raise ValueError(
            f"No patients are in both the landmark +{landmark_day}d cohort and the "
            "complete-case text cohort; cannot build the matched arms."
        )

    labs_only = matched.drop(columns=text_cols)
    # The text arm keeps the outcome/metadata columns (split, duration, event,
    # age) and drops only the lab summary features, so downstream readers see an
    # ordinary landmark frame whose feature columns happen to be embeddings.
    lab_feature_cols = [
        c
        for c in labs_only.columns
        if c not in ca.non_feature_columns(labs_only) and c != ca.ID_COL
    ]
    text_only = matched.drop(columns=lab_feature_cols)

    return {
        "text": text_only,
        "labs": labs_only,
        "labs_text": matched,
    }


def _landmark_manifest(
    base_manifest: dict,
    built_by_landmark: dict[int, pd.DataFrame],
    *,
    feature_set: str,
    n_text_features: int,
    notes_path: str,
    anchor_source: str,
) -> dict:
    """Clone the base manifest with horizons recomputed on the matched cohort.

    The horizon grid is derived from observed follow-up, so it must be
    recomputed: the matched cohort is smaller than the full ADT cohort and its
    event-time quantiles differ. Reusing the base grid would evaluate AUC(t) at
    horizons this cohort cannot estimate.
    """
    manifest = dict(base_manifest)
    time_unit_days = int(manifest["auc_time_unit_days"])
    quantiles = tuple(manifest["auc_quantiles"])
    admin_days = int(manifest["auc_max_time_units"]) * time_unit_days

    horizons_by_landmark: dict[str, dict[str, list[int]]] = {}
    n_patients_by_landmark: dict[str, int] = {}
    split_sizes_by_landmark: dict[str, dict[str, int]] = {}
    for landmark_day, built in sorted(built_by_landmark.items()):
        train_val = built.loc[built["split"].isin(["train", "valid"])]
        horizons: dict[str, list[int]] = {}
        for endpoint, cfg in ca.ENDPOINTS.items():
            if not {cfg["duration_col"], cfg["event_col"]}.issubset(train_val.columns):
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
        key = str(int(landmark_day))
        horizons_by_landmark[key] = horizons
        n_patients_by_landmark[key] = int(len(built))
        n_by_split = built["split"].value_counts().to_dict()
        split_sizes_by_landmark[key] = {
            split: int(n_by_split.get(split, 0)) for split in ("train", "valid", "test")
        }

    manifest.update(
        {
            "feature_set": feature_set,
            "text_anchor": ANCHOR,
            "text_note_time_col": NOTE_TIME_COL,
            "text_note_types": list(NOTE_TYPES),
            "text_pool_strategy": POOL_STRATEGY,
            "text_decay_param": DECAY_PARAM,
            "text_continuous_window": False,
            "text_complete_case_note_types": True,
            "n_text_features": int(n_text_features),
            "notes_path": str(notes_path),
            "treatment_anchor_source": str(anchor_source),
            "prediction_time_origin": "ADT start",
            "cohort_note": (
                "Matched complete-case text cohort: patients in the standard "
                "landmark cohort with all three pre-landmark note modalities. "
                "All three feature sets share this patient set."
            ),
            "landmark_days": sorted(int(day) for day in built_by_landmark),
            "n_patients": int(n_patients_by_landmark.get("0", 0)),
            "n_patients_by_landmark": n_patients_by_landmark,
            "split_sizes_by_landmark": split_sizes_by_landmark,
            "auc_horizons_by_landmark": horizons_by_landmark,
        }
    )
    return manifest


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

    generate_survival_embedding_df, notes_path = _import_embedding_helpers()

    treatment_anchors = load_treatment_anchors(longitudinal_path)
    print(
        f"Recovered ADT anchors for {len(treatment_anchors):,} patients "
        f"from {longitudinal_path}."
    )

    notes_meta, embeddings = load_notes_and_embeddings(notes_path)
    print(
        f"Loaded {notes_meta.height:,} note-embedding rows "
        f"({embeddings.shape[0]:,} x {embeddings.shape[1]} matrix) from {notes_path}."
    )

    notes_meta = attach_adt_relative_note_times(notes_meta, treatment_anchors)
    print(
        f"{notes_meta.height:,} notes dated relative to ADT start for "
        f"{notes_meta['DFCI_MRN'].n_unique():,} cohort patients."
    )

    landmark_days = (
        [int(value) for value in args.landmark_days]
        if args.landmark_days
        else [int(value) for value in base_manifest["landmark_days"]]
    )

    built_by_set: dict[str, dict[int, pd.DataFrame]] = {
        feature_set: {} for feature_set in FEATURE_SETS
    }
    text_feature_names: list[str] = []

    for landmark_day in landmark_days:
        base_path = base_inputs_dir / aggregated_filename(landmark_day)
        labs_path = base_inputs_dir / pre_treatment_lab_filename(landmark_day)
        if not base_path.exists():
            raise FileNotFoundError(f"Missing base landmark input: {base_path}")
        if not labs_path.exists():
            raise FileNotFoundError(f"Missing base pre-landmark lab input: {labs_path}")

        pooled = pool_embeddings_for_landmark(
            notes_meta,
            embeddings,
            landmark_day,
            generate_survival_embedding_df=generate_survival_embedding_df,
        )
        if not text_feature_names:
            text_feature_names = embedding_feature_columns(pooled)

        base = pd.read_csv(base_path, low_memory=False)
        feature_sets = build_landmark_feature_sets(
            base, pooled, landmark_day=landmark_day
        )

        n_base = base[ca.ID_COL].nunique() if ca.ID_COL in base.columns else len(base)
        n_matched = len(feature_sets["labs_text"])
        print(
            f"  landmark +{landmark_day}d: matched cohort {n_matched:,} of "
            f"{n_base:,} standard-cohort patients "
            f"({100.0 * n_matched / n_base:.1f}% retained)"
        )

        source_labs = pd.read_csv(labs_path, low_memory=False)
        for feature_set, built in feature_sets.items():
            set_dir = output_dir / feature_set
            set_dir.mkdir(parents=True, exist_ok=True)
            built.to_csv(set_dir / aggregated_filename(landmark_day), index=False)
            built_by_set[feature_set][landmark_day] = built

            # The lab-bearing arms need the long-format lab frame for canonical
            # lab selection, restricted to the matched cohort. The text-only arm
            # selects no canonical labs, so it gets a well-formed empty frame --
            # matching how build_somatic_gleason_inputs.py stubs this file.
            if feature_set == "text":
                pd.DataFrame(columns=[ca.ID_COL, "LAB_NAME"]).to_csv(
                    set_dir / pre_treatment_lab_filename(landmark_day), index=False
                )
            else:
                matched_labs = source_labs.loc[
                    source_labs[ca.ID_COL].isin(set(built[ca.ID_COL]))
                ]
                matched_labs.to_csv(
                    set_dir / pre_treatment_lab_filename(landmark_day), index=False
                )

    manifest_rows = [
        {
            "feature": feature,
            "feature_kind": (
                "text_note_vintage"
                if str(feature).startswith("PERCENT_")
                else "text_embedding"
            ),
            "source": str(notes_path),
        }
        for feature in text_feature_names
    ]

    for feature_set in FEATURE_SETS:
        set_dir = output_dir / feature_set
        pd.DataFrame(manifest_rows).to_csv(
            set_dir / FEATURE_MANIFEST_FILENAME, index=False
        )
        manifest = _landmark_manifest(
            base_manifest,
            built_by_set[feature_set],
            feature_set=feature_set,
            n_text_features=len(text_feature_names),
            notes_path=str(notes_path),
            anchor_source=str(longitudinal_path),
        )
        manifest["base_inputs_dir"] = str(base_inputs_dir)
        (set_dir / BUILD_MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))

    print(
        f"\nWrote {len(FEATURE_SETS)} feature sets x {len(landmark_days)} landmark(s) "
        f"to {output_dir}\n"
        f"  {len(text_feature_names):,} text feature columns "
        f"({len(NOTE_TYPES)} note types)"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--base-inputs-dir",
        required=True,
        help="Standard COMPASS prediction-inputs directory (prediction_inputs_adt).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination for the text/, labs/, and labs_text/ trees.",
    )
    parser.add_argument(
        "--landmark-days",
        nargs="*",
        default=None,
        help="Landmarks to build. Defaults to the base manifest's landmark_days.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
