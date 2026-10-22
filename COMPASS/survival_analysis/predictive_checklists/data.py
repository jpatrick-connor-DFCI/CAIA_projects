"""Read-only adapters for the full ADT cohort's prebuilt landmark tables.

Durations in these files are already landmark-relative days. This module
validates their provenance and split assignments; it never rebases outcomes,
resplits patients, or removes patients with missing predictors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ENDPOINT_COLUMNS = {
    "platinum": ("t_platinum", "PLATINUM"),
    "nepc": ("t_nepc", "NEPC"),
}
RESOLVED_NEPC_STATUSES = frozenset({
    "adjudicated", "reviewed", "resolved", "completed", "complete",
    "success", "succeeded", "done", "ok",
})
_BASE_COHORT_FILES = frozenset({"prostate_adt_survival_cohort_adt.csv", "adt_mrns.csv"})


@dataclass
class LandmarkData:
    frame: pd.DataFrame
    manifest: dict[str, Any]
    manifest_path: Path
    input_dir: Path
    audit: dict[str, Any] = field(default_factory=dict)


def resolved_nepc_mask(frame: pd.DataFrame) -> pd.Series:
    """Conservatively identify successfully reviewed labels for sensitivity runs.

    Unknown statuses are unresolved, including a completely absent provenance
    column. This does not alter primary full-cohort inclusion. The recognized
    statuses are emitted in the audit so a zero-sized sensitivity set is clear.
    """
    if "NEPC_LABEL_SOURCE" not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    status = frame["NEPC_LABEL_SOURCE"].astype("string").str.strip().str.lower()
    return status.isin(RESOLVED_NEPC_STATUSES).fillna(False).astype(bool)


def _validate_manifest(manifest: dict[str, Any], endpoint: str, landmark: int) -> None:
    if endpoint not in ENDPOINT_COLUMNS:
        raise ValueError("Checklist endpoint must be 'platinum' or 'nepc'.")
    required = {
        "data", "anchor_col", "endpoint", "landmark_days", "split_mode",
        "restrict_to_mrns", "exclude_mrns", "time_unit_days",
        "auc_horizons_by_landmark", "auc_time_unit_days", "auc_max_time_units",
        "auc_timeline_schema_version",
    }
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"Incomplete build manifest ({', '.join(sorted(missing))}); rebuild prediction inputs.")
    if manifest["endpoint"] != endpoint:
        raise ValueError(
            f"Build manifest endpoint is {manifest['endpoint']!r}, expected {endpoint!r}. "
            "Rebuild endpoint-specific prediction inputs; older builders could mislabel this field."
        )
    if "require_nepc" in manifest and bool(manifest["require_nepc"]) != (endpoint == "nepc"):
        raise ValueError("Manifest require_nepc disagrees with endpoint; rebuild prediction inputs.")
    source_name = Path(str(manifest["data"])).name
    anchors = [str(manifest[key]).lower() for key in ("arm", "anchor", "treatment_anchor") if key in manifest]
    if any(anchor != "adt" for anchor in anchors):
        raise ValueError("Manifest identifies a non-ADT treatment arm.")
    if source_name != "longitudinal_prediction_data_adt.csv" and "adt" not in anchors:
        raise ValueError("Full ADT inputs require the ADT longitudinal source or explicit arm='adt' provenance.")
    if manifest["anchor_col"] is not None and str(manifest["anchor_col"]).lower() != "none":
        raise ValueError("COMPASS ADT times must already be anchor-relative (anchor_col='none').")
    if manifest.get("cohort", "all") != "all" or manifest.get("exclusion", "none") != "none":
        raise ValueError("Predictive checklists require the full ADT cohort with exclusion='none'.")
    restriction = manifest["restrict_to_mrns"]
    if restriction and Path(str(restriction)).name not in _BASE_COHORT_FILES:
        raise ValueError("Manifest restricts patients beyond the full ADT base cohort.")
    if manifest["exclude_mrns"] or manifest.get("exclude_pre_adt_castrate", False):
        raise ValueError("Full ADT analysis does not permit additional patient exclusions.")
    if landmark not in manifest["landmark_days"]:
        raise ValueError(f"Landmark {landmark} is absent from the build manifest.")
    if manifest["split_mode"] != "shared":
        raise ValueError("Use shared patient splits across landmarks; rebuild with --split-mode shared.")
    if manifest["auc_timeline_schema_version"] != 3:
        raise ValueError("Expected timeline schema version 3; rebuild prediction inputs.")
    for key in ("time_unit_days", "auc_time_unit_days", "auc_max_time_units"):
        value = float(manifest[key])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Manifest {key} must be positive and finite.")
    if float(manifest["time_unit_days"]) != float(manifest["auc_time_unit_days"]):
        raise ValueError("Manifest outcome and metric time units disagree.")
    horizon_map = manifest["auc_horizons_by_landmark"].get(str(landmark), {})
    if endpoint not in horizon_map:
        raise ValueError("Manifest lacks the selected endpoint's requested evaluation horizon grid.")
    grid = np.asarray(horizon_map[endpoint], dtype=float)
    if grid.ndim != 1 or not np.isfinite(grid).all() or np.any(grid <= 0):
        raise ValueError("Manifest evaluation horizons must be a finite positive one-dimensional grid.")
    if len(grid) and (np.any(np.diff(grid) <= 0) or np.any(grid >= float(manifest["auc_max_time_units"]))):
        raise ValueError("Manifest horizons must increase strictly and remain below the evaluation cap.")


def load_landmark(
    inputs_dir: str | Path,
    endpoint: str,
    landmark: int,
    id_col: str = "DFCI_MRN",
    age_col: str = "AGE_AT_TREATMENTSTART",
) -> LandmarkData:
    """Load a full-ADT table, preserving IDs, every patient, and existing splits."""
    if isinstance(landmark, bool) or int(landmark) != landmark or landmark < 0:
        raise ValueError("landmark must be a nonnegative integer number of days.")
    landmark = int(landmark)
    input_dir = Path(inputs_dir).expanduser().resolve()
    manifest_path = input_dir / "build_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    _validate_manifest(manifest, endpoint, landmark)
    table_path = input_dir / f"aggregated_landmark{landmark}.csv"
    frame = pd.read_csv(table_path, dtype={id_col: "string", "split": "string"})
    duration_col, event_col = ENDPOINT_COLUMNS[endpoint]
    missing = {id_col, age_col, "split", duration_col, event_col} - set(frame.columns)
    if missing:
        raise ValueError(f"Aggregated inputs are missing required columns: {', '.join(sorted(missing))}.")
    if frame.empty:
        raise ValueError("Aggregated inputs contain no patients.")
    if frame[id_col].isna().any() or frame[id_col].str.strip().eq("").any():
        raise ValueError("Aggregated inputs contain missing patient identifiers.")
    if frame[id_col].duplicated().any():
        raise ValueError("Aggregated inputs must have one row per patient; duplicate IDs found.")
    if not frame["split"].isin(["train", "valid", "test"]).all():
        raise ValueError("Aggregated splits must be train, valid, or test, with no missing labels.")
    if set(frame["split"]) != {"train", "valid", "test"}:
        raise ValueError("Aggregated inputs must retain nonempty train, valid, and test partitions.")
    duration = pd.to_numeric(frame[duration_col], errors="coerce")
    events = pd.to_numeric(frame[event_col], errors="coerce")
    if not np.isfinite(duration).all() or not duration.gt(0).all():
        raise ValueError("Selected endpoint durations must be finite positive landmark-relative days.")
    if not events.isin([0, 1]).all():
        raise ValueError("Selected endpoint indicators must be binary, with no missing values.")
    frame[duration_col] = duration.astype(float)
    frame[event_col] = events.astype(int)
    counts = frame["split"].value_counts().to_dict()
    expected_n = manifest.get("n_patients_by_landmark", {}).get(str(landmark))
    if expected_n is not None and int(expected_n) != len(frame):
        raise ValueError("Patient count differs from the build manifest; inputs may be stale or truncated.")
    expected_splits = manifest.get("split_sizes_by_landmark", {}).get(str(landmark), {})
    for split, expected in expected_splits.items():
        if split in {"train", "valid", "test"} and int(expected) != counts.get(split, 0):
            raise ValueError("Split counts differ from the build manifest.")
    split_path = input_dir / f"split_assignments_landmark{landmark}.csv"
    if split_path.exists():
        assignments = pd.read_csv(split_path, dtype={id_col: "string", "split": "string"})
        if not {id_col, "split"}.issubset(assignments) or assignments[id_col].duplicated().any():
            raise ValueError("Invalid stored landmark split assignments.")
        saved = assignments.set_index(id_col)["split"]
        current = frame.set_index(id_col)["split"]
        if set(saved.index) != set(current.index) or not saved.reindex(current.index).equals(current):
            raise ValueError("Aggregated splits disagree with the stored landmark split assignments.")
    sources = (
        frame["NEPC_LABEL_SOURCE"].astype("string").fillna("missing").value_counts().to_dict()
        if "NEPC_LABEL_SOURCE" in frame else {"provenance_unavailable": len(frame)}
    )
    audit = {
        "input_dir": str(input_dir), "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_data": manifest["data"], "endpoint": endpoint, "landmark_days": landmark,
        "n_patients": len(frame), "n_events": int(events.sum()),
        "split_counts": {str(key): int(value) for key, value in counts.items()},
        "events_by_split": {str(key): int(value) for key, value in frame.groupby("split")[event_col].sum().items()},
        "duration_units": "days_after_landmark", "auc_time_unit_days": float(manifest["auc_time_unit_days"]),
        "auc_horizons_days": [float(value) * float(manifest["auc_time_unit_days"])
                              for value in manifest["auc_horizons_by_landmark"][str(landmark)][endpoint]],
        "auc_max_days": float(manifest["auc_time_unit_days"]) * float(manifest["auc_max_time_units"]),
        "max_followup_days": manifest.get("max_followup_days"),
        "nepc_label_source_counts": {str(key): int(value) for key, value in sources.items()},
        "nepc_date_source_counts": (
            frame["NEPC_DATE_SOURCE"].astype("string").fillna("missing").value_counts().to_dict()
            if "NEPC_DATE_SOURCE" in frame else None
        ),
        "nepc_date_precision_counts": (
            frame["NEPC_DATE_PRECISION"].astype("string").fillna("missing").value_counts().to_dict()
            if "NEPC_DATE_PRECISION" in frame else None
        ),
        "undated_positive_demotions": "Not recoverable from aggregated non-events; consult Stage-1 build logs.",
        "resolved_nepc_n": int(resolved_nepc_mask(frame).sum()),
        "resolved_nepc_statuses": sorted(RESOLVED_NEPC_STATUSES),
        "stored_split_assignments_checked": split_path.exists(),
    }
    return LandmarkData(frame, manifest, manifest_path, input_dir, audit)
