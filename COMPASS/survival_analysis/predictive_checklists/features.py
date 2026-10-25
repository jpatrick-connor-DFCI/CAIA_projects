"""Training-only, raw-unit binary candidates for interpretable checklists.

Only age, the established pre-landmark lab summary schema, and explicitly
configured clinical columns (e.g. Gleason, TP53/RB1/PTEN) are eligible.
Patients with missing labs remain in the matrix: numeric comparisons use fitted
training medians, and missingness is an explicit candidate when it varies in
training. All candidates from one lab share a group, so the optimizer can limit
a checklist to one item per lab rather than repeating correlated summaries.
Each clinical column is its own group; a low-cardinality one (binary
alteration flags, Gleason) is thresholded at every observed training value,
since quantiles of a rare 0/1 flag never reach 1.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


SUMMARY_STATS = frozenset({"mean", "min", "max", "last", "delta", "n_observations"})
DEFAULT_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
MAX_DISCRETE_LEVELS = 10
_NON_LAB_SOURCES = frozenset({
    "DFCI_MRN", "MRN", "ID", "SPLIT", "GENDER", "FIRST_TREATMENT",
    "FIRST_RECORD_DATE", "DIAGNOSIS_DATE", "TREATMENT_ANCHOR_DATE",
    "LAST_CONTACT_DATE", "PRIMARY_LABEL", "EITHER", "PANEL_VERSION",
})


def _group(feature: str, age_col: str, clinical_features: Iterable[str] = ()) -> str | None:
    if feature == age_col or feature in clinical_features:
        return feature
    source, separator, stat = feature.rpartition("__")
    if not separator or not source or stat not in SUMMARY_STATS:
        return None
    normalized = source.upper()
    if normalized in _NON_LAB_SOURCES or normalized.startswith(
        ("T_", "NEPC", "AVPC", "PLATINUM", "DEATH")
    ):
        return None
    return source


def _numeric(frame: pd.DataFrame, feature: str) -> pd.Series:
    if feature not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[feature], errors="coerce").astype(float).replace(
        [np.inf, -np.inf], np.nan
    )


def schema_feature_columns(
    frame: pd.DataFrame, age_col: str = "AGE_AT_TREATMENTSTART",
    clinical_features: Iterable[str] = (),
) -> list[str]:
    """Whitelist raw baseline inputs without learning anything from their values.

    Baseline runners can pass this schema-only list into inner CV, where
    coverage, missingness, and variability selection must be fitted anew.
    """
    clinical_features = tuple(clinical_features)
    missing = set(clinical_features) - set(map(str, frame.columns))
    if missing:
        raise ValueError(f"Configured clinical features are absent: {', '.join(sorted(missing))}.")
    return sorted(str(column) for column in frame.columns
                  if _group(str(column), age_col, clinical_features) is not None)


class FeatureBinarizer:
    """Learn candidate definitions from one training block only.

    ``fit(frame, lab_long=...)`` optionally uses the canonical shared lab
    coverage rule on the training patients' pre-landmark long rows. Without
    that table, observed summary coverage is recomputed on ``frame``; a
    caller-supplied canonical list is only a restriction, never a substitute
    for this training coverage check. ``transform`` never changes candidates.
    """

    def __init__(
        self,
        min_patient_coverage: float = 0.20,
        quantiles: Iterable[float] = DEFAULT_QUANTILES,
        age_col: str = "AGE_AT_TREATMENTSTART",
        canonical_labs: Iterable[str] | None = None,
        id_col: str = "DFCI_MRN",
        clinical_features: Iterable[str] = (),
    ) -> None:
        self.min_patient_coverage = float(min_patient_coverage)
        self.quantiles = tuple(sorted(set(float(q) for q in quantiles)))
        if not 0 <= self.min_patient_coverage <= 1:
            raise ValueError("min_patient_coverage must be between zero and one.")
        if not self.quantiles or any(not 0 < q < 1 for q in self.quantiles):
            raise ValueError("quantiles must be finite values strictly between zero and one.")
        self.age_col = str(age_col)
        self.id_col = str(id_col)
        self.clinical_features = tuple(map(str, clinical_features))
        if self.age_col in self.clinical_features or len(set(self.clinical_features)) != len(self.clinical_features):
            raise ValueError("clinical_features must be distinct and exclude the age column.")
        self.canonical_labs = (
            None if canonical_labs is None else sorted(set(map(str, canonical_labs)))
        )
        self.feature_columns: list[str] = []
        self.items: list[dict[str, Any]] = []
        self.groups: list[str] = []
        self.medians: dict[str, float] = {}
        self.coverage: dict[str, float] = {}
        self._fitted = False

    def fit(self, frame: pd.DataFrame, lab_long: pd.DataFrame | None = None) -> "FeatureBinarizer":
        if frame.empty:
            raise ValueError("Cannot fit checklist features on an empty training block.")
        if frame.columns.duplicated().any():
            raise ValueError("Training frame has duplicate feature columns.")
        missing = set(self.clinical_features) - set(map(str, frame.columns))
        if missing:
            raise ValueError(f"Configured clinical features are absent: {', '.join(sorted(missing))}.")
        self.feature_columns, self.items, self.groups = [], [], []
        self.medians, self.coverage = {}, {}
        eligible_labs = None if self.canonical_labs is None else set(self.canonical_labs)
        if lab_long is not None:
            from survival_common.helper import select_canonical_labs

            if self.id_col not in frame:
                raise ValueError(f"Training frame must contain {self.id_col} for lab coverage.")
            long_rows = lab_long.copy()
            long_rows[self.id_col] = long_rows[self.id_col].astype(str)
            train_ids = frame[self.id_col].astype(str)
            long_rows = long_rows.loc[long_rows[self.id_col].isin(train_ids)]
            train_labs = set() if long_rows.empty else set(select_canonical_labs(
                long_rows, mrns=train_ids, min_coverage=self.min_patient_coverage,
                id_col=self.id_col,
            ))
            eligible_labs = train_labs if eligible_labs is None else eligible_labs & train_labs

        for feature in sorted(str(c) for c in frame.columns):
            group = _group(feature, self.age_col, self.clinical_features)
            is_lab = feature != self.age_col and feature not in self.clinical_features
            if group is None or (is_lab and eligible_labs is not None and group not in eligible_labs):
                continue
            values = _numeric(frame, feature)
            observed = values.dropna()
            coverage = float(values.notna().mean())
            if observed.empty or (is_lab and coverage < self.min_patient_coverage):
                continue
            self.feature_columns.append(feature)
            self.medians[feature] = float(observed.median())
            self.coverage[feature] = coverage
            imputed = values.fillna(self.medians[feature]).to_numpy()
            if observed.nunique() >= 2:
                if feature in self.clinical_features and observed.nunique() <= MAX_DISCRETE_LEVELS:
                    thresholds = set(float(v) for v in observed.unique())
                else:
                    thresholds = set(float(v) for v in observed.quantile(self.quantiles))
                for threshold in sorted(thresholds):
                    for op in ("<=", ">="):
                        binary = imputed <= threshold if op == "<=" else imputed >= threshold
                        if binary.all() or not binary.any():
                            continue
                        self.items.append({
                            "name": f"{feature} {op} {threshold:.12g}",
                            "feature": feature, "op": op, "threshold": threshold, "group": group,
                        })
            if values.isna().any():
                self.items.append({
                    "name": f"{feature} is missing", "feature": feature,
                    "op": "is_missing", "threshold": None, "group": group,
                })
        self.groups = [item["group"] for item in self.items]
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise ValueError("FeatureBinarizer must be fitted before transform.")
        values = {feature: _numeric(frame, feature) for feature in self.feature_columns}
        result = np.zeros((len(frame), len(self.items)), dtype=np.int8)
        for i, item in enumerate(self.items):
            raw = values[item["feature"]]
            if item["op"] == "is_missing":
                result[:, i] = raw.isna().to_numpy(dtype=np.int8)
            else:
                imputed = raw.fillna(self.medians[item["feature"]]).to_numpy()
                threshold = item["threshold"]
                result[:, i] = imputed <= threshold if item["op"] == "<=" else imputed >= threshold
        return result

    def to_dict(self) -> dict[str, Any]:
        if not self._fitted:
            raise ValueError("FeatureBinarizer must be fitted before serialization.")
        return {
            "schema_version": 1, "age_col": self.age_col, "id_col": self.id_col,
            "min_patient_coverage": self.min_patient_coverage, "quantiles": list(self.quantiles),
            "canonical_labs": self.canonical_labs, "clinical_features": list(self.clinical_features),
            "feature_columns": list(self.feature_columns),
            "medians": dict(self.medians), "coverage": dict(self.coverage),
            "items": [dict(item) for item in self.items],
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "FeatureBinarizer":
        if state.get("schema_version") != 1:
            raise ValueError("Unsupported checklist feature schema_version.")
        result = cls(**{key: state[key] for key in (
            "age_col", "id_col", "min_patient_coverage", "quantiles", "canonical_labs"
        )}, clinical_features=state.get("clinical_features", ()))
        result.feature_columns = list(state["feature_columns"])
        result.medians = {key: float(value) for key, value in state["medians"].items()}
        result.coverage = {key: float(value) for key, value in state["coverage"].items()}
        result.items = [dict(item) for item in state["items"]]
        for feature in result.feature_columns:
            if _group(feature, result.age_col, result.clinical_features) is None or not np.isfinite(result.medians[feature]):
                raise ValueError("Invalid serialized checklist feature or imputation value.")
        for item in result.items:
            if item["feature"] not in result.feature_columns or item["group"] != _group(item["feature"], result.age_col, result.clinical_features):
                raise ValueError("Invalid serialized checklist item feature/group.")
            if item["op"] not in {"<=", ">=", "is_missing"}:
                raise ValueError("Invalid serialized checklist comparison.")
            if item["op"] != "is_missing" and not np.isfinite(float(item["threshold"])):
                raise ValueError("Invalid serialized checklist threshold.")
        result.groups = [item["group"] for item in result.items]
        result._fitted = True
        return result
