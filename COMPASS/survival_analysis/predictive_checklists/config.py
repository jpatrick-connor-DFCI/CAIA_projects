"""Reproducible, validated run settings (all durations are in days)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class RunConfig:
    endpoints: tuple = ("platinum", "nepc")
    landmarks: tuple = (0, 90, 180)
    horizons: tuple = (365, 730, 1095, 1460, 1825)
    max_items: tuple = (3, 5, 8)
    n_folds: int = 5
    seed: int = 42
    solver: str = "highs"
    solver_time_limit: float = 60.0
    min_patient_coverage: float = 0.2
    quantiles: tuple = (0.1, 0.25, 0.5, 0.75, 0.9)
    min_events: int = 10
    min_controls: int = 10
    min_g: float = 0.05
    bootstrap_samples: int = 200
    baselines: tuple = ("age_cox", "elastic_net", "xgboost")
    resolved_nepc_sensitivity: bool = True
    # Non-lab columns (e.g. GLEASON_SCORE, TP53_ANY) admitted as candidates and
    # baseline predictors; each must exist in every input table.
    clinical_features: tuple = ()
    id_col: str = "DFCI_MRN"
    age_col: str = "AGE_AT_TREATMENTSTART"
    # "drop": lab counts (__n_observations), missingness and panel/burden
    # covariates are never checklist items; they adjust the probability
    # calibration only, evaluated at their training mean for every prediction.
    adjust_only_mode: str = "none"
    # Candidate-pool restrictions, matching --exclude-vitals and --feature-stats
    # of the survival models. Baselines are restricted to the same pool.
    exclude_vitals: bool = False
    feature_stats: str = "all"

    def validate(self):
        if not self.endpoints or set(self.endpoints) - {"platinum", "nepc"}:
            raise ValueError("endpoints must contain platinum and/or nepc")
        for name in ("landmarks", "horizons", "max_items"):
            values = getattr(self, name)
            minimum = 0 if name == "landmarks" else 1
            if not values or any(int(x) != x or x < minimum for x in values):
                raise ValueError(f"{name} must contain integers >= {minimum}")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
        if self.n_folds < 2 or self.min_events < 2 or self.min_controls < 2:
            raise ValueError("n_folds, min_events and min_controls must be >= 2")
        if self.solver not in {"highs", "cbc", "cplex"}:
            raise ValueError("solver must be highs, cbc or cplex")
        if self.solver_time_limit <= 0 or self.bootstrap_samples < 0:
            raise ValueError("invalid time limit or bootstrap count")
        if not 0 < self.min_g <= 1 or not 0 < self.min_patient_coverage <= 1:
            raise ValueError("min_g and min_patient_coverage must be in (0, 1]")
        if not self.quantiles or any(not 0 < q < 1 for q in self.quantiles):
            raise ValueError("quantiles must be strictly between zero and one")
        if set(self.baselines) - {"age_cox", "elastic_net", "xgboost"}:
            raise ValueError("unknown baseline model")
        if (len(set(self.clinical_features)) != len(self.clinical_features)
                or self.age_col in self.clinical_features
                or any(not isinstance(x, str) or not x for x in self.clinical_features)):
            raise ValueError("clinical_features must be distinct non-age column names")
        if self.adjust_only_mode not in {"none", "drop"}:
            raise ValueError("adjust_only_mode must be none or drop")
        if self.feature_stats not in {"all", "mean-n-obs"}:
            raise ValueError("feature_stats must be all or mean-n-obs")
        return self

    def to_dict(self):
        values = asdict(self)
        # Omitted at their defaults so existing manifests (and --resume of runs
        # made before these settings existed) still compare equal.
        for key, default in (("adjust_only_mode", "none"), ("exclude_vitals", False),
                             ("feature_stats", "all")):
            if values[key] == default:
                del values[key]
        return values

    @classmethod
    def load(cls, path=None):
        values = json.loads(Path(path).read_text()) if path else {}
        return cls(**values).validate()
