"""Component-based Gleason/somatic/labs feature-set tokens (Plan §2a).

Exercises COMPASS.survival_analysis.cox_aggregated.prepare_landmark_context
directly against a small on-disk fixture (aggregated_landmark{D}.csv,
pre_treatment_lab_long_landmark{D}.csv, somatic_gleason_features.csv) to
confirm each new token selects exactly the expected columns:
  * 'labs' never includes GLEASON_SCORE or somatic SNV/etc columns.
  * 'gleason' selects only the Gleason score.
  * 'somatic' selects only the somatic binary indicators.
  * 'gleason-labs' keeps GLEASON_SCORE past the lab coverage gate via
    always_include, alongside the normal lab columns.
  * 'gleason-somatic' and 'gleason-somatic-labs' combine components.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SURVIVAL_DIR = REPO_ROOT / "COMPASS" / "survival_analysis"
DATA_PREP_DIR = REPO_ROOT / "COMPASS" / "data_preprocessing"
for _p in (str(REPO_ROOT), str(DATA_PREP_DIR), str(SURVIVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cox_aggregated as ca  # noqa: E402

ID_COL = ca.ID_COL
AGE_COL = ca.AGE_COL
N = 120


def _write_fixture(inputs_dir: Path, landmark_day: int = 0) -> None:
    inputs_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    mrns = np.arange(1, N + 1)
    split = np.where(np.arange(N) < N * 0.6, "train", np.where(np.arange(N) < N * 0.8, "valid", "test"))

    agg = pd.DataFrame(
        {
            ID_COL: mrns,
            AGE_COL: rng.normal(65, 8, size=N),
            "GENDER": "M",
            "FIRST_RECORD_DATE": "2015-01-01",
            "DIAGNOSIS_DATE": "2015-01-01",
            "TREATMENT_ANCHOR_DATE": "2015-06-01",
            "LAST_CONTACT_DATE": "2018-01-01",
            "PLATINUM_DATE": pd.NaT,
            "PLATINUM": rng.binomial(1, 0.3, size=N),
            "DEATH": rng.binomial(1, 0.2, size=N),
            "EITHER": 0,
            "t_diagnosis": 0.0,
            "t_platinum": rng.uniform(30, 900, size=N),
            "t_platinum_from_first_record": rng.uniform(30, 900, size=N),
            "t_last_contact": rng.uniform(200, 1400, size=N),
            "t_last_contact_from_first_record": rng.uniform(200, 1400, size=N),
            "t_death": rng.uniform(30, 1200, size=N),
            "t_death_from_first_record": rng.uniform(30, 1200, size=N),
            "t_either": rng.uniform(30, 900, size=N),
            "split": split,
            # Lab summary columns (survive the coverage/variability gate).
            "PSA__mean": rng.uniform(1, 20, size=N),
            "PSA__slope": rng.normal(0, 1, size=N),
            "TESTOSTERONE__mean": rng.uniform(10, 500, size=N),
            # Static Gleason/somatic features (declared via the manifest).
            "GLEASON_SCORE": rng.choice([6, 7, 8, 9, 10], size=N).astype(float),
            "TP53_SNV": rng.binomial(1, 0.25, size=N).astype(float),
            "PTEN_DEL": rng.binomial(1, 0.20, size=N).astype(float),
        }
    )
    agg.to_csv(inputs_dir / f"aggregated_landmark{landmark_day}.csv", index=False)

    pre_lab_rows = []
    for mrn in mrns:
        for lab in ("PSA", "TESTOSTERONE"):
            for _ in range(5):
                pre_lab_rows.append({ID_COL: mrn, "LAB_NAME": lab, "LAB_VALUE": rng.uniform(1, 20), "t_lab": -30.0})
    pd.DataFrame(pre_lab_rows).to_csv(
        inputs_dir / f"pre_treatment_lab_long_landmark{landmark_day}.csv", index=False
    )

    manifest = pd.DataFrame(
        [
            {"feature": "GLEASON_SCORE", "feature_kind": "gleason_continuous", "source": "fixture"},
            {"feature": "TP53_SNV", "feature_kind": "somatic_binary", "source": "fixture"},
            {"feature": "PTEN_DEL", "feature_kind": "somatic_binary", "source": "fixture"},
        ]
    )
    manifest.to_csv(inputs_dir / "somatic_gleason_features.csv", index=False)


@pytest.fixture()
def inputs_dir(tmp_path) -> Path:
    d = tmp_path / "prediction_inputs"
    _write_fixture(d)
    return d


class TestFeatureSetComponents:
    def test_labs_never_includes_gleason_or_somatic(self, inputs_dir):
        ctx = ca.prepare_landmark_context(inputs_dir, 0, min_patient_coverage=0.2, feature_set="labs")
        assert "GLEASON_SCORE" not in ctx.selected_feature_cols
        assert "TP53_SNV" not in ctx.selected_feature_cols
        assert "PTEN_DEL" not in ctx.selected_feature_cols
        assert len(ctx.canonical_labs) > 0

    def test_gleason_selects_only_gleason(self, inputs_dir):
        ctx = ca.prepare_landmark_context(inputs_dir, 0, min_patient_coverage=0.2, feature_set="gleason")
        assert ctx.selected_feature_cols == ["GLEASON_SCORE"]
        assert ctx.canonical_labs == []

    def test_somatic_selects_only_somatic(self, inputs_dir):
        ctx = ca.prepare_landmark_context(inputs_dir, 0, min_patient_coverage=0.2, feature_set="somatic")
        assert set(ctx.selected_feature_cols) == {"TP53_SNV", "PTEN_DEL"}
        assert ctx.canonical_labs == []
        assert set(ctx.genomic_feature_cols) == {"TP53_SNV", "PTEN_DEL"}

    def test_gleason_labs_keeps_gleason_past_lab_gate(self, inputs_dir):
        ctx = ca.prepare_landmark_context(inputs_dir, 0, min_patient_coverage=0.2, feature_set="gleason-labs")
        assert "GLEASON_SCORE" in ctx.selected_feature_cols
        assert len(ctx.canonical_labs) > 0
        assert any(c not in {"GLEASON_SCORE"} for c in ctx.selected_feature_cols)
        assert "TP53_SNV" not in ctx.selected_feature_cols

    def test_somatic_labs_keeps_somatic_past_lab_gate(self, inputs_dir):
        ctx = ca.prepare_landmark_context(inputs_dir, 0, min_patient_coverage=0.2, feature_set="somatic-labs")
        assert {"TP53_SNV", "PTEN_DEL"}.issubset(set(ctx.selected_feature_cols))
        assert "GLEASON_SCORE" not in ctx.selected_feature_cols
        assert len(ctx.canonical_labs) > 0

    def test_gleason_somatic_combines_without_labs(self, inputs_dir):
        ctx = ca.prepare_landmark_context(inputs_dir, 0, min_patient_coverage=0.2, feature_set="gleason-somatic")
        assert set(ctx.selected_feature_cols) == {"GLEASON_SCORE", "TP53_SNV", "PTEN_DEL"}
        assert ctx.canonical_labs == []

    def test_gleason_somatic_labs_combines_all_three(self, inputs_dir):
        ctx = ca.prepare_landmark_context(
            inputs_dir, 0, min_patient_coverage=0.2, feature_set="gleason-somatic-labs"
        )
        assert {"GLEASON_SCORE", "TP53_SNV", "PTEN_DEL"}.issubset(set(ctx.selected_feature_cols))
        assert len(ctx.canonical_labs) > 0

    def test_unsupported_feature_set_raises(self, inputs_dir):
        with pytest.raises(ValueError):
            ca.prepare_landmark_context(inputs_dir, 0, min_patient_coverage=0.2, feature_set="not-a-real-arm")
