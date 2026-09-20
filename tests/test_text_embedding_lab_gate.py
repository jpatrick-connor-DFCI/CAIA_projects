"""The canonical-lab gate must not silently swallow note-embedding features.

This is the failure mode that makes the 03c text arms worth a test of their own.
Pooled embedding columns are named ``CLINICIAN_EMBEDDING_42`` -- no ``__``
separator -- so ``survival_common.cox_engine.parse_feature_name`` parses them as
lab ``CLINICIAN_EMBEDDING_42`` with stat ``"value"``. Under a non-empty
``restrict_to_labs``, every one of the 2304 dimensions is then dropped as an
unrecognized lab. Nothing raises: the fit proceeds on the always-include
demographics alone and reports a plausible C-index, so a broken "text" arm looks
like a weak-signal result rather than a bug.

The guards, and what each test pins:
  * ``cox_aggregated.prepare_landmark_context`` declares the manifest's columns as
    ``always_include_feature_cols`` and passes ``restrict_to_labs=[]`` for the
    text-only arm -- so embedding columns survive the gate either way.
  * ``genomic_feature_cols`` is left empty for both text arms. In
    ``tune_multivariable_model`` it defaults to ``always_include_feature_cols``
    when None, which would apply the 2.5% mutation-prevalence floor to dense
    continuous dimensions and drop nearly all of them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from survival_common.cox_engine import parse_feature_name
from survival_common.cox_models import select_feature_columns

_PREPROCESSING_DIR = Path(__file__).resolve().parents[1] / "COMPASS" / "data_preprocessing"
if str(_PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(_PREPROCESSING_DIR))

N = 200
N_EMB_DIMS = 8
CANONICAL_LABS = ["PSA", "Testosterone"]


def _cohort() -> tuple[pd.DataFrame, list[str], list[str]]:
    rng = np.random.default_rng(11)
    emb_cols = [
        f"{note_type}_EMBEDDING_{i}"
        for note_type in ("CLINICIAN", "IMAGING", "PATHOLOGY")
        for i in range(N_EMB_DIMS)
    ]
    lab_cols = ["PSA__mean", "Testosterone__last"]
    data = pd.DataFrame(
        {col: rng.normal(size=N) for col in emb_cols}
        | {col: rng.normal(size=N) for col in lab_cols}
        | {"AGE_AT_INDEX": rng.uniform(50, 85, size=N)}
    )
    return data, emb_cols, lab_cols


def test_embedding_columns_parse_as_unrecognized_labs():
    """The premise: this is why the gate is dangerous, not a hypothetical."""
    lab_name, stat = parse_feature_name("CLINICIAN_EMBEDDING_42")
    assert lab_name == "CLINICIAN_EMBEDDING_42"
    assert stat == "value"
    assert lab_name not in CANONICAL_LABS


def test_lab_gate_without_exemption_drops_every_embedding_dim():
    """Characterizes the bug the guards exist to prevent."""
    data, emb_cols, lab_cols = _cohort()

    selected, _ = select_feature_columns(
        data,
        emb_cols + lab_cols + ["AGE_AT_INDEX"],
        min_patient_coverage=0.5,
        restrict_to_labs=CANONICAL_LABS,
        always_include=["AGE_AT_INDEX"],
    )

    assert not [c for c in selected if "EMBEDDING" in c]
    # ... and it does so silently, leaving a fittable age-only design matrix.
    assert "AGE_AT_INDEX" in selected


def test_always_include_exemption_keeps_every_embedding_dim():
    """The labs_text arm: labs stay gated, embeddings are exempt."""
    data, emb_cols, lab_cols = _cohort()

    selected, _ = select_feature_columns(
        data,
        emb_cols + lab_cols + ["AGE_AT_INDEX"],
        min_patient_coverage=0.5,
        restrict_to_labs=CANONICAL_LABS,
        always_include=emb_cols + ["AGE_AT_INDEX"],
    )

    assert set(emb_cols) <= set(selected)
    assert set(lab_cols) <= set(selected)


def test_empty_restrict_to_labs_keeps_every_embedding_dim():
    """The text-only arm: canonical_labs is empty, so the gate is a no-op."""
    data, emb_cols, _ = _cohort()

    selected, _ = select_feature_columns(
        data,
        emb_cols + ["AGE_AT_INDEX"],
        min_patient_coverage=0.5,
        restrict_to_labs=[],
        always_include=emb_cols + ["AGE_AT_INDEX"],
    )

    assert set(emb_cols) <= set(selected)


def test_prevalence_floor_would_gut_continuous_embeddings():
    """Why genomic_feature_cols must stay empty for the text arms.

    Continuous dimensions are almost never exactly 1.0, so a floor written for
    binary mutation indicators reads them as ~0% prevalent.
    """
    data, emb_cols, _ = _cohort()

    kept_without_floor, _ = select_feature_columns(
        data,
        emb_cols + ["AGE_AT_INDEX"],
        min_patient_coverage=0.5,
        restrict_to_labs=[],
        always_include=emb_cols + ["AGE_AT_INDEX"],
    )
    kept_with_floor, _ = select_feature_columns(
        data,
        emb_cols + ["AGE_AT_INDEX"],
        min_patient_coverage=0.5,
        restrict_to_labs=[],
        always_include=emb_cols + ["AGE_AT_INDEX"],
        genomic_feature_cols=emb_cols,
        min_genomic_prevalence=0.025,
    )

    assert set(emb_cols) <= set(kept_without_floor)
    assert not [c for c in kept_with_floor if "EMBEDDING" in c]


def test_constant_embedding_dim_is_still_dropped():
    """The exemption covers the lab gate only -- variability filters still apply."""
    data, emb_cols, _ = _cohort()
    dead_dim = emb_cols[0]
    data[dead_dim] = 0.0

    selected, _ = select_feature_columns(
        data,
        emb_cols + ["AGE_AT_INDEX"],
        min_patient_coverage=0.5,
        restrict_to_labs=[],
        always_include=emb_cols + ["AGE_AT_INDEX"],
    )

    assert dead_dim not in selected
    assert set(emb_cols[1:]) <= set(selected)


# --- embedding-project path resolution -------------------------------------
#
# The builder's first default pointed one directory too shallow (beside
# PROFILE-testing rather than beside "Official CAIA Project Repos"), so the
# notebook could not find the checkout without CTEP_REPO_PATH set. These pin the
# resolution order so that regression cannot come back silently.

import importlib

import build_text_embedding_inputs as bte


def _resolve(monkeypatch, existing: set) -> object:
    """Re-resolve the default with only `existing` paths having anchors.py."""
    real_exists = Path.exists

    def fake_exists(self):
        if self.name == "anchors.py":
            return str(self.parent) in existing
        return real_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)
    return bte._default_clinical_embeddings_repo()


def test_default_prefers_the_projects_level_layout(monkeypatch):
    """The real local layout: the project sits beside the CAIA repos folder."""
    outer, inner = bte._CTEP_CANDIDATE_DIRS
    assert _resolve(monkeypatch, {str(outer)}) == outer


def test_default_still_accepts_the_immediate_sibling_layout(monkeypatch):
    outer, inner = bte._CTEP_CANDIDATE_DIRS
    assert _resolve(monkeypatch, {str(inner)}) == inner


def test_default_is_returned_unresolved_when_nothing_is_found(monkeypatch):
    """Must not raise: --help and argparse work without the checkout present."""
    assert _resolve(monkeypatch, set()) == bte._CTEP_CANDIDATE_DIRS[0]


def test_env_var_overrides_the_search(monkeypatch, tmp_path):
    monkeypatch.setenv("CTEP_REPO_PATH", str(tmp_path))
    reloaded = importlib.reload(bte)
    try:
        assert reloaded.CLINICAL_EMBEDDINGS_REPO == tmp_path
    finally:
        monkeypatch.delenv("CTEP_REPO_PATH", raising=False)
        importlib.reload(bte)


def test_missing_checkout_names_the_env_var_and_paths_searched(monkeypatch, tmp_path):
    monkeypatch.setattr(bte, "CLINICAL_EMBEDDINGS_REPO", tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="CTEP_REPO_PATH"):
        bte._import_embedding_helpers()
