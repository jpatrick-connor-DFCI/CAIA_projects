"""Genomic-prevalence-floor scoping test for Finding 10.

COMPASS's ``somatic_gleason`` feature set mixes genuine binary somatic
mutation indicators with ``GLEASON_SCORE`` (a continuous clinical score)
under one undifferentiated ``always_include_feature_cols`` tuple. Applying
IPIO's ``min_genomic_prevalence`` floor (default 0.025, see
``IPIO.survival_analysis.cox_aggregated.DEFAULT_MIN_GENOMIC_PREVALENCE``) to
that whole tuple would incorrectly prevalence-filter the Gleason score as if
it were a binary mutation indicator.

The fix:
  * ``COMPASS.survival_analysis.cox_aggregated.prepare_landmark_context``
    reads the ``feature_kind`` column in ``somatic_gleason_features.csv``
    (written by ``build_somatic_gleason_inputs.py``) to isolate the
    ``"somatic_binary"`` rows as ``genomic_feature_cols``, distinct from
    ``always_include_feature_cols`` (which still contains every declared
    feature, including ``GLEASON_SCORE``).
  * ``survival_common.cox_models.tune_multivariable_model`` gained a
    dedicated ``genomic_feature_cols`` parameter (previously it silently
    reused ``always_include_feature_cols`` as the genomic set, which was
    only safe for callers -- like IPIO's genomics/all feature subsets --
    where every always-include feature genuinely is genomic).

This test exercises ``select_feature_columns`` directly to prove that
scoping the prevalence floor to the true genomic subset spares a rare-valued
non-genomic always-include feature (standing in for GLEASON_SCORE) from
being dropped, while a rare genomic indicator in the same always-include set
is correctly dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from survival_common.cox_models import select_feature_columns

N = 200


def _cohort():
    rng = np.random.default_rng(7)
    data = pd.DataFrame(
        {
            "AGE": rng.normal(65, 8, size=N),
            # Rare binary mutation indicator: mutated in <2.5% of patients.
            "GENE_X_SV": rng.binomial(1, 0.01, size=N).astype(float),
            # Common binary mutation indicator: comfortably above the floor.
            "GENE_Y_SNV": rng.binomial(1, 0.30, size=N).astype(float),
            # Continuous clinical score, standing in for GLEASON_SCORE -- not
            # binary, and must never be prevalence-filtered as if it were.
            "GLEASON_SCORE": rng.choice([6, 7, 8, 9, 10], size=N).astype(float),
        }
    )
    return data


class TestGenomicPrevalenceFloorScoping:
    def test_rare_genomic_feature_dropped_when_correctly_scoped(self):
        data = _cohort()
        raw_feature_cols = ["GENE_X_SV", "GENE_Y_SNV", "GLEASON_SCORE"]
        always_include = raw_feature_cols  # mirrors COMPASS's undifferentiated tuple
        genomic_only = ["GENE_X_SV", "GENE_Y_SNV"]  # mirrors feature_kind == "somatic_binary"

        selected, _ = select_feature_columns(
            data,
            raw_feature_cols,
            min_patient_coverage=0.0,
            always_include=always_include,
            genomic_feature_cols=genomic_only,
            min_genomic_prevalence=0.025,
        )

        assert "GENE_X_SV" not in selected
        assert "GENE_Y_SNV" in selected
        assert "GLEASON_SCORE" in selected

    def test_unscoped_genomic_cols_would_wrongly_filter_gleason_if_binary_valued(self):
        # Guards against regressing to the old behavior of passing the full
        # always_include tuple as genomic_feature_cols: demonstrates that a
        # non-genomic feature whose values happen to intersect {0, 1} *is*
        # vulnerable to the floor when it's incorrectly included in the
        # genomic set, motivating why the fix computes genomic_only
        # separately rather than reusing always_include_feature_cols.
        data = pd.DataFrame(
            {
                "AGE": np.random.default_rng(1).normal(65, 8, size=N),
                "GENE_X_SV": np.random.default_rng(2).binomial(1, 0.30, size=N).astype(float),
                "RARE_BINARY_CLINICAL_FLAG": np.random.default_rng(3)
                .binomial(1, 0.01, size=N)
                .astype(float),
            }
        )
        raw_feature_cols = ["GENE_X_SV", "RARE_BINARY_CLINICAL_FLAG"]

        selected_scoped, _ = select_feature_columns(
            data,
            raw_feature_cols,
            min_patient_coverage=0.0,
            always_include=raw_feature_cols,
            genomic_feature_cols=["GENE_X_SV"],
            min_genomic_prevalence=0.025,
        )
        assert "RARE_BINARY_CLINICAL_FLAG" in selected_scoped

        selected_unscoped, _ = select_feature_columns(
            data,
            raw_feature_cols,
            min_patient_coverage=0.0,
            always_include=raw_feature_cols,
            genomic_feature_cols=raw_feature_cols,  # the old, unscoped behavior
            min_genomic_prevalence=0.025,
        )
        assert "RARE_BINARY_CLINICAL_FLAG" not in selected_unscoped


class TestPrepareLandmarkContextSourceUsesFeatureKind:
    def test_source_reads_feature_kind_for_genomic_scoping(self):
        import inspect

        import COMPASS.survival_analysis.cox_aggregated as compass_ca

        source = inspect.getsource(compass_ca.prepare_landmark_context)
        assert "feature_kind" in source
        assert "somatic_binary" in source
        assert "genomic_feature_cols" in source

    def test_default_min_genomic_prevalence_is_0_025(self):
        # Mirrors IPIO.survival_analysis.cox_aggregated.DEFAULT_MIN_GENOMIC_PREVALENCE
        # (0.025). Not imported directly alongside COMPASS's module here: both
        # adapters mutate sys.path at import time in conflicting ways, so
        # importing both cox_aggregated modules in one process is unsafe.
        import COMPASS.survival_analysis.cox_aggregated as compass_ca

        assert compass_ca.DEFAULT_MIN_GENOMIC_PREVALENCE == 0.025


class TestGenomicFeatureSetArmsThePrevalenceFloor:
    """The 'genomic' feature set must scope genomic_feature_cols.

    genomic_aggregated.csv was previously consumed under feature_set="labs",
    whose branch leaves genomic_feature_cols empty. Because
    select_feature_columns guards the floor on `if genomic_feature_cols and
    ...`, an empty tuple silently disables it -- so no prevalence filtering
    happened on the COMPASS genomic arm at all.
    """

    def test_genomic_branch_scopes_and_excludes_non_snv(self):
        import inspect

        import COMPASS.survival_analysis.cox_aggregated as compass_ca

        source = inspect.getsource(compass_ca.prepare_landmark_context)
        # The branch must exist and must populate the scoping tuple.
        assert 'feature_set == "genomic"' in source
        assert "always_include_feature_cols = genomic_feature_cols" in source

    def test_snv_only_regexes_are_defined(self):
        import COMPASS.survival_analysis.cox_aggregated as compass_ca

        # Testable set is SNV-only; the broad pattern exists so non-SNV
        # columns can be excluded rather than falling into the lab set.
        assert compass_ca.GENOMIC_FEATURE_RE.match("TP53_SNV")
        assert not compass_ca.GENOMIC_FEATURE_RE.match("PTEN_DEL")
        assert not compass_ca.GENOMIC_FEATURE_RE.match("AR_AMP")
        for col in ("TP53_SNV", "PTEN_DEL", "AR_AMP", "TMPRSS2_SV"):
            assert compass_ca.ANY_VARIANT_RE.match(col)
        # Hyphenated / dotted gene symbols must not be silently dropped.
        assert compass_ca.GENOMIC_FEATURE_RE.match("BRCA-2_SNV")
        assert compass_ca.GENOMIC_FEATURE_RE.match("PIK3.CA_SNV")


class TestEmptyGenomicColsDisablesFloor:
    def test_empty_tuple_admits_below_floor_feature(self):
        """Regression guard for the silent-disable behavior itself."""
        import numpy as np
        import pandas as pd

        from survival_common.cox_models import select_feature_columns

        n = 400
        rng = np.random.default_rng(0)
        data = pd.DataFrame({
            "TP53_SNV": (rng.random(n) < 0.30).astype(int),
            "RARE_SNV": [1] * 4 + [0] * (n - 4),
            "ALT_mean": rng.normal(size=n),
        })
        cols = list(data.columns)
        kwargs = dict(min_patient_coverage=0.1, restrict_to_labs=None,
                      always_include=None, min_genomic_prevalence=0.025)

        unscoped, _ = select_feature_columns(
            data, cols, genomic_feature_cols=[], **kwargs)
        scoped, _ = select_feature_columns(
            data, cols, genomic_feature_cols=["TP53_SNV", "RARE_SNV"], **kwargs)

        assert "RARE_SNV" in unscoped, "empty tuple disables the floor"
        assert "RARE_SNV" not in scoped, "scoped floor must drop the 1% feature"
        assert "TP53_SNV" in scoped
