"""One metrics schema across every model family.

Before the cutover, elastic-net Cox wrote ``test_c_index`` /
``n_events_train_val`` / ``landmark_days``, XGBoost wrote ``c_index`` /
``n_train_val_events`` / ``landmark_day``, and Dynamic-DeepHit wrote
``c_index`` and named the endpoint column ``event``. Every downstream reader
carried per-family candidate lists to paper over it, and a rename on one side
silently produced an all-NA figure panel rather than an error.

These tests pin the shared block down at the source level: each writer must
construct its metrics row from the canonical spellings, and the R figure
pipeline must read them without fallbacks. Source-text assertions rather than
end-to-end fits, since fitting any of these families needs mounted PROFILE
data.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from survival_common.metrics_schema import (
    CANONICAL_METRIC_COLUMNS,
    DEFAULT_COHORT,
    canonical_identity,
    missing_canonical_columns,
    order_canonical_first,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Every module that constructs a `*_metrics*.csv` row. survlatent_ode.py is
# deliberately absent: its metrics frame comes from the external repo's
# eval_model, whose schema is not ours to pin.
WRITER_MODULES = {
    "elastic_net_cox": REPO_ROOT / "survival_common" / "cox_runners.py",
    "xgboost_compass": REPO_ROOT / "COMPASS" / "survival_analysis" / "multivariate_analysis.py",
    "xgboost_ipio": REPO_ROOT / "IPIO" / "survival_analysis" / "multivariate_analysis.py",
    "dynamic_deephit": REPO_ROOT / "survival_common" / "longitudinal_runners.py",
}

FIGURE_PIPELINE_R = (
    REPO_ROOT / "COMPASS" / "survival_analysis" / "COMPASS_generate_figures_pipeline.R"
)

# Spellings the cutover retired from the metrics block. A writer reintroducing
# one of these is the exact regression this file exists to catch.
#
# `landmark_day` (singular) is NOT listed: it remains the correct column name
# in the per-landmark CV-fold, patient-risk, and feature-importance frames,
# which are separate files outside this schema. Only the metrics row was
# renamed to `landmark_days`.
RETIRED_METRIC_SPELLINGS = (
    '"n_train_val_events"',
    '"n_test_events"',
    '"c_index":',
    '"mean_auc_t":',
    '"integrated_brier":',
)


class TestCanonicalSchemaHelpers:
    def test_identity_defaults_an_unset_cohort_to_all(self):
        identity = canonical_identity(
            model="elastic_net_cox",
            cohort=None,
            endpoint="nepc",
            landmark_days=90,
            config="both",
        )
        assert identity["cohort"] == DEFAULT_COHORT
        assert identity["landmark_days"] == 90
        assert identity["endpoint"] == "nepc"

    def test_identity_preserves_an_explicit_cohort(self):
        identity = canonical_identity(
            model="xgboost",
            cohort="metastatic",
            endpoint="platinum",
            landmark_days=0,
            config="baseline",
        )
        assert identity["cohort"] == "metastatic"
        assert identity["config"] == "baseline"

    def test_missing_columns_are_reported_in_schema_order(self):
        assert missing_canonical_columns(CANONICAL_METRIC_COLUMNS) == []
        partial = [c for c in CANONICAL_METRIC_COLUMNS if c != "test_c_index"]
        assert missing_canonical_columns(partial) == ["test_c_index"]

    def test_ordering_puts_the_canonical_block_first(self):
        frame = pd.DataFrame(
            [{"xgb_params": "{}", **{c: 0 for c in CANONICAL_METRIC_COLUMNS}}]
        )
        ordered = order_canonical_first(frame)
        assert list(ordered.columns[: len(CANONICAL_METRIC_COLUMNS)]) == list(
            CANONICAL_METRIC_COLUMNS
        )
        assert "xgb_params" in ordered.columns

    def test_ordering_rejects_a_frame_missing_a_canonical_column(self):
        """A writer that drifts must fail at write time, not produce an NA panel."""
        frame = pd.DataFrame(
            [{c: 0 for c in CANONICAL_METRIC_COLUMNS if c != "test_mean_auc_t"}]
        )
        with pytest.raises(ValueError, match="test_mean_auc_t"):
            order_canonical_first(frame)


@pytest.mark.parametrize("family", sorted(WRITER_MODULES))
class TestEveryWriterEmitsTheCanonicalBlock:
    def test_writer_routes_its_metrics_through_the_shared_schema(self, family):
        source = WRITER_MODULES[family].read_text()
        assert "survival_common.metrics_schema" in source, (
            f"{family} must build its metrics row from the shared schema module"
        )
        assert "order_canonical_first" in source, (
            f"{family} must validate/order its metrics frame before writing"
        )

    def test_writer_does_not_reintroduce_a_retired_spelling(self, family):
        source = WRITER_MODULES[family].read_text()
        offenders = [s for s in RETIRED_METRIC_SPELLINGS if s in source]
        assert not offenders, (
            f"{family} reintroduces retired metric spellings {offenders}; "
            f"use the canonical names in {list(CANONICAL_METRIC_COLUMNS)}"
        )

    def test_writer_emits_the_held_out_performance_columns(self, family):
        """Each family names the held-out metrics it is responsible for.

        Elastic-net Cox builds its row inside cox_models.fit_final_multivariable_model
        and Dynamic-DeepHit inside deephit_engine.compute_metrics, so those two
        are checked against the module that actually constructs the row rather
        than the runner that stamps identity onto it.
        """
        row_builders = {
            "elastic_net_cox": REPO_ROOT / "survival_common" / "cox_models.py",
            "dynamic_deephit": REPO_ROOT / "survival_common" / "deephit_engine.py",
        }
        source = row_builders.get(family, WRITER_MODULES[family]).read_text()
        columns = ("test_c_index", "test_mean_auc_t")
        if family != "dynamic_deephit":
            # DeepHit's integrated Brier is joined on in longitudinal_runners
            # from a separate per-cause computation, not built with the row.
            columns += ("test_integrated_brier",)
        for column in columns:
            assert f'"{column}"' in source, f"{family} must emit {column}"


class TestFigurePipelineReadsCanonicalNamesOnly:
    def test_reader_takes_no_per_family_metric_arguments(self):
        source = FIGURE_PIPELINE_R.read_text()
        assert "read_endpoint_performance <- function(path, endpoint) {" in source, (
            "read_endpoint_performance must no longer accept per-family column "
            "candidate lists"
        )

    def test_reader_does_not_union_endpoint_with_event(self):
        source = FIGURE_PIPELINE_R.read_text()
        assert 'intersect(c("endpoint", "event")' not in source, (
            "DeepHit now writes `endpoint`; the dual-spelling union is retired"
        )

    def test_reader_uses_canonical_count_columns(self):
        source = FIGURE_PIPELINE_R.read_text()
        assert 'c("n_events_train_val", "n_train_val_events")' not in source
        assert 'c("n_events_test", "n_test_events")' not in source
        assert 'pick("n_events_train_val")' in source
        assert 'pick("n_events_test")' in source

    def test_reader_reads_the_canonical_performance_columns(self):
        source = FIGURE_PIPELINE_R.read_text()
        for column in ("test_mean_auc_t", "test_c_index", "test_integrated_brier"):
            assert f'metric("{column}")' in source, (
                f"read_endpoint_performance must read {column} directly"
            )


class TestEveryWriterExposesACohortFlag:
    """`cohort` has no other source: unlike model/config it is not derivable
    at the writer, so each runner needs the CLI arg the pipeline passes."""

    @pytest.mark.parametrize(
        "path",
        [
            REPO_ROOT / "survival_common" / "cox_runners.py",
            REPO_ROOT / "COMPASS" / "survival_analysis" / "multivariate_analysis.py",
            REPO_ROOT / "IPIO" / "survival_analysis" / "multivariate_analysis.py",
            REPO_ROOT / "survival_common" / "longitudinal_runners.py",
        ],
        ids=["cox", "xgboost_compass", "xgboost_ipio", "dynamic_deephit"],
    )
    def test_runner_declares_the_cohort_argument(self, path):
        assert '"--cohort"' in path.read_text()

    def test_pipeline_passes_cohort_to_every_canonical_writer(self):
        source = (
            REPO_ROOT / "COMPASS" / "survival_analysis" / "compass_pipeline.py"
        ).read_text()
        assert 'cohort_args = ["--cohort"' in source
        # elastic-net, xgboost, dynamic-deephit -- survlatent-ode excluded.
        assert len(re.findall(r"\*cohort_args", source)) == 3
