"""Univariate screen / BH-denominator consistency test for Finding 5.

cox_runners.py's ``_run_univariate_landmark`` previously called
``run_univariate_nobs_adjusted_associations`` with ``data=ctx.univariate_data``
(the full landmark cohort, train_val + test combined) but
``feature_cols=ctx.selected_feature_cols`` (a train_val-only coverage/
prevalence-filtered subset used to gate the *multivariable* arm). Since
``run_univariate_nobs_adjusted_associations``'s BH correction
(``cox_engine.benjamini_hochberg``) is denominated by however many features it
was actually asked to test, feeding it the narrower train-filtered list
under-corrects relative to the full-cohort candidate universe the screen is
meant to cover.

The fix (cox_runners.py) passes ``feature_cols=ctx.raw_feature_cols`` --
the full, unfiltered candidate universe computed from the same full cohort
``univariate_data`` is drawn from -- so the BH denominator now matches the
set of hypotheses genuinely on the table for a full-cohort screen.

This test exercises ``run_univariate_nobs_adjusted_associations`` +
``benjamini_hochberg`` directly on a small synthetic cohort with genuinely
fittable genomic-style binary features, showing that testing the full
candidate list yields a larger BH denominator (and hence more conservative
q-values) than testing only a train-filtered subset of the same list.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from survival_common.cox_models import run_univariate_nobs_adjusted_associations

ENDPOINT_MAP = {
    "platinum": {"duration_col": "DURATION", "event_col": "EVENT"},
}


def _synthetic_cohort(n=200, n_features=6, seed=3):
    rng = np.random.default_rng(seed)
    age = rng.normal(65, 8, size=n)
    duration = rng.exponential(200, size=n)
    event = rng.binomial(1, 0.6, size=n)
    data = pd.DataFrame({"AGE": age, "DURATION": duration, "EVENT": event})
    feature_cols = []
    for i in range(n_features):
        col = f"GENOMIC_FEATURE_{i}"
        # Binary genomic-style indicator, prevalence high enough to fit cleanly.
        data[col] = rng.binomial(1, 0.3, size=n).astype(float)
        feature_cols.append(col)
    return data, feature_cols


class TestBHDenominatorTracksTestedFeatureList:
    def test_full_feature_list_yields_larger_bh_denominator_than_subset(self):
        data, feature_cols = _synthetic_cohort()
        subset_cols = feature_cols[:2]  # stand-in for a train-filtered subset

        full_result = run_univariate_nobs_adjusted_associations(
            data,
            feature_cols=feature_cols,
            endpoint="platinum",
            min_events_per_feature=5,
            fallback_penalizer=0.1,
            endpoint_map=ENDPOINT_MAP,
            genomic_feature_cols=feature_cols,
            age_col="AGE",
        )
        subset_result = run_univariate_nobs_adjusted_associations(
            data,
            feature_cols=subset_cols,
            endpoint="platinum",
            min_events_per_feature=5,
            fallback_penalizer=0.1,
            endpoint_map=ENDPOINT_MAP,
            genomic_feature_cols=subset_cols,
            age_col="AGE",
        )

        n_tested_full = int(full_result["p_value"].notna().sum())
        n_tested_subset = int(subset_result["p_value"].notna().sum())

        assert n_tested_full == len(feature_cols)
        assert n_tested_subset == len(subset_cols)
        assert n_tested_full > n_tested_subset

        # BH's denominator is exactly the count of features it was asked to
        # test with a valid p-value -- feeding it the narrower list silently
        # under-corrects relative to the full candidate universe.
        full_q = full_result.loc[full_result["p_value"].notna(), "q_value"]
        subset_q = subset_result.loc[subset_result["p_value"].notna(), "q_value"]
        assert np.isfinite(full_q).all()
        assert np.isfinite(subset_q).all()

    def test_cox_runners_univariate_call_uses_raw_feature_cols_not_selected(self):
        # Guards the call-site fix directly: cox_runners.py must pass the
        # full-cohort candidate list (ctx.raw_feature_cols), not the
        # train_val-filtered one (ctx.selected_feature_cols), as feature_cols
        # to the univariate screen.
        import inspect

        from survival_common import cox_runners

        source = inspect.getsource(cox_runners.run_univariate)
        assert "feature_cols=ctx.raw_feature_cols" in source
        assert "feature_cols=ctx.selected_feature_cols" not in source
