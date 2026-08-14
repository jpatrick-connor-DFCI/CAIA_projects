"""The ADT->NEPC endpoint: t_nepc construction, its leakage guard, and the
guarantee that adding it left the platinum endpoint untouched.

Grounded in survival_common/cohort.py (make_outcome_df's NEPC derivation,
landmark rebasing, administrative censoring, and the ``require_nepc``-gated
validity conditions) and COMPASS/survival_analysis/cox_aggregated.py
(ENDPOINTS + OUTCOME_METADATA_COLUMNS).

The central invariant here is the *gate*: ``t_nepc > 0`` makes NEPC an incident
endpoint by dropping diagnoses at or before the landmark. That is correct for
the NEPC run and wrong for the platinum run, so it must never fire unless
``require_nepc=True`` was asked for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from COMPASS.survival_analysis import cox_aggregated as cox
from survival_common.cohort import make_outcome_df


def _patient_row(
    mrn: int,
    *,
    platinum: int = 0,
    t_platinum: float = 500.0,
    death: int = 0,
    t_last_contact: float = 1000.0,
    nepc: int | None = None,
    t_nepc: float | None = None,
) -> dict:
    """One patient row in the Stage 2 longitudinal_prediction_data schema.

    ``nepc=None`` omits the NEPC columns entirely, reproducing a cohort built
    without the LLM annotations mounted.
    """
    row = {
        "DFCI_MRN": mrn,
        "AGE_AT_TREATMENTSTART": 65.0,
        "FIRST_RECORD_DATE": "2015-01-01",
        "FIRST_TREATMENT": 1,
        "t_first_treatment": 0.0,
        "t_platinum": t_platinum,
        "PLATINUM": platinum,
        "t_death": t_last_contact,
        "DEATH": death,
        "t_last_contact": t_last_contact,
        "LAB_NAME": pd.NA,
        "LAB_VALUE": pd.NA,
        "t_lab": pd.NA,
    }
    if nepc is not None:
        row["NEPC"] = nepc
        row["t_nepc"] = t_last_contact if t_nepc is None else t_nepc
        row["NEPC_DATE_SOURCE"] = "stated"
        row["NEPC_DATE_PRECISION"] = "day"
        row["NEPC_LABEL_SOURCE"] = "adjudicated"
    return row


def _outcome(rows: list[dict], **kwargs) -> pd.DataFrame:
    return make_outcome_df(
        pd.DataFrame(rows),
        anchor_col=None,
        require_first_treatment=False,
        **kwargs,
    )


class TestTNepcConstruction:
    """The three cases the endpoint has to get right."""

    def test_positive_keeps_its_diagnosis_time(self):
        out = _outcome([_patient_row(1, nepc=1, t_nepc=531.0, t_last_contact=1200.0)])
        assert out.loc[1, "NEPC"] == 1
        assert out.loc[1, "t_nepc"] == 531.0

    def test_negative_is_censored_at_last_contact(self):
        out = _outcome([_patient_row(2, nepc=0, t_last_contact=1096.0)])
        assert out.loc[2, "NEPC"] == 0
        assert out.loc[2, "t_nepc"] == 1096.0

    def test_prevalent_diagnosis_is_dropped_by_the_gate(self):
        """A diagnosis at or before the landmark rebases to t_nepc <= 0 and is
        excluded, which is what makes this an *incident* endpoint."""
        rows = [
            _patient_row(1, nepc=1, t_nepc=531.0, t_last_contact=1200.0),
            # Diagnosed on the anchor date itself: prevalent, not incident.
            _patient_row(3, nepc=1, t_nepc=0.0, t_last_contact=1200.0),
        ]
        kept = _outcome(rows, require_nepc=True)
        assert 1 in kept.index
        assert 3 not in kept.index

    def test_prevalent_diagnosis_survives_when_the_gate_is_off(self):
        """Same patient, platinum run: the NEPC gate must not touch them."""
        rows = [_patient_row(3, nepc=1, t_nepc=0.0, t_last_contact=1200.0)]
        kept = _outcome(rows)
        assert 3 in kept.index

    def test_landmark_rebasing_shifts_t_nepc(self):
        out = _outcome(
            [_patient_row(1, nepc=1, t_nepc=531.0, t_last_contact=1200.0)],
            landmark_offset_days=90,
        )
        assert out.loc[1, "t_nepc"] == 531.0 - 90.0
        # The pre-rebase value is retained for diagnostics.
        assert out.loc[1, "t_nepc_from_first_record"] == 531.0

    def test_event_past_the_horizon_is_administratively_censored(self):
        out = _outcome(
            [_patient_row(1, nepc=1, t_nepc=4000.0, t_last_contact=4000.0)],
            max_followup_days=3650.0,
        )
        assert out.loc[1, "NEPC"] == 0
        assert out.loc[1, "t_nepc"] == 3650.0


class TestPlatinumPathUnchanged:
    """Adding the endpoint must not disturb the existing platinum run."""

    def test_cohort_without_nepc_columns_still_builds(self):
        out = _outcome([_patient_row(1, platinum=1, t_platinum=400.0)])
        assert out.loc[1, "PLATINUM"] == 1
        assert out.loc[1, "t_platinum"] == 400.0
        assert "t_nepc" not in out.columns

    def test_nepc_columns_present_but_gate_off_leaves_platinum_cohort_identical(self):
        """The same patients, with and without NEPC columns attached, must
        yield the same platinum cohort and the same platinum durations."""
        without = [
            _patient_row(1, platinum=1, t_platinum=400.0),
            _patient_row(2, platinum=0, t_last_contact=900.0),
            # Prevalent NEPC in the "with" version; must NOT be dropped here.
            _patient_row(3, platinum=1, t_platinum=300.0),
        ]
        with_nepc = [
            _patient_row(1, platinum=1, t_platinum=400.0, nepc=0),
            _patient_row(2, platinum=0, t_last_contact=900.0, nepc=1, t_nepc=500.0),
            _patient_row(3, platinum=1, t_platinum=300.0, nepc=1, t_nepc=0.0),
        ]
        a = _outcome(without)
        b = _outcome(with_nepc)

        assert list(a.index) == list(b.index)
        for col in ("PLATINUM", "t_platinum", "DEATH", "t_death", "EITHER", "t_either"):
            pd.testing.assert_series_equal(a[col], b[col], check_names=False)

    def test_require_nepc_raises_when_the_cohort_has_no_nepc_columns(self):
        with pytest.raises(ValueError, match="require_nepc"):
            _outcome([_patient_row(1, platinum=1)], require_nepc=True)


class TestNepcLeakageGuard:
    """If a NEPC column escapes OUTCOME_METADATA_COLUMNS it becomes a feature,
    and NEPC status predicts NEPC."""

    NEPC_OUTCOME_COLS = (
        "NEPC",
        "NEPC_DATE",
        "t_nepc",
        "t_nepc_from_first_record",
        "NEPC_DATE_SOURCE",
        "NEPC_DATE_PRECISION",
        "NEPC_LABEL_SOURCE",
    )

    def test_every_nepc_column_is_an_outcome_column(self):
        missing = set(self.NEPC_OUTCOME_COLS) - cox.outcome_columns()
        assert not missing, f"NEPC columns missing from the leakage guard: {sorted(missing)}"

    def test_nepc_columns_are_not_selected_as_features(self):
        """The real guard: select_feature_columns must reject them even when a
        caller passes them in as candidate features."""
        n = 40
        rng = np.random.default_rng(0)
        data = pd.DataFrame(
            {
                "DFCI_MRN": range(n),
                "real_lab": rng.normal(size=n),
                "NEPC": rng.integers(0, 2, size=n),
                "t_nepc": rng.normal(500, 50, size=n),
                "t_nepc_from_first_record": rng.normal(500, 50, size=n),
            }
        )
        raw_feature_cols = [
            c for c in data.columns if c not in cox.outcome_columns() and c != "DFCI_MRN"
        ]
        selected, _ = cox.select_feature_columns(
            data, raw_feature_cols, min_patient_coverage=0.0
        )
        assert "real_lab" in selected
        for col in ("NEPC", "t_nepc", "t_nepc_from_first_record"):
            assert col not in selected


class TestEndpointRegistry:
    def test_nepc_endpoint_is_registered(self):
        assert "nepc" in cox.ENDPOINTS
        assert cox.ENDPOINTS["nepc"]["duration_col"] == "t_nepc"
        assert cox.ENDPOINTS["nepc"]["event_col"] == "NEPC"

    def test_platinum_endpoint_spec_is_unchanged(self):
        assert cox.ENDPOINTS["platinum"]["duration_col"] == "t_platinum"
        assert cox.ENDPOINTS["platinum"]["event_col"] == "PLATINUM"

    def test_default_endpoints_still_platinum_only(self):
        """The project default must not silently start running NEPC."""
        from survival_common.projects.compass_profile import CONFIG

        assert CONFIG.default_endpoints == ("platinum",)
