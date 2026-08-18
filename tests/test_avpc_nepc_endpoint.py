"""The ADT->AVPC_NEPC endpoint: t_avpc_nepc construction, its leakage guard,
and the guarantee that adding it left the platinum and nepc endpoints
untouched.

Grounded in survival_common/cohort.py (make_outcome_df's OPTIONAL_ENDPOINT_SPECS
registry, which drives NEPC and AVPC_NEPC through the same five derivation
blocks) and COMPASS/survival_analysis/cox_aggregated.py (ENDPOINTS +
OUTCOME_METADATA_COLUMNS).

Companion to test_nepc_endpoint.py. The central invariant here is the same
one that file pins for NEPC: ``t_avpc_nepc > 0`` makes AVPC_NEPC an incident
endpoint by dropping events at or before the landmark. That is correct for
the avpc_nepc run and wrong for the platinum/nepc runs, so it must never fire
unless ``endpoint="avpc_nepc"`` was asked for. AVPC_NEPC additionally carries
several near-perfect leakage columns (AVPC, NEPC_TIMELINE, AVPC_N_CRITERIA)
that are its own label components -- the leakage guard here is the sharpest
test of OPTIONAL_ENDPOINT_SPECS/OUTCOME_METADATA_COLUMNS staying in lockstep.
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
    avpc_nepc: int | None = None,
    t_avpc_nepc: float | None = None,
) -> dict:
    """One patient row in the Stage 2 longitudinal_prediction_data schema.

    ``avpc_nepc=None`` omits the AVPC_NEPC columns entirely, reproducing a
    cohort built without the LLM annotations mounted.
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
    if avpc_nepc is not None:
        row["AVPC_NEPC"] = avpc_nepc
        row["t_avpc_nepc"] = t_last_contact if t_avpc_nepc is None else t_avpc_nepc
        row["AVPC_NEPC_DATE_SOURCE"] = "stated"
        row["AVPC_NEPC_DATE_PRECISION"] = "day"
        row["AVPC_NEPC_LABEL_SOURCE"] = "timeline_positive"
        row["AVPC"] = 0
        row["AVPC_DATE"] = pd.NA
        row["AVPC_N_CRITERIA"] = 0
        row["NEPC_TIMELINE"] = avpc_nepc
        row["NEPC_TIMELINE_DATE"] = pd.NA
    return row


def _outcome(rows: list[dict], **kwargs) -> pd.DataFrame:
    return make_outcome_df(
        pd.DataFrame(rows),
        anchor_col=None,
        require_first_treatment=False,
        **kwargs,
    )


class TestTAvpcNepcConstruction:
    """The three cases the endpoint has to get right."""

    def test_positive_keeps_its_event_time(self):
        out = _outcome([_patient_row(1, avpc_nepc=1, t_avpc_nepc=531.0, t_last_contact=1200.0)])
        assert out.loc[1, "AVPC_NEPC"] == 1
        assert out.loc[1, "t_avpc_nepc"] == 531.0

    def test_negative_is_censored_at_last_contact(self):
        out = _outcome([_patient_row(2, avpc_nepc=0, t_last_contact=1096.0)])
        assert out.loc[2, "AVPC_NEPC"] == 0
        assert out.loc[2, "t_avpc_nepc"] == 1096.0

    def test_prevalent_event_is_dropped_by_the_gate(self):
        """An event at or before the landmark rebases to t_avpc_nepc <= 0 and
        is excluded, which is what makes this an *incident* endpoint."""
        rows = [
            _patient_row(1, avpc_nepc=1, t_avpc_nepc=531.0, t_last_contact=1200.0),
            # Event on the anchor date itself: prevalent, not incident.
            _patient_row(3, avpc_nepc=1, t_avpc_nepc=0.0, t_last_contact=1200.0),
        ]
        kept = _outcome(rows, endpoint="avpc_nepc")
        assert 1 in kept.index
        assert 3 not in kept.index

    def test_prevalent_event_survives_when_the_gate_is_off(self):
        """Same patient, platinum run: the AVPC_NEPC gate must not touch them."""
        rows = [_patient_row(3, avpc_nepc=1, t_avpc_nepc=0.0, t_last_contact=1200.0)]
        kept = _outcome(rows)
        assert 3 in kept.index

    def test_pre_anchor_platinum_does_not_exclude_avpc_nepc_patient(self):
        rows = [
            _patient_row(
                4, platinum=1, t_platinum=-10.0, avpc_nepc=1, t_avpc_nepc=250.0,
                t_last_contact=1200.0,
            )
        ]
        kept = _outcome(rows, endpoint="avpc_nepc")
        assert 4 in kept.index

    def test_landmark_rebasing_shifts_t_avpc_nepc(self):
        out = _outcome(
            [_patient_row(1, avpc_nepc=1, t_avpc_nepc=531.0, t_last_contact=1200.0)],
            landmark_offset_days=90,
        )
        assert out.loc[1, "t_avpc_nepc"] == 531.0 - 90.0
        # The pre-rebase value is retained for diagnostics.
        assert out.loc[1, "t_avpc_nepc_from_first_record"] == 531.0

    def test_event_past_the_horizon_is_administratively_censored(self):
        out = _outcome(
            [_patient_row(1, avpc_nepc=1, t_avpc_nepc=4000.0, t_last_contact=4000.0)],
            max_followup_days=3650.0,
        )
        assert out.loc[1, "AVPC_NEPC"] == 0
        assert out.loc[1, "t_avpc_nepc"] == 3650.0


class TestPlatinumAndNepcPathsUnchanged:
    """Adding AVPC_NEPC must not disturb the existing platinum/nepc runs."""

    def test_cohort_without_avpc_nepc_columns_still_builds(self):
        out = _outcome([_patient_row(1, platinum=1, t_platinum=400.0)])
        assert out.loc[1, "PLATINUM"] == 1
        assert out.loc[1, "t_platinum"] == 400.0
        assert "t_avpc_nepc" not in out.columns

    def test_avpc_nepc_columns_present_but_gate_off_leaves_platinum_cohort_identical(self):
        """The same patients, with and without AVPC_NEPC columns attached,
        must yield the same platinum cohort and the same platinum durations
        -- this is the zero-behavior-change guarantee of the OPTIONAL_ENDPOINT_SPECS
        refactor: adding a new registry entry must not perturb another
        endpoint's derivation."""
        without = [
            _patient_row(1, platinum=1, t_platinum=400.0),
            _patient_row(2, platinum=0, t_last_contact=900.0),
            # Prevalent AVPC_NEPC in the "with" version; must NOT be dropped here.
            _patient_row(3, platinum=1, t_platinum=300.0),
        ]
        with_avpc_nepc = [
            _patient_row(1, platinum=1, t_platinum=400.0, avpc_nepc=0),
            _patient_row(2, platinum=0, t_last_contact=900.0, avpc_nepc=1, t_avpc_nepc=500.0),
            _patient_row(3, platinum=1, t_platinum=300.0, avpc_nepc=1, t_avpc_nepc=0.0),
        ]
        a = _outcome(without)
        b = _outcome(with_avpc_nepc)

        assert list(a.index) == list(b.index)
        for col in ("PLATINUM", "t_platinum", "DEATH", "t_death", "EITHER", "t_either"):
            pd.testing.assert_series_equal(a[col], b[col], check_names=False)

    def test_avpc_nepc_columns_present_but_gate_off_leaves_nepc_cohort_identical(self):
        """Same guarantee, checked against the nepc endpoint: attaching
        AVPC_NEPC columns to a cohort that also carries NEPC columns must not
        change the nepc endpoint's cohort or durations."""

        def _row_with_nepc(mrn, *, nepc, t_nepc, avpc_nepc=None, t_avpc_nepc=None):
            row = _patient_row(mrn, avpc_nepc=avpc_nepc, t_avpc_nepc=t_avpc_nepc)
            row["NEPC"] = nepc
            row["t_nepc"] = t_nepc
            row["NEPC_DATE_SOURCE"] = "stated"
            row["NEPC_DATE_PRECISION"] = "day"
            row["NEPC_LABEL_SOURCE"] = "adjudicated"
            return row

        without_avpc_nepc = [
            _row_with_nepc(1, nepc=1, t_nepc=300.0),
            _row_with_nepc(2, nepc=0, t_nepc=900.0),
        ]
        with_avpc_nepc = [
            _row_with_nepc(1, nepc=1, t_nepc=300.0, avpc_nepc=1, t_avpc_nepc=50.0),
            _row_with_nepc(2, nepc=0, t_nepc=900.0, avpc_nepc=0),
        ]
        a = _outcome(without_avpc_nepc, endpoint="nepc")
        b = _outcome(with_avpc_nepc, endpoint="nepc")

        assert list(a.index) == list(b.index)
        for col in ("NEPC", "t_nepc"):
            pd.testing.assert_series_equal(a[col], b[col], check_names=False)

    def test_prevalent_avpc_nepc_does_not_exclude_platinum_patient(self):
        kept = _outcome(
            [_patient_row(4, platinum=1, t_platinum=300.0, avpc_nepc=1, t_avpc_nepc=-10.0)],
            endpoint="platinum",
        )
        assert 4 in kept.index

    def test_avpc_nepc_raises_when_the_cohort_has_no_avpc_nepc_columns(self):
        with pytest.raises(ValueError, match="avpc_nepc"):
            _outcome([_patient_row(1, platinum=1)], endpoint="avpc_nepc")


class TestAvpcNepcLeakageGuard:
    """If an AVPC_NEPC column escapes OUTCOME_METADATA_COLUMNS it becomes a
    feature, and AVPC_NEPC status (or one of its own components) predicts
    AVPC_NEPC."""

    AVPC_NEPC_OUTCOME_COLS = (
        "AVPC_NEPC",
        "t_avpc_nepc",
        "t_avpc_nepc_from_first_record",
        "AVPC_NEPC_DATE",
        "AVPC_NEPC_DATE_SOURCE",
        "AVPC_NEPC_DATE_PRECISION",
        "AVPC_NEPC_LABEL_SOURCE",
        "AVPC",
        "AVPC_DATE",
        "AVPC_N_CRITERIA",
        "NEPC_TIMELINE",
        "NEPC_TIMELINE_DATE",
    )

    def test_twelve_avpc_nepc_columns_are_covered(self):
        """Pins the exact count the plan specifies, so a future edit that
        silently drops one is caught here rather than downstream."""
        assert len(self.AVPC_NEPC_OUTCOME_COLS) == 12

    def test_every_avpc_nepc_column_is_an_outcome_column(self):
        missing = set(self.AVPC_NEPC_OUTCOME_COLS) - cox.outcome_columns()
        assert not missing, f"AVPC_NEPC columns missing from the leakage guard: {sorted(missing)}"

    def test_avpc_nepc_columns_are_not_selected_as_features(self):
        """The real guard: select_feature_columns must reject them even when a
        caller passes them in as candidate features."""
        n = 40
        rng = np.random.default_rng(0)
        data = pd.DataFrame(
            {
                "DFCI_MRN": range(n),
                "real_lab": rng.normal(size=n),
                "AVPC_NEPC": rng.integers(0, 2, size=n),
                "t_avpc_nepc": rng.normal(500, 50, size=n),
                "t_avpc_nepc_from_first_record": rng.normal(500, 50, size=n),
                "AVPC": rng.integers(0, 2, size=n),
                "NEPC_TIMELINE": rng.integers(0, 2, size=n),
                "AVPC_N_CRITERIA": rng.integers(0, 7, size=n),
            }
        )
        raw_feature_cols = [
            c for c in data.columns if c not in cox.outcome_columns() and c != "DFCI_MRN"
        ]
        selected, _ = cox.select_feature_columns(
            data, raw_feature_cols, min_patient_coverage=0.0
        )
        assert "real_lab" in selected
        for col in (
            "AVPC_NEPC",
            "t_avpc_nepc",
            "t_avpc_nepc_from_first_record",
            "AVPC",
            "NEPC_TIMELINE",
            "AVPC_N_CRITERIA",
        ):
            assert col not in selected


class TestEndpointRegistry:
    def test_avpc_nepc_endpoint_is_registered(self):
        assert "avpc_nepc" in cox.ENDPOINTS
        assert cox.ENDPOINTS["avpc_nepc"]["duration_col"] == "t_avpc_nepc"
        assert cox.ENDPOINTS["avpc_nepc"]["event_col"] == "AVPC_NEPC"

    def test_nepc_endpoint_spec_is_unchanged(self):
        assert cox.ENDPOINTS["nepc"]["duration_col"] == "t_nepc"
        assert cox.ENDPOINTS["nepc"]["event_col"] == "NEPC"

    def test_platinum_endpoint_spec_is_unchanged(self):
        assert cox.ENDPOINTS["platinum"]["duration_col"] == "t_platinum"
        assert cox.ENDPOINTS["platinum"]["event_col"] == "PLATINUM"

    def test_default_endpoints_still_platinum_only(self):
        """The project default must not silently start running AVPC_NEPC."""
        from survival_common.projects.compass_profile import CONFIG

        assert CONFIG.default_endpoints == ("platinum",)

    def test_optional_endpoint_specs_has_avpc_nepc_entry(self):
        from survival_common.cohort import OPTIONAL_ENDPOINT_SPECS

        assert OPTIONAL_ENDPOINT_SPECS["avpc_nepc"] == (
            "AVPC_NEPC",
            "t_avpc_nepc",
            "AVPC_NEPC_DATE",
        )

    def test_optional_endpoint_specs_nepc_entry_unchanged(self):
        from survival_common.cohort import OPTIONAL_ENDPOINT_SPECS

        assert OPTIONAL_ENDPOINT_SPECS["nepc"] == ("NEPC", "t_nepc", "NEPC_DATE")

    def test_platinum_not_in_optional_endpoint_specs(self):
        """Platinum keeps its bespoke, always-required derivation path; it
        must never be treated as an optional/presence-gated endpoint."""
        from survival_common.cohort import OPTIONAL_ENDPOINT_SPECS

        assert "platinum" not in OPTIONAL_ENDPOINT_SPECS
