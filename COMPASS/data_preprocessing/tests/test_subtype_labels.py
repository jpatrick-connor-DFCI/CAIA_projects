"""Stage 1's reader for the primary-subtype labels, plus the cohort-level
join/censoring/TT_* derivation in build_survival_cohort.

``load_subtype_labels`` reads primary_subtype_labels.parquet (the full join of
the LLM_clinical_annotations per-note NEPC and AVPC pipelines). Covers what
would silently corrupt the join: MRN normalization (labels arrive as strings,
sometimes with a trailing ".0" from a float round-trip), unresolved and
undated positives being censored rather than dropped, AVPC taken as-is from
the runner (no criteria-count gate), and the non-fatal missing-file path that
keeps the platinum pipeline working when the LLM annotations are not mounted.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from COMPASS.data_preprocessing.compile_COMPASS_cohort_data import (
    build_survival_cohort,
    load_subtype_labels,
)

EXPECTED_COLUMNS = [
    "DFCI_MRN",
    "NEPC_DATE",
    "NEPC",
    "NEPC_DATE_SOURCE",
    "NEPC_DATE_PRECISION",
    "NEPC_LABEL_SOURCE",
    "NEPC_STATED_DX",
    "NEPC_STATED_DX_DATE",
    "AVPC_DATE",
    "AVPC",
    "AVPC_LABEL_SOURCE",
    "AVPC_N_CRITERIA",
    "AVPC_CRITERIA",
    "PRIMARY_LABEL",
]


def _label_rows(**overrides) -> dict:
    """One conventional (negative on both arms) patient, columns overridable."""
    row = {
        "DFCI_MRN": ["1001"],
        "primary_label": ["conventional"],
        "nepc_review_status": ["ok"],
        "nepc_status": ["non_nepc"],
        "nepc_date": [None],
        "nepc_date_source": [None],
        "nepc_date_precision": [None],
        "has_stated_nepc_dx": [False],
        "stated_nepc_dx_date": [None],
        "avpc_review_status": ["ok"],
        "is_avpc": [False],
        "avpc_date": [None],
        "n_avpc_criteria": [0],
        "avpc_criteria": [[]],
    }
    row.update(overrides)
    return row


def _write_labels(tmp_path, rows: dict):
    path = tmp_path / "primary_subtype_labels.parquet"
    n = len(rows["DFCI_MRN"])
    schema = {
        "DFCI_MRN": pl.Utf8,
        "primary_label": pl.Utf8,
        "nepc_review_status": pl.Utf8,
        "nepc_status": pl.Utf8,
        "nepc_date": pl.Utf8,
        "nepc_date_source": pl.Utf8,
        "nepc_date_precision": pl.Utf8,
        "has_stated_nepc_dx": pl.Boolean,
        "stated_nepc_dx_date": pl.Utf8,
        "avpc_review_status": pl.Utf8,
        "is_avpc": pl.Boolean,
        "avpc_date": pl.Utf8,
        "n_avpc_criteria": pl.Int64,
        "avpc_criteria": pl.List(pl.Utf8),
    }
    assert all(len(v) == n for v in rows.values())
    pl.DataFrame(rows, schema={k: schema[k] for k in rows}).write_parquet(path)
    return path


class TestLoadSubtypeLabels:
    def test_missing_path_is_non_fatal_and_typed(self, tmp_path):
        """A cohort built without the annotations mounted must still build."""
        out = load_subtype_labels(tmp_path / "does_not_exist.parquet")
        assert out.height == 0
        assert out.columns == EXPECTED_COLUMNS

    def test_none_path_is_non_fatal(self):
        out = load_subtype_labels(None)
        assert out.height == 0
        assert out.columns == EXPECTED_COLUMNS

    def test_mrn_normalization_handles_strings_and_float_roundtrip(self, tmp_path):
        rows = _label_rows()
        rows = {k: v * 3 for k, v in rows.items()}
        # " 1001 " exercises the strip; "1002.0" the float round-trip.
        rows["DFCI_MRN"] = [" 1001 ", "1002.0", "1003"]
        out = load_subtype_labels(_write_labels(tmp_path, rows)).sort("DFCI_MRN")
        assert out["DFCI_MRN"].dtype == pl.Int64
        assert out["DFCI_MRN"].to_list() == [1001, 1002, 1003]

    def test_duplicate_mrns_raise(self, tmp_path):
        rows = {k: v * 2 for k, v in _label_rows().items()}
        rows["DFCI_MRN"] = ["1001", "1001.0"]
        with pytest.raises(ValueError, match="duplicated MRNs"):
            load_subtype_labels(_write_labels(tmp_path, rows))

    def test_missing_required_column_raises(self, tmp_path):
        rows = _label_rows()
        del rows["is_avpc"]
        with pytest.raises(ValueError, match="missing expected columns"):
            load_subtype_labels(_write_labels(tmp_path, rows))

    def test_dated_nepc_positive(self, tmp_path):
        path = _write_labels(
            tmp_path,
            _label_rows(
                primary_label=["nepc"],
                nepc_status=["nepc"],
                nepc_date=["2021-06-15"],
                nepc_date_source=["stated"],
                nepc_date_precision=["day"],
                has_stated_nepc_dx=[True],
                stated_nepc_dx_date=["2021-07-01"],
            ),
        )
        row = load_subtype_labels(path).row(0, named=True)
        assert row["NEPC"] == 1
        assert row["NEPC_DATE"] == dt.datetime(2021, 6, 15)
        assert row["NEPC_DATE_SOURCE"] == "stated"
        assert row["NEPC_LABEL_SOURCE"] == "ok"
        assert row["NEPC_STATED_DX"] == 1
        assert row["NEPC_STATED_DX_DATE"] == dt.datetime(2021, 7, 1)
        assert row["PRIMARY_LABEL"] == "nepc"

    def test_failed_review_is_censored_with_status_kept(self, tmp_path):
        """A null label means the notes were never read: censored as 0, with
        the review status carried so a sensitivity analysis can drop them."""
        path = _write_labels(
            tmp_path,
            _label_rows(
                primary_label=[None],
                nepc_status=[None],
                nepc_review_status=["failed"],
                is_avpc=[None],
                avpc_review_status=["not_run"],
            ),
        )
        row = load_subtype_labels(path).row(0, named=True)
        assert row["NEPC"] == 0
        assert row["NEPC_LABEL_SOURCE"] == "failed"
        assert row["AVPC"] == 0
        assert row["AVPC_LABEL_SOURCE"] == "not_run"
        assert row["PRIMARY_LABEL"] is None

    def test_avpc_taken_from_runner_without_criteria_threshold(self, tmp_path):
        """is_avpc is the diagnosis; a single criterion is not re-gated here."""
        path = _write_labels(
            tmp_path,
            _label_rows(
                primary_label=["avpc"],
                is_avpc=[True],
                avpc_date=["2020-04-10"],
                n_avpc_criteria=[1],
                avpc_criteria=[["C2"]],
            ),
        )
        row = load_subtype_labels(path).row(0, named=True)
        assert row["AVPC"] == 1
        assert row["AVPC_DATE"] == dt.datetime(2020, 4, 10)
        assert row["AVPC_N_CRITERIA"] == 1

    def test_avpc_criteria_list_is_joined(self, tmp_path):
        path = _write_labels(
            tmp_path,
            _label_rows(n_avpc_criteria=[3], avpc_criteria=[["C1", "C3", "C5"]]),
        )
        row = load_subtype_labels(path).row(0, named=True)
        assert row["AVPC_CRITERIA"] == "C1,C3,C5"

    def test_undated_positives_are_demoted_to_censored(self, tmp_path):
        """A positive with no usable date has no event time, so it becomes a
        censored observation rather than a silently-NaT event."""
        rows = {k: v * 2 for k, v in _label_rows().items()}
        rows.update(
            DFCI_MRN=["1001", "1002"],
            nepc_status=["nepc", "nepc"],
            nepc_date=["2021-06-15", None],
            is_avpc=[True, True],
            avpc_date=[None, "2020-01-01"],
        )
        out = load_subtype_labels(_write_labels(tmp_path, rows)).sort("DFCI_MRN")
        assert out["NEPC"].to_list() == [1, 0]
        assert out["AVPC"].to_list() == [0, 1]

    def test_optional_columns_may_be_absent(self, tmp_path):
        path = tmp_path / "minimal.parquet"
        pl.DataFrame(
            {
                "DFCI_MRN": ["1001"],
                "nepc_status": ["nepc"],
                "nepc_date": ["2021-06-15"],
                "is_avpc": [False],
                "avpc_date": [None],
            },
            schema_overrides={"avpc_date": pl.Utf8},
        ).write_parquet(path)
        out = load_subtype_labels(path)
        assert out.columns == EXPECTED_COLUMNS
        assert out["NEPC"].to_list() == [1]
        assert out["NEPC_STATED_DX"].to_list() == [None]


class TestBuildSurvivalCohortLabels:
    """The Stage-1 join/censoring/TT_* logic in build_survival_cohort."""

    def _base_cohort_frames(self, mrns):
        status_df = pl.DataFrame(
            {
                "DFCI_MRN": mrns,
                "BIRTH_DATE": [dt.date(1950, 1, 1)] * len(mrns),
                "GENDER": ["M"] * len(mrns),
                "DEATH_DATE": pl.Series([None] * len(mrns), dtype=pl.Date),
                "LAST_CONTACT_DATE": [dt.date(2022, 1, 1)] * len(mrns),
            }
        )
        anchor_df = pl.DataFrame(
            {
                "DFCI_MRN": mrns,
                "TREATMENT_ANCHOR_DATE": [dt.date(2020, 1, 1)] * len(mrns),
            }
        )
        platinum_df = pl.DataFrame(
            {"DFCI_MRN": [], "PLATINUM_MED": [], "PLATINUM_DATE": []},
            schema={"DFCI_MRN": pl.Int64, "PLATINUM_MED": pl.Utf8, "PLATINUM_DATE": pl.Date},
        )
        return status_df, anchor_df, platinum_df

    def _cohort(self, tmp_path, mrns, rows):
        status_df, anchor_df, platinum_df = self._base_cohort_frames(mrns)
        subtype_df = load_subtype_labels(_write_labels(tmp_path, rows))
        cohort = build_survival_cohort(mrns, anchor_df, platinum_df, status_df, subtype_df)
        return {row["DFCI_MRN"]: row for row in cohort.iter_rows(named=True)}

    def test_join_miss_is_censored_with_missing_label_source(self, tmp_path):
        by_mrn = self._cohort(tmp_path, [1001, 1002], _label_rows(DFCI_MRN=["1001"]))
        for col in ("NEPC", "AVPC", "AVPC_NEPC"):
            assert by_mrn[1002][col] == 0
        assert by_mrn[1002]["NEPC_LABEL_SOURCE"] == "missing_from_labels"
        assert by_mrn[1002]["AVPC_LABEL_SOURCE"] == "missing_from_labels"
        assert by_mrn[1002]["AVPC_NEPC_LABEL_SOURCE"] == "negative"
        # Censored: follow-up end (2022-01-01) minus anchor (2020-01-01) = 731 days.
        assert by_mrn[1002]["TT_NEPC"] == 731
        assert by_mrn[1002]["TT_AVPC"] == 731
        assert by_mrn[1001]["NEPC_LABEL_SOURCE"] == "ok"

    def test_event_times_for_nepc_and_avpc(self, tmp_path):
        rows = {k: v * 2 for k, v in _label_rows().items()}
        rows.update(
            DFCI_MRN=["2001", "2002"],
            nepc_status=["nepc", "non_nepc"],
            nepc_date=["2020-04-10", None],
            is_avpc=[False, True],
            avpc_date=[None, "2020-07-19"],
        )
        by_mrn = self._cohort(tmp_path, [2001, 2002], rows)
        # Anchor 2020-01-01 -> 2020-04-10 is 100 days, -> 2020-07-19 is 200.
        assert by_mrn[2001]["NEPC"] == 1 and by_mrn[2001]["TT_NEPC"] == 100
        assert by_mrn[2001]["AVPC"] == 0 and by_mrn[2001]["TT_AVPC"] == 731
        assert by_mrn[2002]["AVPC"] == 1 and by_mrn[2002]["TT_AVPC"] == 200
        assert by_mrn[2002]["NEPC"] == 0 and by_mrn[2002]["TT_NEPC"] == 731
        assert by_mrn[2001]["AVPC_NEPC_LABEL_SOURCE"] == "nepc"
        assert by_mrn[2002]["AVPC_NEPC_LABEL_SOURCE"] == "avpc"
        assert by_mrn[2002]["TT_AVPC_NEPC"] == 200

    def test_nepc_takes_union_timing_precedence_over_earlier_avpc(self, tmp_path):
        """When both arms fire, the union is dated at NEPC_DATE even though
        AVPC was met earlier; AVPC stays independent for its own endpoint."""
        by_mrn = self._cohort(
            tmp_path,
            [2201],
            _label_rows(
                DFCI_MRN=["2201"],
                nepc_status=["nepc"],
                nepc_date=["2020-07-19"],  # 200 days, later
                is_avpc=[True],
                avpc_date=["2020-04-10"],  # 100 days, earlier
            ),
        )
        row = by_mrn[2201]
        assert row["AVPC_NEPC"] == 1
        assert row["TT_AVPC_NEPC"] == 200
        assert row["AVPC_NEPC_LABEL_SOURCE"] == "nepc"
        assert row["AVPC"] == 1
        assert row["TT_AVPC"] == 100

    def test_cohort_builds_with_no_labels_mounted(self):
        """subtype_df=None (the default) must not break the platinum-only path."""
        mrns = [3001]
        status_df, anchor_df, platinum_df = self._base_cohort_frames(mrns)
        cohort = build_survival_cohort(mrns, anchor_df, platinum_df, status_df)
        row = cohort.row(0, named=True)
        assert row["NEPC"] == 0
        assert row["AVPC"] == 0
        assert row["AVPC_NEPC"] == 0
        assert row["NEPC_LABEL_SOURCE"] == "missing_from_labels"

    def test_output_schema_carries_label_columns(self, tmp_path):
        by_mrn = self._cohort(tmp_path, [4001], _label_rows(DFCI_MRN=["4001"]))
        expected = {
            "NEPC_DATE", "TT_NEPC", "NEPC", "NEPC_DATE_SOURCE",
            "NEPC_DATE_PRECISION", "NEPC_LABEL_SOURCE", "NEPC_STATED_DX",
            "NEPC_STATED_DX_DATE", "AVPC_NEPC_DATE", "TT_AVPC_NEPC", "AVPC_NEPC",
            "AVPC_NEPC_DATE_SOURCE", "AVPC_NEPC_DATE_PRECISION",
            "AVPC_NEPC_LABEL_SOURCE", "AVPC_DATE", "TT_AVPC", "AVPC",
            "AVPC_LABEL_SOURCE", "AVPC_N_CRITERIA", "AVPC_CRITERIA",
            "PRIMARY_LABEL",
        }
        missing = expected - set(by_mrn[4001])
        assert not missing, f"build_survival_cohort dropped label columns: {sorted(missing)}"
        assert not any("TIMELINE" in c for c in by_mrn[4001])
