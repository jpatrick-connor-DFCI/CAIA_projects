"""Stage 1's reader for the broader AVPC/NEPC criteria-timeline labels, plus
the cohort-level join/demotion/TT_AVPC_NEPC derivation.

Companion to test_nepc_dx_labels.py. Covers the same load_*_labels contract
(MRN normalization, undated-positive demotion, non-fatal missing-file path)
for load_avpc_nepc_labels, plus what's specific to this audit label: the extra
audit/provenance component columns (AVPC, AVPC_DATE, NEPC_TIMELINE,
NEPC_TIMELINE_DATE, AVPC_N_CRITERIA), the join-miss -> auto_negative_no_evidence
label_source override that build_survival_cohort applies (this is NOT set by
load_avpc_nepc_labels itself -- see its docstring), and the TT_AVPC_NEPC
derivation that mirrors TT_NEPC/TT_PLATINUM.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from COMPASS.data_preprocessing.compile_COMPASS_cohort_data import (
    build_survival_cohort,
    load_avpc_nepc_labels,
    load_nepc_dx_labels,
)

EXPECTED_COLUMNS = [
    "DFCI_MRN",
    "AVPC_NEPC_DATE",
    "AVPC_NEPC",
    "AVPC_NEPC_DATE_SOURCE",
    "AVPC_NEPC_DATE_PRECISION",
    "AVPC_NEPC_LABEL_SOURCE",
    "AVPC",
    "AVPC_DATE",
    "NEPC_TIMELINE",
    "NEPC_TIMELINE_DATE",
    "AVPC_N_CRITERIA",
]


def _write_labels(tmp_path, rows: dict, name="avpc_nepc_labels.parquet"):
    path = tmp_path / name
    pl.DataFrame(rows).write_parquet(path)
    return path


def _strict_nepc_df(rows: dict | None = None) -> pl.DataFrame:
    """A load_nepc_dx_labels-shaped frame for the *strict* veto-gated NEPC
    label, which is what build_survival_cohort now unions with AVPC to form
    the modeled AVPC_NEPC endpoint."""
    if rows is None:
        return load_nepc_dx_labels(None)
    return pl.DataFrame(rows).with_columns(
        pl.col("DFCI_MRN").cast(pl.Int64),
        pl.col("NEPC_DATE").str.to_datetime("%Y-%m-%d", strict=False),
        pl.col("NEPC").cast(pl.Int64),
        pl.col("NEPC_DATE_SOURCE").cast(pl.Utf8),
        pl.col("NEPC_DATE_PRECISION").cast(pl.Utf8),
        pl.col("NEPC_LABEL_SOURCE").cast(pl.Utf8),
    )


def _full_label_row(**overrides) -> dict:
    row = {
        "DFCI_MRN": ["1001"],
        "has_avpc_nepc": [1],
        "avpc_nepc_date": ["2021-06-15"],
        "date_source": ["stated"],
        "date_precision": ["day"],
        "label_source": ["timeline_positive"],
        "has_avpc": [0],
        "avpc_date": [None],
        "has_nepc_timeline": [1],
        "nepc_timeline_date": ["2021-06-15"],
        "n_avpc_criteria": [2],
    }
    row.update(overrides)
    return row


class TestLoadAvpcNepcLabels:
    """Mirrors test_nepc_dx_labels.py's coverage of load_nepc_dx_labels."""

    def test_missing_path_is_non_fatal_and_typed(self, tmp_path):
        out = load_avpc_nepc_labels(tmp_path / "does_not_exist.parquet")
        assert out.height == 0
        assert out.columns == EXPECTED_COLUMNS

    def test_none_path_is_non_fatal(self, tmp_path):
        out = load_avpc_nepc_labels(None)
        assert out.height == 0
        assert out.columns == EXPECTED_COLUMNS

    def test_mrn_normalization_handles_strings_and_float_roundtrip(self, tmp_path):
        path = _write_labels(
            tmp_path,
            {
                # " 1001 " exercises the strip; "1002.0" the float round-trip.
                "DFCI_MRN": [" 1001 ", "1002.0", "1003"],
                "has_avpc_nepc": [1, 0, 1],
                "avpc_nepc_date": ["2021-06-15", None, "2020-03-01"],
                "date_source": ["stated", "unknown", "note_date"],
                "date_precision": ["day", "unknown", "month"],
                "label_source": ["timeline_positive", "timeline_negative", "timeline_positive"],
                "has_avpc": [1, 0, 1],
                "avpc_date": ["2021-06-15", None, "2020-03-01"],
                "has_nepc_timeline": [0, 0, 0],
                "nepc_timeline_date": [None, None, None],
                "n_avpc_criteria": [3, 1, 3],
            },
        )
        out = load_avpc_nepc_labels(path).sort("DFCI_MRN")
        assert out["DFCI_MRN"].dtype == pl.Int64
        assert out["DFCI_MRN"].to_list() == [1001, 1002, 1003]

    def test_positive_label_with_valid_date(self, tmp_path):
        """A dated positive carries its event flag, date, and components
        through untouched."""
        path = _write_labels(tmp_path, _full_label_row())
        out = load_avpc_nepc_labels(path)
        assert out["AVPC_NEPC"].to_list() == [1]
        assert out["AVPC_NEPC_DATE"].to_list()[0] is not None
        assert out["AVPC_NEPC_LABEL_SOURCE"].to_list() == ["timeline_positive"]
        assert out["NEPC_TIMELINE"].to_list() == [1]
        assert out["AVPC_N_CRITERIA"].to_list() == [2]

    def test_undated_positive_is_demoted_to_censored(self, tmp_path):
        """A positive with no usable date has no event time, so it becomes a
        censored observation rather than a silently-NaT event -- mirrors the
        nepc endpoint's demotion exactly."""
        path = _write_labels(
            tmp_path,
            {
                "DFCI_MRN": ["2001", "2002"],
                "has_avpc_nepc": [1, 1],
                "avpc_nepc_date": ["2021-06-15", None],
                "date_source": ["stated", "unknown"],
                "date_precision": ["day", "unknown"],
                "label_source": ["timeline_positive", "timeline_positive"],
                "has_avpc": [0, 0],
                "avpc_date": [None, None],
                "has_nepc_timeline": [1, 1],
                "nepc_timeline_date": ["2021-06-15", None],
                "n_avpc_criteria": [1, 1],
            },
        )
        out = load_avpc_nepc_labels(path).sort("DFCI_MRN")
        by_mrn = dict(zip(out["DFCI_MRN"].to_list(), out["AVPC_NEPC"].to_list()))
        assert by_mrn[2001] == 1
        assert by_mrn[2002] == 0, "undated positive should be demoted to censored"

    def test_duplicate_mrns_raise(self, tmp_path):
        """The labels are one row per patient; a duplicate would fan out the join."""
        path = _write_labels(
            tmp_path,
            {
                "DFCI_MRN": ["3001", "3001"],
                "has_avpc_nepc": [1, 0],
                "avpc_nepc_date": ["2021-06-15", None],
                "date_source": ["stated", "unknown"],
                "date_precision": ["day", "unknown"],
                "label_source": ["timeline_positive", "timeline_negative"],
                "has_avpc": [1, 0],
                "avpc_date": ["2021-06-15", None],
                "has_nepc_timeline": [0, 0],
                "nepc_timeline_date": [None, None],
                "n_avpc_criteria": [3, 1],
            },
        )
        with pytest.raises(ValueError):
            load_avpc_nepc_labels(path)

    def test_missing_required_column_raises(self, tmp_path):
        path = _write_labels(
            tmp_path,
            {"DFCI_MRN": ["4001"], "has_avpc_nepc": [1]},  # no avpc_nepc_date
        )
        with pytest.raises(ValueError):
            load_avpc_nepc_labels(path)

    def test_optional_provenance_columns_are_backfilled(self, tmp_path):
        """An older labels file without the provenance/component columns
        still loads."""
        path = _write_labels(
            tmp_path,
            {
                "DFCI_MRN": ["5001"],
                "has_avpc_nepc": [1],
                "avpc_nepc_date": ["2021-06-15"],
            },
        )
        out = load_avpc_nepc_labels(path)
        assert out.columns == EXPECTED_COLUMNS
        assert out.height == 1


class TestBuildSurvivalCohortAvpcNepc:
    """The Stage-1 join/demotion/TT_AVPC_NEPC logic in build_survival_cohort."""

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

    def test_join_miss_gets_auto_negative_no_evidence_label_source(self, tmp_path):
        """A patient absent from the timeline entirely (join miss) must be
        censored (AVPC_NEPC=0) with label_source overridden to
        'auto_negative_no_evidence' -- distinct from the timeline builder's
        own 'timeline_negative'/'conventional' outputs for patients it did
        see."""
        mrns = [1001, 1002]
        status_df, anchor_df, platinum_df = self._base_cohort_frames(mrns)
        # Labels file only covers 1001; 1002 is a join miss.
        avpc_nepc_path = _write_labels(tmp_path, _full_label_row(DFCI_MRN=["1001"]))
        avpc_nepc_df = load_avpc_nepc_labels(avpc_nepc_path)
        nepc_df = load_nepc_dx_labels(None)

        cohort = build_survival_cohort(
            mrns, anchor_df, platinum_df, status_df,
            nepc_df=nepc_df, avpc_nepc_df=avpc_nepc_df,
        )
        by_mrn = {row["DFCI_MRN"]: row for row in cohort.iter_rows(named=True)}
        assert by_mrn[1002]["AVPC_NEPC"] == 0
        assert by_mrn[1002]["AVPC_NEPC_LABEL_SOURCE"] == "auto_negative_no_evidence"
        # 1001 is a timeline positive via has_nepc_timeline only, with no
        # strict NEPC label and no AVPC -- under the modeled definition
        # (AVPC or strict NEPC) it is negative, and its label_source reflects
        # the recomputed union, not the timeline builder's own verdict. The
        # timeline's verdict is preserved in the audit column.
        assert by_mrn[1001]["AVPC_NEPC"] == 0
        assert by_mrn[1001]["AVPC_NEPC_LABEL_SOURCE"] == "auto_negative_no_evidence"
        assert by_mrn[1001]["AVPC_NEPC_TIMELINE"] == 1
        assert by_mrn[1001]["AVPC_NEPC_TIMELINE_LABEL_SOURCE"] == "timeline_positive"

    def test_tt_avpc_nepc_derivation_for_event_and_censored(self, tmp_path):
        """TT_AVPC_NEPC = event date minus anchor when the event occurred,
        else follow-up-end minus anchor -- mirrors TT_NEPC/TT_PLATINUM."""
        mrns = [2001, 2002]
        status_df, anchor_df, platinum_df = self._base_cohort_frames(mrns)
        avpc_nepc_path = _write_labels(
            tmp_path,
            {
                "DFCI_MRN": ["2001", "2002"],
                "has_avpc_nepc": [1, 0],
                "avpc_nepc_date": ["2020-04-10", None],
                "date_source": ["stated", "unknown"],
                "date_precision": ["day", "unknown"],
                "label_source": ["timeline_positive", "timeline_negative"],
                "has_avpc": [0, 0],
                "avpc_date": [None, None],
                "has_nepc_timeline": [1, 0],
                "nepc_timeline_date": ["2020-04-10", None],
                "n_avpc_criteria": [1, 0],
            },
        )
        avpc_nepc_df = load_avpc_nepc_labels(avpc_nepc_path)
        # 2001 carries the strict NEPC label so the modeled union fires on it.
        nepc_df = _strict_nepc_df(
            {
                "DFCI_MRN": [2001],
                "NEPC_DATE": ["2020-04-10"],
                "NEPC": [1],
                "NEPC_DATE_SOURCE": ["stated"],
                "NEPC_DATE_PRECISION": ["day"],
                "NEPC_LABEL_SOURCE": ["nepc_dx_positive"],
            }
        )

        cohort = build_survival_cohort(
            mrns, anchor_df, platinum_df, status_df,
            nepc_df=nepc_df, avpc_nepc_df=avpc_nepc_df,
        )
        by_mrn = {row["DFCI_MRN"]: row for row in cohort.iter_rows(named=True)}
        # Anchor 2020-01-01 -> event 2020-04-10 is 100 days.
        assert by_mrn[2001]["AVPC_NEPC"] == 1
        assert by_mrn[2001]["TT_AVPC_NEPC"] == 100
        assert by_mrn[2001]["AVPC_NEPC_LABEL_SOURCE"] == "nepc_strict"
        # Censored: follow-up end (2022-01-01) minus anchor (2020-01-01) = 731 days.
        assert by_mrn[2002]["AVPC_NEPC"] == 0
        assert by_mrn[2002]["TT_AVPC_NEPC"] == 731

    def test_avpc_arm_fires_union_with_avpc_timing(self, tmp_path):
        """AVPC alone (no strict NEPC) makes the union positive, dated at
        AVPC_DATE."""
        mrns = [2101]
        status_df, anchor_df, platinum_df = self._base_cohort_frames(mrns)
        avpc_nepc_path = _write_labels(
            tmp_path,
            _full_label_row(
                DFCI_MRN=["2101"],
                has_avpc=[1],
                avpc_date=["2020-04-10"],
                has_nepc_timeline=[0],
                nepc_timeline_date=[None],
                n_avpc_criteria=[4],
            ),
        )
        cohort = build_survival_cohort(
            mrns, anchor_df, platinum_df, status_df,
            nepc_df=_strict_nepc_df(),
            avpc_nepc_df=load_avpc_nepc_labels(avpc_nepc_path),
        )
        row = cohort.row(0, named=True)
        assert row["AVPC"] == 1
        assert row["AVPC_NEPC"] == 1
        assert row["TT_AVPC_NEPC"] == 100
        assert row["AVPC_NEPC_LABEL_SOURCE"] == "avpc_criteria"

    def test_strict_nepc_takes_timing_precedence_over_earlier_avpc(self, tmp_path):
        """When both arms fire, NEPC precedence dates the event at NEPC_DATE
        even though AVPC crossed threshold earlier."""
        mrns = [2201]
        status_df, anchor_df, platinum_df = self._base_cohort_frames(mrns)
        avpc_nepc_path = _write_labels(
            tmp_path,
            _full_label_row(
                DFCI_MRN=["2201"],
                has_avpc=[1],
                avpc_date=["2020-04-10"],  # 100 days, earlier
                n_avpc_criteria=[4],
            ),
        )
        nepc_df = _strict_nepc_df(
            {
                "DFCI_MRN": [2201],
                "NEPC_DATE": ["2020-07-19"],  # 200 days, later
                "NEPC": [1],
                "NEPC_DATE_SOURCE": ["stated"],
                "NEPC_DATE_PRECISION": ["day"],
                "NEPC_LABEL_SOURCE": ["nepc_dx_positive"],
            }
        )
        cohort = build_survival_cohort(
            mrns, anchor_df, platinum_df, status_df,
            nepc_df=nepc_df,
            avpc_nepc_df=load_avpc_nepc_labels(avpc_nepc_path),
        )
        row = cohort.row(0, named=True)
        assert row["AVPC_NEPC"] == 1
        assert row["TT_AVPC_NEPC"] == 200
        assert row["AVPC_NEPC_LABEL_SOURCE"] == "nepc_strict"

    def test_undated_avpc_positive_is_censored_out_of_union(self, tmp_path):
        """An AVPC positive with no parseable avpc_date carries no event time;
        it must be demoted rather than entering the union undated."""
        mrns = [2301]
        status_df, anchor_df, platinum_df = self._base_cohort_frames(mrns)
        avpc_nepc_path = _write_labels(
            tmp_path,
            _full_label_row(
                DFCI_MRN=["2301"],
                has_avpc=[1],
                avpc_date=[None],
                has_nepc_timeline=[0],
                nepc_timeline_date=[None],
                has_avpc_nepc=[0],
                avpc_nepc_date=[None],
                n_avpc_criteria=[4],
            ),
        )
        cohort = build_survival_cohort(
            mrns, anchor_df, platinum_df, status_df,
            nepc_df=_strict_nepc_df(),
            avpc_nepc_df=load_avpc_nepc_labels(avpc_nepc_path),
        )
        row = cohort.row(0, named=True)
        assert row["AVPC"] == 0
        assert row["AVPC_NEPC"] == 0
        # Censored at follow-up end, not left with a null event time.
        assert row["TT_AVPC_NEPC"] == 731

    def test_cohort_builds_with_no_avpc_nepc_labels_mounted(self):
        """Passing avpc_nepc_df=None (the default) must not break the
        platinum/nepc-only pipeline -- the non-fatal missing-file contract."""
        mrns = [3001]
        status_df, anchor_df, platinum_df = self._base_cohort_frames(mrns)
        cohort = build_survival_cohort(mrns, anchor_df, platinum_df, status_df)
        assert "AVPC_NEPC" in cohort.columns
        assert "TT_AVPC_NEPC" in cohort.columns
        assert cohort["AVPC_NEPC"].to_list() == [0]
        assert cohort["AVPC_NEPC_LABEL_SOURCE"].to_list() == ["auto_negative_no_evidence"]

    def test_output_schema_carries_all_avpc_nepc_columns(self, tmp_path):
        """build_survival_cohort's select list must carry every AVPC_NEPC
        column the survival pipeline downstream expects."""
        mrns = [4001]
        status_df, anchor_df, platinum_df = self._base_cohort_frames(mrns)
        avpc_nepc_path = _write_labels(tmp_path, _full_label_row(DFCI_MRN=["4001"]))
        avpc_nepc_df = load_avpc_nepc_labels(avpc_nepc_path)
        nepc_df = load_nepc_dx_labels(None)

        cohort = build_survival_cohort(
            mrns, anchor_df, platinum_df, status_df,
            nepc_df=nepc_df, avpc_nepc_df=avpc_nepc_df,
        )
        expected = {
            "AVPC_NEPC_DATE",
            "TT_AVPC_NEPC",
            "AVPC_NEPC",
            "AVPC_NEPC_DATE_SOURCE",
            "AVPC_NEPC_DATE_PRECISION",
            "AVPC_NEPC_LABEL_SOURCE",
            "AVPC",
            "AVPC_DATE",
            "NEPC_TIMELINE",
            "NEPC_TIMELINE_DATE",
            "AVPC_N_CRITERIA",
        }
        missing = expected - set(cohort.columns)
        assert not missing, f"build_survival_cohort dropped AVPC_NEPC columns: {sorted(missing)}"
