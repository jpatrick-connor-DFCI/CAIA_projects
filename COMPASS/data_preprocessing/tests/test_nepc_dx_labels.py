"""Stage 1's reader for the strict LLM NEPC diagnosis labels.

Covers the three things that would silently corrupt the join: the MRN
normalization (the labels arrive as strings, sometimes with a trailing ".0"
from a float round-trip), the demotion of undated positives to censored, and
the non-fatal missing-file path that keeps the platinum pipeline working when
the LLM annotations are not mounted.
"""

from __future__ import annotations

import polars as pl
import pytest

from COMPASS.data_preprocessing.compile_COMPASS_cohort_data import load_nepc_dx_labels

EXPECTED_COLUMNS = [
    "DFCI_MRN",
    "NEPC_DATE",
    "NEPC",
    "NEPC_DATE_SOURCE",
    "NEPC_DATE_PRECISION",
    "NEPC_LABEL_SOURCE",
]


def _write_labels(tmp_path, rows: dict):
    path = tmp_path / "nepc_dx_labels.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


def test_missing_path_is_non_fatal_and_typed(tmp_path):
    """A cohort built without the annotations mounted must still build."""
    out = load_nepc_dx_labels(tmp_path / "does_not_exist.parquet")
    assert out.height == 0
    assert out.columns == EXPECTED_COLUMNS


def test_none_path_is_non_fatal(tmp_path):
    out = load_nepc_dx_labels(None)
    assert out.height == 0
    assert out.columns == EXPECTED_COLUMNS


def test_mrn_normalization_handles_strings_and_float_roundtrip(tmp_path):
    path = _write_labels(
        tmp_path,
        {
            # " 1001 " exercises the strip; "1002.0" the float round-trip.
            "DFCI_MRN": [" 1001 ", "1002.0", "1003"],
            "has_nepc_diagnosis": [1, 0, 1],
            "diagnosis_date": ["2021-06-15", None, "2020-03-01"],
            "date_source": ["stated", "unknown", "note_date"],
            "date_precision": ["day", "unknown", "month"],
            "label_source": ["adjudicated", "auto_negative_no_evidence", "adjudicated"],
        },
    )
    out = load_nepc_dx_labels(path).sort("DFCI_MRN")
    assert out["DFCI_MRN"].dtype == pl.Int64
    assert out["DFCI_MRN"].to_list() == [1001, 1002, 1003]


def test_undated_positive_is_demoted_to_censored(tmp_path):
    """A positive with no usable date has no event time, so it becomes a
    censored observation rather than a silently-NaT event."""
    path = _write_labels(
        tmp_path,
        {
            "DFCI_MRN": ["2001", "2002"],
            "has_nepc_diagnosis": [1, 1],
            "diagnosis_date": ["2021-06-15", None],
            "date_source": ["stated", "unknown"],
            "date_precision": ["day", "unknown"],
            "label_source": ["adjudicated", "adjudicated"],
        },
    )
    out = load_nepc_dx_labels(path).sort("DFCI_MRN")
    by_mrn = dict(zip(out["DFCI_MRN"].to_list(), out["NEPC"].to_list()))
    assert by_mrn[2001] == 1
    assert by_mrn[2002] == 0, "undated positive should be demoted to censored"


def test_duplicate_mrns_raise(tmp_path):
    """The labels are one row per patient; a duplicate would fan out the join."""
    path = _write_labels(
        tmp_path,
        {
            "DFCI_MRN": ["3001", "3001"],
            "has_nepc_diagnosis": [1, 0],
            "diagnosis_date": ["2021-06-15", None],
            "date_source": ["stated", "unknown"],
            "date_precision": ["day", "unknown"],
            "label_source": ["adjudicated", "adjudicated"],
        },
    )
    with pytest.raises(ValueError):
        load_nepc_dx_labels(path)


def test_missing_required_column_raises(tmp_path):
    path = _write_labels(
        tmp_path,
        {"DFCI_MRN": ["4001"], "has_nepc_diagnosis": [1]},  # no diagnosis_date
    )
    with pytest.raises(ValueError):
        load_nepc_dx_labels(path)


def test_optional_provenance_columns_are_backfilled(tmp_path):
    """An older labels file without the provenance columns still loads."""
    path = _write_labels(
        tmp_path,
        {
            "DFCI_MRN": ["5001"],
            "has_nepc_diagnosis": [1],
            "diagnosis_date": ["2021-06-15"],
        },
    )
    out = load_nepc_dx_labels(path)
    assert out.columns == EXPECTED_COLUMNS
    assert out.height == 1
