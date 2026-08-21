"""Tests for export_adt_intent_outputs.py -- the CSV/figure export wrapper.

Synthetic fixtures throughout: every value is planted so the expected output
is known by construction.
"""

from __future__ import annotations

import datetime as dt
import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2].parent))

from COMPASS.data_preprocessing.export_adt_intent_outputs import (  # noqa: E402
    OUTPUT_SUBDIR,
    build_labels,
    resolve_out_dir,
    run,
)


def _meds(n: int = 12) -> pl.DataFrame:
    """ADT medication rows; long courses so patients stay METASTATIC."""
    rows = []
    for mrn in range(1, n + 1):
        for i in range(12):
            rows.append(
                (mrn, "LEUPROLIDE ACETATE",
                 (dt.datetime(2015, 1, 1) + dt.timedelta(days=90 * i)).strftime("%Y-%m-%d"))
            )
    return pl.DataFrame(
        {
            "DFCI_MRN": [r[0] for r in rows],
            "NCI_PREFERRED_MED_NM": [r[1] for r in rows],
            "MED_START_DT": [r[2] for r in rows],
        }
    )


def _follow_up(n: int = 12) -> pl.DataFrame:
    """Patient status carrying both FOLLOW_UP_END_DATE and DEATH."""
    return pl.DataFrame(
        {
            "DFCI_MRN": list(range(1, n + 1)),
            "FOLLOW_UP_END_DATE": [dt.datetime(2019, 1, 1)] * n,
            # Every third patient died, so pct_died is known by construction.
            "DEATH": [1 if m % 3 == 0 else 0 for m in range(1, n + 1)],
        }
    )


def _stage_notes(d: str) -> str:
    path = str(Path(d) / "stage_notes.parquet")
    rows = []
    for mrn in range(1, 13):
        rows.append((mrn, dt.datetime(2014, 10, 1), 4 if mrn % 2 else 2))
        if mrn % 3 == 0:
            rows.append((mrn, dt.datetime(2015, 9, 1), 4))
    pl.DataFrame(
        {
            "DFCI_MRN": [r[0] for r in rows],
            "EVENT_DATE": [r[1] for r in rows],
            "DERIVED_STAGE_MERGED": [r[2] for r in rows],
        }
    ).write_parquet(path)
    return path


class TestResolveOutDir(unittest.TestCase):
    def test_output_lands_in_the_named_subdirectory(self):
        """Every artefact for this analysis shares one directory, so a caller
        pointing at FIG_ROOT gets .../ADT_METASTATIC_FILTERING."""
        with tempfile.TemporaryDirectory() as d:
            out = resolve_out_dir(d)
            self.assertEqual(out.name, OUTPUT_SUBDIR)
            self.assertTrue(out.is_dir())


class TestBuildLabels(unittest.TestCase):
    def test_cross_references_are_independently_optional(self):
        """A missing stage file must cost only the stage columns: a partial
        run still has to produce a usable label."""
        labelled = build_labels(_meds(), icds=None, stage_note_level_path=None)
        self.assertIn("ADT_INTENT", labelled.columns)
        self.assertNotIn("MAX_STAGE_BEFORE", labelled.columns)
        self.assertNotIn("N_MET_SITES", labelled.columns)

    def test_stage_columns_appear_when_the_file_exists(self):
        with tempfile.TemporaryDirectory() as d:
            labelled = build_labels(_meds(), stage_note_level_path=_stage_notes(d))
        for col in ("CANCER_STAGE", "MAX_STAGE_BEFORE", "MAX_STAGE_AFTER"):
            self.assertIn(col, labelled.columns)

    def test_a_nonexistent_stage_path_is_skipped_not_raised(self):
        """The path is a cluster default; running locally must not crash."""
        labelled = build_labels(_meds(), stage_note_level_path="/nope/missing.parquet")
        self.assertIn("ADT_INTENT", labelled.columns)
        self.assertNotIn("MAX_STAGE_BEFORE", labelled.columns)


class TestSurvivalPassthrough(unittest.TestCase):
    """DEATH must survive the classify step.

    classify_adt_intent takes follow_up for FOLLOW_UP_END_DATE and does not
    carry DEATH through, so build_labels has to join it on itself. When it
    didn't, report_survival returned an empty frame and the export silently
    skipped summary_survival.csv -- losing the primary go/no-go check while
    reporting success.
    """

    def test_death_column_survives_build_labels(self):
        labelled = build_labels(_meds(), follow_up=_follow_up())
        self.assertIn("DEATH", labelled.columns)
        self.assertEqual(int(labelled["DEATH"].sum()), 4)   # 3, 6, 9, 12

    def test_survival_table_is_written_when_follow_up_has_death(self):
        with tempfile.TemporaryDirectory() as d:
            _, out = run(_meds(), follow_up=_follow_up(), fig_root=d, verbose=False)
            path = out / "summary_survival.csv"
            self.assertTrue(path.exists(), "summary_survival.csv was not written")
            surv = pl.read_csv(path)
        self.assertEqual(int(surv["n_patients"].sum()), 12)
        # 4 of 12 died.
        self.assertAlmostEqual(float(surv["pct_died"][0]), 33.3, places=1)

    def test_survival_still_skipped_when_follow_up_lacks_death(self):
        """The skip message is correct when DEATH genuinely isn't available."""
        with tempfile.TemporaryDirectory() as d:
            _, out = run(_meds(), follow_up=None, fig_root=d, verbose=False)
            self.assertFalse((out / "summary_survival.csv").exists())


class TestRun(unittest.TestCase):
    def test_writes_labels_and_both_figures(self):
        with tempfile.TemporaryDirectory() as d:
            labelled, out = run(
                _meds(), stage_note_level_path=_stage_notes(d),
                fig_root=d, verbose=False,
            )
            self.assertTrue((out / "adt_intent_labels.csv").exists())
            self.assertTrue((out / "stage_metburden.png").exists())
            self.assertTrue((out / "max_stage.png").exists())
            self.assertTrue((out / "summary_stage_max.csv").exists())

    def test_label_csv_round_trips_with_the_max_stage_columns(self):
        with tempfile.TemporaryDirectory() as d:
            labelled, out = run(
                _meds(), stage_note_level_path=_stage_notes(d),
                fig_root=d, verbose=False,
            )
            back = pl.read_csv(out / "adt_intent_labels.csv")
        self.assertEqual(back.height, labelled.height)
        for col in ("MAX_STAGE_BEFORE", "MAX_STAGE_AFTER", "STAGE_UPSTAGED_AFTER_ADT"):
            self.assertIn(col, back.columns)

    def test_absent_inputs_skip_tables_rather_than_writing_empty_ones(self):
        """A header-only CSV would be indistinguishable from "measured,
        found nothing"."""
        with tempfile.TemporaryDirectory() as d:
            _, out = run(_meds(), stage_note_level_path=None, fig_root=d, verbose=False)
            self.assertFalse((out / "summary_stage_max.csv").exists())
            self.assertFalse((out / "summary_met_burden.csv").exists())
            # ...but the label itself and the figures are still produced.
            self.assertTrue((out / "adt_intent_labels.csv").exists())
            self.assertTrue((out / "max_stage.png").exists())

    def test_rerun_overwrites_in_place(self):
        """Regenerating must not accumulate stale variants beside the new ones."""
        with tempfile.TemporaryDirectory() as d:
            _, out = run(_meds(), stage_note_level_path=_stage_notes(d),
                         fig_root=d, verbose=False)
            first = sorted(p.name for p in out.iterdir())
            _, out2 = run(_meds(), stage_note_level_path=_stage_notes(d),
                          fig_root=d, verbose=False)
            self.assertEqual(out, out2)
            self.assertEqual(first, sorted(p.name for p in out2.iterdir()))


if __name__ == "__main__":
    unittest.main()
