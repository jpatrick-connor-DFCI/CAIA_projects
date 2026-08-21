"""Tests for export_adt_intent_outputs.py -- the figure export wrapper.

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
    write_lab_trajectories,
)
from COMPASS.data_preprocessing.adt_intent_trajectories import (  # noqa: E402
    PSA_LAB_NAME,
    TESTOSTERONE_LAB_NAME,
    build_death_km_input,
    build_km_input,
    load_longitudinal,
    summarize_km,
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


def _longitudinal(d: str, n: int = 12) -> pl.DataFrame:
    """Stage 2 frame: labs plus the three cohort-only endpoint pairs.

    LAB_NAME must match the module constants exactly -- a case mismatch
    silently yields an empty trajectory panel rather than an error.
    """
    path = str(Path(d) / "longitudinal.csv")
    rows = []
    for mrn in range(1, n + 1):
        plat = 1 if mrn % 3 == 0 else 0
        for t in (-90.0, 0.0, 90.0, 365.0):
            for lab, val in ((TESTOSTERONE_LAB_NAME, 400.0 if t < 0 else 15.0),
                             (PSA_LAB_NAME, 25.0 if t < 0 else 0.4)):
                rows.append((mrn, t, lab, val, "2015-01-01",
                             900.0, plat, 800.0, 0, 850.0, 0))
    cols = ["DFCI_MRN", "t_lab", "LAB_NAME", "LAB_VALUE", "TREATMENT_ANCHOR_DATE",
            "t_platinum", "PLATINUM", "t_nepc", "NEPC", "t_avpc", "AVPC"]
    pl.DataFrame({c: [r[i] for r in rows] for i, c in enumerate(cols)}).write_csv(path)
    return load_longitudinal(path)


class TestTimeToEvent(unittest.TestCase):
    """KM figures and their summary tables."""

    def test_death_km_needs_only_follow_up(self):
        """Death is computable for every ADT-exposed patient, so it must not
        depend on the Stage 2 longitudinal file."""
        with tempfile.TemporaryDirectory() as d:
            _, out = run(_meds(), follow_up=_follow_up(), longitudinal=None,
                         fig_root=d, verbose=False)
            self.assertTrue((out / "km_death.png").exists())
        # 4 of 12 died (every third patient: 3, 6, 9, 12). No summary table is
        # written any more, so the count is checked at its source.
        km = build_death_km_input(build_labels(_meds(), follow_up=_follow_up()),
                                  _follow_up())
        self.assertEqual(int(summarize_km(km)["n_events"].sum()), 4)

    def test_cohort_only_endpoints_are_skipped_without_longitudinal(self):
        with tempfile.TemporaryDirectory() as d:
            _, out = run(_meds(), follow_up=_follow_up(), longitudinal=None,
                         fig_root=d, verbose=False)
            for endpoint in ("platinum", "nepc", "avpc"):
                self.assertFalse((out / f"km_{endpoint}.png").exists())

    def test_all_four_kms_render_with_longitudinal(self):
        with tempfile.TemporaryDirectory() as d:
            _, out = run(_meds(), follow_up=_follow_up(),
                         longitudinal=_longitudinal(d), fig_root=d, verbose=False)
            for endpoint in ("death", "platinum", "nepc", "avpc"):
                self.assertTrue((out / f"km_{endpoint}.png").exists(),
                                f"km_{endpoint}.png missing")

    def test_km_event_counts_match_the_planted_rate(self):
        """The counts now ride on the figure rather than a CSV, so they are
        checked against the builder the annotation is drawn from."""
        with tempfile.TemporaryDirectory() as d:
            labelled, out = run(_meds(), follow_up=_follow_up(),
                                longitudinal=_longitudinal(d), fig_root=d,
                                verbose=False)
            self.assertTrue((out / "km_platinum.png").exists())
            plat = summarize_km(
                build_km_input(_longitudinal(d), labelled, "platinum")
            )
        # Every third patient of 12 has PLATINUM=1.
        self.assertEqual(int(plat["n_events"].sum()), 4)

    def test_lab_trajectories_render_and_report_coverage(self):
        """Coverage moved from a CSV to the run log, but it must still be
        reported: an empty panel from a lab-name mismatch draws nothing and
        raises nothing, so the log line is the only warning."""
        with tempfile.TemporaryDirectory() as d:
            labelled, out = run(_meds(), follow_up=_follow_up(),
                                longitudinal=_longitudinal(d), fig_root=d,
                                verbose=False)
            self.assertTrue((out / "lab_trajectories.png").exists())
            log = write_lab_trajectories(
                labelled, out, longitudinal=_longitudinal(d)
            )
        joined = "\n".join(log)
        for lab in (TESTOSTERONE_LAB_NAME, PSA_LAB_NAME):
            self.assertIn(f"{lab} coverage:", joined)
        self.assertNotIn("[warn]", joined)

    def test_trajectories_skipped_without_longitudinal(self):
        with tempfile.TemporaryDirectory() as d:
            _, out = run(_meds(), follow_up=_follow_up(), longitudinal=None,
                         fig_root=d, verbose=False)
            self.assertFalse((out / "lab_trajectories.png").exists())


class TestSurvivalPassthrough(unittest.TestCase):
    """DEATH must survive the classify step.

    classify_adt_intent takes follow_up for FOLLOW_UP_END_DATE and does not
    carry DEATH through, so build_labels has to join it on itself. When it
    didn't, every death-based output came back empty while the run still
    reported success. No survival CSV is written now, so the column itself and
    the death KM are what guard the join.
    """

    def test_death_column_survives_build_labels(self):
        labelled = build_labels(_meds(), follow_up=_follow_up())
        self.assertIn("DEATH", labelled.columns)
        self.assertEqual(int(labelled["DEATH"].sum()), 4)   # 3, 6, 9, 12

    def test_death_km_renders_when_follow_up_has_death(self):
        """The death curve is what the DEATH join now feeds; if the join broke
        the KM input would be empty and no figure would be written."""
        with tempfile.TemporaryDirectory() as d:
            labelled, out = run(_meds(), follow_up=_follow_up(), fig_root=d,
                                verbose=False)
            self.assertTrue((out / "km_death.png").exists())
        km = build_death_km_input(labelled, _follow_up())
        self.assertEqual(km.height, 12)
        self.assertEqual(int(km["event"].sum()), 4)   # 4 of 12 died

    def test_death_km_skipped_when_follow_up_is_absent(self):
        """No follow-up frame means no death curve, and the run says so."""
        with tempfile.TemporaryDirectory() as d:
            _, out = run(_meds(), follow_up=None, fig_root=d, verbose=False)
            self.assertFalse((out / "km_death.png").exists())


class TestRun(unittest.TestCase):
    def test_writes_both_figures_and_returns_the_labels(self):
        with tempfile.TemporaryDirectory() as d:
            labelled, out = run(
                _meds(), stage_note_level_path=_stage_notes(d),
                fig_root=d, verbose=False,
            )
            self.assertTrue((out / "stage_metburden.png").exists())
            self.assertTrue((out / "max_stage.png").exists())
        # The label is returned, never written.
        self.assertEqual(labelled.height, 12)

    def test_returned_labels_carry_the_max_stage_columns(self):
        """The returned frame is the only way to reach the label now, so the
        cross-reference joins have to be present on it."""
        with tempfile.TemporaryDirectory() as d:
            labelled, _ = run(
                _meds(), stage_note_level_path=_stage_notes(d),
                fig_root=d, verbose=False,
            )
        for col in ("MAX_STAGE_BEFORE", "MAX_STAGE_AFTER", "STAGE_UPSTAGED_AFTER_ADT"):
            self.assertIn(col, labelled.columns)

    def test_nothing_is_written_as_csv(self):
        """The whole point of the export is PNGs; a stray CSV would mean a
        writer was reintroduced."""
        with tempfile.TemporaryDirectory() as d:
            _, out = run(_meds(), follow_up=_follow_up(),
                         stage_note_level_path=_stage_notes(d),
                         longitudinal=_longitudinal(d), fig_root=d, verbose=False)
            written = sorted(p.name for p in out.iterdir())
        self.assertEqual([p for p in written if not p.endswith(".png")], [])
        self.assertGreater(len(written), 0)

    def test_absent_inputs_skip_figures_rather_than_drawing_empty_ones(self):
        """A missing cross-reference costs its figure, not the run."""
        with tempfile.TemporaryDirectory() as d:
            labelled, out = run(_meds(), stage_note_level_path=None,
                                fig_root=d, verbose=False)
            # The stage joins are absent...
            self.assertNotIn("MAX_STAGE_BEFORE", labelled.columns)
            # ...but the panels that do not need them are still produced.
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
