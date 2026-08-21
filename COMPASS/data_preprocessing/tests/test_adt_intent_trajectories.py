"""Tests for adt_intent_trajectories.py -- lab trajectories and KM inputs
stratified by ADT_INTENT.

Synthetic fixtures throughout, in the style of
test_compass_eligibility_alignment.py: every value is planted so the expected
output is known by construction rather than by running the code first.
"""

from __future__ import annotations

import datetime as dt
import sys
import unittest
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2].parent))

from COMPASS.data_preprocessing.adt_intent_trajectories import (  # noqa: E402
    ID_COL,
    INTENT_COL,
    PSA_LAB_NAME,
    TESTOSTERONE_LAB_NAME,
    build_death_km_input,
    build_km_input,
    build_lab_trajectory,
    km_series_by_intent,
    met_burden_distribution_by_intent,
    plot_stage_metburden_panel,
    stage_distribution_by_intent,
    load_longitudinal,
    logrank_by_intent,
    summarize_km,
    summarize_trajectory_coverage,
)


def _labels(rows) -> pl.DataFrame:
    """rows: (mrn, intent, eligible, adt_first_date)"""
    return pl.DataFrame(
        {
            ID_COL: [r[0] for r in rows],
            INTENT_COL: [r[1] for r in rows],
            "ELIGIBLE": [r[2] for r in rows],
            "ADT_FIRST_DATE": [r[3] for r in rows],
        },
        schema={ID_COL: pl.Int64, INTENT_COL: pl.Utf8,
                "ELIGIBLE": pl.Int8, "ADT_FIRST_DATE": pl.Datetime},
    )


def _long(rows) -> pl.DataFrame:
    """rows: (mrn, t_lab, lab_name, lab_value)"""
    return pl.DataFrame(
        {
            ID_COL: [r[0] for r in rows],
            "t_lab": [float(r[1]) for r in rows],
            "LAB_NAME": [r[2] for r in rows],
            "LAB_VALUE": [float(r[3]) for r in rows],
        },
        schema={ID_COL: pl.Int64, "t_lab": pl.Float64,
                "LAB_NAME": pl.Utf8, "LAB_VALUE": pl.Float64},
    )


class TestLabTrajectory(unittest.TestCase):
    def test_median_and_iqr_are_computed_per_intent_and_bin(self):
        # Five metastatic patients in one bin with values 1..5 -> median 3,
        # q1 2, q3 4.
        labels = _labels([(i, "METASTATIC", 1, dt.datetime(2018, 1, 1)) for i in range(1, 6)])
        rows = [(i, 10, PSA_LAB_NAME, float(i)) for i in range(1, 6)]
        traj = build_lab_trajectory(_long(rows), labels, PSA_LAB_NAME, min_patients_per_bin=5)

        self.assertEqual(traj.height, 1)
        self.assertEqual(traj["median"][0], 3.0)
        self.assertEqual(traj["q1"][0], 2.0)
        self.assertEqual(traj["q3"][0], 4.0)
        self.assertEqual(traj["n_patients"][0], 5)

    def test_repeat_draws_by_one_patient_do_not_outweigh_the_bin(self):
        # Heavily-monitored patients get drawn far more often. Patient 1
        # contributes 10 draws at 100; four others contribute one draw each at
        # 1. Averaging within (patient, bin) first gives per-patient values
        # [100, 1, 1, 1, 1] -> median 1. Pooling raw draws would give 100.
        labels = _labels([(i, "METASTATIC", 1, dt.datetime(2018, 1, 1)) for i in range(1, 6)])
        rows = [(1, 10, PSA_LAB_NAME, 100.0) for _ in range(10)]
        rows += [(i, 10, PSA_LAB_NAME, 1.0) for i in range(2, 6)]
        traj = build_lab_trajectory(_long(rows), labels, PSA_LAB_NAME, min_patients_per_bin=5)

        self.assertEqual(traj["median"][0], 1.0)
        self.assertEqual(traj["n_patients"][0], 5)
        self.assertEqual(traj["n_measurements"][0], 14)

    def test_bins_do_not_straddle_the_anchor(self):
        # A draw at -1 day and one at +1 day are on opposite sides of ADT
        # start and must not land in the same bin, or the pre/post contrast
        # is blurred exactly where it matters most.
        labels = _labels([(i, "METASTATIC", 1, dt.datetime(2018, 1, 1)) for i in range(1, 11)])
        rows = [(i, -1, PSA_LAB_NAME, 50.0) for i in range(1, 6)]
        rows += [(i, 1, PSA_LAB_NAME, 5.0) for i in range(6, 11)]
        traj = build_lab_trajectory(_long(rows), labels, PSA_LAB_NAME,
                                    bin_days=30, min_patients_per_bin=5)

        starts = sorted(traj["bin_start"].to_list())
        self.assertEqual(starts, [-30, 0])
        pre = traj.filter(pl.col("bin_start") == -30)["median"][0]
        post = traj.filter(pl.col("bin_start") == 0)["median"][0]
        self.assertEqual(pre, 50.0)
        self.assertEqual(post, 5.0)

    def test_sparse_bins_are_dropped(self):
        # 4 patients in a bin with a threshold of 5 -> dropped, because a
        # median over a handful of patients is noise plotted as a summary.
        labels = _labels([(i, "METASTATIC", 1, dt.datetime(2018, 1, 1)) for i in range(1, 5)])
        rows = [(i, 10, PSA_LAB_NAME, float(i)) for i in range(1, 5)]
        traj = build_lab_trajectory(_long(rows), labels, PSA_LAB_NAME, min_patients_per_bin=5)
        self.assertEqual(traj.height, 0)

    def test_window_excludes_out_of_range_measurements(self):
        # -400d (before the 1y pre-window) and +2000d (past 5y) are dropped.
        labels = _labels([(i, "METASTATIC", 1, dt.datetime(2018, 1, 1)) for i in range(1, 6)])
        rows = [(i, -400, PSA_LAB_NAME, 9.0) for i in range(1, 6)]
        rows += [(i, 2000, PSA_LAB_NAME, 9.0) for i in range(1, 6)]
        rows += [(i, 100, PSA_LAB_NAME, 3.0) for i in range(1, 6)]
        traj = build_lab_trajectory(_long(rows), labels, PSA_LAB_NAME, min_patients_per_bin=5)

        self.assertEqual(traj.height, 1)
        self.assertEqual(traj["median"][0], 3.0)

    def test_other_labs_are_not_mixed_in(self):
        labels = _labels([(i, "METASTATIC", 1, dt.datetime(2018, 1, 1)) for i in range(1, 6)])
        rows = [(i, 10, PSA_LAB_NAME, 2.0) for i in range(1, 6)]
        rows += [(i, 10, TESTOSTERONE_LAB_NAME, 300.0) for i in range(1, 6)]
        traj = build_lab_trajectory(_long(rows), labels, PSA_LAB_NAME, min_patients_per_bin=5)

        self.assertEqual(traj.height, 1)
        self.assertEqual(traj["median"][0], 2.0)

    def test_intent_classes_are_summarized_separately(self):
        labels = _labels(
            [(i, "METASTATIC", 1, dt.datetime(2018, 1, 1)) for i in range(1, 6)]
            + [(i, "LOCALIZED_ADJUVANT", 1, dt.datetime(2018, 1, 1)) for i in range(6, 11)]
        )
        rows = [(i, 10, PSA_LAB_NAME, 20.0) for i in range(1, 6)]
        rows += [(i, 10, PSA_LAB_NAME, 0.1) for i in range(6, 11)]
        traj = build_lab_trajectory(_long(rows), labels, PSA_LAB_NAME, min_patients_per_bin=5)

        self.assertEqual(traj.height, 2)
        met = traj.filter(pl.col(INTENT_COL) == "METASTATIC")["median"][0]
        loc = traj.filter(pl.col(INTENT_COL) == "LOCALIZED_ADJUVANT")["median"][0]
        self.assertEqual(met, 20.0)
        self.assertEqual(loc, 0.1)

    def test_below_detection_zeros_are_kept_as_real_values(self):
        # Zeros are imputed upstream in dfci_labs from a "<0.1" text result;
        # they are genuine below-detection measurements, not missing data, so
        # dropping them would bias the adjuvant median upward.
        labels = _labels([(i, "LOCALIZED_ADJUVANT", 1, dt.datetime(2018, 1, 1)) for i in range(1, 6)])
        rows = [(i, 10, PSA_LAB_NAME, 0.0) for i in range(1, 6)]
        traj = build_lab_trajectory(_long(rows), labels, PSA_LAB_NAME, min_patients_per_bin=5)

        self.assertEqual(traj.height, 1)
        self.assertEqual(traj["median"][0], 0.0)

    def test_unlabelled_patients_are_excluded(self):
        # A patient with labs but no intent label must not silently join in.
        labels = _labels([(i, "METASTATIC", 1, dt.datetime(2018, 1, 1)) for i in range(1, 6)])
        rows = [(i, 10, PSA_LAB_NAME, 1.0) for i in range(1, 6)]
        rows += [(99, 10, PSA_LAB_NAME, 1000.0)]
        traj = build_lab_trajectory(_long(rows), labels, PSA_LAB_NAME, min_patients_per_bin=5)

        self.assertEqual(traj["n_patients"][0], 5)
        self.assertEqual(traj["median"][0], 1.0)

    def test_empty_input_returns_typed_empty_frame(self):
        labels = _labels([(1, "METASTATIC", 1, dt.datetime(2018, 1, 1))])
        traj = build_lab_trajectory(_long([]), labels, PSA_LAB_NAME)
        self.assertEqual(traj.height, 0)
        self.assertIn("median", traj.columns)
        self.assertIn(INTENT_COL, traj.columns)


class TestCoverage(unittest.TestCase):
    def test_coverage_reports_patients_without_the_lab(self):
        # Testosterone is ordered far less consistently than PSA. Only 2 of 5
        # metastatic patients have a draw -> a flat curve would be missing
        # data, not biology, and this table is what distinguishes them.
        labels = _labels([(i, "METASTATIC", 1, dt.datetime(2018, 1, 1)) for i in range(1, 6)])
        rows = [(1, 10, TESTOSTERONE_LAB_NAME, 20.0), (2, 10, TESTOSTERONE_LAB_NAME, 15.0)]
        cov = summarize_trajectory_coverage(_long(rows), labels, TESTOSTERONE_LAB_NAME)

        row = cov.filter(pl.col(INTENT_COL) == "METASTATIC")
        self.assertEqual(row["n_labelled"][0], 5)
        self.assertEqual(row["n_with_lab"][0], 2)
        self.assertEqual(row["pct_with_lab"][0], 40.0)

    def test_coverage_reports_zero_when_lab_absent(self):
        labels = _labels([(i, "METASTATIC", 1, dt.datetime(2018, 1, 1)) for i in range(1, 6)])
        rows = [(i, 10, PSA_LAB_NAME, 1.0) for i in range(1, 6)]
        cov = summarize_trajectory_coverage(_long(rows), labels, TESTOSTERONE_LAB_NAME)
        self.assertEqual(cov["n_with_lab"][0], 0)


class TestKmInput(unittest.TestCase):
    def _long_with_endpoints(self):
        # Two rows per patient -- the longitudinal frame is one row per lab
        # draw, and endpoint columns are patient-level constants repeated
        # down the block.
        return pl.DataFrame(
            {
                ID_COL: [1, 1, 2, 2, 3, 3],
                "t_lab": [0.0, 10.0] * 3,
                "LAB_NAME": [PSA_LAB_NAME] * 6,
                "LAB_VALUE": [1.0] * 6,
                "t_platinum": [500.0, 500.0, 800.0, 800.0, 300.0, 300.0],
                "PLATINUM": [1, 1, 0, 0, 1, 1],
            },
            schema={ID_COL: pl.Int64, "t_lab": pl.Float64, "LAB_NAME": pl.Utf8,
                    "LAB_VALUE": pl.Float64, "t_platinum": pl.Float64, "PLATINUM": pl.Int64},
        )

    def test_km_input_collapses_to_one_row_per_patient(self):
        labels = _labels([
            (1, "METASTATIC", 1, dt.datetime(2018, 1, 1)),
            (2, "METASTATIC", 1, dt.datetime(2018, 1, 1)),
            (3, "LOCALIZED_ADJUVANT", 1, dt.datetime(2018, 1, 1)),
        ])
        km = build_km_input(self._long_with_endpoints(), labels, "platinum")

        self.assertEqual(km.height, 3)
        self.assertEqual(sorted(km[ID_COL].to_list()), [1, 2, 3])
        self.assertEqual(km.filter(pl.col(ID_COL) == 1)["duration"][0], 500.0)
        self.assertEqual(km.filter(pl.col(ID_COL) == 2)["event"][0], 0)

    def test_missing_endpoint_columns_return_typed_empty_frame(self):
        # NEPC columns are appended only when the cohort carried them, so a
        # platinum-only file must degrade to an empty frame rather than raise.
        labels = _labels([(1, "METASTATIC", 1, dt.datetime(2018, 1, 1))])
        km = build_km_input(self._long_with_endpoints(), labels, "nepc")
        self.assertEqual(km.height, 0)
        self.assertIn("duration", km.columns)
        self.assertIn(INTENT_COL, km.columns)

    def test_unknown_endpoint_raises(self):
        labels = _labels([(1, "METASTATIC", 1, dt.datetime(2018, 1, 1))])
        with self.assertRaises(ValueError):
            build_km_input(self._long_with_endpoints(), labels, "not_an_endpoint")

    def test_negative_durations_are_dropped(self):
        # A prevalent event dated before the anchor yields a negative
        # duration, which is not plottable on a KM from time zero.
        frame = pl.DataFrame(
            {ID_COL: [1, 2], "t_platinum": [-30.0, 100.0], "PLATINUM": [1, 1]},
            schema={ID_COL: pl.Int64, "t_platinum": pl.Float64, "PLATINUM": pl.Int64},
        )
        labels = _labels([
            (1, "METASTATIC", 1, dt.datetime(2018, 1, 1)),
            (2, "METASTATIC", 1, dt.datetime(2018, 1, 1)),
        ])
        km = build_km_input(frame, labels, "platinum")
        self.assertEqual(km.height, 1)
        self.assertEqual(km[ID_COL][0], 2)

    def test_summarize_km_counts_events(self):
        labels = _labels([
            (1, "METASTATIC", 1, dt.datetime(2018, 1, 1)),
            (2, "METASTATIC", 1, dt.datetime(2018, 1, 1)),
            (3, "LOCALIZED_ADJUVANT", 1, dt.datetime(2018, 1, 1)),
        ])
        km = build_km_input(self._long_with_endpoints(), labels, "platinum")
        summary = summarize_km(km)

        met = summary.filter(pl.col(INTENT_COL) == "METASTATIC")
        self.assertEqual(met["n_patients"][0], 2)
        self.assertEqual(met["n_events"][0], 1)
        self.assertEqual(met["pct_event"][0], 50.0)

    def test_summarize_km_on_empty_input(self):
        summary = summarize_km(
            pl.DataFrame(schema={ID_COL: pl.Int64, INTENT_COL: pl.Utf8,
                                 "duration": pl.Float64, "event": pl.Int64})
        )
        self.assertEqual(summary.height, 0)
        self.assertIn("n_events", summary.columns)


class TestDeathKmInput(unittest.TestCase):
    def test_death_duration_is_measured_from_first_adt(self):
        labels = _labels([
            (1, "METASTATIC", 1, dt.datetime(2018, 1, 1)),
            (2, "LOCALIZED_ADJUVANT", 0, dt.datetime(2018, 1, 1)),
        ])
        follow_up = pl.DataFrame(
            {
                ID_COL: [1, 2],
                "FOLLOW_UP_END_DATE": [dt.datetime(2019, 1, 1), dt.datetime(2020, 1, 1)],
                "DEATH": [1, 0],
            },
            schema={ID_COL: pl.Int64, "FOLLOW_UP_END_DATE": pl.Datetime, "DEATH": pl.Int64},
        )
        km = build_death_km_input(labels, follow_up)

        self.assertEqual(km.filter(pl.col(ID_COL) == 1)["duration"][0], 365.0)
        self.assertEqual(km.filter(pl.col(ID_COL) == 1)["event"][0], 1)
        self.assertEqual(km.filter(pl.col(ID_COL) == 2)["duration"][0], 730.0)
        self.assertEqual(km.filter(pl.col(ID_COL) == 2)["event"][0], 0)

    def test_death_km_covers_patients_outside_the_eligible_cohort(self):
        # The whole point of sourcing follow-up from patient status: an
        # ELIGIBLE == 0 patient has no row in the survival cohort but still
        # has a death date, so they belong on the full-population curve.
        labels = _labels([
            (1, "METASTATIC", 1, dt.datetime(2018, 1, 1)),
            (2, "METASTATIC", 0, dt.datetime(2018, 1, 1)),
        ])
        follow_up = pl.DataFrame(
            {
                ID_COL: [1, 2],
                "FOLLOW_UP_END_DATE": [dt.datetime(2019, 1, 1), dt.datetime(2019, 1, 1)],
                "DEATH": [1, 1],
            },
            schema={ID_COL: pl.Int64, "FOLLOW_UP_END_DATE": pl.Datetime, "DEATH": pl.Int64},
        )
        km = build_death_km_input(labels, follow_up)
        self.assertEqual(km.height, 2)

    def test_missing_follow_up_columns_raise(self):
        labels = _labels([(1, "METASTATIC", 1, dt.datetime(2018, 1, 1))])
        bad = pl.DataFrame({ID_COL: [1]}, schema={ID_COL: pl.Int64})
        with self.assertRaises(ValueError):
            build_death_km_input(labels, bad)

    def test_missing_adt_first_date_raises(self):
        labels = pl.DataFrame(
            {ID_COL: [1], INTENT_COL: ["METASTATIC"]},
            schema={ID_COL: pl.Int64, INTENT_COL: pl.Utf8},
        )
        follow_up = pl.DataFrame(
            {ID_COL: [1], "FOLLOW_UP_END_DATE": [dt.datetime(2019, 1, 1)], "DEATH": [1]},
            schema={ID_COL: pl.Int64, "FOLLOW_UP_END_DATE": pl.Datetime, "DEATH": pl.Int64},
        )
        with self.assertRaises(ValueError):
            build_death_km_input(labels, follow_up)

    def test_patients_without_a_status_row_are_dropped_not_zero_filled(self):
        # A null follow-up date must not become a zero-day duration, which
        # would read as an immediate event on the curve.
        labels = _labels([
            (1, "METASTATIC", 1, dt.datetime(2018, 1, 1)),
            (2, "METASTATIC", 1, dt.datetime(2018, 1, 1)),
        ])
        follow_up = pl.DataFrame(
            {ID_COL: [1], "FOLLOW_UP_END_DATE": [dt.datetime(2019, 1, 1)], "DEATH": [1]},
            schema={ID_COL: pl.Int64, "FOLLOW_UP_END_DATE": pl.Datetime, "DEATH": pl.Int64},
        )
        km = build_death_km_input(labels, follow_up)
        self.assertEqual(km.height, 1)
        self.assertEqual(km[ID_COL][0], 1)


class TestAnchorGuard(unittest.TestCase):
    """load_longitudinal must refuse an arpi-arm file.

    Both arms write identical column names, so the only symptom of passing
    the wrong one is that every trajectory is measured from a later origin --
    silent, and invisible in the output.
    """

    def _write(self, tmpdir, anchor_dates):
        path = Path(tmpdir) / "long.csv"
        pl.DataFrame(
            {
                ID_COL: [1, 2, 3],
                "t_lab": [0.0, 0.0, 0.0],
                "LAB_NAME": [PSA_LAB_NAME] * 3,
                "LAB_VALUE": [1.0, 1.0, 1.0],
                "TREATMENT_ANCHOR_DATE": anchor_dates,
            }
        ).write_csv(path)
        return path

    def test_matching_anchor_is_accepted(self):
        import tempfile

        labels = _labels([(i, "METASTATIC", 1, dt.datetime(2018, 1, 1)) for i in (1, 2, 3)])
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, ["2018-01-01"] * 3)
            out = load_longitudinal(path, labels=labels)
            self.assertEqual(out.height, 3)

    def test_arpi_anchor_is_rejected(self):
        import tempfile

        # ARPI exposure typically starts well after first ADT, so the anchor
        # sits years later than ADT_FIRST_DATE for most patients.
        labels = _labels([(i, "METASTATIC", 1, dt.datetime(2018, 1, 1)) for i in (1, 2, 3)])
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, ["2021-06-01"] * 3)
            with self.assertRaises(ValueError) as ctx:
                load_longitudinal(path, labels=labels)
            self.assertIn("arpi", str(ctx.exception).lower())

    def test_a_few_mismatched_patients_do_not_trip_the_guard(self):
        # Real data has some anchor jitter; the guard is a 5% tolerance, not
        # an exact-match assertion.
        import tempfile

        labels = _labels([(i, "METASTATIC", 1, dt.datetime(2018, 1, 1)) for i in (1, 2, 3)])
        with tempfile.TemporaryDirectory() as tmp:
            # 1 of 3 mismatched is 33%, over tolerance -- so use a wider set.
            path = Path(tmp) / "long.csv"
            pl.DataFrame(
                {
                    ID_COL: list(range(1, 41)),
                    "t_lab": [0.0] * 40,
                    "LAB_NAME": [PSA_LAB_NAME] * 40,
                    "LAB_VALUE": [1.0] * 40,
                    "TREATMENT_ANCHOR_DATE": ["2018-01-01"] * 39 + ["2021-06-01"],
                }
            ).write_csv(path)
            wide = _labels([(i, "METASTATIC", 1, dt.datetime(2018, 1, 1)) for i in range(1, 41)])
            out = load_longitudinal(path, labels=wide)
            self.assertEqual(out.height, 40)


class TestKmSeriesConversion(unittest.TestCase):
    def test_series_are_keyed_by_intent(self):
        km = pl.DataFrame(
            {
                ID_COL: [1, 2, 3],
                INTENT_COL: ["METASTATIC", "METASTATIC", "LOCALIZED_ADJUVANT"],
                "duration": [100.0, 200.0, 300.0],
                "event": [1, 0, 1],
            },
            schema={ID_COL: pl.Int64, INTENT_COL: pl.Utf8,
                    "duration": pl.Float64, "event": pl.Int64},
        )
        try:
            series = km_series_by_intent(km)
        except ModuleNotFoundError:
            self.skipTest("pandas not installed")

        self.assertEqual(set(series), {"METASTATIC", "LOCALIZED_ADJUVANT"})
        dur, evt = series["METASTATIC"]
        self.assertEqual(len(dur), 2)
        self.assertEqual(list(evt), [1.0, 0.0])

    def test_absent_intent_classes_are_omitted(self):
        km = pl.DataFrame(
            {ID_COL: [1], INTENT_COL: ["METASTATIC"], "duration": [100.0], "event": [1]},
            schema={ID_COL: pl.Int64, INTENT_COL: pl.Utf8,
                    "duration": pl.Float64, "event": pl.Int64},
        )
        try:
            series = km_series_by_intent(km)
        except ModuleNotFoundError:
            self.skipTest("pandas not installed")
        self.assertEqual(set(series), {"METASTATIC"})


class TestLogrank(unittest.TestCase):
    def test_logrank_returns_typed_frame_without_lifelines(self):
        # lifelines is optional; the report must degrade rather than crash
        # the notebook.
        km = pl.DataFrame(
            {
                ID_COL: [1, 2],
                INTENT_COL: ["METASTATIC", "LOCALIZED_ADJUVANT"],
                "duration": [100.0, 200.0],
                "event": [1, 0],
            },
            schema={ID_COL: pl.Int64, INTENT_COL: pl.Utf8,
                    "duration": pl.Float64, "event": pl.Int64},
        )
        out = logrank_by_intent(km)
        self.assertIn("p_value", out.columns)
        self.assertIn("group_a", out.columns)


class TestStageMetBurdenPlots(unittest.TestCase):
    """Stage / met-burden distribution figures."""

    def _frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                INTENT_COL: ["METASTATIC"] * 3 + ["LOCALIZED_ADJUVANT"] * 2,
                "CANCER_STAGE": ["IV", "IV", "III", "II", None],
                "N_MET_SITES": [3, 1, 0, 0, None],
                "MET_SITE_bone": [1, 1, 0, 0, None],
                "MET_SITE_liver": [1, 0, 0, 0, None],
            }
        )

    def test_distributions_are_within_class_percentages(self):
        """The classes differ in size by design, so raw counts would show
        cohort composition rather than a distributional difference."""
        stage = stage_distribution_by_intent(self._frame())
        met = stage.filter(pl.col(INTENT_COL) == "METASTATIC")
        # 3 metastatic patients are staged; 2 of them are IV.
        self.assertEqual(met["n_in_class"][0], 3)
        self.assertAlmostEqual(
            float(met.filter(pl.col("CANCER_STAGE") == "IV")["pct"][0]), 66.7, places=1
        )
        # Each class's percentages sum to 100.
        for cls in stage[INTENT_COL].unique().to_list():
            total = stage.filter(pl.col(INTENT_COL) == cls)["pct"].sum()
            self.assertAlmostEqual(float(total), 100.0, places=0)

    def test_uncovered_patients_are_excluded_from_denominators(self):
        """A null stage / burden is unobserved, not a zero-burden stage I."""
        stage = stage_distribution_by_intent(self._frame())
        localized = stage.filter(pl.col(INTENT_COL) == "LOCALIZED_ADJUVANT")
        # Two localized patients, but only one carries a stage.
        self.assertEqual(localized["n_in_class"][0], 1)

        burden = met_burden_distribution_by_intent(self._frame())
        self.assertEqual(
            int(burden.filter(pl.col(INTENT_COL) == "LOCALIZED_ADJUVANT")["n_in_class"][0]), 1
        )

    def test_met_burden_top_codes_the_tail(self):
        """Site counts have a long thin tail; top-coding keeps the axis readable."""
        frame = pl.DataFrame(
            {INTENT_COL: ["METASTATIC"] * 2, "N_MET_SITES": [1, 7]}
        )
        got = met_burden_distribution_by_intent(frame, max_sites=4)
        self.assertEqual(sorted(got["n_met_sites"].to_list()), [1, 4])

    def test_panel_renders_with_partial_coverage(self):
        """Missing columns must degrade to a placeholder panel, not raise."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plot_stage_metburden_panel(self._frame())
        self.assertEqual(len(axes), 3)
        plt.close(fig)

        # No stage, no burden at all -- every panel should still draw.
        bare = pl.DataFrame({INTENT_COL: ["METASTATIC", "LOCALIZED_ADJUVANT"]})
        fig, axes = plot_stage_metburden_panel(bare)
        self.assertEqual(len(axes), 3)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
