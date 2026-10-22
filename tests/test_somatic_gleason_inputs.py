"""Source-specific cohorts and stratifiers in build_somatic_gleason_inputs.

Exercises COMPASS.data_preprocessing.build_somatic_gleason_inputs directly.
The Gleason-only cohort stays on the ADT clock; the Gleason ∩ somatic cohort
and the trio stratifiers are anchored on the sequencing report date of the
specimen reported closest to ADT start (select_sequencing_specimen ->
reclock_to_sequencing -> build_sequencing_anchored_case). No somatic cohort is
built on the ADT clock.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PREP_DIR = REPO_ROOT / "COMPASS" / "data_preprocessing"
SURVIVAL_DIR = REPO_ROOT / "COMPASS" / "survival_analysis"
for _p in (str(REPO_ROOT), str(DATA_PREP_DIR), str(SURVIVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_somatic_gleason_inputs as bsg  # noqa: E402
import cox_aggregated as ca  # noqa: E402

ID_COL = ca.ID_COL


def _treatment_anchors(mrns) -> pd.Series:
    return pd.Series(
        pd.Timestamp("2020-01-01"), index=pd.Index(mrns, name=ID_COL), name="anchor"
    )


def _base(mrns) -> pd.DataFrame:
    return pd.DataFrame({ID_COL: mrns})


def _gleason(rows: list[tuple]) -> pd.DataFrame:
    # rows: (mrn, gleason_date, gleason_total)
    df = pd.DataFrame(rows, columns=[ID_COL, "gleason_date", bsg.GLEASON_FEATURE])
    df["gleason_date"] = pd.to_datetime(df["gleason_date"])
    df[bsg.GLEASON_AVAILABLE_DATE] = df["gleason_date"]
    return df


def _somatic(rows: list[tuple], feature="TP53_SNV") -> tuple[pd.DataFrame, list[str]]:
    # rows: (mrn, sequencing_date, feature_value)
    df = pd.DataFrame(rows, columns=[ID_COL, bsg.SEQUENCING_DATE, feature])
    df[bsg.SEQUENCING_DATE] = pd.to_datetime(df[bsg.SEQUENCING_DATE])
    df[bsg.SOMATIC_AVAILABLE_DATE] = df[bsg.SEQUENCING_DATE]
    return df, [feature]


class TestGleasonSomaticAvailableCase:
    def test_gleason_somatic_is_not_built_on_the_adt_clock(self):
        # Anything using somatic data is sequencing-anchored, so the ADT-clock
        # builder refuses the combined cohort instead of silently building it.
        mrns = [1]
        base = _base(mrns)
        anchors = _treatment_anchors(mrns)
        gleason = _gleason([(1, "2019-06-01", 7.0)])
        somatic, somatic_features = _somatic([(1, "2019-07-01", 1.0)])
        with pytest.raises(ValueError, match="sequencing-anchored"):
            bsg.build_available_case_sensitivity(
                base,
                somatic,
                somatic_features,
                gleason,
                treatment_anchors=anchors,
                landmark_day=0,
                analysis="gleason_somatic",
            )

    def test_gleason_only_analysis_unaffected(self):
        mrns = [1, 2]
        base = _base(mrns)
        anchors = _treatment_anchors(mrns)
        gleason = _gleason([(1, "2019-06-01", 7.0), (2, "2019-06-01", 8.0)])
        somatic, somatic_features = _somatic([(1, "2019-07-01", 1.0)])

        out = bsg.build_available_case_sensitivity(
            base,
            somatic,
            somatic_features,
            gleason,
            treatment_anchors=anchors,
            landmark_day=0,
            analysis="gleason",
        )
        assert set(out[ID_COL]) == {1, 2}

    def test_somatic_is_not_built_on_the_adt_clock(self):
        mrns = [1]
        somatic, somatic_features = _somatic([(1, "2019-07-01", 1.0)])
        with pytest.raises(ValueError, match="sequencing-anchored"):
            bsg.build_available_case_sensitivity(
                _base(mrns),
                somatic,
                somatic_features,
                _gleason([(1, "2019-06-01", 7.0)]),
                treatment_anchors=_treatment_anchors(mrns),
                landmark_day=0,
                analysis="somatic",
            )

    def test_unsupported_analysis_raises(self):
        mrns = [1]
        base = _base(mrns)
        anchors = _treatment_anchors(mrns)
        gleason = _gleason([(1, "2019-06-01", 7.0)])
        somatic, somatic_features = _somatic([(1, "2019-07-01", 1.0)])
        with pytest.raises(ValueError):
            bsg.build_available_case_sensitivity(
                base,
                somatic,
                somatic_features,
                gleason,
                treatment_anchors=anchors,
                landmark_day=0,
                analysis="not-a-real-analysis",
            )

    def test_gleason_somatic_available_case_constant_registered(self):
        assert bsg.GLEASON_SOMATIC_AVAILABLE_CASE == "gleason_somatic_available_case"


HORIZON = 3650.0


def _longitudinal(rows: list[tuple]) -> pd.DataFrame:
    # rows: (mrn, t_platinum, platinum, t_last_contact, [t_lab, ...]), all on
    # the ADT clock like the COMPASS longitudinal CSV (one row per lab).
    records = []
    for mrn, t_platinum, platinum, t_last_contact, labs in rows:
        for t_lab in labs:
            records.append({
                ID_COL: mrn,
                "AGE_AT_TREATMENTSTART": 65.0,
                "FIRST_RECORD_DATE": "2015-01-01",
                "FIRST_TREATMENT": 1,
                "t_first_treatment": 0.0,
                "t_platinum": t_platinum,
                "PLATINUM": platinum,
                "t_death": t_last_contact,
                "DEATH": 0,
                "t_last_contact": t_last_contact,
                "LAB_NAME": "PSA",
                "LAB_VALUE": float(t_lab),
                "t_lab": float(t_lab),
            })
    return pd.DataFrame(records)


class TestSequencingAnchoredCase:
    """select_sequencing_specimen -> reclock_to_sequencing ->
    build_sequencing_anchored_case: time 0 is the sequencing report date of
    the specimen reported closest to ADT start, so landmarks are sequencing + D."""

    # ADT starts 2020-01-01 for everyone.
    # MRN 1: sequenced 2019-11-02 (60 d before ADT); a later 2021 specimen is
    #        farther from ADT and must not be selected.
    # MRN 2: sequenced 2020-07-19 (200 d after ADT, i.e. after ADT + 180).
    # MRN 3: sequenced but no Gleason -> excluded.
    # MRN 4: sequenced with Gleason but outside the base split -> excluded.
    MRNS = [1, 2, 3, 4]

    def _inputs(self):
        anchors = _treatment_anchors(self.MRNS)
        somatic, features = _somatic([
            (1, "2019-11-02", 1.0),
            (1, "2021-01-01", 0.0),
            (2, "2020-07-19", 0.0),
            (3, "2019-12-01", 1.0),
            (4, "2019-12-01", 1.0),
        ])
        specimen = bsg.select_sequencing_specimen(somatic, features, anchors)
        gleason = _gleason([
            (1, "2019-06-01", 7.0),
            (1, "2019-12-15", 9.0),  # after sequencing, before sequencing + 90
            (2, "2020-03-01", 8.0),  # after ADT start, before sequencing
            (4, "2019-06-01", 6.0),
        ])
        longitudinal = _longitudinal([
            (1, 500.0, 1, 900.0, [-100.0, -30.0]),
            (2, 1000.0, 0, 1000.0, [150.0, 300.0]),
            (3, 400.0, 0, 400.0, [-10.0]),
            (4, 400.0, 0, 400.0, [-10.0]),
        ])
        split = pd.Series(
            {1: "train", 2: "test", 3: "train"}, name="split"
        ).rename_axis(ID_COL)
        return specimen, features, gleason, longitudinal, split

    def _build(self, landmark_day: int):
        specimen, features, gleason, longitudinal, split = self._inputs()
        reclocked = bsg.reclock_to_sequencing(
            longitudinal, specimen[bsg.SEQUENCING_TO_ADT_DAYS]
        )
        return bsg.build_sequencing_anchored_case(
            reclocked,
            specimen,
            features,
            gleason,
            split=split,
            landmark_day=landmark_day,
            max_followup_days=HORIZON,
        )

    def test_specimen_is_the_one_closest_to_adt(self):
        specimen, *_ = self._inputs()
        assert specimen.loc[1, bsg.SEQUENCING_DATE] == pd.Timestamp("2019-11-02")
        assert specimen.loc[1, bsg.SEQUENCING_TO_ADT_DAYS] == -60.0
        assert specimen.loc[2, bsg.SEQUENCING_TO_ADT_DAYS] == 200.0
        assert specimen.loc[1, "TP53_SNV"] == 1.0

    def test_reclock_keeps_pre_adt_sequencing_and_shifts_every_clock(self):
        specimen, _, _, longitudinal, _ = self._inputs()
        offsets = specimen[bsg.SEQUENCING_TO_ADT_DAYS].drop(index=[3, 4])
        out = bsg.reclock_to_sequencing(longitudinal, offsets)
        assert set(out[ID_COL]) == {1, 2}
        row1 = out.loc[out[ID_COL] == 1].iloc[0]
        assert row1["t_platinum"] == 560.0
        assert row1["t_last_contact"] == 960.0
        assert sorted(out.loc[out[ID_COL] == 1, "t_lab"]) == [-40.0, 30.0]
        assert row1["AGE_AT_TREATMENTSTART"] == pytest.approx(65.0 - 60 / 365.25)
        row2 = out.loc[out[ID_COL] == 2].iloc[0]
        assert row2["t_platinum"] == 800.0
        assert row2["AGE_AT_TREATMENTSTART"] == pytest.approx(65.0 + 200 / 365.25)

    def test_cohort_requires_gleason_and_base_split(self):
        aggregated, _ = self._build(0)
        assert set(aggregated[ID_COL]) == {1, 2}
        # Split is inherited from the base run, not reassigned.
        assert aggregated.set_index(ID_COL)["split"].to_dict() == {1: "train", 2: "test"}

    def test_sequencing_after_adt_plus_180_is_kept_with_sequencing_clock_outcomes(self):
        aggregated, _ = self._build(0)
        row2 = aggregated.set_index(ID_COL).loc[2]
        assert row2["t_platinum"] == 800.0
        assert row2["TP53_SNV"] == 0.0
        row1 = aggregated.set_index(ID_COL).loc[1]
        assert row1["t_platinum"] == 560.0
        assert row1["TP53_SNV"] == 1.0

    def test_gleason_must_be_available_by_sequencing_plus_landmark(self):
        at0 = self._build(0)[0].set_index(ID_COL)
        at90 = self._build(90)[0].set_index(ID_COL)
        # MRN 2's post-ADT Gleason predates its sequencing date, so it counts.
        assert at0.loc[2, bsg.GLEASON_FEATURE] == 8.0
        # MRN 1's 2019-12-15 score is only available from sequencing + 43 d.
        assert at0.loc[1, bsg.GLEASON_FEATURE] == 7.0
        assert at90.loc[1, bsg.GLEASON_FEATURE] == 9.0

    def test_labs_are_restricted_to_before_sequencing_plus_landmark(self):
        _, labs0 = self._build(0)
        _, labs90 = self._build(90)
        assert sorted(labs0.loc[labs0[ID_COL] == 1, "t_lab"]) == [-40.0]
        assert sorted(labs90.loc[labs90[ID_COL] == 1, "t_lab"]) == [-40.0, 30.0]
        # MRN 2's ADT-clock labs (150, 300) sit at -50 and +100 from sequencing.
        assert sorted(labs0.loc[labs0[ID_COL] == 2, "t_lab"]) == [-50.0]
        assert labs0["t_lab"].lt(0).all()
        assert labs90["t_lab"].lt(90).all()


def _somatic_unfiltered(rows: list[tuple], trio_columns: list[str]) -> pd.DataFrame:
    # rows: (mrn, sequencing_date, *trio_values)
    df = pd.DataFrame(rows, columns=[ID_COL, bsg.SEQUENCING_DATE, *trio_columns])
    df[bsg.SEQUENCING_DATE] = pd.to_datetime(df[bsg.SEQUENCING_DATE])
    df[bsg.SOMATIC_AVAILABLE_DATE] = df[bsg.SEQUENCING_DATE]
    return df


def _stage(rows: list[tuple]) -> pd.DataFrame:
    # rows: (mrn, event_date, stage)
    df = pd.DataFrame(rows, columns=[ID_COL, "EVENT_DATE", bsg.STAGE_COLUMN])
    df["EVENT_DATE"] = pd.to_datetime(df["EVENT_DATE"])
    return df


class TestClinicalStratifiers:
    """build_clinical_stratifiers (Plan §4a): stratifiers only, never model
    features, so the trio comes from the UNFILTERED somatic matrix (every
    alteration class) and a patient missing one stratifier still keeps the
    others via independent left joins onto the base cohort."""

    def test_left_joins_each_stratifier_independently(self):
        mrns = [1, 2, 3]
        base = _base(mrns)
        anchors = _treatment_anchors(mrns)
        gleason = _gleason([(1, "2019-06-01", 7.0), (2, "2019-06-01", 8.0)])
        trio_columns = ["TP53_SNV", "TP53_DEL"]
        somatic_unfiltered = _somatic_unfiltered(
            [(1, "2019-07-01", 1.0, 0.0), (3, "2019-07-01", 0.0, 1.0)],
            trio_columns,
        )
        stage = _stage([(2, "2019-05-01", "M1")])

        out = bsg.build_clinical_stratifiers(
            base, somatic_unfiltered, trio_columns, gleason, stage,
            treatment_anchors=anchors, landmark_day=0,
        )

        assert set(out[ID_COL]) == {1, 2, 3}
        # MRN 1: gleason and trio both present.
        row1 = out.loc[out[ID_COL] == 1].iloc[0]
        assert row1[bsg.GLEASON_FEATURE] == 7.0
        assert row1["TP53_SNV"] == 1.0
        # MRN 2: gleason present, trio absent, stage present.
        row2 = out.loc[out[ID_COL] == 2].iloc[0]
        assert row2[bsg.GLEASON_FEATURE] == 8.0
        assert pd.isna(row2["TP53_SNV"])
        assert row2[bsg.STAGE_COLUMN] == "M1"
        # MRN 3: only trio.
        row3 = out.loc[out[ID_COL] == 3].iloc[0]
        assert pd.isna(row3[bsg.GLEASON_FEATURE])
        assert row3["TP53_DEL"] == 1.0

    def test_stage_uses_latest_event_on_or_before_landmark(self):
        mrns = [1]
        base = _base(mrns)
        anchors = _treatment_anchors(mrns)
        gleason = _gleason([])
        trio_columns = ["TP53_SNV"]
        somatic_unfiltered = _somatic_unfiltered([], trio_columns)
        stage = _stage([
            (1, "2019-01-01", "M0"),
            (1, "2019-12-01", "M1"),  # latest on/before landmark (2020-01-01)
            (1, "2020-06-01", "M1c"),  # after landmark -- must not leak in
        ])

        out = bsg.build_clinical_stratifiers(
            base, somatic_unfiltered, trio_columns, gleason, stage,
            treatment_anchors=anchors, landmark_day=0,
        )
        assert out.loc[out[ID_COL] == 1, bsg.STAGE_COLUMN].iloc[0] == "M1"

    def test_trio_uses_latest_specimen_available_by_landmark(self):
        mrns = [1]
        base = _base(mrns)
        anchors = _treatment_anchors(mrns)
        gleason = _gleason([])
        trio_columns = ["TP53_SNV"]
        somatic_unfiltered = _somatic_unfiltered(
            [(1, "2019-01-01", 0.0), (1, "2019-11-01", 1.0)], trio_columns,
        )
        stage = _stage([])

        out = bsg.build_clinical_stratifiers(
            base, somatic_unfiltered, trio_columns, gleason, stage,
            treatment_anchors=anchors, landmark_day=0,
        )
        assert out.loc[out[ID_COL] == 1, "TP53_SNV"].iloc[0] == 1.0

    def test_adt_stratifiers_omit_the_trio_when_no_trio_columns(self):
        # main() writes the ADT-anchored file with trio_columns=[]: the trio is
        # only stratified on the sequencing clock.
        mrns = [1]
        base = _base(mrns)
        anchors = _treatment_anchors(mrns)
        gleason = _gleason([(1, "2019-06-01", 7.0)])
        somatic_unfiltered = _somatic_unfiltered(
            [(1, "2019-07-01", 1.0)], ["TP53_SNV"]
        )
        stage = _stage([(1, "2019-05-01", "M1")])

        out = bsg.build_clinical_stratifiers(
            base, somatic_unfiltered, [], gleason, stage,
            treatment_anchors=anchors, landmark_day=0,
        )
        assert list(out.columns) == [ID_COL, bsg.GLEASON_FEATURE, bsg.STAGE_COLUMN]

    def test_sequencing_stratifiers_use_the_anchoring_specimen(self):
        # Specimen sequenced 2019-11-02; a second specimen on 2020-01-15 is
        # available by sequencing + 90 d but must not replace it.
        mrns = [1]
        base = _base(mrns)
        trio_columns = ["TP53_SNV", "RB1_DEL"]
        somatic_unfiltered = _somatic_unfiltered(
            [(1, "2019-11-02", 1.0, 0.0), (1, "2020-01-15", 0.0, 1.0)],
            trio_columns,
        )
        specimen = pd.DataFrame(
            {bsg.SEQUENCING_DATE: [pd.Timestamp("2019-11-02")]},
            index=pd.Index(mrns, name=ID_COL),
        )
        gleason = _gleason([(1, "2019-06-01", 7.0), (1, "2020-01-20", 9.0)])
        stage = _stage([(1, "2019-10-01", "M0"), (1, "2020-01-20", "M1")])

        at0 = bsg.build_sequencing_clinical_stratifiers(
            base, somatic_unfiltered, trio_columns, gleason, stage,
            specimen=specimen, landmark_day=0,
        ).set_index(ID_COL).loc[1]
        at90 = bsg.build_sequencing_clinical_stratifiers(
            base, somatic_unfiltered, trio_columns, gleason, stage,
            specimen=specimen, landmark_day=90,
        ).set_index(ID_COL).loc[1]

        for row in (at0, at90):
            assert row["TP53_SNV"] == 1.0
            assert row["RB1_DEL"] == 0.0
        # Gleason and stage are those available by sequencing + D.
        assert at0[bsg.GLEASON_FEATURE] == 7.0
        assert at0[bsg.STAGE_COLUMN] == "M0"
        assert at90[bsg.GLEASON_FEATURE] == 9.0
        assert at90[bsg.STAGE_COLUMN] == "M1"

    def test_sequencing_stratifiers_filename(self):
        assert (
            bsg.sequencing_clinical_stratifiers_filename(90)
            == "clinical_stratifiers_sequencing_landmark90.csv"
        )

    def test_tz_aware_event_date_does_not_crash_the_stage_comparison(self):
        """On the cluster, TREATMENT_ANCHOR_DATE comes back UTC-aware while the
        regex stage parquet's EVENT_DATE is naive; pandas raises TypeError on a
        naive-vs-aware comparison rather than assuming a timezone. Both sides
        represent plain calendar dates, so tz must be dropped before comparing.
        """
        mrns = [1]
        base = _base(mrns)
        anchors = _treatment_anchors(mrns).dt.tz_localize("UTC")
        gleason = _gleason([])
        trio_columns = ["TP53_SNV"]
        somatic_unfiltered = _somatic_unfiltered([], trio_columns)
        stage = _stage([
            (1, "2019-12-01", "M1"),
            (1, "2020-06-01", "M1c"),  # after landmark -- must not leak in
        ])
        assert stage["EVENT_DATE"].dt.tz is None

        out = bsg.build_clinical_stratifiers(
            base, somatic_unfiltered, trio_columns, gleason, stage,
            treatment_anchors=anchors, landmark_day=0,
        )
        assert out.loc[out[ID_COL] == 1, bsg.STAGE_COLUMN].iloc[0] == "M1"

    def test_tz_naive_anchors_with_tz_aware_event_date_also_works(self):
        """The reverse mismatch (naive anchors, tz-aware EVENT_DATE) must also
        not crash -- _drop_tz is applied to whichever side carries a tz."""
        mrns = [1]
        base = _base(mrns)
        anchors = _treatment_anchors(mrns)  # naive
        gleason = _gleason([])
        trio_columns = ["TP53_SNV"]
        somatic_unfiltered = _somatic_unfiltered([], trio_columns)
        stage = _stage([(1, "2019-12-01", "M1")])
        stage["EVENT_DATE"] = stage["EVENT_DATE"].dt.tz_localize("UTC")

        out = bsg.build_clinical_stratifiers(
            base, somatic_unfiltered, trio_columns, gleason, stage,
            treatment_anchors=anchors, landmark_day=0,
        )
        assert out.loc[out[ID_COL] == 1, bsg.STAGE_COLUMN].iloc[0] == "M1"


class TestAvailableCaseManifest:
    """_available_case_manifest (called from main() after every analysis):
    available-case cohorts are small subsets of the base cohort, so an
    endpoint with events in the full cohort can have zero here.
    compute_horizon_grid raises ValueError on zero events; the manifest must
    skip that endpoint rather than let the whole build crash.
    """

    def _base_manifest(self) -> dict:
        return {
            "auc_time_unit_days": 30,
            "auc_quantiles": [0.1, 0.9],
            "auc_max_time_units": 24,
        }

    def _available(self, *, n: int, event_col: str, n_events: int) -> pd.DataFrame:
        event = [1] * n_events + [0] * (n - n_events)
        return pd.DataFrame({
            ID_COL: range(n),
            "split": ["train"] * n,
            "t_platinum": range(10, 10 * (n + 1), 10),
            "PLATINUM": event,
            "t_nepc": range(10, 10 * (n + 1), 10),
            "NEPC": [0] * n,  # zero NEPC events in this available-case subset
        })

    def test_endpoint_with_zero_events_is_skipped_not_raised(self):
        available = self._available(n=6, event_col="PLATINUM", n_events=3)
        manifest = bsg._available_case_manifest(
            self._base_manifest(), {0: available}
        )
        horizons = manifest["auc_horizons_by_landmark"]["0"]
        assert "platinum" in horizons
        assert "nepc" not in horizons

    def test_prediction_time_origin_defaults_to_adt_and_is_overridable(self):
        available = self._available(n=6, event_col="PLATINUM", n_events=3)
        adt = bsg._available_case_manifest(self._base_manifest(), {0: available})
        seq = bsg._available_case_manifest(
            self._base_manifest(), {0: available},
            prediction_time_origin="sequencing date",
        )
        assert adt["prediction_time_origin"] == "ADT start"
        assert seq["prediction_time_origin"] == "sequencing date"

    def test_endpoint_with_events_still_gets_a_horizon_grid(self):
        available = self._available(n=6, event_col="PLATINUM", n_events=3)
        manifest = bsg._available_case_manifest(
            self._base_manifest(), {0: available}
        )
        horizons = manifest["auc_horizons_by_landmark"]["0"]
        assert len(horizons["platinum"]) > 0

    def test_endpoint_with_events_all_at_non_positive_duration_is_also_skipped(self):
        """A bare `event == 1` count is not enough: compute_horizon_grid also
        requires duration > 0 (its own validity mask). An event flagged at or
        before the landmark (duration <= 0, e.g. from an NEPC criterion dated
        on the landmark itself) still crashes compute_horizon_grid even though
        n_events > 0 -- this is exactly what happened on the cluster.
        """
        n = 6
        available = pd.DataFrame({
            ID_COL: range(n),
            "split": ["train"] * n,
            "t_platinum": range(10, 10 * (n + 1), 10),
            "PLATINUM": [1, 1, 0, 0, 0, 0],
            "t_nepc": [0, -5, 10, 20, 30, 40],
            "NEPC": [1, 1, 0, 0, 0, 0],  # events exist, but at duration <= 0
        })
        manifest = bsg._available_case_manifest(
            self._base_manifest(), {0: available}
        )
        horizons = manifest["auc_horizons_by_landmark"]["0"]
        assert "platinum" in horizons
        assert "nepc" not in horizons

    def test_endpoint_with_events_all_past_the_admin_censor_cap_is_also_skipped(self):
        """compute_horizon_grid recodes any event past admin_censor_days back
        to censored before checking for observed events. A guard that checks
        raw event==1 & duration>0 without also applying that same cap can see
        "events exist" while every one of them gets zeroed out inside
        compute_horizon_grid -- this is exactly what happened on the cluster
        a second time, with the base manifest's auc_max_time_units * time_unit
        cap (720 days here) smaller than every NEPC event's raw duration.
        """
        n = 6
        admin_days = (
            self._base_manifest()["auc_max_time_units"]
            * self._base_manifest()["auc_time_unit_days"]
        )
        available = pd.DataFrame({
            ID_COL: range(n),
            "split": ["train"] * n,
            "t_platinum": range(10, 10 * (n + 1), 10),
            "PLATINUM": [1, 1, 0, 0, 0, 0],
            "t_nepc": [admin_days + 100, admin_days + 200, 10, 20, 30, 40],
            "NEPC": [1, 1, 0, 0, 0, 0],  # events exist, but past the admin cap
        })
        manifest = bsg._available_case_manifest(
            self._base_manifest(), {0: available}
        )
        horizons = manifest["auc_horizons_by_landmark"]["0"]
        assert "platinum" in horizons
        assert "nepc" not in horizons


class TestLoadStage:
    def test_drops_rows_missing_date_or_stage(self, tmp_path):
        path = tmp_path / "stage.parquet"
        pd.DataFrame({
            ID_COL: [1, 2, 3],
            "EVENT_DATE": [pd.Timestamp("2020-01-01"), pd.NaT, pd.Timestamp("2020-02-01")],
            bsg.STAGE_COLUMN: ["M1", "M1", None],
        }).to_parquet(path)

        out = bsg.load_stage(path)
        assert set(out[ID_COL]) == {1}
        assert list(out.columns) == [ID_COL, "EVENT_DATE", bsg.STAGE_COLUMN]
