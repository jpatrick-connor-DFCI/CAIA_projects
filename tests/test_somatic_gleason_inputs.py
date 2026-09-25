"""build_available_case_sensitivity's Gleason ∩ somatic cohort (Plan §2b).

Exercises COMPASS.data_preprocessing.build_somatic_gleason_inputs directly:
the new analysis="gleason_somatic" inner-joins the Gleason value closest to
the landmark with the latest somatic specimen available by the landmark, so
only patients with both sources by that landmark are retained.
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
    def test_inner_joins_both_sources(self):
        # MRN 1: has both gleason and somatic before the landmark.
        # MRN 2: has only gleason.
        # MRN 3: has only somatic.
        # MRN 4: has both, but somatic is after the landmark (excluded).
        mrns = [1, 2, 3, 4]
        base = _base(mrns)
        anchors = _treatment_anchors(mrns)
        gleason = _gleason(
            [
                (1, "2019-06-01", 7.0),
                (2, "2019-06-01", 8.0),
                (4, "2019-06-01", 9.0),
            ]
        )
        somatic, somatic_features = _somatic(
            [
                (1, "2019-07-01", 1.0),
                (3, "2019-07-01", 0.0),
                (4, "2020-06-01", 1.0),  # after landmark +0d (2020-01-01)
            ]
        )

        out = bsg.build_available_case_sensitivity(
            base,
            somatic,
            somatic_features,
            gleason,
            treatment_anchors=anchors,
            landmark_day=0,
            analysis="gleason_somatic",
        )

        assert set(out[ID_COL]) == {1}
        assert bsg.GLEASON_FEATURE in out.columns
        assert somatic_features[0] in out.columns

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

    def test_endpoint_with_events_still_gets_a_horizon_grid(self):
        available = self._available(n=6, event_col="PLATINUM", n_events=3)
        manifest = bsg._available_case_manifest(
            self._base_manifest(), {0: available}
        )
        horizons = manifest["auc_horizons_by_landmark"]["0"]
        assert len(horizons["platinum"]) > 0


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
