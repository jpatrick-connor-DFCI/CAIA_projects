"""Phase 6: SurvLatent ODE on full follow-up, and the end_of_obs_idx sweep.

The defect these guard against is the same one that bit the person-period
builder earlier: ``add_post_landmark_horizon_columns`` inferred each patient's
landmark as ``max(TIME)``. That is exact on the landmark frame, which ends AT
the landmark -- and wrong on the full-follow-up frame, where ``max(TIME)`` is
the patient's last follow-up visit. The horizon would then shift per patient by
the length of their post-landmark history, producing plausible-looking numbers
against the wrong outcome definition.

Also pinned:

* **The manifest cross-check runs both ways.** Post-landmark labs are valid
  inputs only for predictions made after the landmark, so the two frames must
  not be interchangeable by accident.
* **A sweep point is a distinct fit**, so it needs a distinct run_id, or
  checkpoints collide across cut points. The no-cut spelling must not change,
  because existing cluster checkpoints are named after it.
* **Truncation is how end_of_obs_idx is swept**, since upstream derives it as
  ``tt[-1]`` rather than taking it as a parameter.

These all run without torch and without the external repo: the module is
import-gated so `--help` works bare, and the functions under test are pandas.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(
    0, str(REPO / "COMPASS" / "survival_analysis" / "multivariate_longitudinal")
)

pd = pytest.importorskip("pandas")
slo = pytest.importorskip("survlatent_ode")

ID = "DFCI_MRN"
TIME = "TIME"


def _frame(*, with_landmark_time: bool):
    """Two patients, landmark 10, with post-landmark follow-up.

    Patient 1 is followed to 30, patient 2 only to 12. Their max(TIME) values
    therefore differ from each other AND from the true landmark, which is what
    makes the inference bug visible.
    """
    rows = [
        (1, 0.0), (1, 5.0), (1, 10.0), (1, 20.0), (1, 30.0),
        (2, 0.0), (2, 10.0), (2, 12.0),
    ]
    df = pd.DataFrame(rows, columns=[ID, TIME])
    df["PLATINUM"] = 1
    # Event well past every observation, so horizon censoring has room.
    df["t_platinum"] = 100.0
    if with_landmark_time:
        df["landmark_time"] = 10.0
    return df


def _horizon(df, horizon=50):
    return slo.add_post_landmark_horizon_columns(
        df,
        id_col=ID,
        time_col=TIME,
        event_col="PLATINUM",
        time_to_event_col="t_platinum",
        horizon=horizon,
    )


class TestLandmarkResolution:
    def test_uses_the_explicit_column_when_present(self):
        """The load-bearing test: with landmark_time=10 and horizon=50, the
        censoring time is 60 for BOTH patients, despite max(TIME) of 30 and 12."""
        out, ev, tt = _horizon(_frame(with_landmark_time=True))
        # Event at 100 is past landmark+horizon=60, so both are censored at 60.
        assert out[tt].unique().tolist() == [60.0]
        assert out[ev].eq(0).all()

    def test_inference_would_give_a_different_answer(self):
        """Anti-vacuity for the test above: prove the two paths disagree here,
        so the assertion is actually discriminating between them."""
        out_col, _, tt_col = _horizon(_frame(with_landmark_time=True))
        out_inf, _, tt_inf = _horizon(_frame(with_landmark_time=False))
        assert out_col[tt_col].unique().tolist() != sorted(
            out_inf[tt_inf].unique().tolist()
        )
        # Inference anchors per patient: 30+50 and 12+50.
        assert sorted(out_inf[tt_inf].unique().tolist()) == [62.0, 80.0]

    def test_falls_back_to_max_time_without_the_column(self):
        """The landmark frame has no landmark_time column and must keep working
        exactly as before -- that path is what every published run used."""
        out, _, tt = _horizon(_frame(with_landmark_time=False))
        assert not out.empty
        assert sorted(out[tt].unique().tolist()) == [62.0, 80.0]

    def test_landmark_frame_identity_holds(self):
        """On a frame that ends AT the landmark the two paths must agree, which
        is why the fallback was safe for the landmark arm."""
        df = _frame(with_landmark_time=True)
        landmark = df.loc[df[TIME] <= 10.0].copy()  # truncate at the landmark
        with_col, _, tt_a = _horizon(landmark)
        without_col, _, tt_b = _horizon(landmark.drop(columns=["landmark_time"]))
        assert with_col[tt_a].tolist() == without_col[tt_b].tolist()

    def test_rejects_a_landmark_that_varies_within_a_patient(self):
        df = _frame(with_landmark_time=True)
        df.loc[df.index[0], "landmark_time"] = 99.0
        with pytest.raises(ValueError, match="varies within a patient"):
            _horizon(df)

    def test_rejects_a_non_numeric_landmark(self):
        df = _frame(with_landmark_time=True)
        df["landmark_time"] = "not-a-number"
        with pytest.raises(ValueError, match="non-numeric or missing"):
            _horizon(df)


class TestObservationCut:
    def test_keeps_only_rows_at_or_before_the_cut(self):
        out = slo.truncate_observations_at_cut(
            _frame(with_landmark_time=True), id_col=ID, time_col=TIME, cut_days=5
        )
        # landmark 10 + 5 = 15: patient 1 loses 20 and 30, patient 2 keeps all.
        assert out[TIME].max() <= 15.0
        assert sorted(out.loc[out[ID] == 1, TIME].tolist()) == [0.0, 5.0, 10.0]
        assert sorted(out.loc[out[ID] == 2, TIME].tolist()) == [0.0, 10.0, 12.0]

    def test_a_later_cut_keeps_more(self):
        f = _frame(with_landmark_time=True)
        small = slo.truncate_observations_at_cut(
            f, id_col=ID, time_col=TIME, cut_days=5
        )
        large = slo.truncate_observations_at_cut(
            f, id_col=ID, time_col=TIME, cut_days=25
        )
        assert len(large) > len(small), "the sweep must actually vary the input"

    def test_cut_zero_keeps_history_up_to_the_landmark(self):
        """cut=0 is the landmark arm's input window, which is the sweep's
        natural baseline point."""
        out = slo.truncate_observations_at_cut(
            _frame(with_landmark_time=True), id_col=ID, time_col=TIME, cut_days=0
        )
        assert out[TIME].max() == 10.0

    def test_requires_the_explicit_landmark_column(self):
        """Inferring the landmark here would move each patient's cut point by
        the length of their own post-landmark history."""
        with pytest.raises(ValueError, match="landmark_time column"):
            slo.truncate_observations_at_cut(
                _frame(with_landmark_time=False),
                id_col=ID,
                time_col=TIME,
                cut_days=5,
            )

    def test_rejects_a_negative_cut(self):
        with pytest.raises(ValueError, match="non-negative"):
            slo.truncate_observations_at_cut(
                _frame(with_landmark_time=True), id_col=ID, time_col=TIME, cut_days=-1
            )

    def test_reports_dropped_patients(self, capsys):
        """A cut that silently discards patients would look like a clean run on
        a smaller, healthier sample."""
        df = _frame(with_landmark_time=True)
        # Patient 2's earliest observation is at TIME 0, so a landmark BELOW
        # that (cut = landmark + 0) precedes every row it has.
        df.loc[df[ID] == 2, "landmark_time"] = -5.0
        out = slo.truncate_observations_at_cut(
            df, id_col=ID, time_col=TIME, cut_days=0
        )
        assert out[ID].nunique() == 1
        assert out[ID].unique().tolist() == [1]
        assert "[warn]" in capsys.readouterr().out

    def test_raises_when_nothing_survives(self):
        df = _frame(with_landmark_time=True)
        # Every landmark below every observation time: nothing is at or before
        # the cut, for anyone.
        df["landmark_time"] = -1e6
        with pytest.raises(ValueError, match="left no rows"):
            slo.truncate_observations_at_cut(
                df, id_col=ID, time_col=TIME, cut_days=0
            )


class TestRunId:
    def test_no_cut_keeps_the_historical_spelling(self):
        """Existing cluster checkpoints are named this way; a rename would
        orphan them or trip prepare_run_artifacts."""
        assert (
            slo.default_run_id(config="platinum", landmark_day=90, cut_days=None)
            == "prostate_platinum_landmark90_v1"
        )

    def test_it_matches_the_pipelines_own_constructor(self):
        """compass_pipeline.longitudinal_run_id builds the same name for the
        no-cut case; derive it rather than restating the literal."""
        sys.path.insert(0, str(REPO / "COMPASS" / "survival_analysis"))
        from unittest import mock

        with mock.patch.object(Path, "mkdir", lambda *a, **k: None):
            import compass_pipeline as cp

        assert cp.longitudinal_run_id("platinum", 90) == slo.default_run_id(
            config="platinum", landmark_day=90, cut_days=None
        )

    def test_each_cut_point_gets_its_own_id(self):
        ids = {
            slo.default_run_id(config="platinum", landmark_day=0, cut_days=c)
            for c in (None, 0, 90, 180)
        }
        assert len(ids) == 4, "colliding ids would overwrite each other's fits"

    def test_cut_zero_is_distinct_from_no_cut(self):
        """They are different runs: no-cut reads the landmark frame, cut=0 reads
        the full frame truncated at the landmark."""
        assert slo.default_run_id(
            config="platinum", landmark_day=0, cut_days=0
        ) != slo.default_run_id(config="platinum", landmark_day=0, cut_days=None)


class TestInputSelection:
    def _inputs(self, tmp_path, *, stem, include_post_landmark):
        d = tmp_path / "inputs"
        d.mkdir(exist_ok=True)
        (d / f"{stem}_landmark0.csv").write_text(f"{ID},{TIME}\n1,0\n")
        (d / f"{stem}_landmark0_manifest.json").write_text(
            json.dumps(
                {
                    "id_col": ID,
                    "time_col": TIME,
                    "feat_cont": [],
                    "feat_cat": [],
                    "feat_reconstr": [],
                    "include_post_landmark": include_post_landmark,
                }
            )
        )
        return d

    def _args(self, inputs_dir, *, full_followup):
        return argparse.Namespace(
            inputs_dir=str(inputs_dir), landmark_day=0, full_followup=full_followup
        )

    def test_landmark_flag_reads_the_landmark_file(self, tmp_path):
        d = self._inputs(tmp_path, stem="longitudinal", include_post_landmark=False)
        _df, manifest = slo.load_longitudinal_inputs(
            self._args(d, full_followup=False)
        )
        assert manifest["include_post_landmark"] is False

    def test_full_flag_reads_the_full_file(self, tmp_path):
        d = self._inputs(
            tmp_path, stem="longitudinal_full", include_post_landmark=True
        )
        _df, manifest = slo.load_longitudinal_inputs(
            self._args(d, full_followup=True)
        )
        assert manifest["include_post_landmark"] is True

    def test_missing_full_inputs_name_the_build_flag(self, tmp_path):
        d = self._inputs(tmp_path, stem="longitudinal", include_post_landmark=False)
        with pytest.raises(FileNotFoundError, match="--longitudinal-full-followup"):
            slo.load_longitudinal_inputs(self._args(d, full_followup=True))

    def test_manifest_mismatch_is_refused(self, tmp_path):
        """A full-follow-up file whose manifest says otherwise (or the reverse)
        must fail loudly rather than be scored with the wrong anchor."""
        d = self._inputs(
            tmp_path, stem="longitudinal_full", include_post_landmark=False
        )
        with pytest.raises(ValueError, match="include_post_landmark"):
            slo.load_longitudinal_inputs(self._args(d, full_followup=True))

    def test_landmark_path_refuses_a_full_manifest(self, tmp_path):
        d = self._inputs(tmp_path, stem="longitudinal", include_post_landmark=True)
        with pytest.raises(ValueError, match="include_post_landmark"):
            slo.load_longitudinal_inputs(self._args(d, full_followup=False))

    def test_absent_flag_defaults_to_the_landmark_frame(self, tmp_path):
        """Programmatic callers build a Namespace by hand; absent means off."""
        d = self._inputs(tmp_path, stem="longitudinal", include_post_landmark=False)
        args = argparse.Namespace(inputs_dir=str(d), landmark_day=0)
        _df, manifest = slo.load_longitudinal_inputs(args)
        assert manifest["include_post_landmark"] is False


class TestTorchGating:
    def test_help_exits_zero_without_torch(self):
        """Invariant #7: the adapter's parser must work in an env with no torch
        and no external repo, so a cluster config can be checked cheaply."""
        import subprocess

        script = (
            REPO
            / "COMPASS"
            / "survival_analysis"
            / "multivariate_longitudinal"
            / "survlatent_ode.py"
        )
        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert "--full-followup" in proc.stdout
        assert "--observation-cut-days" in proc.stdout


class TestPipelineWiring:
    """The pipeline passes --run-id explicitly, so the sweep only works if that
    id and the metrics filename both carry the cut. They are built in two
    different functions, which is exactly how they drift apart."""

    @pytest.fixture
    def cp(self):
        from unittest import mock

        sys.path.insert(0, str(REPO / "COMPASS" / "survival_analysis"))
        with mock.patch.object(Path, "mkdir", lambda *a, **k: None):
            import compass_pipeline

        return compass_pipeline

    def _cmd(self, cp, **attrs):
        from unittest import mock

        run = {
            "label": "adt",
            "endpoint": "platinum",
            "cohort": "all",
            "inputs_dir": Path("/tmp/in"),
            "landmarks": [90],
        }
        with mock.patch.multiple(cp, **attrs):
            return [
                str(c)
                for c in cp.build_model_command(
                    "survlatent-ode", 90, "platinum", Path("/tmp/out"), run
                )
            ]

    def test_no_cut_command_is_unchanged(self, cp):
        """Default off: the historical run_id, and neither new flag present."""
        cmd = self._cmd(cp, SURVLATENT_OBSERVATION_CUT_DAYS=None,
                        SURVLATENT_FULL_FOLLOWUP=False)
        assert cmd[cmd.index("--run-id") + 1] == "prostate_platinum_landmark90_v1"
        assert "--observation-cut-days" not in cmd
        assert "--full-followup" not in cmd

    def test_full_followup_flag_alone(self, cp):
        cmd = self._cmd(cp, SURVLATENT_OBSERVATION_CUT_DAYS=None,
                        SURVLATENT_FULL_FOLLOWUP=True)
        assert "--full-followup" in cmd
        assert "--observation-cut-days" not in cmd

    def test_a_cut_carries_into_the_run_id(self, cp):
        cmd = self._cmd(cp, SURVLATENT_OBSERVATION_CUT_DAYS=180,
                        SURVLATENT_FULL_FOLLOWUP=False)
        assert cmd[cmd.index("--observation-cut-days") + 1] == "180"
        assert cmd[cmd.index("--run-id") + 1] == "prostate_platinum_landmark90_cut180_v1"

    def test_each_cut_gets_a_distinct_run_id(self, cp):
        """Otherwise sweep points share checkpoints and silently resume."""
        ids = {
            self._cmd(cp, SURVLATENT_OBSERVATION_CUT_DAYS=c,
                      SURVLATENT_FULL_FOLLOWUP=False)[
                self._cmd(cp, SURVLATENT_OBSERVATION_CUT_DAYS=c,
                          SURVLATENT_FULL_FOLLOWUP=False).index("--run-id") + 1
            ]
            for c in (None, 0, 90, 180)
        }
        assert len(ids) == 4

    def test_metrics_filename_tracks_the_run_id(self, cp):
        """The bug this catches: the summary reading the un-cut filename and
        reporting every swept run as missing."""
        from unittest import mock

        for cut in (None, 0, 180):
            with mock.patch.multiple(
                cp, SURVLATENT_OBSERVATION_CUT_DAYS=cut, SURVLATENT_FULL_FOLLOWUP=False
            ):
                cmd = self._cmd(
                    cp, SURVLATENT_OBSERVATION_CUT_DAYS=cut,
                    SURVLATENT_FULL_FOLLOWUP=False,
                )
                run_id = cmd[cmd.index("--run-id") + 1]
                fname = cp.longitudinal_metrics_filename(
                    "survlatent-ode", "platinum", 90, None
                )
                assert run_id in fname, f"cut={cut}: {fname} does not match {run_id}"

    def test_pipeline_and_adapter_agree_on_the_id(self, cp):
        """Two independent constructors of the same name."""
        for cut in (None, 0, 90, 180):
            assert cp.longitudinal_run_id(
                "platinum", 90, cut_days=cut
            ) == slo.default_run_id(config="platinum", landmark_day=90, cut_days=cut)
