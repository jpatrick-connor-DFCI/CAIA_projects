"""Phase 4: per-prediction-time metrics for the dynamic arm.

Two contracts:

1. **Invariant #9 must survive.** The by-time rows go to their own file. The
   canonical ``*_metrics.csv`` stays one row per cause, so every existing
   reader (summarize_longitudinal_outputs, the figure pipeline) keeps working
   and the dynamic arm's headline number stays comparable to Cox/XGBoost.

2. **The by-time row at the landmark must equal the headline row.** This is
   what anchors the whole series: if the two disagree, either the by-time slice
   or the landmark slice is selecting the wrong rows, and the series cannot be
   read against the other arms. The implementation gets this by construction --
   compute_metrics_by_time calls the same compute_metrics -- and this test is
   what keeps it that way.

Also pinned: the underpowered gate. Late prediction times always thin out, and
a C-index computed on a handful of patients must not be presented as a
comparable number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

engine = pytest.importorskip("survival_common.deephit_engine")
pytest.importorskip("lifelines")

ID_COL = "DFCI_MRN"
EVENTS = ["platinum"]
HORIZONS = {"platinum": np.asarray([4.0, 8.0, 13.0, 26.0])}


def _pred_block(mrns, pred_time, *, rng, event_frac=0.4):
    """One prediction time's worth of rows, with informative risk."""
    n = len(mrns)
    label = (rng.random(n) < event_frac).astype(int)
    duration = rng.uniform(2, 30, n)
    # Risk correlated with the event so the C-index is not degenerate.
    risk = np.clip(0.5 * label + rng.normal(0, 0.25, n), 0, 1)
    row = {
        ID_COL: list(mrns),
        "TIME": float(pred_time),
        "label": label,
        "duration": duration,
        "duration_bin": np.ceil(duration).astype(int),
        "event_1_risk_total": risk,
    }
    for h in HORIZONS["platinum"]:
        row[f"event_1_risk_h{int(h)}"] = np.clip(risk + rng.normal(0, 0.05, n), 0, 1)
    return pd.DataFrame(row)


@pytest.fixture
def dynamic_pred() -> pd.DataFrame:
    """Prediction times 0..4 with a risk set that thins out, as real ones do."""
    rng = np.random.default_rng(0)
    sizes = {0: 200, 1: 160, 2: 120, 3: 60, 4: 8}  # 4 is deliberately tiny
    blocks = [
        _pred_block(range(1000, 1000 + n), t, rng=rng) for t, n in sizes.items()
    ]
    return pd.concat(blocks, ignore_index=True)


@pytest.fixture
def train_val_targets() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 400
    return pd.DataFrame(
        {
            "label": (rng.random(n) < 0.4).astype(int),
            "duration": rng.uniform(2, 40, n),
        },
        index=[str(i) for i in range(n)],
    )


def _by_time(dynamic_pred, train_val_targets, **kw):
    return engine.compute_metrics_by_time(
        dynamic_pred,
        event_names=EVENTS,
        train_val_targets=train_val_targets,
        fixed_horizons_by_event=HORIZONS,
        id_col=ID_COL,
        time_col="TIME",
        **kw,
    )


class TestByTimeShape:
    def test_one_row_per_prediction_time_and_cause(self, dynamic_pred, train_val_targets):
        out = _by_time(dynamic_pred, train_val_targets)
        assert len(out) == dynamic_pred["TIME"].nunique() * len(EVENTS)

    def test_carries_prediction_time_columns(self, dynamic_pred, train_val_targets):
        out = _by_time(dynamic_pred, train_val_targets)
        for col in ("prediction_time", "prediction_day", "n_at_risk", "note"):
            assert col in out.columns

    def test_prediction_day_scales_by_time_unit(self, dynamic_pred, train_val_targets):
        out = _by_time(dynamic_pred, train_val_targets, time_unit_days=7)
        assert np.allclose(out["prediction_day"], out["prediction_time"] * 7)

    def test_sorted_by_prediction_time(self, dynamic_pred, train_val_targets):
        out = _by_time(dynamic_pred, train_val_targets)
        for _, g in out.groupby("endpoint"):
            assert g["prediction_time"].is_monotonic_increasing

    def test_empty_input_gives_empty_frame(self, train_val_targets):
        out = _by_time(pd.DataFrame(), train_val_targets)
        assert out.empty

    def test_missing_time_column_raises(self, dynamic_pred, train_val_targets):
        with pytest.raises(ValueError, match="no 'TIME' column|TIME"):
            _by_time(dynamic_pred.drop(columns=["TIME"]), train_val_targets)

    def test_n_at_risk_tracks_the_block_size(self, dynamic_pred, train_val_targets):
        out = _by_time(dynamic_pred, train_val_targets)
        expected = dynamic_pred.groupby("TIME").size()
        got = out.set_index("prediction_time")["n_at_risk"]
        for t, n in expected.items():
            assert int(got.loc[float(t)]) == int(n)


class TestAnchoringIdentity:
    """The by-time row at a prediction time must equal compute_metrics on that
    same slice. This is what lets the series be read against the other arms."""

    @pytest.mark.parametrize("pred_time", [0, 1, 2, 3])
    def test_matches_compute_metrics_on_the_same_slice(
        self, dynamic_pred, train_val_targets, pred_time
    ):
        block = dynamic_pred.loc[dynamic_pred["TIME"] == float(pred_time)]
        direct, _ = engine.compute_metrics(
            block,
            event_names=EVENTS,
            train_val_targets=train_val_targets,
            quantiles=engine.DEFAULT_AUC_QUANTILES,
            fixed_horizons_by_event=HORIZONS,
        )
        out = _by_time(dynamic_pred, train_val_targets)
        row = out.loc[out["prediction_time"] == float(pred_time)].iloc[0]
        assert row["test_c_index"] == pytest.approx(
            float(direct.iloc[0]["test_c_index"]), nan_ok=True
        )
        assert row["n_test"] == int(direct.iloc[0]["n_test"])
        assert row["n_events_test"] == int(direct.iloc[0]["n_events_test"])


class TestUnderpoweredGate:
    def test_tiny_risk_set_is_gated_not_reported(self, dynamic_pred, train_val_targets):
        """Prediction time 4 has 8 patients; its C-index must not be reported."""
        out = _by_time(dynamic_pred, train_val_targets)
        row = out.loc[out["prediction_time"] == 4.0].iloc[0]
        assert np.isnan(row["test_c_index"])
        assert row["note"].startswith("underpowered")

    def test_gated_rows_are_kept_not_dropped(self, dynamic_pred, train_val_targets):
        """A gap in the series is information; the row must still be present
        with its counts so a reader can see WHY it is missing."""
        out = _by_time(dynamic_pred, train_val_targets)
        assert (out["prediction_time"] == 4.0).any()
        row = out.loc[out["prediction_time"] == 4.0].iloc[0]
        assert row["n_at_risk"] == 8

    def test_well_powered_rows_report_a_number(self, dynamic_pred, train_val_targets):
        out = _by_time(dynamic_pred, train_val_targets)
        healthy = out.loc[out["prediction_time"] <= 3.0]
        assert healthy["note"].eq("").all()
        assert healthy["test_c_index"].notna().all(), "anti-vacuity: all gated"

    def test_the_gate_is_configurable(self, dynamic_pred, train_val_targets):
        """Raising the floor must gate more rows -- proves the gate is live."""
        strict = _by_time(dynamic_pred, train_val_targets, min_risk_set=150)
        gated = strict["note"].str.startswith("underpowered").sum()
        assert gated >= 3  # times 2, 3 and 4 fall below 150


class TestInvariantNine:
    """The canonical metrics file must not gain a prediction_time axis."""

    def test_by_time_is_a_separate_file(self):
        src = (
            __import__("pathlib").Path(__file__).resolve().parents[1]
            / "survival_common"
            / "longitudinal_runners.py"
        ).read_text()
        assert 'f"{prefix}_metrics_by_time_{config_tag}.csv"' in src
        # ...and it is a DIFFERENT path from the canonical metrics file.
        assert 'f"{prefix}_metrics_{config_tag}.csv"' in src

    def test_canonical_metrics_has_no_prediction_time(
        self, dynamic_pred, train_val_targets
    ):
        """compute_metrics -- which writes *_metrics.csv -- must stay per-cause
        only, with no prediction-time axis."""
        landmark = dynamic_pred.loc[dynamic_pred["TIME"] == 0.0]
        metrics, _ = engine.compute_metrics(
            landmark,
            event_names=EVENTS,
            train_val_targets=train_val_targets,
            quantiles=engine.DEFAULT_AUC_QUANTILES,
            fixed_horizons_by_event=HORIZONS,
        )
        assert len(metrics) == len(EVENTS)
        assert "prediction_time" not in metrics.columns
