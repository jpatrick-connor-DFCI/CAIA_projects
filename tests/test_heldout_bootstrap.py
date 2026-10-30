"""Held-out bootstrap: same metrics as the fits, paired across them, honest when it cannot score."""

import numpy as np
import pandas as pd
import pytest
from lifelines.utils import concordance_index

from survival_common.cox_engine import compute_ipcw_auc_t
from survival_common.heldout_bootstrap import (
    assert_reproduces_stored,
    bootstrap_heldout,
    load_test_predictions,
    percentile_ci,
    score_heldout,
    survival_matrix,
)
from survival_common.helper import compute_brier, survival_columns

ID_COL = "DFCI_MRN"
TIME_UNIT_DAYS = 7
MAX_TIME_UNIT = 200
HORIZONS = np.array([26.0, 52.0, 78.0, 104.0])
GRID = dict(horizons=HORIZONS, time_unit_days=TIME_UNIT_DAYS, max_time_unit=MAX_TIME_UNIT)


def _outcomes(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    event_time = 7.0 * rng.exponential(90.0 * np.exp(-signal))
    censor_time = 7.0 * rng.uniform(40.0, 180.0, n)
    return pd.DataFrame({
        ID_COL: np.arange(n),
        "duration_days": np.maximum(np.minimum(event_time, censor_time), 1.0),
        "event": (event_time <= censor_time).astype(int),
        "signal": signal,
    })


def _survival(risk: np.ndarray) -> np.ndarray:
    return np.exp(-np.outer(np.exp(risk), HORIZONS / 100.0))


def _fit(test: pd.DataFrame, noise: float, seed: int, *, with_survival: bool = True) -> pd.DataFrame:
    risk = test["signal"].to_numpy() + noise * np.random.default_rng(seed).normal(size=len(test))
    frame = test[[ID_COL, "duration_days", "event"]].assign(risk_score=risk)
    return frame.assign(**survival_columns(_survival(risk), HORIZONS)) if with_survival else frame


REFERENCE = _outcomes(500, seed=1)[["duration_days", "event"]]
TEST = _outcomes(250, seed=2)


def _bootstrap(fits, *, n_boot=40, seed=7):
    return bootstrap_heldout(
        fits, reference=REFERENCE, n_boot=n_boot, seed=seed, id_col=ID_COL, **GRID,
    )


def test_score_matches_the_engine_functions():
    fit = _fit(TEST, noise=0.5, seed=3)
    duration, event = fit["duration_days"].to_numpy(), fit["event"].to_numpy()
    risk = fit["risk_score"].to_numpy()
    scores = score_heldout(duration, event, risk, survival_matrix(fit, HORIZONS), reference=REFERENCE, **GRID)

    assert scores["test_c_index"] == pytest.approx(concordance_index(duration, -risk, event))
    mean_auc, _ = compute_ipcw_auc_t(
        fit, risk, duration_col="duration_days", event_col="event", reference_df=REFERENCE,
        time_unit_days=TIME_UNIT_DAYS, max_time_unit=MAX_TIME_UNIT, fixed_horizons=HORIZONS,
    )
    assert scores["test_mean_auc_t"] == pytest.approx(mean_auc)
    _, integrated_brier = compute_brier(
        train_event=REFERENCE["event"].to_numpy(),
        train_duration=np.ceil(REFERENCE["duration_days"].to_numpy() / TIME_UNIT_DAYS),
        eval_event=event, eval_duration=np.ceil(duration / TIME_UNIT_DAYS),
        surv_at_horizons=_survival(risk), **GRID,
    )
    assert scores["test_integrated_brier"] == pytest.approx(integrated_brier)
    assert all(np.isfinite(value) for value in scores.values())


def test_identical_fits_have_exactly_zero_paired_difference():
    fit = _fit(TEST, noise=0.5, seed=3)
    # Shuffling proves pairing follows the patient id, not the row order.
    _, draws = _bootstrap({"a": fit, "b": fit.sample(frac=1.0, random_state=0)})
    wide = draws.set_index(["fit", "draw"])
    for metric in ("test_c_index", "test_mean_auc_t", "test_integrated_brier"):
        assert wide.loc["a", metric].equals(wide.loc["b", metric])


def test_pairing_tightens_the_interval_for_correlated_fits():
    """Sharing patients removes their common variance from a difference."""
    a, b = _fit(TEST, noise=0.3, seed=3), _fit(TEST, noise=0.3, seed=4)
    _, draws = _bootstrap({"a": a, "b": b}, n_boot=200)
    _, other_draws = _bootstrap({"b": b}, n_boot=200, seed=8)
    by_fit = draws.set_index(["fit", "draw"])["test_c_index"]
    paired = by_fit.loc["a"] - by_fit.loc["b"]
    unpaired = by_fit.loc["a"] - other_draws.set_index("draw")["test_c_index"]
    assert paired.std() < 0.7 * unpaired.std()


def test_draws_are_reproducible_and_change_with_the_seed():
    fits = {"a": _fit(TEST, noise=0.5, seed=3)}
    _, first = _bootstrap(fits)
    _, again = _bootstrap(fits)
    _, other = _bootstrap(fits, seed=8)
    pd.testing.assert_frame_equal(first, again)
    assert not first["test_c_index"].equals(other["test_c_index"])


def test_interval_covers_the_estimate_and_narrows_with_more_patients():
    def width(n_patients):
        test = _outcomes(n_patients, seed=2)
        observed, draws = _bootstrap({"a": _fit(test, noise=0.5, seed=3)}, n_boot=200)
        low, high, n_valid = percentile_ci(draws["test_c_index"])
        assert n_valid == 200
        assert low < observed.loc[0, "test_c_index"] < high
        return high - low

    assert width(1000) < 0.7 * width(150)


def test_a_fit_without_survival_columns_still_gets_c_and_auc():
    observed, draws = _bootstrap({"a": _fit(TEST, noise=0.5, seed=3, with_survival=False)})
    assert np.isfinite(observed.loc[0, "test_c_index"])
    assert np.isfinite(observed.loc[0, "test_mean_auc_t"])
    assert np.isnan(observed.loc[0, "test_integrated_brier"])
    assert draws["test_integrated_brier"].isna().all()
    assert draws["test_c_index"].notna().all()


def test_degenerate_draws_are_nan_and_stay_aligned():
    """A rare outcome leaves some draws with no events at all."""
    fit = _fit(TEST, noise=0.5, seed=3).assign(event=0)
    fit.loc[fit.index[0], "event"] = 1
    _, draws = _bootstrap({"a": fit, "b": fit}, n_boot=60)
    assert len(draws) == 120
    assert draws["test_c_index"].isna().any()
    assert percentile_ci([np.nan] * 60)[2] == 0
    assert np.isnan(percentile_ci([0.5] * 9 + [np.nan] * 51)[0])


@pytest.mark.parametrize("change", ["patients", "outcomes", "duplicates"])
def test_incomparable_fits_are_refused(change):
    a, b = _fit(TEST, noise=0.5, seed=3), _fit(TEST, noise=0.5, seed=4)
    if change == "patients":
        b = b.iloc[1:]
    elif change == "outcomes":
        b = b.assign(event=1 - b["event"])
    else:
        a = pd.concat([a.iloc[:-1], a.iloc[[0]]], ignore_index=True)
        b = pd.concat([b.iloc[:-1], b.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="differ in test patients|Duplicate test patients"):
        _bootstrap({"a": a, "b": b})


def test_stored_metrics_must_reproduce():
    observed, _ = _bootstrap({"a": _fit(TEST, noise=0.5, seed=3)}, n_boot=1)
    row = observed.iloc[0]
    assert_reproduces_stored(row, row, context="same")
    with pytest.raises(ValueError, match="stale outputs"):
        assert_reproduces_stored(row, {**row, "test_c_index": row["test_c_index"] + 1e-4}, context="stale")
    # NaN on only one side is a mismatch for the metrics every fit can recompute.
    with pytest.raises(ValueError, match="test_mean_auc_t"):
        assert_reproduces_stored(row, {**row, "test_mean_auc_t": np.nan}, context="stale")
    # Older fits have a stored Brier and no way to recompute it.
    assert_reproduces_stored({**row, "test_integrated_brier": np.nan}, row, context="older fit")


@pytest.mark.parametrize("duration_col", ["duration_days", "duration"])
def test_load_keeps_test_rows_from_either_engine(tmp_path, duration_col):
    fit = _fit(TEST, noise=0.5, seed=3).rename(columns={"duration_days": duration_col})
    oof = fit.iloc[:5].drop(columns=survival_columns(_survival(np.zeros(1)), HORIZONS))
    pd.concat([fit.assign(dataset="test"), oof.assign(dataset="cv_oof")]).to_csv(tmp_path / "r.csv", index=False)
    loaded = load_test_predictions(tmp_path / "r.csv")
    assert len(loaded) == len(TEST)
    assert "duration_days" in loaded and loaded["dataset"].eq("test").all()
    np.testing.assert_allclose(survival_matrix(loaded, HORIZONS), _survival(fit["risk_score"].to_numpy()))
    assert survival_matrix(loaded, np.array([999.0])) is None
