"""Guards on DeepHit's CV hyperparameter selection.

Motivating defect: ``cv_run`` caught every per-fold exception into a ``note``
column, and its only failure guard counted folds whose *fit* succeeded
(``best_valid_loss``). Scoring could therefore fail in all of them -- it did,
via a missing ``args.auc_quantiles`` -- leaving the ranking column all-NaN and
the sort to fall through to its tie-break. CV then printed a confident
"CV chose hidden_dim=..." that carried no information at all.

That is the worst shape a bug can take here: no crash, no NaN in the headline
metrics, just silently arbitrary hyperparameters. These tests pin both halves --
the parser supplies what the engine reads, and an unrankable CV refuses to
report a selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from survival_common.longitudinal_runners import build_deephit_parser

engine = pytest.importorskip("survival_common.deephit_engine")


BASE_ARGS = [
    "--inputs-dir", "/nonexistent",
    "--output-dir", "/nonexistent",
]


class TestParserSuppliesWhatCvReads:
    """Every args attribute cv_run reads must survive a default parse."""

    def test_auc_quantiles_is_resolvable(self):
        args = build_deephit_parser().parse_args(BASE_ARGS)
        # cv_run reads this to pass compute_metrics a fallback quantile grid.
        # The parser does not define it, so the engine must tolerate its absence
        # rather than raising inside the per-fold try/except.
        quantiles = getattr(args, "auc_quantiles", engine.DEFAULT_AUC_QUANTILES)
        assert len(tuple(quantiles)) > 0

    @pytest.mark.parametrize(
        "attr",
        ["n_folds", "cv_hidden_dims", "cv_dropouts", "cv_lrs", "seed",
         "batch_size", "epochs", "patience", "min_delta", "weight_decay",
         "max_pred_window"],
    )
    def test_cv_required_attrs_exist(self, attr):
        args = build_deephit_parser().parse_args(BASE_ARGS)
        assert hasattr(args, attr), f"build_deephit_parser does not define {attr}"


class TestUnrankableCvRaises:
    """An all-NaN ranking column must fail loudly, not pick a tie-break winner.

    Drives the real :func:`cv_run` rather than a copy of its selection tail, so
    the guard is pinned where it actually ships. Folds are made to fail by
    monkeypatching ``train_evaluate`` -- which is precisely how the original bug
    manifested: the fit succeeded and the *scoring* raised inside the per-fold
    try/except.
    """

    def _args(self):
        return build_deephit_parser().parse_args(
            BASE_ARGS + [
                "--n-folds", "2",
                "--cv-hidden-dims", "16", "32",
                "--cv-dropouts", "0.1",
                "--cv-lrs", "1e-3",
                "--epochs", "1",
            ]
        )

    def _static(self, n=40) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {
                "PLATINUM": rng.integers(0, 2, n),
                "DEATH": rng.integers(0, 2, n),
            },
            index=[str(1000 + i) for i in range(n)],
        )

    def _run(self, monkeypatch, *, boom: bool):
        if boom:
            # Reproduce the real failure shape: the FIT succeeds (so
            # best_valid_loss is finite and the older n_valid_folds guard is
            # satisfied) and the SCORING raises. That is the case that used to
            # slip through to a confident, meaningless selection.
            def _fit_ok(**kwargs):
                n = len(kwargs["eval_ids"])
                return (
                    pd.DataFrame({"DFCI_MRN": sorted(kwargs["eval_ids"]),
                                  "label": np.ones(n, dtype=int)}),
                    [{"epoch": 1}],
                    1.0,
                )

            def _score_explodes(*a, **kw):
                raise AttributeError(
                    "'Namespace' object has no attribute 'auc_quantiles'"
                )

            monkeypatch.setattr(engine, "train_evaluate", _fit_ok)
            monkeypatch.setattr(engine, "compute_metrics", _score_explodes)

        return engine.cv_run(
            df=pd.DataFrame({"DFCI_MRN": [], "TIME": []}),
            id_col="DFCI_MRN",
            time_col="TIME",
            feature_cols=["PSA"],
            targets=pd.DataFrame(),
            train_val_static=self._static(),
            args=self._args(),
            n_events=1,
            event_names=["platinum"],
            fixed_horizons_by_event={"platinum": np.asarray([10.0])},
        )

    def test_all_folds_failing_raises(self, monkeypatch):
        with pytest.raises(RuntimeError, match="no usable C-index"):
            self._run(monkeypatch, boom=True)

    def test_the_error_names_the_underlying_fold_failure(self, monkeypatch):
        """The message must carry the cause, or the next person debugs blind."""
        with pytest.raises(RuntimeError, match="auc_quantiles"):
            self._run(monkeypatch, boom=True)

    def test_guard_does_not_fire_when_folds_score(self, monkeypatch):
        """Anti-vacuity: a CV that produces real C-indices must still select."""
        def _ok(**kwargs):
            n = len(kwargs["eval_ids"])
            rng = np.random.default_rng(len(kwargs["train_ids"]))
            pred = pd.DataFrame({
                "DFCI_MRN": sorted(kwargs["eval_ids"]),
                "label": rng.integers(0, 2, n),
                "duration": rng.uniform(1, 20, n),
                "duration_bin": rng.integers(1, 10, n),
                "event_1_risk_total": rng.uniform(0, 1, n),
                "event_1_risk_h10": rng.uniform(0, 1, n),
            })
            return pred, [{"epoch": 1}], 1.0

        monkeypatch.setattr(engine, "train_evaluate", _ok)
        # targets must cover the fold ids for the fold_train_targets slice.
        targets = pd.DataFrame(
            {"label": 1, "duration": 5.0, "duration_bin": 5},
            index=[str(1000 + i) for i in range(40)],
        )
        args = self._args()
        fold_df, cv_summary, best_row = engine.cv_run(
            df=pd.DataFrame({"DFCI_MRN": [], "TIME": []}),
            id_col="DFCI_MRN",
            time_col="TIME",
            feature_cols=["PSA"],
            targets=targets,
            train_val_static=self._static(),
            args=args,
            n_events=1,
            event_names=["platinum"],
            fixed_horizons_by_event={"platinum": np.asarray([10.0])},
        )
        assert not fold_df.empty
        assert best_row["hidden_dim"] in (16, 32)
