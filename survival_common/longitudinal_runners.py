"""Shared CLI and orchestration for the Dynamic-DeepHit longitudinal arm.

Mirrors ``cox_runners.py`` / ``xgboost_runners.py`` in shape, but does not
reuse ``add_common_cox_args``: that binds ``--endpoints`` to
``choices=list(cox.ENDPOINTS)`` (platinum-only), which is the wrong axis
here -- these models take ``--config {platinum,competing}`` and run one
landmark per invocation (not a list). See
``survival_common.longitudinal_targets`` for config/target semantics and
``survival_common.deephit_engine`` for the torch-gated model itself.
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from survival_common import cohort, deephit_engine as engine
from survival_common.helper import assert_no_test_leakage, resolve_auc_max_time_units
from survival_common.longitudinal_targets import (
    LONGITUDINAL_CONFIGS,
    manifest_horizons_for_config,
    patient_targets,
    resolve_config,
)

BUILD_MANIFEST_FILENAME = "build_manifest.json"
MIN_LONGITUDINAL_SCHEMA_VERSION = 1
MIN_AUC_TIMELINE_SCHEMA_VERSION = 3


def add_common_longitudinal_args(parser: argparse.ArgumentParser) -> None:
    """~25 lines of intentional duplication with add_common_cox_args.

    Does not import cox_aggregated / cox_runners at all -- this arm's axis is
    --config, not --endpoints, and it operates on a single --landmark-day per
    invocation rather than a list, so add_common_cox_args does not fit.
    """
    parser.add_argument(
        "--id-col",
        default=cohort.ID_COL,
        help=f"Patient identifier column name (default: {cohort.ID_COL}).",
    )
    parser.add_argument(
        "--age-col",
        default=cohort.AGE_COL,
        help=f"Age covariate column name (default: {cohort.AGE_COL}).",
    )
    parser.add_argument(
        "--inputs-dir",
        required=True,
        help="Directory containing prebuilt inputs from build_prediction_inputs.py.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for Dynamic-DeepHit result CSVs.",
    )
    parser.add_argument(
        "--landmark-day",
        type=int,
        default=0,
        help=(
            "Single landmark to analyze (singular -- one model fit per "
            "invocation). Resolves to longitudinal_landmark{D}.csv in --inputs-dir."
        ),
    )
    parser.add_argument(
        "--config",
        choices=sorted(LONGITUDINAL_CONFIGS),
        default="platinum",
        help="platinum (death censored, comparable to Cox/XGBoost) or competing (platinum+death).",
    )


def build_deephit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dynamic-DeepHit recurrent competing-risks model for longitudinal lab histories."
    )
    add_common_longitudinal_args(parser)
    parser.add_argument(
        "--max-pred-window",
        type=int,
        default=engine.DEFAULT_MAX_PRED_WINDOW,
        help=(
            "Discrete prediction window in input time bins. Must cover the "
            "largest manifest AUC(t) horizon at this landmark."
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=engine.DEFAULT_SEED)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--n-folds", type=int, default=engine.DEFAULT_N_FOLDS)
    parser.add_argument(
        "--cv-hidden-dims", nargs="+", type=int, default=list(engine.DEFAULT_CV_HIDDEN_DIMS)
    )
    parser.add_argument(
        "--cv-dropouts", nargs="+", type=float, default=list(engine.DEFAULT_CV_DROPOUTS)
    )
    parser.add_argument("--cv-lrs", nargs="+", type=float, default=list(engine.DEFAULT_CV_LRS))
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Skip 5-fold CV; fit a single model with --hidden-dim/--dropout/--lr.",
    )
    return parser


def load_longitudinal_inputs(args: Namespace) -> tuple[pd.DataFrame, dict, dict]:
    """Load the per-landmark longitudinal CSV + its manifest + the shared build manifest.

    Asserts the additive-only schema contract (plan Sec.5): the shared
    build_manifest.json must declare longitudinal_schema_version and
    auc_timeline_schema_version at or above what this runner was written
    against, so a stale prediction-inputs directory fails loudly rather than
    silently mis-scoring.
    """
    inputs_dir = Path(args.inputs_dir)
    if not inputs_dir.exists():
        raise FileNotFoundError(
            f"Inputs dir {inputs_dir} not found. Run build_prediction_inputs.py first."
        )
    landmark_day = int(args.landmark_day)
    input_csv = inputs_dir / f"longitudinal_landmark{landmark_day}.csv"
    manifest_path = inputs_dir / f"longitudinal_landmark{landmark_day}_manifest.json"
    build_manifest_path = inputs_dir / BUILD_MANIFEST_FILENAME
    for path in (input_csv, manifest_path, build_manifest_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run build_prediction_inputs.py --build-longitudinal first."
            )

    manifest = json.loads(manifest_path.read_text())
    build_manifest = json.loads(build_manifest_path.read_text())

    longitudinal_schema_version = int(build_manifest.get("longitudinal_schema_version", 0))
    if longitudinal_schema_version < MIN_LONGITUDINAL_SCHEMA_VERSION:
        raise ValueError(
            f"build_manifest.json's longitudinal_schema_version="
            f"{longitudinal_schema_version} is older than this runner requires "
            f"(>= {MIN_LONGITUDINAL_SCHEMA_VERSION}). Rebuild prediction inputs."
        )
    auc_timeline_schema_version = int(build_manifest.get("auc_timeline_schema_version", 0))
    if auc_timeline_schema_version < MIN_AUC_TIMELINE_SCHEMA_VERSION:
        raise ValueError(
            f"build_manifest.json's auc_timeline_schema_version="
            f"{auc_timeline_schema_version} is older than this runner requires "
            f"(>= {MIN_AUC_TIMELINE_SCHEMA_VERSION}). Rebuild prediction inputs."
        )
    max_followup_days = build_manifest.get("max_followup_days")
    print(
        f"Loading longitudinal inputs from {input_csv} "
        f"(landmark=+{landmark_day}d, max_followup_days={max_followup_days})"
    )

    df = pd.read_csv(input_csv, low_memory=False)
    return df, manifest, build_manifest


def write_deephit_outputs(
    *,
    output_dir: Path,
    config: str,
    pred: pd.DataFrame,
    metrics: pd.DataFrame,
    auc_t: pd.DataFrame,
    brier_t: pd.DataFrame,
    fold_df: pd.DataFrame,
    cv_summary_df: pd.DataFrame,
    run_manifest: dict,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_tag = config
    pred_path = output_dir / f"dynamic_deephit_patient_risks_{config_tag}.csv"
    metrics_path = output_dir / f"dynamic_deephit_metrics_{config_tag}.csv"
    auc_path = output_dir / f"dynamic_deephit_auc_t_{config_tag}.csv"
    brier_path = output_dir / f"dynamic_deephit_brier_{config_tag}.csv"
    cv_folds_path = output_dir / f"dynamic_deephit_cv_folds_{config_tag}.csv"
    cv_summary_path = output_dir / f"dynamic_deephit_cv_summary_{config_tag}.csv"
    manifest_out_path = output_dir / f"dynamic_deephit_manifest_{config_tag}.json"

    pred.to_csv(pred_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    auc_t.to_csv(auc_path, index=False)
    saved = [metrics_path, auc_path, pred_path]
    if not brier_t.empty:
        brier_t.to_csv(brier_path, index=False)
        saved.append(brier_path)
    if not fold_df.empty:
        fold_df.to_csv(cv_folds_path, index=False)
        saved.append(cv_folds_path)
    if not cv_summary_df.empty:
        cv_summary_df.to_csv(cv_summary_path, index=False)
        saved.append(cv_summary_path)
    manifest_out_path.write_text(json.dumps(run_manifest, indent=2))
    saved.append(manifest_out_path)

    print("\nSaved:")
    for path in saved:
        print(f"  {path}")
    return saved


def run_deephit(args: Namespace) -> None:
    engine.require_torch()
    cohort.configure_id_columns(args.id_col, args.age_col)
    engine.set_seed(args.seed)

    df, manifest, build_manifest = load_longitudinal_inputs(args)
    landmark_day = int(args.landmark_day)
    id_col = manifest["id_col"]
    time_col = manifest["time_col"]
    feature_cols = list(manifest["feat_cont"])

    event_config = resolve_config(args.config, manifest)
    event_cols = event_config.event_cols
    time_cols = event_config.time_cols
    event_names = event_config.event_names

    auc_max_time_units = resolve_auc_max_time_units(build_manifest, None)
    landmark_horizons = build_manifest.get("auc_horizons_by_landmark", {}).get(str(landmark_day))
    if landmark_horizons is None:
        raise KeyError(
            f"build_manifest.json has no auc_horizons_by_landmark entry for landmark +{landmark_day}d."
        )
    max_manifest_horizon = max((h for hs in landmark_horizons.values() for h in hs), default=0)
    if args.max_pred_window < max_manifest_horizon:
        raise ValueError(
            f"--max-pred-window={args.max_pred_window} is shorter than the largest "
            f"manifest horizon ({max_manifest_horizon}) at landmark +{landmark_day}d. "
            "Increase it so DeepHit's AUC grid is comparable to Cox/XGBoost."
        )

    targets = patient_targets(
        df,
        id_col=id_col,
        time_col=time_col,
        event_cols=event_cols,
        time_cols=time_cols,
        max_pred_window=args.max_pred_window,
    )

    patient_split = df.groupby(id_col)["split"].first().astype(str)
    train_ids = set(patient_split.index[patient_split.eq("train")].map(str))
    valid_ids = set(patient_split.index[patient_split.eq("valid")].map(str))
    test_ids = set(patient_split.index[patient_split.eq("test")].map(str))
    train_val_ids = train_ids.union(valid_ids)
    assert_no_test_leakage(
        test_mrns=test_ids,
        train_mrns=train_val_ids,
        context=f"longitudinal_runners.run_deephit[{args.config}]",
    )

    patient_static = df.groupby(id_col)[["PLATINUM", "DEATH"]].first().astype(int)
    patient_static.index = patient_static.index.astype(str)
    train_val_static = patient_static.loc[
        patient_static.index.intersection(sorted(train_val_ids))
    ].copy()

    train_val_targets = targets.loc[targets.index.map(str).isin(train_val_ids)].copy()
    fixed_horizons_by_event = manifest_horizons_for_config(
        build_manifest, landmark_day, event_names, max_pred_window=args.max_pred_window
    )
    for event_name in event_names:
        if event_name not in fixed_horizons_by_event:
            raise KeyError(
                f"build_manifest.json has no usable horizons for event {event_name!r} "
                f"at landmark +{landmark_day}d within --max-pred-window={args.max_pred_window}."
            )

    fold_df = pd.DataFrame()
    cv_summary_df = pd.DataFrame()
    chosen: dict | None = None
    if not args.no_cv:
        n_combos = len(args.cv_hidden_dims) * len(args.cv_dropouts) * len(args.cv_lrs)
        print(
            f"Running {args.n_folds}-fold CV on {len(train_val_ids)} train_val patients "
            f"({n_combos} combos) for config={args.config} ..."
        )
        fold_df, cv_summary_df, best_row = engine.cv_run(
            df=df,
            id_col=id_col,
            time_col=time_col,
            feature_cols=feature_cols,
            targets=targets,
            train_val_static=train_val_static,
            args=args,
            n_events=len(event_names),
            event_names=event_names,
            fixed_horizons_by_event=fixed_horizons_by_event,
            config_label=args.config,
        )
        chosen = {
            "hidden_dim": int(best_row["hidden_dim"]),
            "dropout": float(best_row["dropout"]),
            "lr": float(best_row["lr"]),
            "cv_stratification": str(best_row.get("cv_stratification", "")),
        }
        for event_name in event_names:
            chosen[f"cv_mean_c_index__{event_name}"] = float(
                best_row.get(f"cv_mean_c_index__{event_name}", np.nan)
            )
            chosen[f"cv_mean_auc_t__{event_name}"] = float(
                best_row.get(f"cv_mean_auc_t__{event_name}", np.nan)
            )
            chosen[f"cv_mean_integrated_brier__{event_name}"] = float(
                best_row.get(f"cv_mean_integrated_brier__{event_name}", np.nan)
            )
        print(
            f"CV chose hidden_dim={chosen['hidden_dim']} "
            f"dropout={chosen['dropout']:g} lr={chosen['lr']:g}"
        )

    final_hidden_dim = chosen["hidden_dim"] if chosen is not None else args.hidden_dim
    final_dropout = chosen["dropout"] if chosen is not None else args.dropout
    final_lr = chosen["lr"] if chosen is not None else args.lr

    pred, history, best_valid = engine.train_evaluate(
        df=df,
        id_col=id_col,
        time_col=time_col,
        feature_cols=feature_cols,
        targets=targets,
        train_ids=train_ids,
        valid_ids=valid_ids,
        eval_ids=test_ids,
        args=args,
        n_events=len(event_names),
        horizon=args.max_pred_window,
        hidden_dim=final_hidden_dim,
        dropout=final_dropout,
        lr=final_lr,
        seed=args.seed,
    )

    time_unit_days = int(manifest["time_unit_days"])
    metrics, auc_t = engine.compute_metrics(
        pred,
        event_names=event_names,
        train_val_targets=train_val_targets,
        quantiles=tuple(engine.DEFAULT_AUC_QUANTILES),
        fixed_horizons_by_event=fixed_horizons_by_event,
    )
    brier_t, integrated_brier_by_event = engine.compute_brier_for_pred(
        pred,
        event_names=event_names,
        train_val_targets=train_val_targets,
        horizons_by_event=fixed_horizons_by_event,
        time_unit_days=time_unit_days,
    )
    metrics = metrics.copy()
    metrics["integrated_brier"] = metrics["event"].map(
        lambda name: integrated_brier_by_event.get(name, float("nan"))
    )
    metrics["selected_hidden_dim"] = final_hidden_dim
    metrics["selected_dropout"] = final_dropout
    metrics["selected_lr"] = final_lr
    metrics["config"] = args.config
    metrics["landmark_days"] = landmark_day

    run_manifest = {
        "inputs_dir": str(args.inputs_dir),
        "landmark_day": landmark_day,
        "config": args.config,
        "event_names": event_names,
        "feature_cols": feature_cols,
        "horizon_source": "auc_horizons_by_landmark[landmark]['platinum']",
        "auc_horizons_by_event": {
            event_name: [float(v) for v in horizons]
            for event_name, horizons in fixed_horizons_by_event.items()
        },
        "max_pred_window": args.max_pred_window,
        "seed": args.seed,
        "split_counts": {
            "train": len(train_ids),
            "valid": len(valid_ids),
            "test": len(test_ids),
        },
        "best_valid_loss": best_valid,
        "selected_hyperparameters": {
            "hidden_dim": final_hidden_dim,
            "dropout": final_dropout,
            "lr": final_lr,
            "selected_via_cv": chosen is not None,
            "cv_summary": chosen,
        },
        "integrated_brier_by_event": integrated_brier_by_event,
        "history": history,
    }

    write_deephit_outputs(
        output_dir=Path(args.output_dir),
        config=args.config,
        pred=pred,
        metrics=metrics,
        auc_t=auc_t,
        brier_t=brier_t,
        fold_df=fold_df,
        cv_summary_df=cv_summary_df,
        run_manifest=run_manifest,
    )
