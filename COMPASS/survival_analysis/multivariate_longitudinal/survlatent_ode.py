"""Train and evaluate SurvLatent ODE on the person-period longitudinal input
produced by ``build_prediction_inputs.py --build-longitudinal``.

Assumes:
  * The bundled itmoon7/survlatent_ode checkout is importable (override its path
    with --survlatent-repo; the script adds it to sys.path and chdirs into it so that
    model_performance/<run_id>/ checkpoints land inside the repo workspace).
  * The associated conda env (from survlatent_ode_conda.yml) is active.

Reads longitudinal_landmark{D}.csv + its manifest from --inputs-dir, matching
the same --inputs-dir/--landmark-day convention as dynamic_deephit.py (see
survival_common/longitudinal_runners.py). Emits a test-set metrics CSV and the
cause-specific CIF tensor for head-to-head comparison with the Cox/XGBoost/
Dynamic-DeepHit arms.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent               # .../multivariate_longitudinal
SURVIVAL_DIR = THIS_DIR.parent                            # .../survival_analysis
SURVIVAL_PARENT = SURVIVAL_DIR.parent                      # .../COMPASS
REPO_ROOT = SURVIVAL_PARENT.parent
for _p in (str(REPO_ROOT), str(SURVIVAL_PARENT), str(SURVIVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from survival_common.helper import assert_no_test_leakage  # noqa: E402
from survival_common.longitudinal_targets import LONGITUDINAL_CONFIGS  # noqa: E402

DEFAULT_LR = 0.01
DEFAULT_SURV_LOSS_SCALE = 10.0
DEFAULT_WAIT_UNTIL_FULL_SURV_LOSS = 15
DEFAULT_MAX_PRED_WINDOW = 260
DEFAULT_SURVLATENT_REPO = SURVIVAL_DIR / "survlatent_ode_repo"

# Event configuration, mirroring survival_common.longitudinal_targets (no
# death-alone config -- see that module).
#
# For single-event configs, event_col / time_to_event_col are strings (not
# lists of length 1) because pre_process_data dispatches on isinstance(...,
# list) and a list-of-one would land in the competing-risks branch with
# n_events==1.
EVENT_CONFIGS = {
    "platinum": {
        "event_col": "PLATINUM",
        "time_to_event_col": "t_platinum",
    },
    "competing": {
        "event_col": ["PLATINUM", "DEATH"],
        "time_to_event_col": ["t_platinum", "t_death"],
    },
    "nepc": {
        "event_col": "NEPC",
        "time_to_event_col": "t_nepc",
    },
    "nepc_competing": {
        "event_col": ["NEPC", "DEATH"],
        "time_to_event_col": ["t_nepc", "t_death"],
    },
}
assert set(EVENT_CONFIGS) == set(LONGITUDINAL_CONFIGS), (
    "EVENT_CONFIGS must stay in lockstep with survival_common.longitudinal_targets."
    "LONGITUDINAL_CONFIGS so both models take identical --config values."
)
for _name, _spec in EVENT_CONFIGS.items():
    # Also pin the columns, not just the config names: a config that agreed by
    # name but disagreed on which event it reads would make the two models'
    # identically-labelled outputs mean different things. Normalize the
    # string-vs-list dispatch convention above before comparing.
    _events = _spec["event_col"]
    _times = _spec["time_to_event_col"]
    _events = [_events] if isinstance(_events, str) else list(_events)
    _times = [_times] if isinstance(_times, str) else list(_times)
    assert _events == LONGITUDINAL_CONFIGS[_name]["event_cols"], (
        f"EVENT_CONFIGS[{_name!r}] event columns {_events} disagree with "
        f"LONGITUDINAL_CONFIGS[{_name!r}] {LONGITUDINAL_CONFIGS[_name]['event_cols']}."
    )
    assert _times == LONGITUDINAL_CONFIGS[_name]["time_cols"], (
        f"EVENT_CONFIGS[{_name!r}] time columns {_times} disagree with "
        f"LONGITUDINAL_CONFIGS[{_name!r}] {LONGITUDINAL_CONFIGS[_name]['time_cols']}."
    )


def load_split(
    df: pd.DataFrame, split: str, *, id_col: str, time_col: str
) -> pd.DataFrame:
    sub = df.loc[df["split"] == split].copy()
    if sub.empty:
        raise ValueError(f"Split '{split}' is empty.")
    return sub.sort_values([id_col, time_col], ascending=(True, True)).drop(columns=["split"])


def add_post_landmark_horizon_columns(
    df: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    event_col: str | list[str],
    time_to_event_col: str | list[str],
    horizon: int,
) -> tuple[pd.DataFrame, str | list[str], str | list[str]]:
    """Censor each endpoint at first-treatment-start + horizon.

    SurvLatent's max_pred_window is absolute from TIME=0, but this application
    evaluates a post-landmark horizon from each patient's landmark. We therefore
    add per-patient horizon-censored event/time columns and reserve the larger
    absolute model window only for making room for pre-landmark history.
    """
    if horizon <= 0:
        raise ValueError("--max-pred-window must be positive.")

    event_cols = event_col if isinstance(event_col, list) else [event_col]
    time_cols = time_to_event_col if isinstance(time_to_event_col, list) else [time_to_event_col]
    if len(event_cols) != len(time_cols):
        raise ValueError("event_col and time_to_event_col must have matching lengths.")

    adjusted = df.copy()
    landmark_time = adjusted.groupby(id_col)[time_col].transform("max").astype(float)
    if landmark_time.isna().any():
        raise ValueError("Unable to infer landmark time from patient TIME maxima.")

    adjusted_event_cols: list[str] = []
    adjusted_time_cols: list[str] = []
    for raw_event_col, raw_time_col in zip(event_cols, time_cols):
        if raw_event_col not in adjusted.columns or raw_time_col not in adjusted.columns:
            raise ValueError(f"Missing event/time columns: {raw_event_col}, {raw_time_col}")

        adj_event_col = f"{raw_event_col}__post_landmark_h{horizon}"
        adj_time_col = f"{raw_time_col}__post_landmark_h{horizon}"
        raw_event = pd.to_numeric(adjusted[raw_event_col], errors="coerce").fillna(0).astype(int)
        raw_time = pd.to_numeric(adjusted[raw_time_col], errors="coerce").astype(float)
        post_landmark_time = raw_time - landmark_time
        censor_time = landmark_time + float(horizon)
        if post_landmark_time.isna().any() or post_landmark_time.le(0).any():
            raise ValueError(
                f"Input contains non-positive post-landmark time for {raw_time_col}."
            )

        observed_before_horizon = post_landmark_time.gt(0) & post_landmark_time.lt(horizon)
        adjusted_time = np.where(observed_before_horizon, raw_time, censor_time)
        # Events exactly at the administrative horizon are censored so every
        # patient follows the same strictly-within-horizon convention.
        within_horizon = raw_event.eq(1) & post_landmark_time.gt(0) & post_landmark_time.lt(horizon)
        adjusted[adj_event_col] = within_horizon.astype(int)
        adjusted[adj_time_col] = adjusted_time
        adjusted_post_time = adjusted[adj_time_col] - landmark_time
        if adjusted_post_time.isna().any() or adjusted_post_time.le(0).any():
            raise ValueError(
                f"Non-positive post-landmark time found after horizon censoring for {raw_time_col}."
            )
        if adjusted_post_time.gt(horizon).any():
            raise ValueError(
                f"Post-landmark time exceeds requested horizon after censoring for {raw_time_col}."
            )
        adjusted_event_cols.append(adj_event_col)
        adjusted_time_cols.append(adj_time_col)

    if isinstance(event_col, list):
        return adjusted, adjusted_event_cols, adjusted_time_cols
    return adjusted, adjusted_event_cols[0], adjusted_time_cols[0]


def import_survlatent(repo_path: Path):
    if not repo_path.exists():
        raise FileNotFoundError(f"SurvLatent ODE repo not found at {repo_path}")
    sys.path.insert(0, str(repo_path.resolve()))
    os.chdir(repo_path)
    # Imported lazily so --help works without the conda env active.
    import torch  # noqa: F401
    from lib.neural_ode_surv import SurvLatentODE  # noqa: F401
    from lib.utils import get_ckpt_model  # noqa: F401
    return torch, SurvLatentODE, get_ckpt_model


def prepare_run_artifacts(run_id: str, *, overwrite: bool, resume: bool) -> None:
    performance_dir = Path("model_performance") / run_id
    survival_curves_dir = Path("surv_curves") / run_id
    experiment_paths = [
        Path("experiments") / f"experiment_{run_id}.ckpt",
        Path("experiments") / f"run_{run_id}.ckpt",
    ]
    existing = [
        p for p in [performance_dir, survival_curves_dir, *experiment_paths] if p.exists()
    ]
    if existing:
        if overwrite:
            for path in existing:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            print(f"Removed existing artifacts for run_id={run_id}.")
        elif resume:
            print(f"Resuming with existing artifacts for run_id={run_id}.")
        else:
            existing_str = "\n  ".join(str(p) for p in existing)
            raise RuntimeError(
                f"Existing SurvLatent artifacts found for run_id={run_id}:\n  {existing_str}\n"
                "Reusing a run_id can silently load an old best_model.pt and produce misleading AUCs. "
                "Pass --overwrite-run for a fresh fit or --resume-run if this is intentional."
            )

    # Upstream uses os.mkdir() for run-specific paths and assumes these parent
    # directories already exist in the cloned repository. Fresh clones do not
    # necessarily contain empty directories (notably surv_curves), so create
    # every parent before model.fit().
    for parent in (Path("model_performance"), Path("surv_curves"), Path("experiments")):
        parent.mkdir(parents=True, exist_ok=True)


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _value_at_horizon(series, horizon: int) -> float:
    values = _to_numpy(series).reshape(-1)
    if horizon < 0 or horizon >= len(values):
        return np.nan
    return float(values[horizon])


def pad_patient_curves(curves) -> tuple[np.ndarray, np.ndarray]:
    arrays = [_to_numpy(curve).reshape(-1).astype(float) for curve in curves]
    lengths = np.asarray([len(curve) for curve in arrays], dtype=int)
    max_len = int(lengths.max()) if len(lengths) else 0
    padded = np.full((len(arrays), max_len), np.nan, dtype=float)
    for idx, curve in enumerate(arrays):
        padded[idx, : len(curve)] = curve
    return padded, lengths


def pad_event_patient_curves(curves_by_event) -> tuple[np.ndarray, np.ndarray]:
    event_arrays = [
        [_to_numpy(curve).reshape(-1).astype(float) for curve in event_curves]
        for event_curves in curves_by_event
    ]
    n_events = len(event_arrays)
    n_patients = max((len(event_curves) for event_curves in event_arrays), default=0)
    max_len = max(
        (len(curve) for event_curves in event_arrays for curve in event_curves),
        default=0,
    )
    padded = np.full((n_events, n_patients, max_len), np.nan, dtype=float)
    lengths = np.zeros((n_events, n_patients), dtype=int)
    for event_idx, event_curves in enumerate(event_arrays):
        for patient_idx, curve in enumerate(event_curves):
            lengths[event_idx, patient_idx] = len(curve)
            padded[event_idx, patient_idx, : len(curve)] = curve
    return padded, lengths


def _quantile_horizons(remaining_time: np.ndarray, event_mask: np.ndarray) -> dict[str, int]:
    event_times = remaining_time[event_mask & np.isfinite(remaining_time) & (remaining_time > 0)]
    if len(event_times) == 0:
        event_times = remaining_time[np.isfinite(remaining_time) & (remaining_time > 0)]
    if len(event_times) == 0:
        return {}
    return {
        "25": int(np.quantile(event_times, 0.25)),
        "50": int(np.quantile(event_times, 0.50)),
        "75": int(np.quantile(event_times, 0.75)),
    }


def write_prediction_diagnostics(
    *,
    output_dir: Path,
    run_id: str,
    config: str,
    batch_dict: dict,
    metric_input,
    cs_cif_total,
    n_events: int,
    event_cols: list[str],
    requested_window: int,
    model_window: int,
) -> Path:
    sample_ids = list(batch_dict["sample_ids"])
    labels = _to_numpy(batch_dict["labels"]).reshape(-1).astype(int)
    remaining_time = _to_numpy(batch_dict["remaining_time_to_event"]).reshape(-1).astype(float)
    end_of_obs = np.asarray([float(_to_numpy(v).reshape(-1)[0]) for v in batch_dict["end_of_obs_idx"]])
    observed = _to_numpy(batch_dict["observed_data"]).astype(float)
    observed_mask = _to_numpy(batch_dict["observed_mask"]).astype(float)
    observed_abs_sum = np.abs(observed * observed_mask).sum(axis=(1, 2))
    observed_value_count = observed_mask.sum(axis=(1, 2))
    observed_time_count = (observed_mask.sum(axis=2) > 0).sum(axis=1)

    rows = pd.DataFrame(
        {
            "DFCI_MRN": sample_ids,
            "config": config,
            "label": labels,
            "remaining_time_to_event": remaining_time,
            "end_of_obs_idx": end_of_obs,
            "observed_value_count": observed_value_count,
            "observed_time_count": observed_time_count,
            "observed_abs_sum": observed_abs_sum,
            "requested_max_pred_window": requested_window,
            "model_max_pred_window": model_window,
        }
    )

    if n_events == 1:
        event_name = event_cols[0]
        horizons = _quantile_horizons(remaining_time, labels == 1)
        rows[f"{event_name}_event"] = labels == 1
        for quantile_label, horizon in horizons.items():
            risks = [1.0 - _value_at_horizon(seq, horizon) for seq in metric_input]
            rows[f"{event_name}_risk_q{quantile_label}"] = risks
            rows[f"{event_name}_horizon_q{quantile_label}"] = horizon
    else:
        for event_idx, event_name in enumerate(event_cols):
            event_label = event_idx + 1
            horizons = _quantile_horizons(remaining_time, labels == event_label)
            rows[f"{event_name}_event"] = labels == event_label
            event_cif = cs_cif_total[event_idx]
            for quantile_label, horizon in horizons.items():
                risks = [_value_at_horizon(seq, horizon) for seq in event_cif]
                rows[f"{event_name}_risk_q{quantile_label}"] = risks
                rows[f"{event_name}_horizon_q{quantile_label}"] = horizon

    path = output_dir / f"survlatent_ode_prediction_diagnostics_{run_id}.csv"
    rows.to_csv(path, index=False)

    print("\nPrediction diagnostics:")
    print(
        "  processed input variation: "
        f"observed_value_count_sd={rows['observed_value_count'].std(ddof=0):.6f} "
        f"observed_time_count_sd={rows['observed_time_count'].std(ddof=0):.6f} "
        f"observed_abs_sum_sd={rows['observed_abs_sum'].std(ddof=0):.6f} "
        f"end_of_obs_sd={rows['end_of_obs_idx'].std(ddof=0):.6f}"
    )
    risk_sds = []
    for col in rows.columns:
        if "_risk_q" not in col:
            continue
        values = pd.to_numeric(rows[col], errors="coerce")
        risk_sds.append(float(values.std(ddof=0)))
        print(
            f"  {col}: n={int(values.notna().sum())} "
            f"mean={values.mean():.4f} sd={values.std(ddof=0):.6f} "
            f"min={values.min():.4f} max={values.max():.4f}"
        )
    if risk_sds and max(risk_sds) < 1e-5:
        print(
            "  WARNING: risk variation is essentially zero; AUC(t) will be near 0.5 "
            "because every patient is receiving the same risk curve."
        )
    print(f"  saved: {path}")
    return path


def summarize_training_curve(*, output_dir: Path, run_id: str, n_events: int) -> Path | None:
    curve_rows = []
    performance_dir = Path("model_performance") / run_id
    metric_names = ["c_idx", "ibs", "mean_auc", "reconstr_loss", "survival_loss"]
    for event_idx in range(n_events):
        perf_path = performance_dir / f"model_performance_{event_idx}_{run_id}.npy"
        if not perf_path.exists():
            continue
        performance = np.load(perf_path, allow_pickle=True)
        if len(performance) < len(metric_names):
            continue
        n_epochs = max(len(values) for values in performance[: len(metric_names)])
        for epoch_idx in range(n_epochs):
            row = {"event_idx": event_idx + 1, "epoch": epoch_idx + 1}
            for metric_name, values in zip(metric_names, performance[: len(metric_names)]):
                if epoch_idx < len(values):
                    row[metric_name] = values[epoch_idx]
                else:
                    row[metric_name] = np.nan
            curve_rows.append(row)

    if not curve_rows:
        return None

    curve = pd.DataFrame(curve_rows)
    path = output_dir / f"survlatent_ode_training_curve_{run_id}.csv"
    curve.to_csv(path, index=False)

    print("\nTraining curve diagnostics:")
    for event_idx, event_curve in curve.groupby("event_idx", sort=True):
        auc = pd.to_numeric(event_curve["mean_auc"], errors="coerce")
        valid_auc = auc.dropna()
        if valid_auc.empty:
            continue
        best_pos = int(valid_auc.to_numpy().argmax())
        best_epoch = int(valid_auc.index[best_pos])
        best_row = event_curve.loc[best_epoch]
        first_auc = float(valid_auc.iloc[0])
        best_auc = float(best_row["mean_auc"])
        last_auc = float(valid_auc.iloc[-1])
        print(
            f"  event_{int(event_idx)}: first_auc={first_auc:.4f} "
            f"best_auc={best_auc:.4f} at epoch={int(best_row['epoch'])} "
            f"last_auc={last_auc:.4f}"
        )
        if int(best_row["epoch"]) < 3:
            print(
                "    WARNING: peak validation AUC occurred before epoch 3; "
                "the upstream SurvLatent trainer does not save best_model.pt "
                "for epochs 1-2."
            )
        if best_auc - last_auc > 0.05:
            print(
                "    WARNING: validation AUC fell substantially after its peak; "
                "reduce --surv-loss-scale, lengthen --wait-until-full-surv-loss, "
                "or evaluate --checkpoint best instead of latest."
            )
    print(f"  saved: {path}")
    return path


def load_longitudinal_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    """Resolve longitudinal_landmark{D}.csv + its manifest from --inputs-dir.

    Mirrors survival_common.longitudinal_runners.load_longitudinal_inputs so
    both models are invoked identically by compass_pipeline.py, but this
    adapter does not need the shared build_manifest.json's schema-version
    guards -- it reads no AUC(t)/Brier horizons from it (SurvLatent's own
    metrics are computed by the external repo's eval_model).
    """
    inputs_dir = Path(args.inputs_dir)
    if not inputs_dir.exists():
        raise FileNotFoundError(
            f"Inputs dir {inputs_dir} not found. Run build_prediction_inputs.py first."
        )
    landmark_day = int(args.landmark_day)
    input_csv = inputs_dir / f"longitudinal_landmark{landmark_day}.csv"
    manifest_path = inputs_dir / f"longitudinal_landmark{landmark_day}_manifest.json"
    for path in (input_csv, manifest_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run build_prediction_inputs.py --build-longitudinal first."
            )
    manifest = json.loads(manifest_path.read_text())
    print(f"Loading longitudinal inputs from {input_csv} (landmark=+{landmark_day}d)")
    df = pd.read_csv(input_csv, low_memory=False)
    return df, manifest


def main(args: argparse.Namespace) -> None:
    # Resolved to absolute paths before import_survlatent() chdirs into the
    # external repo -- a relative --output-dir would otherwise resolve inside
    # the repo's own workspace.
    output_dir = Path(args.output_dir).resolve()

    run_id = args.run_id or f"prostate_{args.config}_landmark{args.landmark_day}_v1"
    metrics_path = output_dir / f"survlatent_ode_test_metrics_{run_id}.csv"
    if not args.overwrite and metrics_path.exists():
        print(f"[skip] {metrics_path} already exists (pass --overwrite to refit)")
        return

    df, manifest = load_longitudinal_inputs(args)

    id_col = manifest["id_col"]
    time_col = manifest["time_col"]
    feat_cont = manifest["feat_cont"]
    feat_cat = manifest["feat_cat"]
    feat_reconstr = manifest["feat_reconstr"]
    max_landmark_time = int(manifest.get("max_landmark_time", 0))
    model_max_pred_window = int(args.max_pred_window) + max_landmark_time

    event_cfg = EVENT_CONFIGS[args.config]
    raw_event_col = event_cfg["event_col"]
    raw_time_to_event_col = event_cfg["time_to_event_col"]
    n_events = len(raw_event_col) if isinstance(raw_event_col, list) else 1

    # For saving artifacts and printing, always use list form.
    event_cols_list = raw_event_col if isinstance(raw_event_col, list) else [raw_event_col]

    data_info_dic = {
        "id_col": id_col,
        "event_col": raw_event_col,
        "time_to_event_col": raw_time_to_event_col,
        "time_col": time_col,
        "feat_cat": feat_cat,
        "feat_cont": feat_cont,
    }
    feats_dim = len(feat_cat) + len(feat_cont)
    reconstr_dim = len(feat_reconstr)
    print(f"Config: {args.config}  events={event_cols_list}  n_events={n_events}")
    print(
        f"Post-landmark prediction window={args.max_pred_window}; "
        f"model absolute window={model_max_pred_window} "
        f"(max landmark offset={max_landmark_time})"
    )

    df, event_col, time_to_event_col = add_post_landmark_horizon_columns(
        df,
        id_col=id_col,
        time_col=time_col,
        event_col=raw_event_col,
        time_to_event_col=raw_time_to_event_col,
        horizon=args.max_pred_window,
    )
    data_info_dic["event_col"] = event_col
    data_info_dic["time_to_event_col"] = time_to_event_col
    adjusted_event_cols_list = event_col if isinstance(event_col, list) else [event_col]
    print(f"Horizon-censored event columns for modeling: {adjusted_event_cols_list}")

    data_train = load_split(df, "train", id_col=id_col, time_col=time_col)
    data_valid = load_split(df, "valid", id_col=id_col, time_col=time_col)
    data_test = load_split(df, "test", id_col=id_col, time_col=time_col)

    assert_no_test_leakage(
        test_mrns=data_test[id_col],
        train_mrns=data_train[id_col],
        context="survlatent_ode.main train/test",
    )
    assert_no_test_leakage(
        test_mrns=data_test[id_col],
        train_mrns=data_valid[id_col],
        context="survlatent_ode.main valid/test",
    )

    print(
        f"Loaded splits: "
        f"train={data_train[id_col].nunique()} patients / {len(data_train)} rows, "
        f"valid={data_valid[id_col].nunique()} / {len(data_valid)}, "
        f"test={data_test[id_col].nunique()} / {len(data_test)}"
    )
    print(f"feats_dim={feats_dim}, reconstr_dim={reconstr_dim}, n_events={n_events}")

    torch, SurvLatentODE, get_ckpt_model = import_survlatent(Path(args.survlatent_repo))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SurvLatentODE(
        input_dim=feats_dim,
        reconstr_dim=reconstr_dim,
        dec_latent_dim=args.dec_latent_dim,
        enc_latent_dim=args.enc_latent_dim,
        enc_f_nn_layers=args.enc_f_nn_layers,
        dec_g_nn_layers=args.dec_g_nn_layers,
        num_units_ode=args.num_units_ode,
        num_units_gru=args.num_units_gru,
        device=device,
        n_events=n_events,
        haz_dec_layers=args.haz_dec_layers,
    )

    if not args.skip_train:
        prepare_run_artifacts(run_id, overwrite=args.overwrite_run, resume=args.resume_run)
        print(f"\nTraining run_id={run_id} for up to {args.n_epochs} epochs ...")
        model.fit(
            data_train,
            data_valid,
            data_info_dic,
            max_pred_window=model_max_pred_window,
            run_id=run_id,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            surv_loss_scale=args.surv_loss_scale,
            early_stopping=args.early_stopping,
            feat_reconstr=feat_reconstr,
            wait_until_full_surv_loss=args.wait_until_full_surv_loss,
            survival_loss_exp=args.survival_loss_exp,
            random_seed=args.seed,
        )

    ckpt = Path("model_performance") / run_id / f"{args.checkpoint}_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt.resolve()}; training may have failed."
        )
    print(f"\nLoading {args.checkpoint} checkpoint: {ckpt.resolve()}")
    model_info = get_ckpt_model(str(ckpt), model, device)
    print(
        "Checkpoint info: "
        f"selected={args.checkpoint} "
        f"best_epoch={model_info.get('best_epoch', 'NA')} "
        f"itr={model_info.get('itr', 'NA')} "
        f"max_obs_time={model_info.get('max_obs_time', 'NA')}"
    )

    print("Processing held-out test set ...")
    batch_dict_test = model.process_eval_data(
        data_test,
        data_info_dic,
        max_pred_window=model_max_pred_window,
        run_id=run_id,
        feat_reconstr=feat_reconstr,
        model_info=model_info,
    )

    print("Sampling survival probabilities and cause-specific CIFs ...")
    try:
        from lib.neural_ode_surv import eval_model  # imported here to keep top-light
    except ImportError:
        from lib.utils import eval_model

    if n_events == 1:
        surv_prob = model.get_surv_prob(
            batch_dict_test,
            model_info=model_info,
            max_pred_window=model_max_pred_window,
            filename_suffix=run_id,
            device=device,
            n_events=n_events,
        )
        metric_input = surv_prob
        cs_cif_total = None
    else:
        ef_surv_prob, cs_cif_total = model.get_surv_prob(
            batch_dict_test,
            model_info=model_info,
            max_pred_window=model_max_pred_window,
            filename_suffix=run_id,
            device=device,
            n_events=n_events,
        )
        surv_prob = None
        metric_input = ef_surv_prob

    df_test_result = eval_model(
        model_info,
        batch_dict_test,
        metric_input,
        run_id=run_id,
        cs_cif_total=cs_cif_total,
        # Metrics are reported on the requested post-landmark horizon; the
        # larger model horizon only compensates for pre-landmark history.
        max_pred_window=args.max_pred_window,
        n_events=n_events,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    cif_path = output_dir / f"survlatent_ode_cif_{run_id}.npz"
    training_curve_path = summarize_training_curve(
        output_dir=output_dir,
        run_id=run_id,
        n_events=n_events,
    )
    diagnostics_path = write_prediction_diagnostics(
        output_dir=output_dir,
        run_id=run_id,
        config=args.config,
        batch_dict=batch_dict_test,
        metric_input=metric_input,
        cs_cif_total=cs_cif_total,
        n_events=n_events,
        event_cols=event_cols_list,
        requested_window=args.max_pred_window,
        model_window=model_max_pred_window,
    )
    df_test_result.to_csv(metrics_path, index=False)
    if n_events == 1:
        surv_prob_padded, surv_prob_lengths = pad_patient_curves(surv_prob)
        np.savez_compressed(
            cif_path,
            surv_prob=surv_prob_padded,
            surv_prob_lengths=surv_prob_lengths,
            event_cols=np.array(event_cols_list),
            model_event_cols=np.array(adjusted_event_cols_list),
            post_landmark_max_pred_window=np.asarray(args.max_pred_window),
            model_max_pred_window=np.asarray(model_max_pred_window),
        )
    else:
        ef_surv_prob_padded, ef_surv_prob_lengths = pad_patient_curves(metric_input)
        cs_cif_total_padded, cs_cif_total_lengths = pad_event_patient_curves(cs_cif_total)
        np.savez_compressed(
            cif_path,
            ef_surv_prob=ef_surv_prob_padded,
            ef_surv_prob_lengths=ef_surv_prob_lengths,
            cs_cif_total=cs_cif_total_padded,
            cs_cif_total_lengths=cs_cif_total_lengths,
            event_cols=np.array(event_cols_list),
            model_event_cols=np.array(adjusted_event_cols_list),
            post_landmark_max_pred_window=np.asarray(args.max_pred_window),
            model_max_pred_window=np.asarray(model_max_pred_window),
        )
    saved_paths = [metrics_path, cif_path, diagnostics_path]
    if training_curve_path is not None:
        saved_paths.append(training_curve_path)
    print("\nSaved:")
    for path in saved_paths:
        print(f"  {path}")
    print("\nTest metrics:")
    print(df_test_result.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SurvLatent ODE recurrent survival model for longitudinal lab histories."
    )
    parser.add_argument(
        "--survlatent-repo",
        type=Path,
        default=DEFAULT_SURVLATENT_REPO,
        help=(
            "Path to the editable itmoon7/survlatent_ode checkout used for imports "
            f"and checkpoints (default: {DEFAULT_SURVLATENT_REPO})."
        ),
    )
    parser.add_argument(
        "--inputs-dir",
        required=True,
        help="Directory containing prebuilt inputs from build_prediction_inputs.py.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write metrics/CIF artifacts into.",
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
        choices=sorted(EVENT_CONFIGS),
        required=True,
        help="platinum (death censored, comparable to Cox/XGBoost) or competing (platinum+death).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Training identifier. Defaults to prostate_<config>_landmark<D>_v1 so "
            "checkpoints do not collide across configs or landmarks."
        ),
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=False,
        help=(
            "Refit even if this run_id's survlatent_ode_test_metrics_*.csv already "
            "exists in --output-dir. Default: pick up where left off, skipping a "
            "run whose output is already present. Distinct from --overwrite-run, "
            "which controls SurvLatent's own checkpoint artifacts."
        ),
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Skip the run if its output already exists (the default).",
    )
    parser.add_argument("--skip-train", action="store_true", help="Reuse an existing checkpoint.")
    parser.add_argument(
        "--overwrite-run",
        action="store_true",
        help="Delete existing SurvLatent artifacts for this run_id before training.",
    )
    parser.add_argument(
        "--resume-run",
        action="store_true",
        help="Allow training to reuse/append existing SurvLatent artifacts for this run_id.",
    )
    parser.add_argument(
        "--checkpoint",
        choices=["best", "latest"],
        default="best",
        help="Checkpoint to evaluate after training or with --skip-train.",
    )

    parser.add_argument("--seed", type=int, default=1991)
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--surv-loss-scale", type=float, default=DEFAULT_SURV_LOSS_SCALE)
    parser.add_argument(
        "--wait-until-full-surv-loss",
        type=int,
        default=DEFAULT_WAIT_UNTIL_FULL_SURV_LOSS,
        help="Epochs used to ramp survival loss from 0.01 to --surv-loss-scale.",
    )
    parser.add_argument(
        "--early-stopping",
        dest="early_stopping",
        action="store_true",
        default=True,
        help="Enable SurvLatent's validation-AUC early stopping.",
    )
    parser.add_argument(
        "--no-early-stopping",
        dest="early_stopping",
        action="store_false",
        help="Disable SurvLatent's validation-AUC early stopping.",
    )
    parser.add_argument(
        "--survival-loss-exp",
        dest="survival_loss_exp",
        action="store_true",
        default=True,
        help="Use SurvLatent's exponential survival-loss warmup.",
    )
    parser.add_argument(
        "--no-survival-loss-exp",
        dest="survival_loss_exp",
        action="store_false",
        help="Disable SurvLatent's exponential survival-loss warmup.",
    )
    parser.add_argument(
        "--max-pred-window",
        type=int,
        default=DEFAULT_MAX_PRED_WINDOW,
        help="Prediction horizon in the same time unit as TIME, post-landmark (default 260).",
    )

    parser.add_argument("--enc-latent-dim", type=int, default=50)
    parser.add_argument("--enc-f-nn-layers", type=int, default=5)
    parser.add_argument("--num-units-gru", type=int, default=80)
    parser.add_argument("--dec-latent-dim", type=int, default=40)
    parser.add_argument("--dec-g-nn-layers", type=int, default=7)
    parser.add_argument("--haz-dec-layers", type=int, default=3)
    parser.add_argument("--num-units-ode", type=int, default=70)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
