"""
Train and evaluate SurvLatent ODE on the person-period input produced by
build_input.py.

Assumes:
  * The itmoon7/survlatent_ode repo is cloned and importable (pass its path via
    --survlatent-repo; the script adds it to sys.path and chdirs into it so that
    model_performance/<run_id>/ checkpoints land inside the repo workspace).
  * The associated conda env (from survlatent_ode_conda.yml) is active.

Runs with paper defaults unless overridden. Emits a test-set metrics CSV and the
cause-specific CIF tensor for head-to-head comparison with the Cox pipeline.
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

DEFAULT_RESULTS = Path("/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis")

# Event configuration.
# For single-event configs, event_col / time_to_event_col are strings (not lists
# of length 1) because pre_process_data dispatches on isinstance(..., list) and
# a list-of-one would land in the competing-risks branch with n_events==1.
EVENT_CONFIGS = {
    "death": {
        "event_col": "DEATH",
        "time_to_event_col": "t_death",
    },
    "platinum": {
        "event_col": "PLATINUM",
        "time_to_event_col": "t_platinum",
    },
    "competing": {
        "event_col": ["PLATINUM", "DEATH"],
        "time_to_event_col": ["t_platinum", "t_death"],
    },
}


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
    evaluates a post-treatment horizon from each patient's landmark. We therefore
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
    experiment_paths = [
        Path("experiments") / f"experiment_{run_id}.ckpt",
        Path("experiments") / f"run_{run_id}.ckpt",
    ]
    existing = [p for p in [performance_dir, *experiment_paths] if p.exists()]
    if not existing:
        return
    if overwrite:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        print(f"Removed existing artifacts for run_id={run_id}.")
        return
    if resume:
        print(f"Resuming with existing artifacts for run_id={run_id}.")
        return
    existing_str = "\n  ".join(str(p) for p in existing)
    raise RuntimeError(
        f"Existing SurvLatent artifacts found for run_id={run_id}:\n  {existing_str}\n"
        "Reusing a run_id can silently load an old best_model.pt and produce misleading AUCs. "
        "Pass --overwrite-run for a fresh fit or --resume-run if this is intentional."
    )


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _value_at_horizon(series, horizon: int) -> float:
    values = _to_numpy(series).reshape(-1)
    if horizon < 0 or horizon >= len(values):
        return np.nan
    return float(values[horizon])


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

    rows = pd.DataFrame(
        {
            "DFCI_MRN": sample_ids,
            "config": config,
            "label": labels,
            "remaining_time_to_event": remaining_time,
            "end_of_obs_idx": end_of_obs,
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
    for col in rows.columns:
        if "_risk_q" not in col:
            continue
        values = pd.to_numeric(rows[col], errors="coerce")
        print(
            f"  {col}: n={int(values.notna().sum())} "
            f"mean={values.mean():.4f} sd={values.std(ddof=0):.6f} "
            f"min={values.min():.4f} max={values.max():.4f}"
        )
    print(f"  saved: {path}")
    return path


def main(args: argparse.Namespace) -> None:
    input_csv = Path(args.input_csv)
    manifest_path = Path(args.manifest)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())

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
        f"Post-treatment prediction window={args.max_pred_window}; "
        f"model absolute window={model_max_pred_window} "
        f"(max landmark offset={max_landmark_time})"
    )

    df = pd.read_csv(input_csv)
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

    run_id = args.run_id or f"prostate_{args.config}_v1"

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
            random_seed=args.seed,
        )

    ckpt = Path("model_performance") / run_id / "best_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt.resolve()}; training may have failed."
        )
    print(f"\nLoading best checkpoint: {ckpt.resolve()}")
    model_info = get_ckpt_model(str(ckpt), model, device)

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
        # Metrics are reported on the requested post-treatment horizon; the
        # larger model horizon only compensates for pre-landmark history.
        max_pred_window=args.max_pred_window,
        n_events=n_events,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"survlatent_ode_test_metrics_{run_id}.csv"
    cif_path = output_dir / f"survlatent_ode_cif_{run_id}.npz"
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
        np.savez_compressed(
            cif_path,
            surv_prob=np.asarray(surv_prob),
            event_cols=np.array(event_cols_list),
            model_event_cols=np.array(adjusted_event_cols_list),
            post_treatment_max_pred_window=np.asarray(args.max_pred_window),
            model_max_pred_window=np.asarray(model_max_pred_window),
        )
    else:
        np.savez_compressed(
            cif_path,
            ef_surv_prob=np.asarray(metric_input),
            cs_cif_total=np.asarray(cs_cif_total),
            event_cols=np.array(event_cols_list),
            model_event_cols=np.array(adjusted_event_cols_list),
            post_treatment_max_pred_window=np.asarray(args.max_pred_window),
            model_max_pred_window=np.asarray(model_max_pred_window),
        )
    print(f"\nSaved:\n  {metrics_path}\n  {cif_path}\n  {diagnostics_path}")
    print("\nTest metrics:")
    print(df_test_result.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--survlatent-repo",
        required=True,
        help="Path to the cloned itmoon7/survlatent_ode repository (used for imports and checkpoint dir).",
    )
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_RESULTS / "survlatent_ode_input.csv"),
        help="Output of build_input.py.",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_RESULTS / "survlatent_ode_manifest.json"),
        help="Manifest emitted by build_input.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RESULTS),
        help="Directory to write metrics/CIF artifacts into.",
    )
    parser.add_argument(
        "--config",
        choices=list(EVENT_CONFIGS),
        required=True,
        help="Event configuration: 'death' / 'platinum' (single-event) or 'competing' (platinum vs death).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Training identifier. Defaults to prostate_<config>_v1 so checkpoints do not collide across configs.",
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

    parser.add_argument("--seed", type=int, default=1991)
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--surv-loss-scale", type=float, default=100.0)
    parser.add_argument("--wait-until-full-surv-loss", type=int, default=3)
    parser.add_argument("--early-stopping", action="store_true", default=True)
    parser.add_argument(
        "--max-pred-window",
        type=int,
        default=260,
        help="Prediction horizon in the same time unit as TIME (default 260 weeks ≈ 5 years).",
    )

    parser.add_argument("--enc-latent-dim", type=int, default=50)
    parser.add_argument("--enc-f-nn-layers", type=int, default=5)
    parser.add_argument("--num-units-gru", type=int, default=80)
    parser.add_argument("--dec-latent-dim", type=int, default=40)
    parser.add_argument("--dec-g-nn-layers", type=int, default=7)
    parser.add_argument("--haz-dec-layers", type=int, default=3)
    parser.add_argument("--num-units-ode", type=int, default=70)

    main(parser.parse_args())
