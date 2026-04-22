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
    event_col = event_cfg["event_col"]
    time_to_event_col = event_cfg["time_to_event_col"]
    n_events = len(event_col) if isinstance(event_col, list) else 1

    # For saving artifacts and printing, always use list form.
    event_cols_list = event_col if isinstance(event_col, list) else [event_col]

    data_info_dic = {
        "id_col": id_col,
        "event_col": event_col,
        "time_to_event_col": time_to_event_col,
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
    from lib.neural_ode_surv import eval_model  # imported here to keep top-light

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
    df_test_result.to_csv(metrics_path, index=False)
    if n_events == 1:
        np.savez_compressed(
            cif_path,
            surv_prob=np.asarray(surv_prob),
            event_cols=np.array(event_cols_list),
            post_treatment_max_pred_window=np.asarray(args.max_pred_window),
            model_max_pred_window=np.asarray(model_max_pred_window),
        )
    else:
        np.savez_compressed(
            cif_path,
            ef_surv_prob=np.asarray(metric_input),
            cs_cif_total=np.asarray(cs_cif_total),
            event_cols=np.array(event_cols_list),
            post_treatment_max_pred_window=np.asarray(args.max_pred_window),
            model_max_pred_window=np.asarray(model_max_pred_window),
        )
    print(f"\nSaved:\n  {metrics_path}\n  {cif_path}")
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
