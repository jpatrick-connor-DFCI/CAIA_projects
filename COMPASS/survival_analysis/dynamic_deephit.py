"""
Dynamic-DeepHit-style recurrent survival model for longitudinal lab histories.

This script consumes the per-landmark person-period CSV produced by
build_prediction_inputs.py and trains a compact GRU model with a discrete
competing-risks likelihood. It is intentionally self-contained so it does not
depend on the original TensorFlow 1.x Dynamic-DeepHit reference code.

Runs 5-fold stratified CV on train_val (combined PLATINUM+DEATH stratification)
over a (hidden_dim x dropout x lr) grid, then fits the chosen hyperparameter
combo with the train/valid split from the input manifest for early stopping and
evaluates on the held-out test fold. --no-cv falls back to the legacy single-fit
path. Per-config outputs are suffixed by args.config so the platinum / death /
competing runs do not collide.

Outputs (per --config):
  dynamic_deephit_metrics_{config}.csv
  dynamic_deephit_auc_t_{config}.csv
  dynamic_deephit_brier_{config}.csv
  dynamic_deephit_cv_folds_{config}.csv
  dynamic_deephit_cv_summary_{config}.csv
  dynamic_deephit_patient_risks_{config}.csv
  dynamic_deephit_manifest_{config}.json
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence
    from torch.utils.data import DataLoader, Dataset

    TORCH_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    torch = None
    nn = None
    Dataset = object
    DataLoader = None
    pack_padded_sequence = None
    TORCH_IMPORT_ERROR = exc

try:
    from lifelines.utils import concordance_index

    LIFELINES_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    concordance_index = None
    LIFELINES_IMPORT_ERROR = exc

try:
    from tqdm.auto import tqdm

    TQDM_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - tqdm is optional
    TQDM_AVAILABLE = False

    def tqdm(iterable=None, **kwargs):  # type: ignore[no-redef]
        """No-op tqdm fallback — yields the iterable unchanged."""
        if iterable is None:
            class _Null:
                def update(self, *_a, **_kw): pass
                def set_postfix(self, *_a, **_kw): pass
                def set_description(self, *_a, **_kw): pass
                def close(self): pass
                def __enter__(self): return self
                def __exit__(self, *_): return False
            return _Null()
        return iterable

try:
    from sksurv.metrics import cumulative_dynamic_auc

    SKSURV_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    cumulative_dynamic_auc = None
    SKSURV_IMPORT_ERROR = exc

SURVIVAL_DIR = Path(__file__).resolve().parent
if str(SURVIVAL_DIR) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_DIR))

from cox_aggregated import _make_survival_array  # noqa: E402
from helper import (  # noqa: E402
    assert_disjoint_folds,
    assert_no_test_leakage,
    compute_brier,
    iter_stratified_folds,
)

DEFAULT_RESULTS = Path("/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis")
DEFAULT_SEED = 42
DEFAULT_MAX_PRED_WINDOW = 260
DEFAULT_AUC_QUANTILES = (0.25, 0.375, 0.50, 0.625, 0.75)
DEFAULT_N_FOLDS = 5
DEFAULT_CV_HIDDEN_DIMS = [32, 64, 128]
DEFAULT_CV_DROPOUTS = [0.10, 0.20, 0.30]
DEFAULT_CV_LRS = [5e-4, 1e-3, 2e-3]


def require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError(
            "torch is required for dynamic_deephit.py."
        ) from TORCH_IMPORT_ERROR


def require_lifelines() -> None:
    if concordance_index is None:
        raise ModuleNotFoundError(
            "lifelines is required to compute concordance indices."
        ) from LIFELINES_IMPORT_ERROR


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def event_config(config: str, manifest: dict) -> tuple[list[str], list[str], list[str]]:
    event_cols = list(manifest["event_cols"])
    time_cols = list(manifest["time_to_event_cols"])
    normalized = config.lower()
    if normalized == "competing":
        return event_cols, time_cols, event_cols
    event_lookup = {col.lower(): idx for idx, col in enumerate(event_cols)}
    if normalized not in event_lookup:
        valid = ", ".join(["competing", *event_cols])
        raise ValueError(f"Unsupported --config '{config}'. Choose from: {valid}")
    idx = event_lookup[normalized]
    return [event_cols[idx]], [time_cols[idx]], [config]


def patient_targets(
    df: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    event_cols: list[str],
    time_cols: list[str],
    max_pred_window: int,
) -> pd.DataFrame:
    rows = []
    for mrn, group in df.groupby(id_col, sort=False):
        landmark = float(pd.to_numeric(group[time_col], errors="coerce").max())
        durations = []
        observed = []
        for event_col, event_time_col in zip(event_cols, time_cols):
            event = int(pd.to_numeric(group[event_col], errors="coerce").fillna(0).iloc[0])
            event_time = float(pd.to_numeric(group[event_time_col], errors="coerce").iloc[0])
            durations.append(event_time - landmark)
            observed.append(event == 1)

        durations_arr = np.asarray(durations, dtype=float)
        observed_arr = np.asarray(observed, dtype=bool)
        valid_times = np.isfinite(durations_arr) & (durations_arr > 0)
        if not valid_times.any():
            continue

        if observed_arr.any():
            candidate = np.where(observed_arr & valid_times, durations_arr, np.inf)
            event_idx = int(np.argmin(candidate))
            duration = float(candidate[event_idx])
            label = event_idx + 1 if np.isfinite(duration) else 0
        else:
            duration = float(np.nanmin(durations_arr[valid_times]))
            label = 0

        uncensored_duration = duration
        uncensored_label = label

        if duration > max_pred_window:
            duration = float(max_pred_window)
            label = 0

        duration_bin = int(np.clip(np.ceil(duration), 1, max_pred_window))
        rows.append(
            {
                id_col: mrn,
                "landmark_time": landmark,
                "duration": duration,
                "duration_bin": duration_bin,
                "label": label,
                "uncensored_duration": uncensored_duration,
                "uncensored_label": uncensored_label,
            }
        )
    return pd.DataFrame(rows).set_index(id_col)


def fixed_horizon_grid_from_targets(
    train_val_targets: pd.DataFrame,
    *,
    event_idx: int,
    event_name: str,
    quantiles: tuple[float, ...],
    max_pred_window: int,
) -> np.ndarray:
    """Match Cox/XGBoost horizon selection: train/valid event-time quantiles.

    DeepHit durations are already in the same integer time unit as the
    longitudinal input. We use uncensored event durations for quantile selection
    and require the model prediction window to cover the selected grid.
    """
    if {"uncensored_label", "uncensored_duration"}.issubset(train_val_targets.columns):
        event_mask = train_val_targets["uncensored_label"].eq(event_idx)
        event_durations = train_val_targets.loc[event_mask, "uncensored_duration"]
    else:
        event_mask = train_val_targets["label"].eq(event_idx)
        event_durations = train_val_targets.loc[event_mask, "duration"]
    event_durations = pd.to_numeric(event_durations, errors="coerce").to_numpy(dtype=float)
    event_durations = event_durations[np.isfinite(event_durations) & (event_durations > 0)]
    if len(event_durations) == 0:
        return np.asarray([], dtype=float)
    horizons = np.asarray(
        [int(v) for v in np.quantile(event_durations, list(quantiles))],
        dtype=float,
    )
    horizons = np.unique(horizons[horizons > 0])
    if len(horizons) and horizons.max() > float(max_pred_window):
        raise ValueError(
            f"DeepHit --max-pred-window={max_pred_window} is shorter than the "
            f"largest train/valid quantile horizon for {event_name} ({int(horizons.max())}). "
            "Increase --max-pred-window so DeepHit and Cox/XGBoost AUC horizons are comparable."
        )
    return horizons


def fit_normalization(df: pd.DataFrame, *, feature_cols: list[str], train_ids: set) -> tuple[pd.Series, pd.Series]:
    train = df.loc[df["DFCI_MRN"].astype(str).isin(train_ids), feature_cols]
    mean = train.mean(skipna=True).fillna(0.0)
    std = train.std(skipna=True, ddof=0).replace(0, np.nan).fillna(1.0)
    return mean, std


def build_sequences(
    df: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    feature_cols: list[str],
    targets: pd.DataFrame,
    mean: pd.Series,
    std: pd.Series,
    max_observed_time: float,
) -> dict[str, dict]:
    sequences: dict[str, dict] = {}
    denom = max(float(max_observed_time), 1.0)
    for mrn, group in df.groupby(id_col, sort=False):
        if mrn not in targets.index:
            continue
        group = group.sort_values(time_col)
        raw = group[feature_cols].astype(float)
        mask = raw.notna().astype(float)
        values = ((raw - mean) / std).fillna(0.0)
        rel_time = (pd.to_numeric(group[time_col], errors="coerce").to_numpy(dtype=float) / denom)
        rel_time = rel_time.reshape(-1, 1)
        x = np.hstack([values.to_numpy(dtype=np.float32), mask.to_numpy(dtype=np.float32), rel_time.astype(np.float32)])
        target = targets.loc[mrn]
        sequences[str(mrn)] = {
            "x": x.astype(np.float32),
            "length": int(len(x)),
            "label": int(target["label"]),
            "duration_bin": int(target["duration_bin"]),
            "duration": float(target["duration"]),
        }
    return sequences


class SequenceDataset(Dataset):
    def __init__(self, sequences: dict[str, dict], ids: list[str]):
        self.sequences = sequences
        self.ids = [str(v) for v in ids if str(v) in sequences]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        sample_id = self.ids[idx]
        item = self.sequences[sample_id]
        return {
            "id": sample_id,
            "x": item["x"],
            "length": item["length"],
            "label": item["label"],
            "duration_bin": item["duration_bin"],
            "duration": item["duration"],
        }


def collate_batch(batch: list[dict]) -> dict:
    max_len = max(item["length"] for item in batch)
    feat_dim = batch[0]["x"].shape[1]
    x = np.zeros((len(batch), max_len, feat_dim), dtype=np.float32)
    for idx, item in enumerate(batch):
        x[idx, : item["length"], :] = item["x"]
    return {
        "ids": [item["id"] for item in batch],
        "x": torch.tensor(x, dtype=torch.float32),
        "length": torch.tensor([item["length"] for item in batch], dtype=torch.long),
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "duration_bin": torch.tensor([item["duration_bin"] for item in batch], dtype=torch.long),
        "duration": torch.tensor([item["duration"] for item in batch], dtype=torch.float32),
    }


if nn is not None:
    class DynamicDeepHitGRU(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, n_events: int, horizon: int, dropout: float):
            super().__init__()
            self.n_events = n_events
            self.horizon = horizon
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_events * horizon + 1),
            )

        def forward(self, x, length):
            packed = pack_padded_sequence(
                x,
                length.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, h_n = self.gru(packed)
            logits = self.head(h_n[-1])
            return logits

        def probabilities(self, logits):
            probs = torch.softmax(logits, dim=1)
            event_probs = probs[:, : self.n_events * self.horizon].reshape(
                -1, self.n_events, self.horizon
            )
            no_event = probs[:, -1]
            return event_probs, no_event
else:
    class DynamicDeepHitGRU:  # pragma: no cover - only used when torch is missing
        def __init__(self, *args, **kwargs):
            require_torch()


def deephit_nll(model: DynamicDeepHitGRU, logits, label, duration_bin) -> torch.Tensor:
    event_probs, no_event = model.probabilities(logits)
    eps = 1e-8
    idx = torch.clamp(duration_bin, min=1, max=model.horizon) - 1
    observed = label > 0
    losses = []
    if observed.any():
        event_idx = label[observed] - 1
        time_idx = idx[observed]
        prob = event_probs[observed, event_idx, time_idx]
        losses.append(-torch.log(prob + eps))
    if (~observed).any():
        censor_idx = idx[~observed]
        future_mass = []
        censored_probs = event_probs[~observed]
        for row_idx, c_idx in enumerate(censor_idx):
            future_mass.append(censored_probs[row_idx, :, c_idx + 1 :].sum() + no_event[~observed][row_idx])
        losses.append(-torch.log(torch.stack(future_mass) + eps))
    return torch.cat(losses).mean()


def run_epoch(model, loader, optimizer, device: str) -> float:
    model.train(optimizer is not None)
    losses = []
    for batch in loader:
        x = batch["x"].to(device)
        length = batch["length"].to(device)
        label = batch["label"].to(device)
        duration_bin = batch["duration_bin"].to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(x, length)
            loss = deephit_nll(model, logits, label, duration_bin)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else np.nan


def predict(model, loader, device: str) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["x"].to(device), batch["length"].to(device))
            event_probs, no_event = model.probabilities(logits)
            event_probs_np = event_probs.cpu().numpy()
            no_event_np = no_event.cpu().numpy()
            for row_idx, sample_id in enumerate(batch["ids"]):
                row = {
                    "DFCI_MRN": sample_id,
                    "duration": float(batch["duration"][row_idx]),
                    "duration_bin": int(batch["duration_bin"][row_idx]),
                    "label": int(batch["label"][row_idx]),
                    "no_event_probability": float(no_event_np[row_idx]),
                }
                for event_idx in range(event_probs_np.shape[1]):
                    pmf = event_probs_np[row_idx, event_idx, :]
                    cif = np.cumsum(pmf)
                    row[f"event_{event_idx + 1}_risk_total"] = float(cif[-1])
                    for horizon, risk in enumerate(cif, start=1):
                        row[f"event_{event_idx + 1}_risk_h{horizon}"] = float(risk)
                rows.append(row)
    return pd.DataFrame(rows)


def train_evaluate(
    *,
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    feature_cols: list[str],
    targets: pd.DataFrame,
    train_ids: set,
    valid_ids: set,
    eval_ids: set,
    args: argparse.Namespace,
    n_events: int,
    horizon: int,
    hidden_dim: int,
    dropout: float,
    lr: float,
    seed: int,
) -> tuple[pd.DataFrame, dict, float]:
    """Train DeepHit on `train_ids` watching `valid_ids` for early stopping,
    predict on `eval_ids`. Normalization is fit on train_ids only — never on
    valid or eval. Returns (pred_df, history, best_valid_loss).
    """
    set_seed(seed)
    mean, std = fit_normalization(df, feature_cols=feature_cols, train_ids=train_ids)
    train_rows = df.loc[df[id_col].astype(str).isin(train_ids), time_col]
    if train_rows.empty:
        raise ValueError("Empty fold_train when fitting DeepHit normalization.")
    max_observed_time = float(train_rows.max())
    sequences = build_sequences(
        df,
        id_col=id_col,
        time_col=time_col,
        feature_cols=feature_cols,
        targets=targets,
        mean=mean,
        std=std,
        max_observed_time=max_observed_time,
    )

    train_ds = SequenceDataset(sequences, sorted(train_ids))
    valid_ds = SequenceDataset(sequences, sorted(valid_ids))
    eval_ds = SequenceDataset(sequences, sorted(eval_ids))
    if len(train_ds) == 0 or len(valid_ds) == 0 or len(eval_ds) == 0:
        raise ValueError(
            f"Empty split: train={len(train_ds)} valid={len(valid_ds)} eval={len(eval_ds)}"
        )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch
    )

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    input_dim = next(iter(train_loader))["x"].shape[-1]
    model = DynamicDeepHitGRU(
        input_dim=input_dim,
        hidden_dim=int(hidden_dim),
        n_events=n_events,
        horizon=horizon,
        dropout=float(dropout),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(lr), weight_decay=args.weight_decay
    )

    best_state = None
    best_valid = float("inf")
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        valid_loss = run_epoch(model, valid_loader, None, device)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss}
        )
        if valid_loss < best_valid - args.min_delta:
            best_valid = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    pred = predict(model, eval_loader, device)
    return pred, history, best_valid


def cv_run(
    *,
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    feature_cols: list[str],
    targets: pd.DataFrame,
    train_val_static: pd.DataFrame,
    args: argparse.Namespace,
    n_events: int,
    event_names: list[str],
    fixed_horizons_by_event: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """5-fold stratified CV over (hidden_dim x dropout x lr).

    `train_val_static` is patient-indexed with PLATINUM/DEATH for stratification
    (combined 4-cell label). For each fold and each combo, the model is trained
    on fold_train MRNs with fold_val watched for early stopping (and used as
    the metric set). Returns (fold_df, cv_summary_df, best_row).
    """
    fold_partitions = list(
        iter_stratified_folds(train_val_static, n_folds=args.n_folds, seed=args.seed)
    )
    if not fold_partitions:
        raise RuntimeError("No CV folds produced for DeepHit.")
    cv_stratification = fold_partitions[0][3]

    grid = list(
        product(args.cv_hidden_dims, args.cv_dropouts, args.cv_lrs)
    )
    fold_rows: list[dict] = []
    total_runs = len(grid) * len(fold_partitions)
    cv_bar = tqdm(
        total=total_runs,
        desc=f"DeepHit CV[{args.config}]",
        dynamic_ncols=True,
    )
    for hidden_dim, dropout, lr in grid:
        for fold, tr_idx, val_idx, _ in fold_partitions:
            fold_train_ids = set(train_val_static.index[tr_idx].astype(str))
            fold_val_ids = set(train_val_static.index[val_idx].astype(str))
            assert_disjoint_folds(
                fold_train_mrns=fold_train_ids,
                fold_val_mrns=fold_val_ids,
                fold=fold,
            )
            row = {
                "fold": fold,
                "hidden_dim": int(hidden_dim),
                "dropout": float(dropout),
                "lr": float(lr),
                "n_train": len(fold_train_ids),
                "n_val": len(fold_val_ids),
                "cv_stratification": cv_stratification,
                "best_valid_loss": np.nan,
                "n_epochs": 0,
                "note": "",
            }
            for event_name in event_names:
                row[f"c_index_val__{event_name}"] = np.nan
                row[f"mean_auc_t_val__{event_name}"] = np.nan
                row[f"integrated_brier_val__{event_name}"] = np.nan
            try:
                pred, history, best_valid = train_evaluate(
                    df=df,
                    id_col=id_col,
                    time_col=time_col,
                    feature_cols=feature_cols,
                    targets=targets,
                    train_ids=fold_train_ids,
                    valid_ids=fold_val_ids,
                    eval_ids=fold_val_ids,
                    args=args,
                    n_events=n_events,
                    horizon=args.max_pred_window,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    lr=lr,
                    seed=args.seed + fold,
                )
                row["best_valid_loss"] = float(best_valid)
                row["n_epochs"] = int(len(history))
                fold_train_targets = targets.loc[
                    targets.index.map(str).isin(fold_train_ids)
                ].copy()
                metrics_df, _ = compute_metrics(
                    pred,
                    event_names=event_names,
                    train_val_targets=fold_train_targets,
                    quantiles=tuple(args.auc_quantiles),
                    fixed_horizons_by_event=fixed_horizons_by_event,
                )
                _, ibs_by_event = compute_brier_for_pred(
                    pred,
                    event_names=event_names,
                    train_val_targets=fold_train_targets,
                    horizons_by_event=fixed_horizons_by_event,
                )
                for _, mrow in metrics_df.iterrows():
                    event_name = mrow["event"]
                    row[f"c_index_val__{event_name}"] = float(mrow.get("c_index", np.nan))
                    row[f"mean_auc_t_val__{event_name}"] = float(
                        mrow.get("mean_auc_t", np.nan)
                    )
                    row[f"integrated_brier_val__{event_name}"] = float(
                        ibs_by_event.get(event_name, np.nan)
                    )
            except Exception as exc:  # pragma: no cover - defensive
                row["note"] = f"fold_failed: {exc}"
            fold_rows.append(row)
            if hasattr(cv_bar, "set_postfix"):
                cv_bar.set_postfix(
                    {
                        "hd": int(hidden_dim),
                        "dr": f"{float(dropout):.2f}",
                        "lr": f"{float(lr):g}",
                        "fold": fold,
                        "best_valid": (
                            f"{row['best_valid_loss']:.4f}"
                            if np.isfinite(row.get("best_valid_loss", np.nan))
                            else "nan"
                        ),
                    }
                )
            cv_bar.update(1)
    cv_bar.close()

    fold_df = pd.DataFrame(fold_rows)
    agg_cols = {
        "best_valid_loss_mean": ("best_valid_loss", "mean"),
        "n_epochs_mean": ("n_epochs", "mean"),
        "n_valid_folds": ("best_valid_loss", lambda s: int(s.notna().sum())),
        "cv_stratification": ("cv_stratification", "first"),
    }
    for event_name in event_names:
        agg_cols[f"cv_mean_c_index__{event_name}"] = (
            f"c_index_val__{event_name}",
            "mean",
        )
        agg_cols[f"cv_std_c_index__{event_name}"] = (
            f"c_index_val__{event_name}",
            "std",
        )
        agg_cols[f"cv_mean_auc_t__{event_name}"] = (
            f"mean_auc_t_val__{event_name}",
            "mean",
        )
        agg_cols[f"cv_mean_integrated_brier__{event_name}"] = (
            f"integrated_brier_val__{event_name}",
            "mean",
        )
    cv_df = (
        fold_df.groupby(["hidden_dim", "dropout", "lr"], dropna=False)
        .agg(**agg_cols)
        .reset_index()
    )
    cv_df["all_folds_valid"] = cv_df["n_valid_folds"].eq(int(args.n_folds))

    if cv_df["n_valid_folds"].eq(0).all():
        raise RuntimeError("All DeepHit CV fits failed.")
    candidate = cv_df.loc[cv_df["all_folds_valid"]]
    if candidate.empty:
        candidate = cv_df.sort_values("n_valid_folds", ascending=False)

    # Score by mean C-index averaged across causes (ignores NaNs).
    cindex_cols = [
        f"cv_mean_c_index__{name}" for name in event_names
    ]
    candidate = candidate.copy()
    candidate["__rank_score"] = candidate[cindex_cols].mean(axis=1, skipna=True)
    best_row = (
        candidate.sort_values(
            ["__rank_score", "n_valid_folds", "hidden_dim", "dropout", "lr"],
            ascending=[False, False, True, True, True],
            na_position="last",
        )
        .drop(columns="__rank_score")
        .iloc[0]
        .to_dict()
    )
    return fold_df, cv_df, best_row


def compute_brier_for_pred(
    pred: pd.DataFrame,
    *,
    event_names: list[str],
    train_val_targets: pd.DataFrame,
    horizons_by_event: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Per-cause IPCW Brier on DeepHit's PMF predictions.

    For each cause k, treats the event as binary (1 iff cause==k, 0 otherwise
    including censored AND competing) and feeds 1 - CIF_k(h) as the survival
    estimate at horizon h. For competing-risk configs this is the binary
    "cause-of-interest" Brier — not the cumulative-incidence Brier — but it's
    the convention consistent with how the existing per-cause AUC(t) is
    computed. Train arrays come from `train_val_targets` so the IPCW weights
    are train-only.
    """
    rows: list[dict] = []
    integrated_by_event: dict[str, float] = {}
    ref_event_any = train_val_targets["label"].to_numpy(dtype=int)
    ref_duration = train_val_targets["duration"].to_numpy(dtype=float)
    ref_valid = np.isfinite(ref_duration) & (ref_duration > 0)

    for event_idx, event_name in enumerate(event_names, start=1):
        horizons = np.asarray(
            horizons_by_event.get(event_name, np.asarray([], dtype=float)),
            dtype=float,
        )
        horizons = np.unique(horizons[horizons > 0])
        if len(horizons) == 0:
            integrated_by_event[event_name] = float("nan")
            continue

        event = pred["label"].eq(event_idx).astype(int).to_numpy()
        duration = pred["duration"].to_numpy(dtype=float)
        valid = np.isfinite(duration) & (duration > 0)
        if not valid.any():
            integrated_by_event[event_name] = float("nan")
            continue

        surv_cols = []
        for h in horizons:
            risk_col = f"event_{event_idx}_risk_h{int(h)}"
            if risk_col not in pred.columns:
                surv_cols = []
                break
            cif = pred.loc[valid, risk_col].to_numpy(dtype=float)
            surv_cols.append(1.0 - cif)
        if not surv_cols:
            # Fall back to the cumulative total risk if per-horizon columns
            # aren't materialized (legacy paths).
            total = pred.loc[valid, f"event_{event_idx}_risk_total"].to_numpy(
                dtype=float
            )
            surv_cols = [1.0 - total for _ in horizons]
        surv_at_horizons = np.column_stack(surv_cols)

        train_event = (ref_event_any[ref_valid] == event_idx).astype(int)
        train_duration = ref_duration[ref_valid]
        eval_event = event[valid]
        eval_duration = duration[valid]

        brier_df, ibs = compute_brier(
            train_event=train_event,
            train_duration=train_duration,
            eval_event=eval_event,
            eval_duration=eval_duration,
            surv_at_horizons=surv_at_horizons,
            horizons=horizons,
            time_unit_days=1,  # DeepHit horizons are already in time-unit bins.
        )
        if not brier_df.empty:
            brier_df = brier_df.copy()
            brier_df.insert(0, "event", event_name)
            rows.extend(brier_df.to_dict("records"))
        integrated_by_event[event_name] = ibs

    return pd.DataFrame(rows), integrated_by_event


def compute_metrics(
    pred: pd.DataFrame,
    *,
    event_names: list[str],
    train_val_targets: pd.DataFrame,
    quantiles: tuple[float, ...],
    fixed_horizons_by_event: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_lifelines()
    metric_rows = []
    auc_rows = []
    ref_event_any = train_val_targets["label"].to_numpy(dtype=int)
    ref_duration = train_val_targets["duration"].to_numpy(dtype=float)

    for event_idx, event_name in enumerate(event_names, start=1):
        event = pred["label"].eq(event_idx).astype(int).to_numpy()
        duration = pred["duration"].to_numpy(dtype=float)
        risk = pred[f"event_{event_idx}_risk_total"].to_numpy(dtype=float)
        valid = np.isfinite(duration) & (duration > 0) & np.isfinite(risk)
        c_index = np.nan
        if valid.sum() > 0 and event[valid].sum() > 0:
            c_index = float(concordance_index(duration[valid], -risk[valid], event[valid]))

        mean_auc = np.nan
        event_times = duration[valid & (event == 1)]
        event_times = event_times[np.isfinite(event_times) & (event_times > 0)]
        if cumulative_dynamic_auc is not None and len(event_times) > 0:
            ref_valid = np.isfinite(ref_duration) & (ref_duration > 0)
            ref_surv = _make_survival_array(
                (ref_event_any[ref_valid] == event_idx),
                ref_duration[ref_valid],
            )
            eval_surv = _make_survival_array(event[valid], duration[valid])
            if fixed_horizons_by_event is None:
                horizons = np.asarray([int(v) for v in np.quantile(event_times, quantiles)], dtype=float)
                horizon_quantiles = tuple(quantiles)
            else:
                horizons = np.asarray(
                    fixed_horizons_by_event.get(event_name, np.asarray([], dtype=float)),
                    dtype=float,
                )
                horizons = np.unique(horizons[horizons > 0])
                horizon_quantiles = tuple([np.nan] * len(horizons))
            for quantile, horizon in zip(horizon_quantiles, horizons):
                auc_t = np.nan
                note = ""
                if horizon <= 0:
                    note = "non_positive_horizon"
                else:
                    risk_col = f"event_{event_idx}_risk_h{int(horizon)}"
                    if risk_col in pred.columns:
                        horizon_risk = pred.loc[valid, risk_col].to_numpy(dtype=float)
                    else:
                        horizon_risk = risk[valid]
                    try:
                        auc_values, _ = cumulative_dynamic_auc(
                            ref_surv,
                            eval_surv,
                            horizon_risk,
                            np.asarray([horizon], dtype=float),
                        )
                        auc_t = float(auc_values[0])
                    except ValueError as exc:
                        note = f"auc_failed: {exc}"
                auc_rows.append(
                    {
                        "event": event_name,
                        "horizon_quantile": quantile,
                        "horizon_time_unit": horizon,
                        "auc_t": auc_t,
                        "note": note,
                    }
                )
            if len(horizons) >= 2 and horizons[-1] > horizons[0]:
                times = np.arange(horizons[0], horizons[-1], dtype=float)
                try:
                    risk_cols = []
                    for t in times:
                        risk_col = f"event_{event_idx}_risk_h{int(t)}"
                        if risk_col in pred.columns:
                            risk_cols.append(pred.loc[valid, risk_col].to_numpy(dtype=float))
                        else:
                            risk_cols.append(risk[valid])
                    time_risk = np.column_stack(risk_cols)
                    _, mean_auc = cumulative_dynamic_auc(ref_surv, eval_surv, time_risk, times)
                    mean_auc = float(mean_auc)
                except ValueError:
                    mean_auc = np.nan

        metric_rows.append(
            {
                "event": event_name,
                "n_test": int(valid.sum()),
                "n_test_events": int(event[valid].sum()),
                "c_index": c_index,
                "mean_auc_t": mean_auc,
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(auc_rows)


def main(args: argparse.Namespace) -> None:
    require_torch()
    set_seed(args.seed)
    input_csv = Path(args.input_csv)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    df = pd.read_csv(input_csv, low_memory=False)

    # Per-(landmark, endpoint) horizons live in build_manifest.json (sibling of
    # the longitudinal CSV). Loading them here so DeepHit, Cox, and XGBoost
    # evaluate mean AUC(t) on the identical horizon set.
    build_manifest_path = input_csv.parent / "build_manifest.json"
    if not build_manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {build_manifest_path}. Run build_prediction_inputs.py first."
        )
    build_manifest = json.loads(build_manifest_path.read_text())
    args.auc_quantiles = list(build_manifest["auc_quantiles"])
    landmark_horizons = build_manifest["auc_horizons_by_landmark"].get(
        str(int(manifest["landmark_days"]))
    )
    if landmark_horizons is None:
        raise KeyError(
            f"build_manifest.json has no auc_horizons_by_landmark entry for landmark "
            f"+{manifest['landmark_days']}d."
        )
    max_manifest_horizon = max(
        (h for hs in landmark_horizons.values() for h in hs), default=0
    )
    if args.max_pred_window < max_manifest_horizon:
        raise ValueError(
            f"--max-pred-window={args.max_pred_window} is shorter than the largest "
            f"manifest horizon ({max_manifest_horizon}). Increase it so DeepHit's "
            "AUC grid is comparable to Cox/XGBoost."
        )

    id_col = manifest["id_col"]
    time_col = manifest["time_col"]
    event_cols, time_cols, event_names = event_config(args.config, manifest)
    feature_cols = list(manifest["feat_cont"])
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
        context=f"dynamic_deephit.main[{args.config}]",
    )

    # Patient-level static frame for stratification (combined PLATINUM+DEATH).
    patient_static = (
        df.groupby(id_col)[["PLATINUM", "DEATH"]]
        .first()
        .astype(int)
    )
    patient_static.index = patient_static.index.astype(str)
    train_val_static = patient_static.loc[
        patient_static.index.intersection(sorted(train_val_ids))
    ].copy()

    # AUC horizons sourced from build_manifest.json so Cox / XGBoost / DeepHit
    # evaluate mean AUC(t) on the identical (landmark, endpoint) horizon set.
    train_val_targets = targets.loc[
        targets.index.map(str).isin(train_val_ids)
    ].copy()
    fixed_horizons_by_event: dict[str, np.ndarray] = {}
    for event_name in event_names:
        manifest_lookup_key = event_name.lower()
        manifest_horizons = landmark_horizons.get(manifest_lookup_key)
        if manifest_horizons is None:
            raise KeyError(
                f"build_manifest.json has no horizons for event {event_name!r} "
                f"at landmark +{manifest['landmark_days']}d."
            )
        horizons = np.asarray(manifest_horizons, dtype=float)
        horizons = horizons[(horizons > 0) & (horizons <= float(args.max_pred_window))]
        if len(horizons):
            fixed_horizons_by_event[event_name] = horizons

    fold_df = pd.DataFrame()
    cv_summary_df = pd.DataFrame()
    chosen: dict | None = None
    if not args.no_cv:
        print(
            f"Running 5-fold CV on {len(train_val_ids)} train_val patients "
            f"({len(args.cv_hidden_dims)*len(args.cv_dropouts)*len(args.cv_lrs)} combos) "
            f"for config={args.config} ..."
        )
        fold_df, cv_summary_df, best_row = cv_run(
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

    # Final fit uses the manifest's train split and watches valid for early stopping.
    final_hidden_dim = chosen["hidden_dim"] if chosen is not None else args.hidden_dim
    final_dropout = chosen["dropout"] if chosen is not None else args.dropout
    final_lr = chosen["lr"] if chosen is not None else args.lr

    pred, history, best_valid = train_evaluate(
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

    metrics, auc_t = compute_metrics(
        pred,
        event_names=event_names,
        train_val_targets=train_val_targets,
        quantiles=tuple(args.auc_quantiles),
        fixed_horizons_by_event=fixed_horizons_by_event,
    )
    brier_t, integrated_brier_by_event = compute_brier_for_pred(
        pred,
        event_names=event_names,
        train_val_targets=train_val_targets,
        horizons_by_event=fixed_horizons_by_event,
    )
    metrics = metrics.copy()
    metrics["integrated_brier"] = metrics["event"].map(
        lambda name: integrated_brier_by_event.get(name, float("nan"))
    )
    metrics["selected_hidden_dim"] = final_hidden_dim
    metrics["selected_dropout"] = final_dropout
    metrics["selected_lr"] = final_lr
    metrics["config"] = args.config

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_tag = args.config
    pred_path = output_dir / f"dynamic_deephit_patient_risks_{config_tag}.csv"
    metrics_path = output_dir / f"dynamic_deephit_metrics_{config_tag}.csv"
    auc_path = output_dir / f"dynamic_deephit_auc_t_{config_tag}.csv"
    brier_path = output_dir / f"dynamic_deephit_brier_{config_tag}.csv"
    cv_folds_path = output_dir / f"dynamic_deephit_cv_folds_{config_tag}.csv"
    cv_summary_path = output_dir / f"dynamic_deephit_cv_summary_{config_tag}.csv"
    manifest_out = output_dir / f"dynamic_deephit_manifest_{config_tag}.json"
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
    manifest_out.write_text(
        json.dumps(
            {
                "input_csv": str(input_csv),
                "source_manifest": str(manifest_path),
                "config": args.config,
                "event_names": event_names,
                "feature_cols": feature_cols,
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
            },
            indent=2,
        )
    )
    saved.append(manifest_out)
    print("\nSaved:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs-dir",
        default=str(DEFAULT_RESULTS / "prediction_inputs"),
        help="Directory containing prebuilt inputs from build_prediction_inputs.py.",
    )
    parser.add_argument(
        "--landmark-day",
        type=int,
        default=0,
        help="Landmark to load. Resolves to longitudinal_landmark{D}.csv in --inputs-dir unless overridden.",
    )
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Override longitudinal CSV path (defaults to longitudinal_landmark{landmark_days}.csv).",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Override manifest path (defaults to longitudinal_landmark{landmark_days}_manifest.json).",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--config", default="competing", help="competing, PLATINUM, or DEATH")
    parser.add_argument(
        "--max-pred-window",
        type=int,
        default=DEFAULT_MAX_PRED_WINDOW,
        help=(
            "Discrete prediction window in input time bins. Must cover the "
            "train/valid event-time quantile horizons used for Cox/XGBoost-comparable AUC."
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cuda", action="store_true")
    # auc_quantiles loaded from build_manifest.json so all three models share one grid.
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument(
        "--cv-hidden-dims",
        nargs="+",
        type=int,
        default=list(DEFAULT_CV_HIDDEN_DIMS),
    )
    parser.add_argument(
        "--cv-dropouts",
        nargs="+",
        type=float,
        default=list(DEFAULT_CV_DROPOUTS),
    )
    parser.add_argument(
        "--cv-lrs",
        nargs="+",
        type=float,
        default=list(DEFAULT_CV_LRS),
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Skip 5-fold CV; fit a single model with --hidden-dim/--dropout/--lr.",
    )
    parsed = parser.parse_args()
    inputs_dir = Path(parsed.inputs_dir)
    if parsed.input_csv is None:
        parsed.input_csv = str(inputs_dir / f"longitudinal_landmark{int(parsed.landmark_day)}.csv")
    if parsed.manifest is None:
        parsed.manifest = str(
            inputs_dir / f"longitudinal_landmark{int(parsed.landmark_day)}_manifest.json"
        )
    main(parsed)
