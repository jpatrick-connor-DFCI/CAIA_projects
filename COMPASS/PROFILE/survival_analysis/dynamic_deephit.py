"""
Dynamic-DeepHit-style recurrent survival model for longitudinal lab histories.

This script consumes the person-period CSV produced by
survlatent_ode/build_input.py and trains a compact GRU model with a discrete
competing-risks likelihood. It is intentionally self-contained so it does not
depend on the original TensorFlow 1.x Dynamic-DeepHit reference code.

Outputs:
  dynamic_deephit_metrics.csv
  dynamic_deephit_auc_t.csv
  dynamic_deephit_patient_risks.csv
  dynamic_deephit_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
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
    from sksurv.metrics import cumulative_dynamic_auc

    SKSURV_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    cumulative_dynamic_auc = None
    SKSURV_IMPORT_ERROR = exc

SURVIVAL_DIR = Path(__file__).resolve().parent
if str(SURVIVAL_DIR) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_DIR))

from cox_aggregated import _make_survival_array  # noqa: E402

DEFAULT_RESULTS = Path("/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/survival_analysis")
DEFAULT_SEED = 42
DEFAULT_MAX_PRED_WINDOW = 52
DEFAULT_AUC_QUANTILES = (0.25, 0.375, 0.50, 0.625, 0.75)


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
            }
        )
    return pd.DataFrame(rows).set_index(id_col)


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


def compute_metrics(
    pred: pd.DataFrame,
    *,
    event_names: list[str],
    train_val_targets: pd.DataFrame,
    quantiles: tuple[float, ...],
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
            horizons = np.asarray([int(v) for v in np.quantile(event_times, quantiles)], dtype=float)
            for quantile, horizon in zip(quantiles, horizons):
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
                    _, mean_auc = cumulative_dynamic_auc(ref_surv, eval_surv, risk[valid], times)
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
    mean, std = fit_normalization(df, feature_cols=feature_cols, train_ids=train_ids)
    max_observed_time = float(df.loc[df[id_col].astype(str).isin(train_ids), time_col].max())
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
    test_ds = SequenceDataset(sequences, sorted(test_ids))
    if len(train_ds) == 0 or len(valid_ds) == 0 or len(test_ds) == 0:
        raise ValueError(
            f"Empty split after sequence assembly: train={len(train_ds)} valid={len(valid_ds)} test={len(test_ds)}"
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    input_dim = next(iter(train_loader))["x"].shape[-1]
    model = DynamicDeepHitGRU(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        n_events=len(event_names),
        horizon=args.max_pred_window,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_valid = np.inf
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        valid_loss = run_epoch(model, valid_loader, None, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
        if valid_loss < best_valid - args.min_delta:
            best_valid = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}")
        if epochs_without_improvement >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    pred = predict(model, test_loader, device)
    train_val_targets = targets.loc[
        targets.index.map(str).isin(train_ids.union(valid_ids))
    ].copy()
    metrics, auc_t = compute_metrics(
        pred,
        event_names=event_names,
        train_val_targets=train_val_targets,
        quantiles=tuple(args.auc_quantiles),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"dynamic_deephit_patient_risks_{args.config}.csv"
    metrics_path = output_dir / f"dynamic_deephit_metrics_{args.config}.csv"
    auc_path = output_dir / f"dynamic_deephit_auc_t_{args.config}.csv"
    manifest_out = output_dir / f"dynamic_deephit_manifest_{args.config}.json"
    pred.to_csv(pred_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    auc_t.to_csv(auc_path, index=False)
    manifest_out.write_text(
        json.dumps(
            {
                "input_csv": str(input_csv),
                "source_manifest": str(manifest_path),
                "config": args.config,
                "event_names": event_names,
                "feature_cols": feature_cols,
                "max_pred_window": args.max_pred_window,
                "seed": args.seed,
                "split_counts": {
                    "train": len(train_ds),
                    "valid": len(valid_ds),
                    "test": len(test_ds),
                },
                "best_valid_loss": best_valid,
                "history": history,
            },
            indent=2,
        )
    )
    print(f"\nSaved:\n  {metrics_path}\n  {auc_path}\n  {pred_path}\n  {manifest_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default=str(DEFAULT_RESULTS / "survlatent_ode_input.csv"))
    parser.add_argument("--manifest", default=str(DEFAULT_RESULTS / "survlatent_ode_manifest.json"))
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--config", default="competing", help="competing, PLATINUM, or DEATH")
    parser.add_argument("--max-pred-window", type=int, default=DEFAULT_MAX_PRED_WINDOW)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--auc-quantiles", nargs="+", type=float, default=list(DEFAULT_AUC_QUANTILES))
    main(parser.parse_args())
