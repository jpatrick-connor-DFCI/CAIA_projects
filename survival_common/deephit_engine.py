"""Dynamic-DeepHit-style recurrent survival model for longitudinal lab histories.

Consumes the person-period frame produced by
:func:`survival_common.cohort.build_person_period_wide` and the discrete-time
targets produced by :func:`survival_common.longitudinal_targets.patient_targets`,
and trains a compact GRU model with a discrete competing-risks likelihood. It
is intentionally self-contained so it does not depend on the original
TensorFlow 1.x Dynamic-DeepHit reference code.

Torch-gated (README invariant): this module must import cleanly with no torch
installed. Three layers make that true: (1) the try/except below, with
``Dataset = object`` so ``class SequenceDataset(Dataset)`` still evaluates at
import time; (2) :class:`DynamicDeepHitGRU` is only defined with real torch
ops when ``nn is not None``, else a stub that raises on construction; (3)
:func:`require_torch` is called only inside functions (``set_seed``,
``train_evaluate``, ...), never at module import.

Does not construct targets or resolve ``--config`` -- that lives in
``survival_common.longitudinal_targets`` (torch-free, unit-tested without
this module). Does not parse CLI args or read manifests -- that lives in
``survival_common.longitudinal_runners``.
"""

from __future__ import annotations

from itertools import product
from typing import Any

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
        """No-op tqdm fallback -- yields the iterable unchanged."""
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

from survival_common.helper import (
    MIN_IPCW_TIMELINE_COVERAGE,
    MIN_IPCW_VALID_HORIZONS,
    _make_survival_array,
    assert_disjoint_folds,
    compute_brier,
    iter_stratified_folds,
)

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
            "torch is required for survival_common.deephit_engine."
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


def fit_normalization(
    df: pd.DataFrame, *, id_col: str, feature_cols: list[str], train_ids: set
) -> tuple[pd.Series, pd.Series]:
    train = df.loc[df[id_col].astype(str).isin(train_ids), feature_cols]
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


def deephit_nll(model: "DynamicDeepHitGRU", logits, label, duration_bin) -> "torch.Tensor":
    """Discrete-time competing-risks negative log-likelihood.

    Observed branch: -log P(event at its bin). Censored branch: -log P(no
    event through the censoring bin), i.e. the survival-plus-no-event mass
    strictly after the censoring bin. Vectorized as a reverse-cumsum + gather
    (equivalent to summing event_probs[:, censor_idx+1:] per row) rather than
    a per-row Python loop, since this runs inside every fold of a hyperparameter
    grid search.
    """
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
        censored_probs = event_probs[~observed]
        # reverse-cumsum over the horizon axis: rev_cumsum[..., t] = sum_{t' >= t} probs[..., t']
        rev_cumsum = torch.flip(torch.cumsum(torch.flip(censored_probs, dims=[-1]), dim=-1), dims=[-1])
        n_rows = censored_probs.shape[0]
        horizon = censored_probs.shape[-1]
        future_start = censor_idx + 1
        # future_mass per row = sum over events of rev_cumsum[row, event, future_start] when
        # future_start < horizon, else 0 (no bins remain strictly after the censoring bin).
        in_range = future_start < horizon
        gather_idx = torch.clamp(future_start, max=horizon - 1)
        gathered = rev_cumsum[torch.arange(n_rows), :, gather_idx]  # [n_rows, n_events]
        future_mass = torch.where(
            in_range, gathered.sum(dim=-1), torch.zeros(n_rows, dtype=gathered.dtype, device=gathered.device)
        )
        future_mass = future_mass + no_event[~observed]
        losses.append(-torch.log(future_mass + eps))
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


def predict(model, loader, device: str, *, id_col: str) -> pd.DataFrame:
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
                    id_col: sample_id,
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
    args: Any,
    n_events: int,
    horizon: int,
    hidden_dim: int,
    dropout: float,
    lr: float,
    seed: int,
) -> tuple[pd.DataFrame, list[dict], float]:
    """Train DeepHit on `train_ids` watching `valid_ids` for early stopping,
    predict on `eval_ids`. Normalization is fit on train_ids only -- never on
    valid or eval. Returns (pred_df, history, best_valid_loss).
    """
    require_torch()
    set_seed(seed)
    mean, std = fit_normalization(df, id_col=id_col, feature_cols=feature_cols, train_ids=train_ids)
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

    device = "cuda" if getattr(args, "cuda", False) and torch.cuda.is_available() else "cpu"
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
    history: list[dict] = []
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
    pred = predict(model, eval_loader, device, id_col=id_col)
    return pred, history, best_valid


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
            # Mean AUC(t) is integrated over exactly the fixed requested
            # timeline (never a re-derived np.arange grid), masking any
            # horizon whose risk_h{t} column isn't materialized rather than
            # substituting the time-constant total risk -- matching the
            # oracle contract in cox_engine.compute_ipcw_auc_t (:194-207).
            estimable_times = np.asarray(
                [
                    t
                    for t in horizons
                    if f"event_{event_idx}_risk_h{int(t)}" in pred.columns
                ],
                dtype=float,
            )
            if len(estimable_times) >= MIN_IPCW_VALID_HORIZONS and len(
                estimable_times
            ) / len(horizons) >= MIN_IPCW_TIMELINE_COVERAGE:
                try:
                    risk_cols = [
                        pred.loc[valid, f"event_{event_idx}_risk_h{int(t)}"].to_numpy(
                            dtype=float
                        )
                        for t in estimable_times
                    ]
                    time_risk = np.column_stack(risk_cols)
                    _, mean_auc = cumulative_dynamic_auc(
                        ref_surv, eval_surv, time_risk, estimable_times
                    )
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


def compute_brier_for_pred(
    pred: pd.DataFrame,
    *,
    event_names: list[str],
    train_val_targets: pd.DataFrame,
    horizons_by_event: dict[str, np.ndarray],
    time_unit_days: int = 7,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Per-cause IPCW Brier on DeepHit's PMF predictions.

    For each cause k, treats the event as binary (1 iff cause==k, 0 otherwise
    including censored AND competing) and feeds 1 - CIF_k(h) as the survival
    estimate at horizon h. For competing-risk configs this is the binary
    "cause-of-interest" Brier -- not the cumulative-incidence Brier -- but
    it's the convention consistent with how the per-cause AUC(t) above is
    computed. Train arrays come from `train_val_targets` so the IPCW weights
    are train-only. `time_unit_days` only labels the `horizon_days` display
    column (see helper.compute_brier); it does not rescale any horizon.
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

        # Mask any horizon whose risk_h{t} column isn't materialized rather
        # than substituting the time-constant total risk -- matching the
        # oracle contract's masking behavior (never a substitute summary).
        estimable_horizons = np.asarray(
            [h for h in horizons if f"event_{event_idx}_risk_h{int(h)}" in pred.columns],
            dtype=float,
        )
        if len(estimable_horizons) == 0:
            integrated_by_event[event_name] = float("nan")
            continue
        surv_cols = [
            1.0 - pred.loc[valid, f"event_{event_idx}_risk_h{int(h)}"].to_numpy(dtype=float)
            for h in estimable_horizons
        ]
        horizons = estimable_horizons
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
            time_unit_days=time_unit_days,
        )
        if not brier_df.empty:
            brier_df = brier_df.copy()
            brier_df.insert(0, "event", event_name)
            rows.extend(brier_df.to_dict("records"))
        integrated_by_event[event_name] = ibs

    return pd.DataFrame(rows), integrated_by_event


def cv_run(
    *,
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    feature_cols: list[str],
    targets: pd.DataFrame,
    train_val_static: pd.DataFrame,
    args: Any,
    n_events: int,
    event_names: list[str],
    fixed_horizons_by_event: dict[str, np.ndarray],
    config_label: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """5-fold stratified CV over (hidden_dim x dropout x lr).

    `train_val_static` is patient-indexed with PLATINUM/DEATH for stratification
    (combined 4-cell label via iter_stratified_folds(event_col=None)). For each
    fold and each combo, the model is trained on fold_train MRNs with fold_val
    watched for early stopping (and used as the metric set). Returns
    (fold_df, cv_summary_df, best_row).
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
        desc=f"DeepHit CV[{config_label}]",
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
