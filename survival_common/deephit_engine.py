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
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    from torch.utils.data import DataLoader, Dataset

    TORCH_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    torch = None
    nn = None
    Dataset = object
    DataLoader = None
    pack_padded_sequence = None
    pad_packed_sequence = None
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


def select_device() -> str:
    """Use the first CUDA GPU when available, otherwise fall back to CPU."""
    require_torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


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
    dynamic: bool = False,
) -> dict[str, dict]:
    """Build per-patient model input sequences.

    The feature layout is identical in both arms:
    ``[normalized values | observed mask | relative time]``, width ``2F + 1``.

    ``dynamic=False`` attaches one scalar target per patient (from
    :func:`~survival_common.longitudinal_targets.patient_targets`, indexed by
    patient). ``dynamic=True`` attaches per-step target arrays aligned to the
    sequence (from
    :func:`~survival_common.longitudinal_targets.patient_targets_dynamic`,
    indexed by ``(patient, TIME)``), plus a ``step_mask`` marking which steps
    have a target at all -- a step whose patient is no longer at risk is carried
    in ``x`` (the GRU still needs it to build state) but contributes no loss.
    """
    sequences: dict[str, dict] = {}
    denom = max(float(max_observed_time), 1.0)
    for mrn, group in df.groupby(id_col, sort=False):
        if not dynamic and mrn not in targets.index:
            continue
        group = group.sort_values(time_col)
        raw = group[feature_cols].astype(float)
        mask = raw.notna().astype(float)
        values = ((raw - mean) / std).fillna(0.0)
        times = pd.to_numeric(group[time_col], errors="coerce").to_numpy(dtype=float)
        rel_time = (times / denom).reshape(-1, 1)
        x = np.hstack([values.to_numpy(dtype=np.float32), mask.to_numpy(dtype=np.float32), rel_time.astype(np.float32)])

        if not dynamic:
            target = targets.loc[mrn]
            sequences[str(mrn)] = {
                "x": x.astype(np.float32),
                "length": int(len(x)),
                "label": int(target["label"]),
                "duration_bin": int(target["duration_bin"]),
                "duration": float(target["duration"]),
            }
            continue

        # Per-step targets, aligned positionally to the sorted sequence. Steps
        # with no target row (patient not at risk at that time) get step_mask=0
        # and placeholder target values that the loss never reads.
        n_steps = len(x)
        step_label = np.zeros(n_steps, dtype=np.int64)
        step_bin = np.ones(n_steps, dtype=np.int64)
        step_duration = np.zeros(n_steps, dtype=np.float32)
        step_mask = np.zeros(n_steps, dtype=np.float32)
        for pos, t in enumerate(times):
            key = (mrn, float(t))
            if key not in targets.index:
                continue
            target = targets.loc[key]
            step_label[pos] = int(target["label"])
            step_bin[pos] = int(target["duration_bin"])
            step_duration[pos] = float(target["duration"])
            step_mask[pos] = 1.0
        if not step_mask.any():
            # Patient is never at risk anywhere in their sequence.
            continue
        sequences[str(mrn)] = {
            "x": x.astype(np.float32),
            "length": int(n_steps),
            "label": step_label,
            "duration_bin": step_bin,
            "duration": step_duration,
            "step_mask": step_mask,
            "times": times.astype(np.float32),
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
        out = {
            "id": sample_id,
            "x": item["x"],
            "length": item["length"],
            "label": item["label"],
            "duration_bin": item["duration_bin"],
            "duration": item["duration"],
        }
        # Dynamic arm only; their presence is what collate_batch dispatches on.
        for key in ("step_mask", "times"):
            if key in item:
                out[key] = item[key]
        return out


def collate_batch(batch: list[dict]) -> dict:
    """Right-zero-pad a batch of sequences to the batch's max length.

    Handles both target shapes. In the landmark arm ``label``/``duration_bin``/
    ``duration`` are per-patient scalars and collate to ``[B]``. In the dynamic
    arm they are per-step arrays and collate to ``[B, T]``, padded with a
    ``step_mask`` of zeros so the loss ignores padded positions. ``length`` is
    carried separately either way for ``pack_padded_sequence``.
    """
    max_len = max(item["length"] for item in batch)
    feat_dim = batch[0]["x"].shape[1]
    x = np.zeros((len(batch), max_len, feat_dim), dtype=np.float32)
    for idx, item in enumerate(batch):
        x[idx, : item["length"], :] = item["x"]

    out = {
        "ids": [item["id"] for item in batch],
        "x": torch.tensor(x, dtype=torch.float32),
        "length": torch.tensor([item["length"] for item in batch], dtype=torch.long),
    }

    if "step_mask" not in batch[0]:
        out["label"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        out["duration_bin"] = torch.tensor(
            [item["duration_bin"] for item in batch], dtype=torch.long
        )
        out["duration"] = torch.tensor(
            [item["duration"] for item in batch], dtype=torch.float32
        )
        return out

    n = len(batch)
    label = np.zeros((n, max_len), dtype=np.int64)
    duration_bin = np.ones((n, max_len), dtype=np.int64)
    duration = np.zeros((n, max_len), dtype=np.float32)
    step_mask = np.zeros((n, max_len), dtype=np.float32)
    times = np.zeros((n, max_len), dtype=np.float32)
    for idx, item in enumerate(batch):
        end = item["length"]
        label[idx, :end] = item["label"]
        duration_bin[idx, :end] = item["duration_bin"]
        duration[idx, :end] = item["duration"]
        step_mask[idx, :end] = item["step_mask"]
        times[idx, :end] = item["times"]
    out["label"] = torch.tensor(label, dtype=torch.long)
    out["duration_bin"] = torch.tensor(duration_bin, dtype=torch.long)
    out["duration"] = torch.tensor(duration, dtype=torch.float32)
    out["step_mask"] = torch.tensor(step_mask, dtype=torch.float32)
    out["times"] = torch.tensor(times, dtype=torch.float32)
    return out


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

        def forward(self, x, length, *, per_step: bool = False):
            """Return head logits for the sequence.

            ``per_step=False`` (the landmark arm) reads the final hidden state
            and returns ``[B, n_events*horizon + 1]`` -- one prediction per
            patient. ``per_step=True`` (the dynamic arm) returns
            ``[B, T, n_events*horizon + 1]`` -- one prediction per observation
            time.

            The GRU is causal, so step ``t``'s output depends only on steps
            ``<= t``; no future masking is needed to make the per-step
            predictions honest. ``tests/test_dynamic_no_future_leakage.py``
            asserts that property directly rather than trusting it.
            """
            packed = pack_padded_sequence(
                x,
                length.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            out, h_n = self.gru(packed)
            if not per_step:
                return self.head(h_n[-1])
            out, _ = pad_packed_sequence(out, batch_first=True)
            return self.head(out)

        def probabilities(self, logits):
            """Softmax the head logits into (event CIF pmf, no-event mass).

            Shapes follow the input: ``[B, K]`` -> ``([B, n_events, horizon],
            [B])``; ``[B, T, K]`` -> ``([B, T, n_events, horizon], [B, T])``.
            The softmax is always over the final axis, which is the one the
            head emits.
            """
            probs = torch.softmax(logits, dim=-1)
            lead = probs.shape[:-1]
            event_probs = probs[..., : self.n_events * self.horizon].reshape(
                *lead, self.n_events, self.horizon
            )
            no_event = probs[..., -1]
            return event_probs, no_event
else:
    class DynamicDeepHitGRU:  # pragma: no cover - only used when torch is missing
        def __init__(self, *args, **kwargs):
            require_torch()


def deephit_nll(
    model: "DynamicDeepHitGRU", logits, label, duration_bin, *, reduction: str = "mean"
) -> "torch.Tensor":
    """Discrete-time competing-risks negative log-likelihood.

    Observed branch: -log P(event at its bin). Censored branch: -log P(no
    event through the censoring bin), i.e. the survival-plus-no-event mass
    strictly after the censoring bin. Vectorized as a reverse-cumsum + gather
    (equivalent to summing event_probs[:, censor_idx+1:] per row) rather than
    a per-row Python loop, since this runs inside every fold of a hyperparameter
    grid search.

    ``reduction="none"`` returns the per-row loss **in input order** rather than
    the batch mean, for callers that need to weight rows (the dynamic arm
    weights each step by 1/T_i). Note the two branches are computed on disjoint
    subsets and must be scattered back, not concatenated: concatenating would
    silently reorder rows relative to their weights.
    """
    event_probs, no_event = model.probabilities(logits)
    eps = 1e-8
    idx = torch.clamp(duration_bin, min=1, max=model.horizon) - 1
    observed = label > 0
    per_row = torch.zeros(logits.shape[0], dtype=logits.dtype, device=logits.device)
    if observed.any():
        event_idx = label[observed] - 1
        time_idx = idx[observed]
        prob = event_probs[observed, event_idx, time_idx]
        per_row[observed] = -torch.log(prob + eps)
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
        per_row[~observed] = -torch.log(future_mass + eps)
    if reduction == "none":
        return per_row
    if reduction == "mean":
        return per_row.mean()
    raise ValueError(f"Unsupported reduction {reduction!r}; use 'mean' or 'none'.")


def deephit_nll_dynamic(
    model: "DynamicDeepHitGRU", logits, label, duration_bin, step_mask
) -> "torch.Tensor":
    """Per-step discrete-time NLL for the dynamic arm.

    ``logits`` is ``[B, T, K]`` and ``label``/``duration_bin``/``step_mask`` are
    ``[B, T]``. Valid steps are flattened and scored by the identical likelihood
    :func:`deephit_nll` applies -- the observed/censored branches and the
    reverse-cumsum over future bins are reused verbatim rather than reimplemented,
    so the two arms cannot drift apart on the loss.

    Each step is weighted by ``1 / T_i`` (its own patient's valid-step count) so
    that every patient contributes equally regardless of how much lab history
    they have. Without this a handful of heavily-sampled patients would dominate
    the objective, and the model would fit them at everyone else's expense.
    """
    valid = step_mask.bool()
    if not valid.any():
        raise ValueError("deephit_nll_dynamic received a batch with no valid steps.")

    # 1/T_i for each patient, broadcast to that patient's steps.
    steps_per_patient = valid.sum(dim=1, keepdim=True).clamp(min=1)
    weights = (1.0 / steps_per_patient.to(logits.dtype)).expand_as(valid)[valid]

    flat_logits = logits[valid]
    flat_label = label[valid]
    flat_bin = duration_bin[valid]

    per_step = deephit_nll(
        model, flat_logits, flat_label, flat_bin, reduction="none"
    )
    # Normalize by total weight so the loss scale is independent of batch size
    # and of how many steps the batch happens to carry.
    return (per_step * weights).sum() / weights.sum()


def run_epoch(model, loader, optimizer, device: str, *, dynamic: bool = False) -> float:
    model.train(optimizer is not None)
    losses = []
    for batch in loader:
        x = batch["x"].to(device)
        length = batch["length"].to(device)
        label = batch["label"].to(device)
        duration_bin = batch["duration_bin"].to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(x, length, per_step=dynamic)
            if dynamic:
                loss = deephit_nll_dynamic(
                    model, logits, label, duration_bin, batch["step_mask"].to(device)
                )
            else:
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


def predict_dynamic(
    model, loader, device: str, *, id_col: str, time_col: str = "TIME"
) -> pd.DataFrame:
    """One prediction row per (patient, prediction time).

    Emits the same ``event_{k}_risk_h{h}`` / ``event_{k}_risk_total`` columns
    :func:`predict` does, plus ``time_col``, so downstream metric code can treat
    a single-time slice of this frame exactly like a landmark prediction frame.
    Padded and not-at-risk steps are skipped via ``step_mask``.
    """
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["x"].to(device), batch["length"].to(device), per_step=True
            )
            event_probs, no_event = model.probabilities(logits)
            event_probs_np = event_probs.cpu().numpy()   # [B, T, n_events, horizon]
            no_event_np = no_event.cpu().numpy()         # [B, T]
            step_mask_np = batch["step_mask"].numpy()
            times_np = batch["times"].numpy()
            label_np = batch["label"].numpy()
            bin_np = batch["duration_bin"].numpy()
            duration_np = batch["duration"].numpy()
            for row_idx, sample_id in enumerate(batch["ids"]):
                for step in range(int(batch["length"][row_idx])):
                    if step_mask_np[row_idx, step] <= 0:
                        continue
                    row = {
                        id_col: sample_id,
                        time_col: float(times_np[row_idx, step]),
                        "duration": float(duration_np[row_idx, step]),
                        "duration_bin": int(bin_np[row_idx, step]),
                        "label": int(label_np[row_idx, step]),
                        "no_event_probability": float(no_event_np[row_idx, step]),
                    }
                    for event_idx in range(event_probs_np.shape[2]):
                        pmf = event_probs_np[row_idx, step, event_idx, :]
                        cif = np.cumsum(pmf)
                        row[f"event_{event_idx + 1}_risk_total"] = float(cif[-1])
                        for horizon, risk in enumerate(cif, start=1):
                            row[f"event_{event_idx + 1}_risk_h{horizon}"] = float(risk)
                    rows.append(row)
    return pd.DataFrame(rows)


def landmark_slice(
    pred: pd.DataFrame,
    landmark_targets: pd.DataFrame,
    *,
    id_col: str,
    time_col: str = "TIME",
) -> pd.DataFrame:
    """Reduce a dynamic prediction frame to one row per patient, at the landmark.

    This is the row the landmark arm would have produced for the same patient,
    so the result can be handed to :func:`compute_metrics` unchanged and the
    dynamic arm's headline numbers stay comparable to Cox/XGBoost. Patients whose
    landmark time is absent from ``pred`` (not at risk there) are dropped, which
    is the same exclusion the landmark arm applies.
    """
    landmark_by_id = (
        landmark_targets["landmark_time"].astype(float).rename("__landmark_time")
    )
    landmark_by_id.index = landmark_by_id.index.map(str)
    out = pred.copy()
    out["__landmark_time"] = out[id_col].map(str).map(landmark_by_id)
    out = out.loc[
        out["__landmark_time"].notna()
        & np.isclose(out[time_col].astype(float), out["__landmark_time"])
    ]
    return out.drop(columns="__landmark_time").reset_index(drop=True)


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
    dynamic: bool = False,
) -> tuple[pd.DataFrame, list[dict], float]:
    """Train DeepHit on `train_ids` watching `valid_ids` for early stopping,
    predict on `eval_ids`. Normalization is fit on train_ids only -- never on
    valid or eval. Returns (pred_df, history, best_valid_loss).

    ``dynamic=True`` trains against per-step targets and returns one prediction
    row per (patient, time) instead of per patient; ``targets`` must then be a
    ``(patient, TIME)``-indexed frame from ``patient_targets_dynamic``.
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
        dynamic=dynamic,
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

    device = select_device()
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
        train_loss = run_epoch(model, train_loader, optimizer, device, dynamic=dynamic)
        valid_loss = run_epoch(model, valid_loader, None, device, dynamic=dynamic)
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
    if dynamic:
        pred = predict_dynamic(model, eval_loader, device, id_col=id_col, time_col=time_col)
    else:
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
    """Compute per-event C-index and IPCW mean AUC(t) (see ``cox_engine.compute_ipcw_auc_t``
    for the shared fixed-horizon oracle contract this honors).

    Competing-risks convention: the IPCW censoring-weight reference distribution
    binarizes on ``(label == event_idx)``, so any competing event (a different
    cause, or true censoring) is folded into "censored" for the KM censoring-weight
    estimator -- a cause-specific convention, not a CIF/Aalen-Johansen-based one.
    ``cox_engine.compute_ipcw_auc_t`` uses the identical convention on its
    pre-supplied cause-specific ``event_col``, so this bias is consistent in
    direction and magnitude across the Cox, XGBoost, and DeepHit arms wherever
    competing events exist in the cohort (e.g. death without platinum). This is a
    stated methodological choice, not a defect.
    """
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
                # Canonical spelling (survival_common.metrics_schema). This
                # frame is per-cause, so `endpoint` names the cause -- the
                # column was called `event` before the schema cutover.
                "endpoint": event_name,
                "n_test": int(valid.sum()),
                "n_events_test": int(event[valid].sum()),
                # DeepHit scores the held-out block only; the train-side twins
                # are NaN so readers can assume the canonical columns exist.
                "train_val_c_index": float("nan"),
                "test_c_index": c_index,
                "train_val_mean_auc_t": float("nan"),
                "test_mean_auc_t": mean_auc,
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


# A prediction time is reported only if its risk set is at least this large and
# carries at least this many events of the cause. Below that, C-index and IPCW
# AUC(t) are dominated by sampling noise, and the late prediction times in a
# dynamic sweep always thin out this way -- the risk set shrinks monotonically
# as patients have their event, die, or are censored. Reporting a 0.93 C-index
# computed on 6 patients and 2 events next to a 0.61 on 400 invites exactly the
# wrong reading, so those rows are emitted with NaN metrics and a `note` naming
# the shortfall rather than being silently dropped (a gap in the series is
# itself information) or silently reported.
MIN_RISK_SET_FOR_BY_TIME = 25
MIN_EVENTS_FOR_BY_TIME = 5


def compute_metrics_by_time(
    dynamic_pred: pd.DataFrame,
    *,
    event_names: list[str],
    train_val_targets: pd.DataFrame,
    fixed_horizons_by_event: dict[str, np.ndarray],
    id_col: str,
    time_col: str = "TIME",
    quantiles: tuple[float, ...] = DEFAULT_AUC_QUANTILES,
    time_unit_days: int = 7,
    min_risk_set: int = MIN_RISK_SET_FOR_BY_TIME,
    min_events: int = MIN_EVENTS_FOR_BY_TIME,
) -> pd.DataFrame:
    """Score the dynamic arm separately at each prediction time.

    This is the output that makes the dynamic arm worth running: it shows
    whether a patient's risk estimate actually sharpens as their lab history
    accrues, which a single landmark number cannot express.

    Reuses :func:`compute_metrics` and :func:`compute_brier_for_pred`
    unmodified, once per prediction time, so a by-time row at
    ``prediction_time == landmark_time`` is computed by exactly the same code
    that produces the headline ``*_metrics.csv`` row. That identity is what
    makes the series anchorable to the other model arms, and it is asserted in
    tests/test_dynamic_metrics_schema.py.

    The returned frame carries the canonical metric columns plus
    ``prediction_time``, ``prediction_day``, ``n_at_risk`` and ``note``. It is
    written to a SEPARATE ``*_metrics_by_time.csv`` -- invariant #9 fixes the
    canonical block on ``*_metrics.csv``, which stays one row per cause.

    Interpretation caveat, which belongs in any write-up of these numbers: the
    risk set at a late prediction time is not a sample of the cohort, it is the
    subset that had not yet had the event. Rising AUC across prediction time is
    therefore not a clean "the model gets better" claim -- the population being
    scored changes underneath it. The held-back-window ablation
    (``incremental_risk.py``) is the cleaner contrast, because it holds the risk
    set fixed and varies only the input window.
    """
    require_lifelines()
    if dynamic_pred.empty:
        return pd.DataFrame()
    if time_col not in dynamic_pred.columns:
        raise ValueError(
            f"Dynamic predictions have no {time_col!r} column, so they cannot be "
            "grouped by prediction time. Was this frame produced with --dynamic?"
        )

    rows: list[dict] = []
    for pred_time, block in dynamic_pred.groupby(time_col, sort=True):
        pred_time = float(pred_time)
        n_at_risk = int(len(block))
        for event_idx, event_name in enumerate(event_names, start=1):
            n_events = int(block["label"].eq(event_idx).sum())
            base = {
                "prediction_time": pred_time,
                # Days since the per-patient TIME origin. The origin is
                # landmark_time, so this is days relative to the landmark, and
                # is negative for pre-landmark prediction times.
                "prediction_day": pred_time * float(time_unit_days),
                "endpoint": event_name,
                "n_at_risk": n_at_risk,
            }
            if n_at_risk < min_risk_set or n_events < min_events:
                rows.append({
                    **base,
                    "n_test": n_at_risk,
                    "n_events_test": n_events,
                    "test_c_index": float("nan"),
                    "test_mean_auc_t": float("nan"),
                    "test_integrated_brier": float("nan"),
                    "note": (
                        f"underpowered: n_at_risk={n_at_risk} "
                        f"(min {min_risk_set}), n_events={n_events} "
                        f"(min {min_events})"
                    ),
                })
                continue

            # One cause at a time: compute_metrics loops over every name it is
            # given, and the power gate above is per-cause, so passing the full
            # list here would resurrect a cause this prediction time cannot
            # support.
            metrics, _auc = compute_metrics(
                block,
                event_names=[event_name],
                train_val_targets=train_val_targets,
                quantiles=quantiles,
                fixed_horizons_by_event=fixed_horizons_by_event,
            )
            _brier, ibs_by_event = compute_brier_for_pred(
                block,
                event_names=[event_name],
                train_val_targets=train_val_targets,
                horizons_by_event=fixed_horizons_by_event,
                time_unit_days=time_unit_days,
            )
            if metrics.empty:
                rows.append({**base, "note": "compute_metrics returned no row"})
                continue
            row = metrics.iloc[0]
            rows.append({
                **base,
                "n_test": int(row["n_test"]),
                "n_events_test": int(row["n_events_test"]),
                "test_c_index": float(row["test_c_index"]),
                "test_mean_auc_t": float(row["test_mean_auc_t"]),
                "test_integrated_brier": float(
                    ibs_by_event.get(event_name, float("nan"))
                ),
                "note": "",
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["endpoint", "prediction_time"]).reset_index(drop=True)


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
    dynamic: bool = False,
    landmark_targets: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """5-fold stratified CV over (hidden_dim x dropout x lr).

    ``dynamic=True`` fits each fold against the per-step targets in ``targets``
    but *scores* it on the landmark slice, so a fold's reported C-index means the
    same thing in both arms and hyperparameters are not selected by a metric the
    landmark arm never computes. ``landmark_targets`` (patient-indexed, required
    when ``dynamic``) supplies both that slice and the IPCW reference.

    `train_val_static` is patient-indexed with PLATINUM/DEATH for stratification
    (combined 4-cell label via iter_stratified_folds(event_col=None)). For each
    fold and each combo, the model is trained on fold_train MRNs with fold_val
    watched for early stopping (and used as the metric set). Returns
    (fold_df, cv_summary_df, best_row).

    Caveat (Finding 6): because the early-stopping watch set and the reported
    metric set are the same fold, each fold's c_index_val/mean_auc_t_val is
    optimistically biased relative to a genuinely held-out 3-way split -- the
    model is tuned (via early stopping) specifically to perform well on the
    exact fold whose score is then reported. This is the standard, if slightly
    optimistic, convention for CV-with-early-stopping at this cohort size (see
    the identical convention and caveat in multivariate_analysis.py's
    run_xgboost) and is a documented methodological choice, not a bug.
    """
    if dynamic and landmark_targets is None:
        raise ValueError("cv_run(dynamic=True) requires landmark_targets.")
    # Patient-indexed targets used for fold scoring and as the IPCW reference.
    # In the landmark arm this is `targets` itself.
    scoring_targets = landmark_targets if dynamic else targets

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
                    dynamic=dynamic,
                )
                row["best_valid_loss"] = float(best_valid)
                row["n_epochs"] = int(len(history))
                if dynamic:
                    pred = landmark_slice(
                        pred,
                        landmark_targets,
                        id_col=id_col,
                        time_col=time_col,
                    )
                fold_train_targets = scoring_targets.loc[
                    scoring_targets.index.map(str).isin(fold_train_ids)
                ].copy()
                metrics_df, _ = compute_metrics(
                    pred,
                    event_names=event_names,
                    train_val_targets=fold_train_targets,
                    # build_deephit_parser exposes no --auc-quantiles, so read it
                    # defensively. These quantiles are only a fallback anyway:
                    # fixed_horizons_by_event is always supplied here, and
                    # compute_metrics ignores `quantiles` whenever it is.
                    quantiles=tuple(
                        getattr(args, "auc_quantiles", DEFAULT_AUC_QUANTILES)
                    ),
                    fixed_horizons_by_event=fixed_horizons_by_event,
                )
                _, ibs_by_event = compute_brier_for_pred(
                    pred,
                    event_names=event_names,
                    train_val_targets=fold_train_targets,
                    horizons_by_event=fixed_horizons_by_event,
                )
                for _, mrow in metrics_df.iterrows():
                    # compute_metrics emits the canonical spelling; index
                    # directly rather than .get() so a future schema drift
                    # raises instead of quietly filling these with NaN.
                    event_name = mrow["endpoint"]
                    row[f"c_index_val__{event_name}"] = float(mrow["test_c_index"])
                    row[f"mean_auc_t_val__{event_name}"] = float(
                        mrow["test_mean_auc_t"]
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

    # Refuse to "select" hyperparameters off an all-NaN ranking column. The
    # n_valid_folds guard above only proves the fits ran -- scoring can still
    # have failed in every fold (a missing args attribute, an IPCW failure),
    # leaving the sort to fall through to its tie-break and report a confident
    # choice that carries no information. Fail loudly instead.
    if not np.isfinite(candidate["__rank_score"]).any():
        notes = (
            fold_df["note"].fillna("").astype(str).replace("", np.nan).dropna().unique()
        )
        detail = f" First fold error: {notes[0]}" if len(notes) else ""
        raise RuntimeError(
            "DeepHit CV produced no usable C-index in any fold, so no "
            "hyperparameter combination can be ranked."
            f"{detail}"
        )
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
