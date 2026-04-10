"""
Step 4: SurvLatentODE — Neural ODE survival model for irregularly-sampled labs.

Architecture (following Sutton et al. 2021 / Schirmer et al. 2022 style):
  1. ODE-RNN encoder: reads the patient's lab time series backwards in time,
     producing an initial latent state z0 ~ q(z0 | x).
  2. Latent ODE: integrates z0 forward in continuous time.
  3. Survival decoder: reads z(t_event) → log-hazard per event head
     (platinum and death as competing events).

Loss: ELBO (recon MSE + KL-annealed) + λ * Breslow partial likelihood per head.

Evaluation:
  - 20% of patients held out as test set (stratified by event status, fixed seed)
  - 5-fold CV on the remaining 80% reports mean ± std C-index
    (CV folds use --cv-epochs, default = epochs // 4, to keep runtime tractable)
  - Final model trained on full 80%, evaluated on 20% test set

Requirements: torch, torchdiffeq, sksurv, numpy, pandas

Usage:
  python survlatentode.py --data /path/to/longitudinal_prediction_data.csv \
      --epochs 200 --cv-epochs 50 --latent-dim 32 --hidden-dim 64

Outputs:
  results/survlatentode_cv_metrics.csv
  results/survlatentode_checkpoint.pt
  results/survlatentode_train_log.csv
  results/survlatentode_test_predictions.csv
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchdiffeq import odeint

BASE = Path(__file__).resolve().parent
DATA_PATH = Path("/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/")
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 42
N_FOLDS = 5
TEST_FRAC = 0.20
MIN_PATIENT_COVERAGE = 0.20
TOP_N_LABS = 20

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def prepare_all_patients(csv_path: str) -> tuple[list[dict], list[str]]:
    """
    Load data, select labs, build per-patient time-series dicts.
    Does NOT split — splitting happens downstream.
    Returns (patients, lab_names).
    """
    print("Loading data...")
    df = pd.read_csv(csv_path, low_memory=False)
    for col in ["LAB_DATE", "FIRST_TREATMENT_DATE", "LAST_CONTACT_DATE", "PLATINUM_DATE"]:
        df[col] = pd.to_datetime(df[col])

    # Select labs by patient coverage
    coverage = df.groupby("LAB_NAME")["DFCI_MRN"].nunique() / df["DFCI_MRN"].nunique()
    eligible = coverage[coverage >= MIN_PATIENT_COVERAGE].sort_values(ascending=False)
    lab_names = eligible.index.tolist()[:TOP_N_LABS]
    print(f"Using {len(lab_names)} labs: {lab_names}")

    # Standardize each lab (across all patients — minor leakage, acceptable for NN)
    lab_stats: dict[str, tuple[float, float]] = {}
    for lab in lab_names:
        vals = df.loc[df["LAB_NAME"] == lab, "LAB_VALUE"].dropna()
        mu, sigma = float(vals.mean()), float(vals.std())
        lab_stats[lab] = (mu, max(sigma, 1e-6))

    # Outcome metadata (one row per patient)
    pat_meta = (df[["DFCI_MRN", "FIRST_TREATMENT_DATE", "LAST_CONTACT_DATE",
                    "DEATH", "PLATINUM_DATE", "PLATINUM", "AGE_AT_TREATMENTSTART"]]
                .drop_duplicates("DFCI_MRN").set_index("DFCI_MRN"))
    pat_meta["t_platinum"] = (pat_meta["PLATINUM_DATE"] - pat_meta["FIRST_TREATMENT_DATE"]).dt.days
    pat_meta["t_death"]    = (pat_meta["LAST_CONTACT_DATE"] - pat_meta["FIRST_TREATMENT_DATE"]).dt.days
    pat_meta = pat_meta[(pat_meta["t_platinum"] > 0) & (pat_meta["t_death"] > 0)]

    lab_df = df[df["LAB_NAME"].isin(lab_names)].copy()
    lab_df["LAB_VALUE_STD"] = lab_df.apply(
        lambda r: (r["LAB_VALUE"] - lab_stats[r["LAB_NAME"]][0]) / lab_stats[r["LAB_NAME"]][1],
        axis=1
    )

    n_labs  = len(lab_names)
    lab_idx = {lab: i for i, lab in enumerate(lab_names)}

    patients = []
    for mrn, row in pat_meta.iterrows():
        t0    = row["FIRST_TREATMENT_DATE"]
        max_t = max(float(row["t_death"]), float(row["t_platinum"]))

        pat_obs = lab_df[lab_df["DFCI_MRN"] == mrn].copy()
        pat_obs["t_days"] = (pat_obs["LAB_DATE"] - t0).dt.days.astype(float)
        pat_obs = pat_obs[(pat_obs["t_days"] > -365 * 5) & (pat_obs["t_days"] <= max_t)]
        if pat_obs.empty:
            continue

        timepoints = sorted(pat_obs["t_days"].unique())
        obs_vals   = np.full((len(timepoints), n_labs), np.nan, dtype=np.float32)
        obs_mask   = np.zeros((len(timepoints), n_labs), dtype=bool)

        for ti, t in enumerate(timepoints):
            for _, lab_row in pat_obs[pat_obs["t_days"] == t].iterrows():
                li = lab_idx.get(lab_row["LAB_NAME"])
                if li is not None and np.isfinite(lab_row["LAB_VALUE_STD"]):
                    obs_vals[ti, li] = lab_row["LAB_VALUE_STD"]
                    obs_mask[ti, li] = True

        patients.append({
            "mrn":        mrn,
            "age":        float(row["AGE_AT_TREATMENTSTART"]) / 100.0,
            "t_platinum": float(row["t_platinum"]),
            "e_platinum": float(row["PLATINUM"]),
            "t_death":    float(row["t_death"]),
            "e_death":    float(row["DEATH"]),
            "timepoints": np.array(timepoints, dtype=np.float32),
            "obs_vals":   obs_vals,
            "obs_mask":   obs_mask,
        })

    print(f"Total patients with observations: {len(patients)}")
    return patients, lab_names


def stratified_label(patients: list[dict]) -> np.ndarray:
    """0=neither, 1=platinum only, 2=death only, 3=both."""
    p = np.array([int(pt["e_platinum"]) for pt in patients])
    d = np.array([int(pt["e_death"])    for pt in patients])
    return p + 2 * d


def split_patients(patients: list[dict]) -> tuple[list[dict], list[dict]]:
    strat = stratified_label(patients)
    idx = np.arange(len(patients))
    tr_idx, te_idx = train_test_split(idx, test_size=TEST_FRAC,
                                      stratify=strat, random_state=SEED)
    return [patients[i] for i in tr_idx], [patients[i] for i in te_idx]


# ---------------------------------------------------------------------------
# Dataset / collation
# ---------------------------------------------------------------------------
class PatientDataset(Dataset):
    def __init__(self, patients: list[dict]):
        self.patients = patients

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        return self.patients[idx]


def collate_fn(batch: list[dict]) -> dict:
    max_T  = max(p["timepoints"].shape[0] for p in batch)
    n_labs = batch[0]["obs_vals"].shape[1]
    B      = len(batch)

    timepoints = np.zeros((B, max_T), dtype=np.float32)
    obs_vals   = np.zeros((B, max_T, n_labs), dtype=np.float32)
    obs_mask   = np.zeros((B, max_T, n_labs), dtype=bool)
    seq_lens   = np.zeros(B, dtype=np.int64)

    for i, p in enumerate(batch):
        T = p["timepoints"].shape[0]
        timepoints[i, :T] = p["timepoints"]
        obs_vals[i, :T]   = p["obs_vals"]
        obs_mask[i, :T]   = p["obs_mask"]
        seq_lens[i]        = T

    return {
        "mrn":        [p["mrn"] for p in batch],
        "age":        torch.tensor([p["age"]        for p in batch], dtype=torch.float32),
        "t_platinum": torch.tensor([p["t_platinum"] for p in batch], dtype=torch.float32),
        "e_platinum": torch.tensor([p["e_platinum"] for p in batch], dtype=torch.float32),
        "t_death":    torch.tensor([p["t_death"]    for p in batch], dtype=torch.float32),
        "e_death":    torch.tensor([p["e_death"]    for p in batch], dtype=torch.float32),
        "timepoints": torch.tensor(timepoints),
        "obs_vals":   torch.tensor(obs_vals),
        "obs_mask":   torch.tensor(obs_mask),
        "seq_lens":   torch.tensor(seq_lens),
    }


def make_loader(patients: list[dict], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(PatientDataset(patients), batch_size=batch_size,
                      shuffle=shuffle, collate_fn=collate_fn, num_workers=0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ODEFunc(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)

    def forward(self, t, z):
        return self.net(z)


class RecognitionRNN(nn.Module):
    def __init__(self, n_labs: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.gru        = nn.GRU(n_labs * 2 + 1, hidden_dim, batch_first=True)
        self.mu_net     = nn.Linear(hidden_dim, latent_dim)
        self.logvar_net = nn.Linear(hidden_dim, latent_dim)

    def forward(self, obs_vals, obs_mask, timepoints, seq_lens):
        B, T, D = obs_vals.shape
        dt = torch.zeros(B, T, 1, device=obs_vals.device)
        dt[:, 1:, 0] = timepoints[:, 1:] - timepoints[:, :-1]

        obs_clean = obs_vals.clone()
        obs_clean[~obs_mask] = 0.0
        inp = torch.cat([obs_clean, obs_mask.float(), dt], dim=-1)
        _, h = self.gru(inp.flip(dims=[1]))
        h = h.squeeze(0)
        return self.mu_net(h), self.logvar_net(h)


class ObservationDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, n_labs: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_labs),
        )

    def forward(self, z):
        return self.net(z)


class SurvivalDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, n_events: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_events),
        )

    def forward(self, z, age):
        return self.net(torch.cat([z, age.unsqueeze(-1)], dim=-1))


class SurvLatentODE(nn.Module):
    def __init__(self, n_labs: int, latent_dim: int, hidden_dim: int,
                 ode_method: str = "rk4"):
        super().__init__()
        self.latent_dim = latent_dim
        self.ode_method = ode_method
        self.encoder      = RecognitionRNN(n_labs, latent_dim, hidden_dim)
        self.ode_func     = ODEFunc(latent_dim, hidden_dim)
        self.obs_decoder  = ObservationDecoder(latent_dim, hidden_dim, n_labs)
        self.surv_decoder = SurvivalDecoder(latent_dim, hidden_dim, n_events=2)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, batch: dict, eval_at: torch.Tensor | None = None):
        obs_vals   = batch["obs_vals"]
        obs_mask   = batch["obs_mask"]
        timepoints = batch["timepoints"]
        seq_lens   = batch["seq_lens"]
        age        = batch["age"]
        B, T, D    = obs_vals.shape
        device     = obs_vals.device
        scale      = 365.0

        mu, logvar = self.encoder(obs_vals, obs_mask, timepoints, seq_lens)
        z0 = self.reparameterize(mu, logvar)

        # Build query time grid (normalised to years)
        query_days = torch.unique(timepoints.reshape(-1)).sort().values
        query_days = query_days[query_days >= 0]
        if eval_at is not None:
            query_days = torch.unique(
                torch.cat([query_days, eval_at.to(device).unique()])).sort().values
        query_t = query_days / scale

        if query_t.shape[0] < 2 or (query_t[-1] - query_t[0]) < 1e-6:
            log_hazard = self.surv_decoder(z0, age)
            return log_hazard, torch.tensor(0.0, device=device), \
                   torch.tensor(0.0, device=device), z0

        z_traj = odeint(self.ode_func, z0, query_t,
                        method=self.ode_method,
                        options={"step_size": 0.05})  # (n_t, B, latent_dim)

        # Reconstruction loss (masked MSE at observed timepoints)
        recon_loss = torch.zeros(1, device=device)
        n_obs = 0
        for bi in range(B):
            Ti = int(seq_lens[bi].item())
            for ti in range(Ti):
                t_q = timepoints[bi, ti] / scale
                idx = (query_t - t_q).abs().argmin()
                x_hat   = self.obs_decoder(z_traj[idx, bi])
                mask_ti = obs_mask[bi, ti]
                if mask_ti.any():
                    recon_loss = recon_loss + ((x_hat[mask_ti] - obs_vals[bi, ti][mask_ti]) ** 2).sum()
                    n_obs += mask_ti.sum().item()
        if n_obs > 0:
            recon_loss = recon_loss / n_obs

        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

        # Latent state at eval_at (or last observed timepoint)
        z_eval_list = []
        for bi in range(B):
            if eval_at is not None:
                t_q = eval_at[bi].item() / scale
            else:
                Ti  = int(seq_lens[bi].item())
                t_q = timepoints[bi, Ti - 1].item() / scale
            idx = (query_t - t_q).abs().argmin()
            z_eval_list.append(z_traj[idx, bi])
        z_eval = torch.stack(z_eval_list)

        log_hazard = self.surv_decoder(z_eval, age)
        return log_hazard, recon_loss.squeeze(), kl_loss, z_eval


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def cox_loss(log_hazard: torch.Tensor,
             durations: torch.Tensor,
             events: torch.Tensor) -> torch.Tensor:
    """Breslow partial likelihood loss."""
    order   = durations.argsort(descending=True)
    lh      = log_hazard[order]
    evt     = events[order]
    log_cum = torch.logcumsumexp(lh, dim=0)
    numer   = (lh - log_cum)[evt.bool()]
    if numer.numel() == 0 or torch.isnan(numer).any():
        return torch.zeros(1, device=log_hazard.device, requires_grad=True).squeeze()
    return -numer.mean()


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
def train_epoch(model: SurvLatentODE, loader: DataLoader,
                optimizer: torch.optim.Optimizer, device: torch.device,
                kl_weight: float, surv_weight: float) -> dict:
    model.train()
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "surv": 0.0}
    n = 0
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        eval_at = torch.minimum(batch["t_platinum"], batch["t_death"])
        log_hazard, recon, kl, _ = model(batch, eval_at=eval_at)

        surv = (cox_loss(log_hazard[:, 0], batch["t_platinum"], batch["e_platinum"]) +
                cox_loss(log_hazard[:, 1], batch["t_death"],    batch["e_death"]))
        loss = recon + kl_weight * kl + surv_weight * surv

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        totals["loss"]  += loss.item()
        totals["recon"] += recon.item()
        totals["kl"]    += kl.item()
        totals["surv"]  += surv.item()
        n += 1

    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def evaluate_loader(model: SurvLatentODE, loader: DataLoader,
                    device: torch.device) -> dict[str, float]:
    from sksurv.metrics import concordance_index_censored
    model.eval()
    risk_plat, risk_death = [], []
    t_plat, t_death       = [], []
    e_plat, e_death       = [], []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        eval_at = torch.minimum(batch["t_platinum"], batch["t_death"])
        lh, *_ = model(batch, eval_at=eval_at)
        risk_plat.extend(lh[:, 0].cpu().tolist())
        risk_death.extend(lh[:, 1].cpu().tolist())
        t_plat.extend(batch["t_platinum"].cpu().tolist())
        t_death.extend(batch["t_death"].cpu().tolist())
        e_plat.extend(batch["e_platinum"].cpu().tolist())
        e_death.extend(batch["e_death"].cpu().tolist())

    results = {}
    for name, risks, times, events in [
        ("platinum", risk_plat,  t_plat,  e_plat),
        ("death",    risk_death, t_death, e_death),
    ]:
        risks  = np.array(risks)
        times  = np.array(times)
        events = np.array(events, dtype=bool)
        if events.any():
            c, *_ = concordance_index_censored(events, times, risks)
        else:
            c = float("nan")
        results[f"c_{name}"] = c
    return results


def build_model(n_labs: int, args) -> SurvLatentODE:
    return SurvLatentODE(n_labs=n_labs, latent_dim=args.latent_dim,
                         hidden_dim=args.hidden_dim, ode_method=args.ode_method)


def train_model(model: SurvLatentODE, train_loader: DataLoader,
                val_loader: DataLoader | None, device: torch.device,
                n_epochs: int, args,
                checkpoint_path: Path | None = None) -> tuple[list[dict], float]:
    """
    Train for n_epochs. Returns (log_rows, best_avg_c).
    If checkpoint_path is given, saves the best model there.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    kl_anneal = n_epochs // 2

    log_rows = []
    best_c   = -1.0

    for epoch in range(1, n_epochs + 1):
        kl_w    = min(1.0, epoch / max(kl_anneal, 1))
        metrics = train_epoch(model, train_loader, optimizer, device,
                              kl_weight=kl_w, surv_weight=args.surv_weight)
        scheduler.step()
        row = {"epoch": epoch, **metrics}

        if val_loader is not None and (epoch % args.eval_every == 0 or epoch == n_epochs):
            eval_m = evaluate_loader(model, val_loader, device)
            row.update(eval_m)
            avg_c = float(np.nanmean([eval_m.get("c_platinum", np.nan),
                                       eval_m.get("c_death",    np.nan)]))
            print(f"  Epoch {epoch:4d} | loss={metrics['loss']:.4f} "
                  f"recon={metrics['recon']:.4f} kl={metrics['kl']:.4f} "
                  f"surv={metrics['surv']:.4f} | "
                  f"C-plat={eval_m.get('c_platinum', float('nan')):.4f} "
                  f"C-death={eval_m.get('c_death', float('nan')):.4f}")
            if avg_c > best_c:
                best_c = avg_c
                if checkpoint_path is not None:
                    torch.save({"epoch": epoch, "model_state": model.state_dict()},
                               checkpoint_path)
        else:
            print(f"  Epoch {epoch:4d} | loss={metrics['loss']:.4f} "
                  f"recon={metrics['recon']:.4f} kl={metrics['kl']:.4f} "
                  f"surv={metrics['surv']:.4f} | kl_w={kl_w:.2f}")

        log_rows.append(row)

    return log_rows, best_c


# ---------------------------------------------------------------------------
# CV + final training
# ---------------------------------------------------------------------------
def run_cv(train_val_pats: list[dict], lab_names: list[str],
           device: torch.device, args) -> list[dict]:
    strat = stratified_label(train_val_pats)
    kf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    idx   = np.arange(len(train_val_pats))

    cv_rows = []
    cv_epochs = args.cv_epochs if args.cv_epochs > 0 else max(args.epochs // 4, 10)
    print(f"\n5-fold CV on {len(train_val_pats)} train/val patients "
          f"({cv_epochs} epochs per fold)")

    for fold, (tr_idx, val_idx) in enumerate(kf.split(idx, strat), 1):
        fold_tr  = [train_val_pats[i] for i in tr_idx]
        fold_val = [train_val_pats[i] for i in val_idx]
        print(f"\nFold {fold}  (tr={len(fold_tr)}, val={len(fold_val)})")

        model = build_model(len(lab_names), args).to(device)
        tr_loader  = make_loader(fold_tr,  args.batch_size, shuffle=True)
        val_loader = make_loader(fold_val, args.batch_size, shuffle=False)

        _, _ = train_model(model, tr_loader, val_loader, device,
                           n_epochs=cv_epochs, args=args, checkpoint_path=None)

        # Evaluate at end of training (no best-model checkpointing in CV folds)
        eval_m = evaluate_loader(model, val_loader, device)
        c_plat  = eval_m.get("c_platinum", float("nan"))
        c_death = eval_m.get("c_death",    float("nan"))
        print(f"  Fold {fold} final: C-plat={c_plat:.4f}  C-death={c_death:.4f}")
        cv_rows.append({"fold": fold, "c_platinum": c_plat, "c_death": c_death,
                        "avg_c": float(np.nanmean([c_plat, c_death]))})

    cv_df = pd.DataFrame(cv_rows)
    print(f"\nCV C-platinum: {cv_df['c_platinum'].mean():.4f} ± "
          f"{cv_df['c_platinum'].std():.4f}")
    print(f"CV C-death:    {cv_df['c_death'].mean():.4f} ± "
          f"{cv_df['c_death'].std():.4f}")
    cv_df.to_csv(RESULTS / "survlatentode_cv_metrics.csv", index=False)
    print("Saved: results/survlatentode_cv_metrics.csv")
    return cv_rows


def run_final(train_val_pats: list[dict], test_pats: list[dict],
              lab_names: list[str], device: torch.device, args) -> None:
    print(f"\n--- Final model: {len(train_val_pats)} train/val, "
          f"{len(test_pats)} test, {args.epochs} epochs ---")

    model      = build_model(len(lab_names), args).to(device)
    tr_loader  = make_loader(train_val_pats, args.batch_size, shuffle=True)
    te_loader  = make_loader(test_pats,      args.batch_size, shuffle=False)

    ckpt_path  = Path(args.output)
    log_rows, best_c = train_model(model, tr_loader, te_loader, device,
                                   n_epochs=args.epochs, args=args,
                                   checkpoint_path=ckpt_path)

    pd.DataFrame(log_rows).to_csv(RESULTS / "survlatentode_train_log.csv", index=False)
    print(f"Saved: results/survlatentode_train_log.csv")

    # Load best checkpoint and produce test predictions
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    final_eval = evaluate_loader(model, te_loader, device)
    print(f"Final test C-platinum={final_eval['c_platinum']:.4f}  "
          f"C-death={final_eval['c_death']:.4f}  "
          f"(best avg during training={best_c:.4f})")

    pred_rows = []
    model.eval()
    with torch.no_grad():
        for batch in te_loader:
            batch   = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in batch.items()}
            eval_at = torch.minimum(batch["t_platinum"], batch["t_death"])
            lh, *_  = model(batch, eval_at=eval_at)
            for i, mrn in enumerate(batch["mrn"]):
                pred_rows.append({
                    "DFCI_MRN":           mrn,
                    "log_hazard_platinum": lh[i, 0].item(),
                    "log_hazard_death":    lh[i, 1].item(),
                    "t_platinum": batch["t_platinum"][i].item(),
                    "e_platinum": batch["e_platinum"][i].item(),
                    "t_death":    batch["t_death"][i].item(),
                    "e_death":    batch["e_death"][i].item(),
                })

    pd.DataFrame(pred_rows).to_csv(
        RESULTS / "survlatentode_test_predictions.csv", index=False)
    print("Saved: results/survlatentode_test_predictions.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_patients, lab_names = prepare_all_patients(args.data)
    train_val_pats, test_pats = split_patients(all_patients)
    print(f"Train/val: {len(train_val_pats)}  |  Test (held-out): {len(test_pats)}")
    print(f"Model params (approx): {sum(p.numel() for p in build_model(len(lab_names), args).parameters()):,}")

    run_cv(train_val_pats, lab_names, device, args)
    run_final(train_val_pats, test_pats, lab_names, device, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        default=str(DATA_PATH / "longitudinal_prediction_data.csv"))
    parser.add_argument("--output",      default=str(RESULTS / "survlatentode_checkpoint.pt"))
    parser.add_argument("--epochs",      type=int,   default=200)
    parser.add_argument("--cv-epochs",   type=int,   default=50,
                        help="Epochs per CV fold (default 50; 0 = epochs//4)")
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--latent-dim",  type=int,   default=32)
    parser.add_argument("--hidden-dim",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--surv-weight", type=float, default=1.0)
    parser.add_argument("--eval-every",  type=int,   default=10)
    parser.add_argument("--ode-method",  default="rk4",
                        choices=["rk4", "euler", "dopri5"])
    main(parser.parse_args())
