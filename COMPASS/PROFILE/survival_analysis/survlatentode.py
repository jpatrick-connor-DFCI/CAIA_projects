"""
Step 4: SurvLatentODE — Neural ODE survival model for irregularly-sampled labs.

Architecture (following Sutton et al. 2021 / Schirmer et al. 2022 style):
  1. ODE-RNN encoder: reads the patient's lab time series backwards in time,
     producing an initial latent state z0 ~ q(z0 | x).
  2. Latent ODE: a neural ODE integrates z0 forward in continuous time.
  3. Survival decoder: reads z(t) and outputs the log-hazard for each event.
     For competing risks we use two output heads (platinum, death).

Observation model:
  - Each lab value is modelled as Gaussian with learned mean/variance from z(t).
  - Missing labs at a time step use a mask (no gradient from missing values).

Loss:
  - ELBO = reconstruction log-likelihood + KL(q(z0) || p(z0))
  - Survival log-likelihood (cause-specific, censored) per event head
  - Total = ELBO + λ * survival_loss

Requirements:
  torch, torchdiffeq, scikit-survival (for c-index), numpy, pandas, scipy

Usage:
  python survlatentode.py --data /path/to/longitudinal_prediction_data.csv \
      --epochs 200 --latent-dim 32 --hidden-dim 64 --lr 1e-3 \
      --output results/survlatentode_checkpoint.pt

Outputs:
  results/survlatentode_checkpoint.pt       (model weights)
  results/survlatentode_train_log.csv       (per-epoch losses)
  results/survlatentode_test_predictions.csv
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchdiffeq import odeint

BASE = Path(__file__).resolve().parent
DATA_PATH = Path("/data/gusev/USERS/jpconnor/data/CAIA/COMPASS/")
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
MIN_PATIENT_COVERAGE = 0.20
TOP_N_LABS = 20


def prepare_dataset(csv_path: str, test_frac: float = 0.2):
    """
    Returns:
      train_patients, test_patients : list of patient dicts
      lab_names : list[str]  (ordered feature dimensions)
      n_labs : int
    """
    print("Loading data...")
    df = pd.read_csv(csv_path, low_memory=False)
    for col in ["LAB_DATE", "FIRST_TREATMENT_DATE", "LAST_CONTACT_DATE", "PLATINUM_DATE"]:
        df[col] = pd.to_datetime(df[col])

    # Select labs by coverage
    coverage = df.groupby("LAB_NAME")["DFCI_MRN"].nunique() / df["DFCI_MRN"].nunique()
    eligible = coverage[coverage >= MIN_PATIENT_COVERAGE].sort_values(ascending=False)
    lab_names = eligible.index.tolist()[:TOP_N_LABS]
    print(f"Using {len(lab_names)} labs: {lab_names}")

    # Standardize each lab across the full dataset
    lab_stats: dict[str, tuple[float, float]] = {}
    for lab in lab_names:
        vals = df.loc[df["LAB_NAME"] == lab, "LAB_VALUE"].dropna()
        mu, sigma = float(vals.mean()), float(vals.std())
        lab_stats[lab] = (mu, max(sigma, 1e-6))

    # Build per-patient records
    pat_meta = (df[["DFCI_MRN", "FIRST_TREATMENT_DATE", "LAST_CONTACT_DATE",
                    "DEATH", "PLATINUM_DATE", "PLATINUM", "AGE_AT_TREATMENTSTART"]]
                .drop_duplicates("DFCI_MRN").set_index("DFCI_MRN"))
    pat_meta["t_platinum"] = (pat_meta["PLATINUM_DATE"] - pat_meta["FIRST_TREATMENT_DATE"]).dt.days
    pat_meta["t_death"]    = (pat_meta["LAST_CONTACT_DATE"] - pat_meta["FIRST_TREATMENT_DATE"]).dt.days
    pat_meta = pat_meta[(pat_meta["t_platinum"] > 0) & (pat_meta["t_death"] > 0)]

    lab_df = df[df["LAB_NAME"].isin(lab_names)].copy()
    # Standardize
    lab_df["LAB_VALUE_STD"] = lab_df.apply(
        lambda r: (r["LAB_VALUE"] - lab_stats[r["LAB_NAME"]][0]) / lab_stats[r["LAB_NAME"]][1],
        axis=1
    )

    n_labs = len(lab_names)
    lab_idx = {lab: i for i, lab in enumerate(lab_names)}

    patients = []
    for mrn, row in pat_meta.iterrows():
        t0 = row["FIRST_TREATMENT_DATE"]
        pat_obs = lab_df[lab_df["DFCI_MRN"] == mrn].copy()
        pat_obs["t_days"] = (pat_obs["LAB_DATE"] - t0).dt.days.astype(float)

        # Keep all observations (pre and post T0)
        # Clip to reasonable window: drop obs > max follow-up
        max_t = max(float(row["t_death"]), float(row["t_platinum"]))
        pat_obs = pat_obs[(pat_obs["t_days"] > -365 * 5) & (pat_obs["t_days"] <= max_t)]
        if pat_obs.empty:
            continue

        # Group observations by date → dict of time_point -> {lab: value}
        timepoints = sorted(pat_obs["t_days"].unique())
        obs_vals  = np.full((len(timepoints), n_labs), np.nan)
        obs_mask  = np.zeros((len(timepoints), n_labs), dtype=bool)

        for ti, t in enumerate(timepoints):
            obs_at_t = pat_obs[pat_obs["t_days"] == t]
            for _, lab_row in obs_at_t.iterrows():
                li = lab_idx.get(lab_row["LAB_NAME"])
                if li is not None and np.isfinite(lab_row["LAB_VALUE_STD"]):
                    obs_vals[ti, li] = lab_row["LAB_VALUE_STD"]
                    obs_mask[ti, li] = True

        patients.append({
            "mrn":         mrn,
            "age":         float(row["AGE_AT_TREATMENTSTART"]) / 100.0,  # normalize
            "t_platinum":  float(row["t_platinum"]),
            "e_platinum":  float(row["PLATINUM"]),
            "t_death":     float(row["t_death"]),
            "e_death":     float(row["DEATH"]),
            "timepoints":  np.array(timepoints, dtype=np.float32),  # days from T0
            "obs_vals":    obs_vals.astype(np.float32),             # (T, D)
            "obs_mask":    obs_mask,                                 # (T, D) bool
        })

    print(f"Total patients: {len(patients)}")
    random.shuffle(patients)
    n_test = int(len(patients) * test_frac)
    return patients[n_test:], patients[:n_test], lab_names


# ---------------------------------------------------------------------------
# Dataset + collation
# ---------------------------------------------------------------------------
class PatientDataset(Dataset):
    def __init__(self, patients: list[dict]):
        self.patients = patients

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        return self.patients[idx]


def collate_fn(batch: list[dict]) -> dict:
    """Pad time series to same length within batch."""
    max_T = max(p["timepoints"].shape[0] for p in batch)
    n_labs = batch[0]["obs_vals"].shape[1]
    B = len(batch)

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
        "age":        torch.tensor([p["age"] for p in batch], dtype=torch.float32),
        "t_platinum": torch.tensor([p["t_platinum"] for p in batch], dtype=torch.float32),
        "e_platinum": torch.tensor([p["e_platinum"] for p in batch], dtype=torch.float32),
        "t_death":    torch.tensor([p["t_death"] for p in batch], dtype=torch.float32),
        "e_death":    torch.tensor([p["e_death"] for p in batch], dtype=torch.float32),
        "timepoints": torch.tensor(timepoints),
        "obs_vals":   torch.tensor(obs_vals),
        "obs_mask":   torch.tensor(obs_mask),
        "seq_lens":   torch.tensor(seq_lens),
    }


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------
class ODEFunc(nn.Module):
    """Neural ODE dynamics function: dz/dt = f(z)."""
    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Initialize small to encourage slow dynamics initially
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)

    def forward(self, t, z):
        return self.net(z)


class RecognitionRNN(nn.Module):
    """
    GRU-based encoder that reads observations backwards in time.
    Produces mean and log-variance of q(z0).
    """
    def __init__(self, n_labs: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        # Input: observed values + mask + time delta = n_labs*2 + 1
        self.gru = nn.GRU(input_size=n_labs * 2 + 1,
                          hidden_size=hidden_dim, batch_first=True)
        self.mu_net  = nn.Linear(hidden_dim, latent_dim)
        self.logvar_net = nn.Linear(hidden_dim, latent_dim)

    def forward(self, obs_vals: torch.Tensor, obs_mask: torch.Tensor,
                timepoints: torch.Tensor, seq_lens: torch.Tensor):
        """
        obs_vals:   (B, T, D)
        obs_mask:   (B, T, D)
        timepoints: (B, T)   — days from T0
        seq_lens:   (B,)
        """
        B, T, D = obs_vals.shape
        # Time deltas (between consecutive timepoints, reversed)
        dt = torch.zeros(B, T, 1, device=obs_vals.device)
        dt[:, 1:, 0] = timepoints[:, 1:] - timepoints[:, :-1]

        # Replace NaN in obs_vals with 0 (mask handles the contribution)
        obs_vals_clean = obs_vals.clone()
        obs_vals_clean[~obs_mask] = 0.0
        mask_float = obs_mask.float()

        inp = torch.cat([obs_vals_clean, mask_float, dt], dim=-1)  # (B, T, 2D+1)
        # Reverse time order for encoding (read past → present)
        inp_rev = inp.flip(dims=[1])

        out, h = self.gru(inp_rev)   # h: (1, B, hidden_dim)
        h = h.squeeze(0)             # (B, hidden_dim)

        mu     = self.mu_net(h)      # (B, latent_dim)
        logvar = self.logvar_net(h)  # (B, latent_dim)
        return mu, logvar


class ObservationDecoder(nn.Module):
    """Decodes latent state z(t) → predicted lab values."""
    def __init__(self, latent_dim: int, hidden_dim: int, n_labs: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_labs),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SurvivalDecoder(nn.Module):
    """
    Maps latent state z → log-hazard for each event.
    Two heads: platinum, death.
    Also accepts static covariates (age).
    """
    def __init__(self, latent_dim: int, hidden_dim: int, n_events: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),   # +1 for age
            nn.ReLU(),
            nn.Linear(hidden_dim, n_events),
        )

    def forward(self, z: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim), age: (B,) → log_hazard: (B, 2)"""
        inp = torch.cat([z, age.unsqueeze(-1)], dim=-1)
        return self.net(inp)


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

        # Prior: z0 ~ N(0, I)
        self.register_buffer("prior_mu",     torch.zeros(latent_dim))
        self.register_buffer("prior_logvar", torch.zeros(latent_dim))

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, batch: dict, eval_at: torch.Tensor | None = None):
        """
        eval_at: (B,) — time at which to evaluate survival head (e.g., event time).
                 If None, uses the last observed timepoint per patient.
        Returns:
          log_hazard: (B, 2)   — [platinum_log_hazard, death_log_hazard]
          recon_loss: scalar   — reconstruction loss
          kl_loss:    scalar   — KL divergence
          z_eval:     (B, latent_dim)  — latent state at eval_at
        """
        obs_vals   = batch["obs_vals"]    # (B, T, D)
        obs_mask   = batch["obs_mask"]    # (B, T, D)
        timepoints = batch["timepoints"]  # (B, T)  days
        seq_lens   = batch["seq_lens"]    # (B,)
        age        = batch["age"]         # (B,)

        B, T, D = obs_vals.shape
        device = obs_vals.device

        # Encode
        mu, logvar = self.encoder(obs_vals, obs_mask, timepoints, seq_lens)
        z0 = self.reparameterize(mu, logvar)   # (B, latent_dim)

        # ODE integration: need a common time grid per batch
        # We integrate from min to max timepoint seen, at all unique timepoints
        # For efficiency, use a shared fine grid clipped to [min_t, max_t]
        all_times = timepoints.reshape(-1)
        t_min = all_times[all_times != 0].min().item() if (all_times != 0).any() else 0.0
        t_max = timepoints.max().item()

        # Build sorted unique query times (in days, normalized to years for ODE stability)
        scale = 365.0
        query_times_days = torch.unique(timepoints.reshape(-1)).sort().values
        query_times_days = query_times_days[query_times_days >= 0]
        if eval_at is not None:
            extra = eval_at.unique().sort().values
            query_times_days = torch.unique(torch.cat([query_times_days, extra])).sort().values

        query_times = query_times_days / scale  # normalise to years

        if query_times.shape[0] == 0 or (query_times.max() - query_times.min()) < 1e-6:
            # Degenerate batch: no valid time range
            log_hazard = self.surv_decoder(z0, age)
            return log_hazard, torch.tensor(0.0), torch.tensor(0.0), z0

        # Integrate ODE: shape (n_times, B, latent_dim)
        # torchdiffeq expects initial value shape (B, latent_dim)
        z_traj = odeint(self.ode_func, z0, query_times.to(device),
                        method=self.ode_method,
                        options={"step_size": 0.05})  # ~18d steps

        # Reconstruction loss: for each patient, at each observed timepoint,
        # decode z and compute MSE against observed (masked) values
        recon_loss = torch.tensor(0.0, device=device)
        n_obs = 0

        for bi in range(B):
            Ti = int(seq_lens[bi].item())
            for ti in range(Ti):
                t_query = timepoints[bi, ti] / scale
                # Find closest index in query_times
                idx = (query_times - t_query).abs().argmin()
                z_ti = z_traj[idx, bi, :]          # (latent_dim,)
                x_hat = self.obs_decoder(z_ti)     # (D,)
                mask_ti = obs_mask[bi, ti]          # (D,) bool
                if mask_ti.any():
                    diff = (x_hat[mask_ti] - obs_vals[bi, ti][mask_ti]) ** 2
                    recon_loss = recon_loss + diff.sum()
                    n_obs += mask_ti.sum().item()

        if n_obs > 0:
            recon_loss = recon_loss / n_obs

        # KL divergence: KL( N(mu, sigma^2) || N(0, I) )
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

        # Survival head: evaluate at eval_at (e.g., event time)
        if eval_at is not None:
            z_eval_list = []
            for bi in range(B):
                t_q = eval_at[bi].item() / scale
                idx = (query_times - t_q).abs().argmin()
                z_eval_list.append(z_traj[idx, bi, :])
            z_eval = torch.stack(z_eval_list, dim=0)
        else:
            # Use latent state at last observed timepoint per patient
            z_eval_list = []
            for bi in range(B):
                Ti = int(seq_lens[bi].item())
                t_q = timepoints[bi, Ti - 1].item() / scale
                idx = (query_times - t_q).abs().argmin()
                z_eval_list.append(z_traj[idx, bi, :])
            z_eval = torch.stack(z_eval_list, dim=0)

        log_hazard = self.surv_decoder(z_eval, age)  # (B, 2)
        return log_hazard, recon_loss, kl_loss, z_eval


# ---------------------------------------------------------------------------
# Survival loss (cause-specific, censored negative log-likelihood)
# ---------------------------------------------------------------------------
def cox_survival_loss(log_hazard: torch.Tensor,
                      durations: torch.Tensor,
                      events: torch.Tensor) -> torch.Tensor:
    """
    Breslow approximation of the partial likelihood loss.
    log_hazard: (B,)  — log partial hazard scores
    durations:  (B,)  — observed times
    events:     (B,)  — 1 if event occurred, 0 if censored
    """
    # Sort by duration descending
    order   = durations.argsort(descending=True)
    lh      = log_hazard[order]
    evt     = events[order]
    log_cum = torch.logcumsumexp(lh, dim=0)
    loss    = -(lh - log_cum)[evt.bool()].mean()
    if torch.isnan(loss):
        return torch.tensor(0.0, requires_grad=True)
    return loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(model: SurvLatentODE, train_loader: DataLoader,
          optimizer: torch.optim.Optimizer, device: torch.device,
          kl_weight: float = 1.0, surv_weight: float = 1.0) -> dict:
    model.train()
    total_loss = total_recon = total_kl = total_surv = 0.0
    n_batches = 0

    for batch in train_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Evaluate survival head at event/censoring time (separately per event)
        eval_platinum = batch["t_platinum"]
        eval_death    = batch["t_death"]
        # Use the minimum of the two to avoid ODE integration past observed time
        eval_at = torch.minimum(eval_platinum, eval_death)

        log_hazard, recon_loss, kl_loss, _ = model(batch, eval_at=eval_at)

        # Survival losses (one per event head)
        surv_loss_plat = cox_survival_loss(
            log_hazard[:, 0], batch["t_platinum"], batch["e_platinum"])
        surv_loss_death = cox_survival_loss(
            log_hazard[:, 1], batch["t_death"], batch["e_death"])
        surv_loss = surv_loss_plat + surv_loss_death

        loss = recon_loss + kl_weight * kl_loss + surv_weight * surv_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss  += loss.item()
        total_recon += recon_loss.item()
        total_kl    += kl_loss.item()
        total_surv  += surv_loss.item()
        n_batches   += 1

    return {
        "loss":  total_loss / n_batches,
        "recon": total_recon / n_batches,
        "kl":    total_kl / n_batches,
        "surv":  total_surv / n_batches,
    }


@torch.no_grad()
def evaluate(model: SurvLatentODE, loader: DataLoader,
             device: torch.device) -> dict:
    model.eval()
    from sksurv.metrics import concordance_index_censored

    all_risk_plat  = []
    all_risk_death = []
    all_t_plat     = []
    all_e_plat     = []
    all_t_death    = []
    all_e_death    = []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        eval_at = torch.minimum(batch["t_platinum"], batch["t_death"])
        log_hazard, *_ = model(batch, eval_at=eval_at)

        all_risk_plat.extend(log_hazard[:, 0].cpu().numpy().tolist())
        all_risk_death.extend(log_hazard[:, 1].cpu().numpy().tolist())
        all_t_plat.extend(batch["t_platinum"].cpu().numpy().tolist())
        all_e_plat.extend(batch["e_platinum"].cpu().numpy().tolist())
        all_t_death.extend(batch["t_death"].cpu().numpy().tolist())
        all_e_death.extend(batch["e_death"].cpu().numpy().tolist())

    results = {}
    for name, risks, times, events in [
        ("platinum", all_risk_plat, all_t_plat, all_e_plat),
        ("death",    all_risk_death, all_t_death, all_e_death),
    ]:
        risks  = np.array(risks)
        times  = np.array(times)
        events = np.array(events, dtype=bool)
        if events.any():
            c, *_ = concordance_index_censored(events, times, risks)
            results[f"c_{name}"] = c
        else:
            results[f"c_{name}"] = float("nan")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_pats, test_pats, lab_names = prepare_dataset(
        args.data, test_frac=args.test_frac)

    train_ds = PatientDataset(train_pats)
    test_ds  = PatientDataset(test_pats)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=0)

    model = SurvLatentODE(
        n_labs=len(lab_names),
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        ode_method=args.ode_method,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    log_rows = []
    best_c = -1.0
    # Anneal KL weight from 0 → 1 over first half of training
    kl_anneal_epochs = args.epochs // 2

    for epoch in range(1, args.epochs + 1):
        kl_weight = min(1.0, epoch / kl_anneal_epochs)
        metrics = train(model, train_loader, optimizer, device,
                        kl_weight=kl_weight, surv_weight=args.surv_weight)
        scheduler.step()

        row = {"epoch": epoch, **metrics}

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            eval_m = evaluate(model, test_loader, device)
            row.update(eval_m)
            avg_c = np.nanmean([eval_m.get("c_platinum", np.nan),
                                 eval_m.get("c_death", np.nan)])
            print(f"Epoch {epoch:4d} | loss={metrics['loss']:.4f} "
                  f"recon={metrics['recon']:.4f} kl={metrics['kl']:.4f} "
                  f"surv={metrics['surv']:.4f} | "
                  f"C-plat={eval_m.get('c_platinum', float('nan')):.4f} "
                  f"C-death={eval_m.get('c_death', float('nan')):.4f}")
            if avg_c > best_c:
                best_c = avg_c
                torch.save({
                    "epoch":      epoch,
                    "model_state": model.state_dict(),
                    "lab_names":  lab_names,
                    "args":       vars(args),
                }, args.output)
                print(f"  → Saved best model (avg C={best_c:.4f})")
        else:
            print(f"Epoch {epoch:4d} | loss={metrics['loss']:.4f} "
                  f"recon={metrics['recon']:.4f} kl={metrics['kl']:.4f} "
                  f"surv={metrics['surv']:.4f} | kl_w={kl_weight:.2f}")

        log_rows.append(row)

    pd.DataFrame(log_rows).to_csv(RESULTS / "survlatentode_train_log.csv", index=False)
    print("Training complete.")
    print(f"Best avg C-index: {best_c:.4f}")

    # Save test predictions from best checkpoint
    ckpt = torch.load(args.output, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    pred_rows = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            eval_at = torch.minimum(batch["t_platinum"], batch["t_death"])
            log_hazard, *_ = model(batch, eval_at=eval_at)
            for i, mrn in enumerate(batch["mrn"]):
                pred_rows.append({
                    "DFCI_MRN":        mrn,
                    "log_hazard_plat": log_hazard[i, 0].item(),
                    "log_hazard_death": log_hazard[i, 1].item(),
                    "t_platinum":      batch["t_platinum"][i].item(),
                    "e_platinum":      batch["e_platinum"][i].item(),
                    "t_death":         batch["t_death"][i].item(),
                    "e_death":         batch["e_death"][i].item(),
                })

    pd.DataFrame(pred_rows).to_csv(
        RESULTS / "survlatentode_test_predictions.csv", index=False)
    print("Saved: results/survlatentode_test_predictions.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        default=str(DATA_PATH / "longitudinal_prediction_data.csv"))
    parser.add_argument("--output",      default=str(RESULTS / "survlatentode_checkpoint.pt"))
    parser.add_argument("--epochs",      type=int,   default=200)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--latent-dim",  type=int,   default=32)
    parser.add_argument("--hidden-dim",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--surv-weight", type=float, default=1.0,
                        help="Weight on survival loss relative to ELBO")
    parser.add_argument("--test-frac",   type=float, default=0.2)
    parser.add_argument("--eval-every",  type=int,   default=10)
    parser.add_argument("--ode-method",  default="rk4",
                        choices=["rk4", "euler", "dopri5"])
    main(parser.parse_args())
