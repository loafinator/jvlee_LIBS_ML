from __future__ import annotations

"""
jvlee_LIBS_ML > LIBS > p3VAE > p3vae_003.py
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, get_worker_info

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from setup import add_project_root_to_path
add_project_root_to_path(parent_generation=1)

# torch.autograd.set_detect_anomaly(True)

# region Global Variables
max_epochs = 5
batch_size = 256
learning_rate = 1e-05
weight_decay = 1e-04

ALL_COLUMNS = [
    'conc_Ba_wt%', 'conc_Ca_wt%', 'conc_Ce_wt%', 'conc_Cr_wt%', 
    'conc_Cs_wt%', 'conc_Fe_wt%', 'conc_Gd_wt%', 'conc_K_wt%', 
    'conc_La_wt%', 'conc_Li_wt%', 'conc_Mg_wt%', 'conc_Mn_wt%', 
    'conc_Nd_wt%', 'conc_Ni_wt%', 'conc_Sm_wt%', 'conc_Sr_wt%', 
    'conc_U_wt%',  'conc_Y_wt%',  'conc_Cl_wt%',
]

master_table_path = Path('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/master_line_table.csv').resolve()
# endregion


class LIBSSpectraDataset(Dataset):
    def __init__(self, h5_path: str | Path, split: str = 'train'):
        self.h5_path = str(h5_path)
        self.split = split
        self.hf = None  # Lazy file handle per worker process

        with h5py.File(self.h5_path, 'r') as hf:
            if split not in hf:
                raise KeyError(
                    f"Split '{split}' not found in {self.h5_path}. Available: {list(hf.keys())}"
                )
            self.length = hf[split]['spectra'].shape[0]     # type: ignore
            self.n_features = hf[split]['spectra'].shape[1]     # type: ignore
            self.elem_names = hf[split]['metadata']['elem_names'][:]     # type: ignore

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.hf is None:
            self.hf = h5py.File(self.h5_path, 'r')

        group = self.hf[self.split]
        spectrum = group['spectra'][idx]     # type: ignore
        targets = group['metadata']['elem_comp_wt%'][idx]     # type: ignore
        is_synth = group['metadata']['is_synthetic'][idx]     # type: ignore

        return {
            'spectrum': torch.from_numpy(spectrum).float(),
            'targets': torch.from_numpy(targets).float(),
            'is_synthetic': torch.tensor(is_synth, dtype=torch.long),
        }

    def __del__(self):
        if self.hf is not None:
            self.hf.close()

class LIBSSpectraDatasetInMemory(Dataset):
    def __init__(self, h5_path: str | Path, split: str = 'train'):
        h5_path = str(h5_path)
        print(f"Loading '{split}' dataset into memory...")
        
        with h5py.File(h5_path, 'r') as hf:
            # Read entire arrays directly into RAM (NumPy) -> convert to torch Tensors
            self.spectra = torch.from_numpy(hf[split]['spectra'][:]).float()        # type: ignore
            self.targets = torch.from_numpy(hf[split]['metadata']['elem_comp_wt%'][:]).float()        # type: ignore
            self.is_synth = torch.from_numpy(hf[split]['metadata']['is_synthetic'][:]).long()        # type: ignore

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            'spectrum': self.spectra[idx],
            'targets': self.targets[idx],
            'is_synthetic': self.is_synth[idx],
        }

class PhysicsSpectralDecoder(nn.Module):
    def __init__(self, wl_grid, master_table_path):
        super().__init__()

        master_df = pd.read_csv(master_table_path)

        nist_wls = torch.tensor(master_df['wavelength_nm'].values, dtype=torch.float32)
        nist_intens = torch.tensor(master_df['base_intensity'].values, dtype=torch.float32)
        elem_indices = torch.tensor(master_df['elem_idx'].values, dtype=torch.long)

        # 1. Grid shape: (1, 1, N_channels) -> broadcastable across batch and lines
        self.register_buffer('grid', wl_grid.view(1, 1, -1))
        
        # 2. Line centers shape: (1, N_lines, 1) -> broadcastable across batch and channels
        self.register_buffer('line_centers', nist_wls.view(1, -1, 1))
        
        # 3. Line base amps shape: (1, N_lines)
        max_amp = nist_intens.max() if nist_intens.max() > 0 else 1.0
        self.register_buffer('line_base_amps', (nist_intens / max_amp).unsqueeze(0))
        
        # 4. Element indices mapping elements to line table rows
        self.register_buffer('elem_indices', elem_indices)

        self.num_lines = len(master_df)

    def forward(self, concentrations, broadening, chunk_size=128):
        concentrations = concentrations.float()
        broadening = broadening.float()

        line_concs = concentrations[:, self.elem_indices]
        amps_all = (line_concs * self.line_base_amps).unsqueeze(2) # (B, N_lines, 1)
        gammas = torch.clamp(broadening.unsqueeze(2), min=0.01, max=5.0)

        grid_f32 = self.grid.float()            # (1, 1, 10000)
        centers_f32 = self.line_centers.float()  # (1, N_lines, 1)

        total_lines = centers_f32.size(1)
        recon_spectrum = 0.0

        # Chunk lines to fit comfortably within GPU cache
        for i in range(0, total_lines, chunk_size):
            amps_chunk = amps_all[:, i:i+chunk_size, :]
            centers_chunk = centers_f32[:, i:i+chunk_size, :]

            denom = ((grid_f32 - centers_chunk) ** 2) + (gammas ** 2) + 1e-6
            profiles_chunk = amps_chunk * (gammas / torch.pi) / denom
            
            recon_spectrum = recon_spectrum + torch.sum(profiles_chunk, dim=1)

        return recon_spectrum

class PhysicsInformedVAE(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        n_elements: int, 
        wl_grid: torch.Tensor, 
        master_table_path: str | Path,
        n_broadening: int = 1,  # 1 for global scalar gamma, or n_lines for line-specific
    ):
        super().__init__()
        self.n_elements = n_elements
        self.n_broadening = n_broadening

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Output dimensions: N_elements (concentrations) + N_broadening (gamma)
        phys_dim = self.n_elements + self.n_broadening
        self.fc_mean_phys = nn.Linear(128, phys_dim)
        self.fc_logvar_phys = nn.Linear(128, phys_dim)

        self.fc_mean_matrix = nn.Linear(128, 8)
        self.fc_logvar_matrix = nn.Linear(128, 8)

        self.physics_decoder = PhysicsSpectralDecoder(
            wl_grid=wl_grid,
            master_table_path=master_table_path
        )

        self.matrix_decoder = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

        nn.init.constant_(self.fc_logvar_phys.weight, 0.0)
        nn.init.constant_(self.fc_logvar_phys.bias, 0.0)
        nn.init.constant_(self.fc_logvar_matrix.weight, 0.0)
        nn.init.constant_(self.fc_logvar_matrix.bias, 0.0)

    def forward(self, x):
        h = self.encoder(x)

        mu_p = torch.clamp(self.fc_mean_phys(h), min=-10.0, max=10.0)
        logvar_p = torch.clamp(self.fc_logvar_phys(h), min=-6.0, max=6.0)
        z_phys = self.reparameterize(mu_p, logvar_p)

        # Limit concentrations and broadening to reasonable physical ranges
        concentrations = torch.clamp(nn.functional.softplus(z_phys[:, :self.n_elements]), min=1e-5, max=100.0)
        broadening = torch.clamp(nn.functional.softplus(z_phys[:, self.n_elements:]), min=1e-3, max=10.0)

        mu_m = self.fc_mean_matrix(h)
        logvar_m = torch.clamp(self.fc_logvar_matrix(h), min=-8.0, max=8.0)
        z_matrix = self.reparameterize(mu_m, logvar_m)

        ideal_physics_spectrum = self.physics_decoder(concentrations, broadening)
        matrix_residual = self.matrix_decoder(z_matrix)

        reconstructed_spectrum = ideal_physics_spectrum + matrix_residual

        return reconstructed_spectrum, mu_p, logvar_p, mu_m, logvar_m

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class PhysicsVAELoss(nn.Module):
    def __init__(self, alpha_recon: float = 1.0, beta_sup: float = 10.0, gamma_kl: float = 1e-4):
        super().__init__()
        self.alpha_recon = alpha_recon
        self.beta_sup = beta_sup
        self.gamma_kl = gamma_kl
        self.mse = nn.MSELoss()

    def forward(
        self,
        x_recon: torch.Tensor,
        x_target: torch.Tensor,
        pred_conc: torch.Tensor,
        true_conc: torch.Tensor,
        mu_p: torch.Tensor,
        logvar_p: torch.Tensor,
        mu_m: torch.Tensor,
        logvar_m: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        
        recon_loss = nn.functional.mse_loss(x_recon, x_target, reduction='mean')
        sup_loss = self.mse(pred_conc, true_conc)

        # Numerically stable KL calculation averaged across the batch and features
        logvar_p_clamped = torch.clamp(logvar_p, min=-6.0, max=6.0)
        kl_phys = -0.5 * torch.mean(1 + logvar_p_clamped - mu_p.pow(2) - torch.exp(logvar_p_clamped))
        kl_matrix = -0.5 * torch.mean(1 + logvar_m - mu_m.pow(2) - torch.exp(logvar_m))
        kl_loss = kl_phys + kl_matrix

        total_loss = (self.alpha_recon * recon_loss) + (self.beta_sup * sup_loss) + (self.gamma_kl * kl_loss)

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "sup_loss": sup_loss,
            "kl_loss": kl_loss,
        }

def worker_init_fn(worker_id: int):
    worker_info = get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        dataset.hf = None     # type: ignore

def train_physics_vae(
    h5_path: str | Path,
    input_dim: int = 10000,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = learning_rate,
    weight_decay: float = 1e-4,
    device: str | None = None,
    debug: bool = False,
) -> PhysicsInformedVAE:

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    print(f"🚀 Training on device: {device_obj}")

    train_dataset = LIBSSpectraDatasetInMemory(h5_path, split='train')
    val_dataset = LIBSSpectraDatasetInMemory(h5_path, split='val')

    n_elements = len(ALL_COLUMNS)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device_obj.type == 'cuda'),
        worker_init_fn=worker_init_fn,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device_obj.type == 'cuda'),
        worker_init_fn=worker_init_fn,
        persistent_workers=True
    )

    wl_grid = torch.linspace(200, 900, input_dim, device=device_obj)

    model = PhysicsInformedVAE(
        input_dim=input_dim,
        n_elements=n_elements,
        wl_grid=wl_grid,
        master_table_path=master_table_path,
        n_broadening=1,  # Predict 1 global Lorentz broadening parameter gamma
    ).to(device_obj)

    criterion = PhysicsVAELoss(alpha_recon=1.0, beta_sup=10.0, gamma_kl=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)

    best_val_loss = float("inf")

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # --- Training ---
        model.train()
        train_running_loss, train_recon_acc, train_sup_acc = 0.0, 0.0, 0.0

        with torch.autograd.set_detect_anomaly(debug):      # type: ignore
            for batch in train_loader:
                x = batch["spectrum"].to(device_obj, non_blocking=True)
                y_conc = batch["targets"].to(device_obj, non_blocking=True)

                if torch.isnan(x).any() or torch.isnan(y_conc).any():
                    raise ValueError("Input batch contains NaN values!")

                optimizer.zero_grad()

                # Autocast context for forward pass & loss computation
                with torch.amp.autocast(device_type='cuda', enabled=(device_obj.type == 'cuda'), dtype=torch.bfloat16):      # type: ignore
                    x_recon, mu_p, logvar_p, mu_m, logvar_m = model(x)
                    
                    # Slicing target concentrations
                    pred_conc = nn.functional.softplus(mu_p[:, :n_elements])

                    losses = criterion(
                        x_recon=x_recon,
                        x_target=x,
                        pred_conc=pred_conc,
                        true_conc=y_conc,
                        mu_p=mu_p,
                        logvar_p=logvar_p,
                        mu_m=mu_m,
                        logvar_m=logvar_m,
                    )

                    loss = losses["loss"]

                if torch.isnan(loss):
                    print(f"⚠️ Warning: NaN detected in loss at epoch {epoch}. Skipping step.")
                    continue

                # 1. Scale loss and run backward pass
                scaler.scale(loss).backward()

                # 2. Unscale gradients before clipping!
                scaler.unscale_(optimizer)

                # 3. Clip unscaled gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # 4. Optimizer step via scaler & update scale factor
                scaler.step(optimizer)
                scaler.update()

                train_running_loss += loss.item() * x.size(0)
                train_recon_acc += losses["recon_loss"].item() * x.size(0)
                train_sup_acc += losses["sup_loss"].item() * x.size(0)

        # --- Validation ---
        model.eval()
        val_running_loss, val_recon_acc, val_sup_acc = 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["spectrum"].to(device_obj, non_blocking=True)
                y_conc = batch["targets"].to(device_obj, non_blocking=True)

                x_recon, mu_p, logvar_p, mu_m, logvar_m = model(x)
                pred_conc = nn.functional.softplus(mu_p[:, :n_elements])

                losses = criterion(
                    x_recon=x_recon,
                    x_target=x,
                    pred_conc=pred_conc,
                    true_conc=y_conc,
                    mu_p=mu_p,
                    logvar_p=logvar_p,
                    mu_m=mu_m,
                    logvar_m=logvar_m,
                )

                val_running_loss += losses["loss"].item() * x.size(0)
                val_recon_acc += losses["recon_loss"].item() * x.size(0)
                val_sup_acc += losses["sup_loss"].item() * x.size(0)

        epoch_train_loss = train_running_loss / len(train_dataset)
        epoch_val_loss = val_running_loss / len(val_dataset)
        elapsed = time.time() - start_time
        scheduler.step(epoch_val_loss)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | "
            f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}"
        )

    return model


if __name__ == "__main__":
    print('Starting run')
    loop_time_start = time.perf_counter()
    h5_file = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/exp_syn_train_val_test_dataset_clean.h5"

    trained_model = train_physics_vae(
        h5_path=h5_file,
        input_dim=10000,
        epochs=5,
        batch_size=64,
        lr=learning_rate,
        weight_decay=0.0001,
        device='cuda',
        debug=False
    )

    loop_time_end = time.perf_counter()
    print(f"Run completed in {(loop_time_end - loop_time_start) / 60:.2f} minutes.")

    # print(f'length of all_cols is {len(ALL_COLUMNS)}')