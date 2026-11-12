from __future__ import annotations

"""
jvlee_LIBS_ML > LIBS > p3VAE > p3vae_004.py
"""

# region Imports
import h5py
import sys
import time
import torch
import math

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path
from typing import Any
from torch.utils.data import DataLoader, Dataset, get_worker_info
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from setup import add_project_root_to_path
add_project_root_to_path(parent_generation=1)

from utils import (
    get_worker_logger,
    log,
    logging,
)
# endregion

# region Global Variables
    # B 32, C 16, no compile wrape --> 47 min/epoch
BATCH_SIZE = 128         # Prevents GPU OOM on your local GPU
CHUNK_SIZE = 128         # Keeps broadcast tensor overhead tiny
MAX_EPOCHS = 200          # Fast execution loop
MIN_LR = 1e-06
# ****************************************************************************************************
cpus_per_task = 8   # TODO: MUST UPDATE FOR ANY CHANGE IN SLURM/SRUN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ****************************************************************************************************
NUM_WORKERS = cpus_per_task - 2
subset_size = 500       # NOTE: not currently using
learning_rate = 1e-03
weight_decay = 1e-04

ELEMENT_WEIGHTS_REFERENCE = {
    b'Ba': 5.0, b'Ca': 5.0, b'Ce': 5.0, b'Cr': 5.0, b'Cs': 5.0,
    b'Fe': 5.0, b'Gd': 5.0, b'K':  0.2, b'La': 5.0, b'Li': 0.2,
    b'Mg': 5.0, b'Mn': 5.0, b'Nd': 5.0, b'Ni': 5.0, b'Sm': 5.0, 
    b'Sr': 5.0, b'U':  5.0, b'Y':  5.0, b'Cl':0.2,
}

ELEMENT_WEIGHTS = torch.tensor([
    5.0, 5.0, 5.0, 5.0, 5.0,
    5.0, 5.0, 0.2, 5.0, 0.2,
    5.0, 5.0, 5.0, 5.0, 5.0,
    5.0, 5.0, 5.0, 0.2,
], dtype=torch.float32)


h5_file = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/cts_noleak_xandy.h5"
master_table_path = Path('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/master_line_table.csv').resolve()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_save_path = Path(f'/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/p3VAE/trained_models/best_p3vae_model_{timestamp}.pt').resolve()
model_save_path.parent.mkdir(parents=True, exist_ok=True)
# endregion

class BankedPhysicsDecoder(nn.Module):
    def __init__(self, wl_grid, master_table_path, n_elements,
                 n_gamma=64, g_min=0.01, g_max=5.0):
        super().__init__()
        df = pd.read_csv(master_table_path)
        centers = torch.tensor(df['wavelength_nm'].values, dtype=torch.float32)
        amps = torch.tensor(df['base_intensity'].values, dtype=torch.float32)
        amps = amps / amps.max()
        elem = torch.tensor(df['elem_idx'].values)
        wl = wl_grid.detach().cpu().float()
        gammas = torch.logspace(math.log10(g_min), math.log10(g_max), n_gamma)

        bank = torch.zeros(n_gamma, n_elements, wl.numel())
        for k, g in enumerate(gammas):
            for e in range(n_elements):
                m = elem == e
                c, a = centers[m], amps[m]
                for i in range(0, len(c), 256):
                    d = (wl[None, :] - c[i:i+256, None])**2 + g**2 + 1e-6
                    bank[k, e] += (a[i:i+256, None] * (g / math.pi) / d).sum(0)
        self.register_buffer('bank', bank)
        self.register_buffer('log_g', gammas.log())
        self.G = n_gamma

    def forward(self, conc, broadening, chunk_size=None):
        with torch.autocast(device_type=conc.device.type, enabled=False):
            conc = conc.float()
            lg = torch.log(broadening.float().clamp(0.01, 5.0)).reshape(-1)
            step = self.log_g[1] - self.log_g[0]
            pos = (lg - self.log_g[0]) / step
            lo = pos.floor().clamp(0, self.G - 2).long()
            w = (pos - lo).clamp(0, 1)[:, None, None]
            prof = (1 - w) * self.bank[lo] + w * self.bank[lo + 1]   # [B, E, W]
            return torch.einsum('be,bew->bw', conc, prof)


class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class LIBSSpectraDataset(Dataset):
    def __init__(
            self, 
            h5_path: str | Path, 
            log_path: str | Path,
            split: str = 'train', 
            in_memory: bool = False,
    ):
        if log_path is None:
            log_path = Path(r"/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/logs/p3vae_logs/default.txt").resolve()
        logger = get_worker_logger(Path(log_path).stem)

        self.h5_path = str(h5_path)
        self.split = split
        self.in_memory = in_memory
        self.hf = None

        self.y_key = f"y_{self.split}"
        self.x_key = f"X_{self.split}"
        self.syn_key = f"{self.split}_syn"

        with h5py.File(self.h5_path, 'r') as hf:
            if self.y_key not in hf:
                available_splits = [
                    k.replace('y_', '') for k in hf.keys() if k.startswith('y_') and not k.endswith('_backup')
                ]
                raise KeyError(
                    f"Split '{split}' (key '{self.y_key}') not found in {self.h5_path}. "
                    f"Available splits: {available_splits}"
                )

            self.length = hf[self.y_key].shape[0]     # type: ignore
            self.n_features = hf['wavelengths'].shape[0]     # type: ignore
            self.n_targets = hf['feature_cols'].shape[0]     # type: ignore

            self.wavelengths = hf['wavelengths'][:]     # type: ignore
            self.target_names = hf['feature_cols'][:]     # type: ignore

            if self.in_memory:
                log(logger=logger, msg=f"Loading '{self.split}' dataset ({self.length} samples) into memory...")
                self.spectra = torch.from_numpy(hf[self.y_key][:]).float()     # type: ignore
                self.targets = torch.from_numpy(hf[self.x_key][:]).float()     # type: ignore
                self.is_synth = torch.from_numpy(hf[self.syn_key][:]).long()     # type: ignore

        # Ensure self.hf starts clean for workers
        self.hf = None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.in_memory:
            return {
                'spectrum': self.spectra[idx],
                'targets': self.targets[idx],
                'is_synthetic': self.is_synth[idx],
            }

        if self.hf is None:
            self.hf = h5py.File(self.h5_path, 'r', swmr=True)

        spectrum = self.hf[self.y_key][idx]     # type: ignore
        targets = self.hf[self.x_key][idx]     # type: ignore
        is_synth = self.hf[self.syn_key][idx]     # type: ignore

        return {
            'spectrum': torch.from_numpy(spectrum).float(),
            'targets': torch.from_numpy(targets).float(),
            'is_synthetic': torch.tensor(is_synth, dtype=torch.long),
        }

    def __del__(self):
        if self.hf is not None:
            self.hf.close()


class PhysicsSpectralDecoder(nn.Module):
    def __init__(self, wl_grid: torch.Tensor | list | None, master_table_path: str | Path):
        super().__init__()

        if not isinstance(wl_grid, torch.Tensor):
            wl_grid = torch.tensor(wl_grid, dtype=torch.float32)

        master_df = pd.read_csv(master_table_path)

        nist_wls = torch.tensor(master_df['wavelength_nm'].values, dtype=torch.float32)
        nist_intens = torch.tensor(master_df['base_intensity'].values, dtype=torch.float32)
        elem_indices = torch.tensor(master_df['elem_idx'].values, dtype=torch.long)

        self.register_buffer('grid', wl_grid.view(1, 1, -1))
        self.register_buffer('line_centers', nist_wls.view(1, -1, 1))

        max_amp = nist_intens.max() if nist_intens.max() > 0 else 1.0
        self.register_buffer('line_base_amps', (nist_intens / max_amp).unsqueeze(0))
        self.register_buffer('elem_indices', elem_indices)

        self.num_lines = len(master_df)

    @staticmethod
    def _compute_chunk_profile(
        amps_chunk: torch.Tensor, 
        gammas: torch.Tensor, 
        grid: torch.Tensor, 
        centers_chunk: torch.Tensor
    ) -> torch.Tensor:
        denom = ((grid - centers_chunk) ** 2) + (gammas ** 2) + 1e-6
        profiles_chunk = amps_chunk * (gammas / torch.pi) / denom
        return torch.sum(profiles_chunk, dim=1)

    def forward(
        self, 
        concentrations: torch.Tensor, 
        broadening: torch.Tensor, 
        chunk_size: int = CHUNK_SIZE,
    ) -> torch.Tensor:
        batch_size = concentrations.size(0)

        line_concs = concentrations[:, self.elem_indices]
        amps_all = (line_concs * self.line_base_amps).unsqueeze(2)

        if broadening.dim() == 1:
            broadening = broadening.unsqueeze(1)
        gammas = torch.clamp(broadening.unsqueeze(2), min=0.01, max=5.0)

        # Match precision across all input tensors for AMP/Checkpoint compatibility
        target_dtype = concentrations.dtype
        grid_in = self.grid.to(dtype=target_dtype)
        centers_in = self.line_centers.to(dtype=target_dtype)
        # amps_all = amps_all.to(dtype=target_dtype)
        # gammas = gammas.to(dtype=target_dtype)

        total_lines = centers_in.size(1)
        recon_spectrum = torch.zeros(
            (batch_size, grid_in.size(2)), 
            device=concentrations.device, 
            dtype=target_dtype
        )

        for i in range(0, total_lines, chunk_size):
            amps_chunk = amps_all[:, i:i + chunk_size, :]
            centers_chunk = centers_in[:, i:i + chunk_size, :]

            if self.training:
                chunk_sum = checkpoint(
                    self._compute_chunk_profile,
                    amps_chunk,
                    gammas,
                    grid_in,
                    centers_chunk,
                    use_reentrant=False
                )
            else:
                chunk_sum = self._compute_chunk_profile(
                    amps_chunk, gammas, grid_in, centers_chunk
                )

            recon_spectrum = recon_spectrum + chunk_sum

        return recon_spectrum


class PhysicsInformedVAE(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        n_elements: int, 
        wl_grid: torch.Tensor, 
        master_table_path: str | Path,
        n_broadening: int = 1,
        matrix_latent_dim: int = 8
    ):
        super().__init__()
        self.n_elements = n_elements
        self.n_broadening = n_broadening
        self.matrix_latent_dim = matrix_latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        phys_dim = self.n_elements + self.n_broadening
        self.fc_mean_phys = nn.Linear(128, phys_dim)
        self.fc_logvar_phys = nn.Linear(128, phys_dim)

        self.fc_mean_matrix = nn.Linear(128, self.matrix_latent_dim)
        self.fc_logvar_matrix = nn.Linear(128, self.matrix_latent_dim)

        # self.physics_decoder = PhysicsSpectralDecoder(
        #     wl_grid=wl_grid,
        #     master_table_path=master_table_path
        # )

        self.physics_decoder = BankedPhysicsDecoder(
            wl_grid=wl_grid, master_table_path=master_table_path,
            n_elements=n_elements, n_gamma=32, g_min=0.01, g_max=5.0,
        )

        self.matrix_decoder = nn.Sequential(
            nn.Linear(self.matrix_latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

        self.residual_scale = nn.Parameter(torch.tensor([1.0]))

        nn.init.constant_(self.fc_logvar_phys.weight, 0.0)
        nn.init.constant_(self.fc_logvar_phys.bias, 0.0)
        nn.init.constant_(self.fc_logvar_matrix.weight, 0.0)
        nn.init.constant_(self.fc_logvar_matrix.bias, 0.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def _extract_physical_params(self, z_phys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        concs_raw = z_phys[:, :self.n_elements]
        gamma_raw = z_phys[:, self.n_elements:]

        concentrations = torch.clamp(F.softplus(concs_raw), min=1e-5, max=100.0)
        broadening = torch.clamp(F.softplus(gamma_raw), min=1e-3, max=10.0)

        return concentrations, broadening

    def forward(
            self, 
            x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        h = self.encoder(x)

        mu_p = torch.clamp(self.fc_mean_phys(h), min=-10.0, max=10.0)
        logvar_p = torch.clamp(self.fc_logvar_phys(h), min=-6.0, max=6.0)
        z_phys = self.reparameterize(mu_p, logvar_p)

        concentrations, broadening = self._extract_physical_params(z_phys)

        mu_m = self.fc_mean_matrix(h)
        logvar_m = torch.clamp(self.fc_logvar_matrix(h), min=-8.0, max=8.0)
        z_matrix = self.reparameterize(mu_m, logvar_m)

        ideal_physics_spectrum = self.physics_decoder(concentrations, broadening, chunk_size=CHUNK_SIZE)
        matrix_residual = self.matrix_decoder(z_matrix) * self.residual_scale

        reconstructed_spectrum = ideal_physics_spectrum + matrix_residual

        return {
            'reconstructed_spectrum': reconstructed_spectrum,
            'ideal_physics_spectrum': ideal_physics_spectrum,
            'matrix_residual': matrix_residual,
            'concentrations': concentrations,
            'broadening': broadening,
            'mu_p': mu_p,
            'logvar_p': logvar_p,
            'mu_m': mu_m,
            'logvar_m': logvar_m,
        }


class PhysicsVAELoss(nn.Module):

    def __init__(
        self,
        element_weights: torch.Tensor | None = None,
        alpha_recon: float = 1.0,
        beta_sup: float = 10.0,
        gamma_kl: float = 1e-4,
        use_log_conc: bool = True,
    ):
        super().__init__()
        self.alpha_recon = alpha_recon
        self.beta_sup = beta_sup
        self.gamma_kl = gamma_kl
        self.use_log_conc = use_log_conc

        if element_weights is not None:
            if not isinstance(element_weights, torch.Tensor):
                element_weights = torch.tensor(element_weights, dtype=torch.float32)
            self.register_buffer("element_weights", element_weights)
        else:
            self.element_weights = None

    @staticmethod
    def _kl_divergence(
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """Computes KL divergence between standard Gaussian N(0, I) and N(mu, std^2)."""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def forward(
        self, 
        model_outputs: dict[str, torch.Tensor], 
        batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        x_recon = model_outputs["reconstructed_spectrum"]
        x_target = batch["spectrum"]

        pred_conc = model_outputs["concentrations"]
        true_conc = batch["targets"]

        # 1. Spectral Reconstruction Loss
        recon_loss = F.mse_loss(x_recon, x_target, reduction="mean")

        # 2. Dopant-Focused Concentration Loss
        if self.use_log_conc:
            diff = torch.log1p(pred_conc) - torch.log1p(true_conc)
        else:
            diff = pred_conc - true_conc

        element_errors = diff**2  # [Batch, N_elements]

        if self.element_weights is not None:
            element_errors = element_errors * self.element_weights

        sup_loss = element_errors.mean()

        # 3. KL Divergence
        kl_phys = self._kl_divergence(
            model_outputs["mu_p"], model_outputs["logvar_p"]
        )
        kl_matrix = self._kl_divergence(
            model_outputs["mu_m"], model_outputs["logvar_m"]
        )
        kl_loss = kl_phys + kl_matrix

        total_loss = (
            (self.alpha_recon * recon_loss)
            + (self.beta_sup * sup_loss)
            + (self.gamma_kl * kl_loss)
        )

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
        dataset.hf = None   # type: ignore
        # if hasattr(dataset, 'in_memory') and not dataset.in_memory:     # type: ignore
        #     dataset.hf = None     # type: ignore

def train_physics_vae(
    h5_path: str | Path,
    master_table_path: str | Path,
    log_path: str | Path | None = None,
    save_path: str | Path | None = None,
    input_dim: int = 10000,
    epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = learning_rate,
    weight_decay: float = weight_decay,
    in_memory: bool = True,
    num_workers: int = NUM_WORKERS,
    device: str | None = None,
) -> PhysicsInformedVAE:

    if not save_path:
        save_path = model_save_path

    # region Logger Setup
    if log_path is None:
        log_path = Path(f"/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/logs/p3vae_logs/default_{timestamp}.txt").resolve()
    else:
        log_path = Path(str(log_path)).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = get_worker_logger(Path(log_path).stem)
    log(logger=logger, msg='hi')
    # endregion

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    log(logger=logger, msg=f"🚀 Training on device: {device_obj}")

    num_workers = 0 if in_memory else num_workers

    log(logger=logger, msg='Loading training dataset')
    train_dataset = LIBSSpectraDataset(h5_path, log_path=log_path, split='train', in_memory=in_memory)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device_obj.type == 'cuda'),
        worker_init_fn=worker_init_fn if not in_memory else None,
        persistent_workers=(num_workers > 0),
    )
    
    log(logger=logger, msg='Loading validation dataset')
    val_dataset = LIBSSpectraDataset(h5_path, log_path=log_path, split='val', in_memory=in_memory)
    val_num_workers = max(0, num_workers // 2)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers // 2),
        pin_memory=(device_obj.type == 'cuda'),
        worker_init_fn=worker_init_fn if not in_memory else None,
        persistent_workers=(val_num_workers > 0),
    )

    wl_grid = torch.from_numpy(train_dataset.wavelengths).to(device_obj)
    n_elements = train_dataset.n_targets

    log(logger=logger, msg='Initiate p3vae model.')
    model = PhysicsInformedVAE(
        input_dim=input_dim,
        n_elements=n_elements,
        wl_grid=wl_grid,
        master_table_path=master_table_path,
        n_broadening=1,
    ).to(device_obj)

    model = torch.compile(model=model, mode='reduce-overhead')

    log(logger=logger, msg='Initiate custom loss function')
    criterion = PhysicsVAELoss(
        element_weights=ELEMENT_WEIGHTS,
        alpha_recon=1.0, 
        beta_sup=10.0, 
        gamma_kl=1e-4,
        use_log_conc=True
    ).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)     # type: ignore
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=MIN_LR)

    use_amp = device_obj.type == 'cuda'
    # Use GradScaler for float16 mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)     # type: ignore

    best_val_loss = float("inf")

    early_stopper = EarlyStopping(patience=15)
    patience_counter = 0

    log(logger=logger, msg='Start Training and Validation Loops')
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        batch_counter = 0

        # --- TRAINING ---
        model.train()     # type: ignore
        train_loss, train_recon, train_sup = 0.0, 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d}",miniters=25, leave=True)

        for batch in pbar:
            batch_counter += 1
            batch = {k: v.to(device_obj, non_blocking=True) for k, v in batch.items()}

            if torch.isnan(batch['spectrum']).any():
                raise ValueError("Input spectrum contains NaN values!")

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', enabled=use_amp, dtype=torch.float16):     # type: ignore
                model_outputs = model(batch['spectrum'])
                loss_dict = criterion(model_outputs, batch)
                loss = loss_dict["loss"]

            if torch.isnan(loss):
                log(logger=logger, msg=f"⚠️ Warning: NaN detected in loss at epoch {epoch}. Skipping step.")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)     # type: ignore
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

            batch_sz = batch['spectrum'].size(0)
            train_loss += loss.item() * batch_sz
            train_recon += loss_dict["recon_loss"].item() * batch_sz
            train_sup += loss_dict["sup_loss"].item() * batch_sz

        # --- VALIDATION ---
        model.eval()     # type: ignore
        val_loss, val_recon, val_sup = 0.0, 0.0, 0.0
        val_batch_counter = 0

        with torch.no_grad():
            for batch in val_loader:
                val_batch_counter += 1
                batch = {k: v.to(device_obj, non_blocking=True) for k, v in batch.items()}

                with torch.amp.autocast(device_type='cuda', enabled=use_amp, dtype=torch.float16):     # type: ignore
                    model_outputs = model(batch['spectrum'])
                    loss_dict = criterion(model_outputs, batch)

                batch_sz = batch['spectrum'].size(0)
                val_loss += loss_dict["loss"].item() * batch_sz
                val_recon += loss_dict["recon_loss"].item() * batch_sz
                val_sup += loss_dict["sup_loss"].item() * batch_sz

        epoch_train_loss = train_loss / len(train_dataset)
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_recon = val_recon / len(val_dataset)
        epoch_val_sup = val_sup / len(val_dataset)

        elapsed = time.time() - start_time
        scheduler.step(epoch_val_loss)

        log(logger=logger, msg=f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | (Recon MSE: {epoch_val_recon:.4f}, Conc MSE: {epoch_val_sup:.4f})")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            raw_model = getattr(model, '_orig_mod', model)
            torch.save({
                'epoch': epoch,
                'model_state_dict': raw_model.state_dict(),     # type: ignore
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, save_path)
            log(logger=logger, msg=f"  💾 Saved new best model checkpoint to {save_path}")

        if optimizer.param_groups[0]['lr'] <= MIN_LR:
            patience_counter += 1
            if patience_counter >= 10:
                log(logger=logger, msg=f'🛑 Early stopping triggered at epoch {epoch}: model converged.')

        if early_stopper(epoch_val_loss):
            log(logger=logger, msg=f"🛑 Early stopping triggered at epoch {epoch}: model converged.")
            break

    checkpoint = torch.load(save_path, map_location=device_obj)
    raw_model = getattr(model, '_orig_mod', model)
    raw_model.load_state_dict(checkpoint['model_state_dict'])     # type: ignore
    log(logger=logger, msg=f"✨ Training complete. Loaded best model from epoch {checkpoint['epoch']} (Val Loss: {best_val_loss:.4f})")

    return model     # type: ignore

def evaluate_and_plot(
    checkpoint_path: str | Path = "best_p3vae_model.pt",
    h5_path: str | Path = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/cts_xandy.h5",
    master_table_path: str | Path = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/master_line_table.csv",
    split: str = "val",
    output_dir: str | Path = "plots",
    log_path: str | Path = '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/logs/p3vae_logs/default.txt',
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # region Logger Setup
    if log_path is None:
        log_path = Path(r"/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/logs/p3vae_logs/default.txt").resolve()
    logger = get_worker_logger(Path(log_path).stem)
    # endregion

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(logger=logger, msg=f"Running evaluation on: {device}")

    # 1. Load Dataset
    val_dataset = LIBSSpectraDataset(h5_path, log_path=log_path, split=split, in_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    wl_grid = torch.from_numpy(val_dataset.wavelengths).to(device)
    target_names = [
        name.decode("utf-8") if isinstance(name, bytes) else str(name)
        for name in val_dataset.target_names    # type: ignore
    ]

    # 2. Instantiate and Load Model Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = PhysicsInformedVAE(
        input_dim=val_dataset.n_features,
        n_elements=val_dataset.n_targets,
        wl_grid=wl_grid,
        master_table_path=master_table_path,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    log(logger=logger, msg=f"Loaded model from epoch {checkpoint.get('epoch', 'N/A')} | (Val Loss: {checkpoint.get('val_loss', 0.0):.6f})")

    # 3. Predict across dataset
    all_true_conc = []
    all_pred_conc = []
    all_spectra = []
    all_recons = []
    all_physics = []
    all_residuals = []

    with torch.no_grad():
        for batch in val_loader:
            x_target = batch["spectrum"].to(device)
            true_conc = batch["targets"].cpu().numpy()

            outputs = model(x_target)

            all_true_conc.append(true_conc)
            all_pred_conc.append(outputs["concentrations"].cpu().numpy())
            all_spectra.append(x_target.cpu().numpy())
            all_recons.append(outputs["reconstructed_spectrum"].cpu().numpy())
            all_physics.append(outputs["ideal_physics_spectrum"].cpu().numpy())
            all_residuals.append(outputs["matrix_residual"].cpu().numpy())

    true_conc = np.concatenate(all_true_conc, axis=0)
    pred_conc = np.concatenate(all_pred_conc, axis=0)
    true_spectra = np.concatenate(all_spectra, axis=0)
    recon_spectra = np.concatenate(all_recons, axis=0)
    phys_spectra = np.concatenate(all_physics, axis=0)
    res_spectra = np.concatenate(all_residuals, axis=0)

    # =========================================================================
    # PLOT 1: Concentration Parity Grid (Pred vs Reality)
    # =========================================================================
    n_elements = len(target_names)
    cols = 5
    rows = int(np.ceil(n_elements / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.2))
    axes = axes.flatten()

    for idx, name in enumerate(target_names):
        ax = axes[idx]
        y_true = true_conc[:, idx]
        y_pred = pred_conc[:, idx]

        # Calculate R^2 score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-8
        r2 = 1.0 - (ss_res / ss_tot)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        # Plot points and 1:1 parity line
        ax.scatter(y_true, y_pred, alpha=0.4, s=12, color="#1f77b4", edgecolors="none")
        max_val = max(np.max(y_true), np.max(y_pred))
        min_val = min(np.min(y_true), np.min(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.7, lw=1)

        ax.set_title(f"{name}\n$R^2$: {r2:.3f} | RMSE: {rmse:.2f}%", fontsize=9, fontweight="bold")
        ax.set_xlabel("True Conc (%)", fontsize=8)
        ax.set_ylabel("Pred Conc (%)", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)

    # Hide unused subplots
    for idx in range(n_elements, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    parity_path = output_dir / f"concentration_parity_plots_{split}.png"
    plt.savefig(parity_path, dpi=300)
    plt.close()
    log(logger=logger, msg=f"Saved Concentration Parity Plot to: {parity_path}")

    # =========================================================================
    # PLOT 2: Spectral Reconstruction Overlay (Random Sample)
    # =========================================================================
    sample_idx = np.random.randint(0, len(true_spectra))
    wl = val_dataset.wavelengths

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top panel: Measured vs Total Reconstruction
    ax1.plot(wl, true_spectra[sample_idx], label="Measured Spectrum", color="black", alpha=0.7, lw=1)
    ax1.plot(wl, recon_spectra[sample_idx], label="Model Reconstruction", color="crimson", alpha=0.8, lw=1)
    ax1.set_ylabel("Intensity (a.u.)", fontsize=10)
    ax1.set_title(f"Sample #{sample_idx} - Full Spectrum Overlay", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Bottom panel: Physics model vs Matrix Residual breakdown
    ax2.plot(wl, phys_spectra[sample_idx], label="Physics Emission Component", color="forestgreen", lw=1)
    ax2.plot(wl, res_spectra[sample_idx], label="Matrix Residual Component", color="orange", lw=1)
    ax2.set_xlabel("Wavelength (nm)", fontsize=10)
    ax2.set_ylabel("Intensity (a.u.)", fontsize=10)
    ax2.set_title("Model Component Breakdown", fontsize=11)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    spec_path = output_dir / f"sample_{sample_idx}_spectral_breakdown_{split}.png"
    plt.savefig(spec_path, dpi=300)
    plt.close()
    log(logger=logger, msg=f"Saved Spectral Breakdown Plot to: {spec_path}")

if __name__ == "__main__":
    print('Starting run')
    loop_time_start = time.perf_counter()

    # # ---------------------------------------------------
    # # Compare new and old physics decoders
    # # ---------------------------------------------------
    # log_path= '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/logs/p3vae_logs/004_log.txt'
    # train_dataset = LIBSSpectraDataset(h5_file, log_path=log_path, split='train', in_memory=True)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # device_obj = torch.device(device)
    # wl_grid = torch.from_numpy(train_dataset.wavelengths).to(device_obj)

    # old = PhysicsSpectralDecoder(wl_grid, master_table_path).cuda().eval()
    # new = BankedPhysicsDecoder(wl_grid, master_table_path, 19).cuda().eval()
    # conc = torch.rand(8, 19, device='cuda') * 2
    # gam  = torch.rand(8, 1, device='cuda') * 2 + 0.05
    # a, b = old(conc, gam), new(conc, gam)
    # print(((a - b).abs().max() / a.abs().max()).item())

    # --------------------------------------------------
    # Train Model
    # --------------------------------------------------
    model = train_physics_vae(
        h5_path=h5_file,
        master_table_path=master_table_path,
        log_path= '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/logs/p3vae_logs/004_noleak_log.txt',
        save_path= model_save_path,
        input_dim = 10000,
        epochs = MAX_EPOCHS,
        batch_size = BATCH_SIZE,
        lr = learning_rate,
        weight_decay = 1e-4,
        in_memory = True,
        num_workers = NUM_WORKERS,
        device = 'cuda',
    )

    # --------------------------------------------------
    # Evaluate Model
    # --------------------------------------------------
    evaluate_and_plot(
        checkpoint_path=model_save_path,
        h5_path=h5_file,
        master_table_path=master_table_path,
        split='test',
        output_dir='/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/p3VAE/plots'
    )

    loop_time_end = time.perf_counter()
    print(f"Run completed in {(loop_time_end - loop_time_start) / 60:.2f} minutes.")