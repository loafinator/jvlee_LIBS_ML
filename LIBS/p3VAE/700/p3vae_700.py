from __future__ import annotations

"""
jvlee_LIBS_ML > LIBS > p3VAE > 700 > p3vae_700_3.py
Memory-safe, benchmark-aligned P3VAE pipeline for processed-data_20261110b:
- Uses blake2b-fingerprinted 7-fold splits shared across PLS, 1D-CNN, Rev-PLS, and Rev-MLP
- Evaluates both Encoder (Spectra -> Conc) and Decoders (Latent / Conc -> Spectra)
- Accumulates chunked float64 SSE/SST and caches only representative plotting rows (<15 MB .npz)
"""

import os
import sys
import time
import math
import gc
import random
import hashlib
import h5py
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from typing import Any, cast
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from setup import add_project_root_to_path
add_project_root_to_path(parent_generation=1)

from utils import (
    get_worker_logger,
    log,
    get_h5_ds,
)


# =============================================================================
# Global Configuration & File Paths
# =============================================================================
H5_FILE = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/exp-syn_processed-data_20261110b.h5"
SPLIT_DIR = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/fold_splits/processed-data_20261110b")
OUTPUT_DIR = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/plots/P3VAE/processed-data_20261110b")
MASTER_TABLE_PATH = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/master_line_table.csv").resolve()
TRAINED_MODEL_DIR = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/p3VAE/trained_models/processed-data_20261110b")
LOG_DIR = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/logs/p3vae_logs")

NUM_FOLDS = 7
SEEDS = [42]
SEED = 42
MAX_WT_PERCENT = 12.0

HOST_ELEMENTS = {'Li', 'K', 'Cl'}
EXP_DOPANTS = {'Ce', 'Gd', 'Sm', 'U'}
TARGET_NAMES = ['Ce', 'Gd', 'Sm', 'U']
DOPANT_NAMES = {'Ce', 'Gd', 'Sm', 'U'}

# P3VAE Hyperparameters
BATCH_SIZE = 128
CHUNK_SIZE = 128
MAX_EPOCHS = 200
MIN_LR = 1e-06
LEARNING_RATE = 1e-03
WEIGHT_DECAY = 1e-04
DROPOUT = 0.15
LR_PATIENCE = 4
STOP_PATIENCE = 20

# Worker setup (update CPUS_PER_TASK to match SLURM allocation)
CPUS_PER_TASK = 8
NUM_WORKERS = max(0, min(CPUS_PER_TASK - 2, (os.cpu_count() or 2) - 1))

ELEMENT_WEIGHTS_REFERENCE = {
    b'Ba': 5.0, b'Ca': 5.0, b'Ce': 5.0, b'Cr': 5.0, b'Cs': 5.0,
    b'Fe': 5.0, b'Gd': 5.0, b'K':  0.2, b'La': 5.0, b'Li': 0.2,
    b'Mg': 5.0, b'Mn': 5.0, b'Nd': 5.0, b'Ni': 5.0, b'Sm': 5.0,
    b'Sr': 5.0, b'U':  5.0, b'Y':  5.0, b'Cl': 0.2,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Model & Loss Definitions
# =============================================================================
class BankedPhysicsDecoder(nn.Module):
    def __init__(
        self,
        wl_grid: torch.Tensor,
        master_table_path: str | Path,
        n_elements: int,
        elem_names: list[str] | None = None,
        keep_idx: list[int] | None = None,
        n_gamma: int = 32,
        g_min: float = 0.01,
        g_max: float = 5.0,
    ):
        super().__init__()
        df = pd.read_csv(master_table_path)
        centers = torch.tensor(df['wavelength_nm'].values, dtype=torch.float32)
        amps = torch.tensor(df['base_intensity'].values, dtype=torch.float32)
        wl = wl_grid.detach().cpu().float()
        wl_min, wl_max = float(wl.min()), float(wl.max())
        gammas = torch.logspace(math.log10(g_min), math.log10(g_max), n_gamma)

        elem_col_clean = None
        if 'element' in df.columns:
            elem_col_clean = (
                df['element']
                .astype(str)
                .str.strip()
                .str.replace(r"^b'|'$", "", regex=True)
                .values
            )

        bank = torch.zeros(n_gamma, n_elements, wl.numel())
        mid_k = n_gamma // 2

        for e in range(n_elements):
            elem_label = elem_names[e] if elem_names is not None else str(e)
            if elem_col_clean is not None and elem_names is not None:
                m = (elem_col_clean == elem_names[e])
            elif keep_idx is not None and 'elem_idx' in df.columns:
                m = (df['elem_idx'].values == keep_idx[e])
            else:
                m = (df['elem_idx'].values == e)

            if int(np.sum(m)) == 0:
                raise ValueError(
                    f"No spectral lines matched for element '{elem_label}' in {master_table_path}!"
                )

            c, a = centers[m], amps[m]
            in_range = int(((c >= wl_min) & (c <= wl_max)).sum().item())
            if in_range == 0:
                print(f"⚠️ Element '{elem_label}' has {len(c)} lines in table, but 0 inside [{wl_min:.1f}, {wl_max:.1f}] nm!")

            # Keep lines within +/- 5*g_max of detector bounds so edge Lorentzian wings aren't clipped
            near_grid = (c >= wl_min - 5.0 * g_max) & (c <= wl_max + 5.0 * g_max)
            c, a = c[near_grid], a[near_grid]
            if len(c) == 0:
                continue

            # Per-element amplitude pre-normalization before accumulating Lorentzian profiles
            a = a / a.max().clamp(min=1e-8)

            for k, g in enumerate(gammas):
                for i in range(0, len(c), 256):
                    d = (wl[None, :] - c[i:i + 256, None]) ** 2 + g ** 2 + 1e-6
                    bank[k, e] += (a[i:i + 256, None] * (g / math.pi) / d).sum(0)

            # Normalize each element's bank slice by its in-grid peak at reference broadening
            ref_peak = bank[mid_k, e].max().clamp(min=1e-8)
            bank[:, e] /= ref_peak

        self.register_buffer('bank', bank)
        self.register_buffer('log_g', gammas.log())
        self.G = n_gamma

        # Learnable positive per-element emission gain, initialized so softplus(raw_gain) == 1.0
        init_raw_gain = math.log(math.e - 1.0)
        self.raw_element_gain = nn.Parameter(torch.full((n_elements,), init_raw_gain, dtype=torch.float32))

    def forward(self, conc: torch.Tensor, broadening: torch.Tensor, chunk_size: int | None = None) -> torch.Tensor:
        with torch.autocast(device_type=conc.device.type, enabled=False):
            conc = conc.float()
            lg = torch.log(broadening.float().clamp(0.01, 5.0)).reshape(-1)
            step = self.log_g[1] - self.log_g[0]
            pos = (lg - self.log_g[0]) / step
            lo = pos.floor().clamp(0, self.G - 2).long()
            w = (pos - lo).clamp(0, 1)[:, None, None]
            prof = (1 - w) * self.bank[lo] + w * self.bank[lo + 1]  # [B, E, W]
            gain = F.softplus(self.raw_element_gain)                  # [E]
            return torch.einsum('be,e,bew->bw', conc, gain, prof)


class PhysicsInformedVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_elements: int,
        wl_grid: torch.Tensor,
        master_table_path: str | Path,
        elem_names: list[str] | None = None,
        keep_idx: list[int] | None = None,
        n_broadening: int = 1,
        matrix_latent_dim: int = 8,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.n_elements = n_elements
        self.n_broadening = n_broadening
        self.matrix_latent_dim = matrix_latent_dim

        # Feature extraction funnel: 10,000 -> 4096 -> 1024 -> 256 -> 128
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.LayerNorm(4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(4096, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        phys_dim = self.n_elements + self.n_broadening
        self.fc_mean_phys = nn.Linear(128, phys_dim)
        self.fc_logvar_phys = nn.Linear(128, phys_dim)

        self.fc_mean_matrix = nn.Linear(128, self.matrix_latent_dim)
        self.fc_logvar_matrix = nn.Linear(128, self.matrix_latent_dim)

        self.physics_decoder = BankedPhysicsDecoder(
            wl_grid=wl_grid,
            master_table_path=master_table_path,
            n_elements=n_elements,
            elem_names=elem_names,
            keep_idx=keep_idx,
            n_gamma=32,
            g_min=0.01,
            g_max=5.0,
        )

        # Keep matrix decoder small (128 hidden units) to limit unphysical overfitting
        self.matrix_decoder = nn.Sequential(
            nn.Linear(self.matrix_latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid(),
        )

        self.residual_scale = nn.Parameter(torch.tensor([0.05]))

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

        # Normalized [0, 1] concentration range (1.0 == elem_scale_wt)
        concentrations = torch.clamp(F.softplus(concs_raw), min=0.0, max=1.0)
        broadening = torch.clamp(F.softplus(gamma_raw), min=1e-3, max=10.0)
        return concentrations, broadening

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
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
        beta_sup: float = 20.0,
        gamma_kl: float = 1e-4,
        use_log_conc: bool = False,
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
    def _kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def forward(
        self,
        model_outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        x_recon = model_outputs["reconstructed_spectrum"]
        x_target = batch["spectrum"]

        pred_conc = model_outputs["concentrations"]
        true_conc = batch["targets"]

        recon_loss = F.mse_loss(x_recon, x_target, reduction="none").sum(dim=-1).mean()

        if self.use_log_conc:
            diff = torch.log1p(pred_conc) - torch.log1p(true_conc)
        else:
            diff = pred_conc - true_conc

        element_errors = diff ** 2
        if self.element_weights is not None:
            element_errors = element_errors * self.element_weights
        sup_loss = element_errors.mean()

        kl_phys = self._kl_divergence(model_outputs["mu_p"], model_outputs["logvar_p"])
        kl_matrix = self._kl_divergence(model_outputs["mu_m"], model_outputs["logvar_m"])
        kl_loss = kl_phys + kl_matrix
        res_reg = 0.1 * model_outputs["matrix_residual"].pow(2).sum(dim=-1).mean()

        total_loss = (
            (self.alpha_recon * recon_loss)
            + (self.beta_sup * sup_loss)
            + (self.gamma_kl * kl_loss)
            + (res_reg)
        )

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "sup_loss": sup_loss,
            "kl_loss": kl_loss,
        }


class SpectralScaler:
    """Scales L2-normalized spectra to align peak magnitude with the physics decoder without leakage."""
    def __init__(self, mode: str = "robust_max", q: float = 99.9):
        self.mode = mode
        self.q = q
        self.scale_: torch.Tensor | None = None

    def fit(self, x_train: torch.Tensor):
        if self.mode == "global_max":
            self.scale_ = torch.clamp(x_train.max(), min=1e-6)
        elif self.mode == "mean_max":
            self.scale_ = torch.clamp(x_train.max(dim=1).values.mean(), min=1e-6)
        elif self.mode == "robust_max":
            row_maxes = x_train.max(dim=1).values.cpu().numpy()
            self.fit_from_row_maxes(row_maxes)
        else:
            self.scale_ = torch.clamp(x_train.max(), min=1e-6)

    def fit_from_row_maxes(self, row_maxes: np.ndarray):
        """Fits scale from precomputed 1D row maxima without copying multi-GB spectrum slices."""
        if self.mode == "global_max":
            scale_val = float(np.max(row_maxes))
        elif self.mode == "mean_max":
            scale_val = float(np.mean(row_maxes))
        else:
            scale_val = float(np.percentile(row_maxes, self.q))
        self.scale_ = torch.tensor(scale_val, dtype=torch.float32).clamp(min=1e-6)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale_ is None:
            raise RuntimeError("SpectralScaler must be fitted before calling transform!")
        return x / self.scale_.to(x.device)

    def inverse_transform(self, x_scaled: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        if self.scale_ is None:
            raise RuntimeError("SpectralScaler must be fitted before calling inverse_transform!")
        if isinstance(x_scaled, np.ndarray):
            return x_scaled * float(self.scale_.cpu().item())
        return x_scaled * self.scale_.to(x_scaled.device)

    def state_dict(self) -> dict:
        return {"scale_": self.scale_, "mode": self.mode, "q": self.q}

    def load_state_dict(self, state: dict):
        self.scale_ = state["scale_"]
        self.mode = state["mode"]
        self.q = state.get("q", 99.9)


class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 5e-6):
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
    def __init__(self, spectra: torch.Tensor, targets: torch.Tensor, is_synth: np.ndarray, indices: np.ndarray):
        self.spectra = spectra
        self.targets = targets
        self.is_synth = is_synth
        self.idx = np.asarray(indices)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int) -> dict[str, Any]:
        j = self.idx[i]
        return {
            'spectrum': self.spectra[j],
            'targets': self.targets[j],
            'is_synthetic': self.is_synth[j],
        }


# =============================================================================
# Data Loading, Fingerprinted Splits & Chunked Memory-Safe Utilities
# =============================================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_dataset_fingerprint(data: dict) -> str:
    """Computes a fast cryptographic hash of targets, groups, and lockbox flags."""
    h = hashlib.blake2b(digest_size=16)
    h.update(data['y'].numpy().tobytes())
    h.update(np.ascontiguousarray(data['group_id']).tobytes())
    h.update(np.ascontiguousarray(data['is_lockbox']).tobytes())
    return h.hexdigest()


def load_crossval_data(h5_path: str) -> dict:
    """Loads pre-cleaned dataset as float32 tensors stored in System CPU RAM (matching benchmark scripts)."""
    with h5py.File(h5_path, 'r') as hf:
        spectra_raw = get_h5_ds(hf, 'X')[:].astype(np.float32)
        targets_raw = get_h5_ds(hf, 'y')[:].astype(np.float32)
        name_key = 'target_elements' if 'target_elements' in hf else 'feature_cols'

        if 'meta/is_synthetic' in hf:
            is_synth = hf['meta/is_synthetic'][:]  # type: ignore
        elif 'is_synthetic' in hf:
            is_synth = get_h5_ds(hf, 'is_synthetic')[:]
        else:
            is_synth = np.zeros(len(spectra_raw), dtype=np.int32)

        group_key = 'groups' if 'groups' in hf else 'group_id'
        group_id = get_h5_ds(hf, group_key)[:] if group_key in hf else np.arange(len(spectra_raw))
        wavelengths = get_h5_ds(hf, 'wavelengths')[:]
        names = [
            n.decode('utf-8') if isinstance(n, bytes) else str(n)
            for n in get_h5_ds(hf, name_key)[:]
        ]

        if 'is_lockbox' not in hf:
            raise KeyError(
                "HDF5 file missing 'is_lockbox'. Re-run data preprocessing to write "
                "a group-stratified lockbox split."
            )
        is_lockbox = get_h5_ds(hf, 'is_lockbox')[:].astype(bool)

    X_tensor = torch.from_numpy(spectra_raw)
    l2_norms = torch.norm(X_tensor, p=2, dim=1, keepdim=True).clamp(min=1e-08)
    X_tensor = X_tensor / l2_norms

    return dict(
        X=X_tensor,
        y=torch.from_numpy(targets_raw),
        is_synth=is_synth,
        is_lockbox=is_lockbox,
        group_id=group_id,
        wavelengths=wavelengths,
        names=names,
    )


def assign_group_folds(
    X, group_id, is_synthetic, dopant_idx, elem_names=None,
    n_splits=5, n_trials=2000, min_groups_per_dopant=1, seed=42
):
    """Stratified group fold assignment matching all benchmark scripts."""
    exp = np.where(is_synthetic == 0)[0]
    groups, inv = np.unique(group_id[exp], return_inverse=True)
    inv = inv.ravel()
    n_groups = len(groups)
    shots = np.bincount(inv, minlength=n_groups)

    first = np.zeros(n_groups, dtype=int)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    present = X[exp[first]][:, dopant_idx] > 0
    total_g_per_dopant = present.sum(axis=0)
    active_dopant_mask = total_g_per_dopant > 0
    present = present[:, active_dopant_mask]
    active_dopant_idx = [d for i, d in enumerate(dopant_idx) if active_dopant_mask[i]]
    sig = (present.astype(int) * (1 << np.arange(len(active_dopant_idx)))).sum(axis=1)
    classes = np.unique(sig)

    best_fog: np.ndarray | None = None
    best_score = None
    for t in range(n_trials):
        rng = np.random.default_rng(seed + t)
        fog = np.empty(n_groups, dtype=int)
        offset = int(rng.integers(n_splits))
        for s in rng.permutation(classes):
            members = rng.permutation(np.where(sig == s)[0])
            fog[members] = (offset + np.arange(len(members))) % n_splits
            offset += len(members)
        fold_shots = np.bincount(fog, weights=shots, minlength=n_splits)
        imbalance = np.abs(fold_shots / shots.sum() - 1.0 / n_splits).max()
        cov = np.array([present[fog == k].sum(axis=0) for k in range(n_splits)])
        score = (-min(int(cov.min()), min_groups_per_dopant), imbalance)
        if best_score is None or score < best_score:
            best_fog, best_score = fog, score

    if best_fog is None:
        raise ValueError("Failed to assign fold groups: n_trials must be > 0.")

    fold_id = np.full(len(X), -1, dtype=np.int32)
    fold_id[exp] = best_fog[inv]
    return fold_id


def build_splits_for_seed(data, split_dir, num_folds, seed, allow_cache: bool = True):
    """Loads existing pre-computed CV splits for a given seed or creates them if absent or stale."""
    split_dir = Path(split_dir)
    split_path = split_dir / f'max12_{num_folds}_folds_seed{seed}.npz'
    current_fp = compute_dataset_fingerprint(data)

    if allow_cache and split_path.exists():
        d = np.load(split_path)
        cached_fp = str(d['fingerprint']) if 'fingerprint' in d else None
        if cached_fp == current_fp and d.get('num_folds', None) == num_folds:
            return d['test_idx'], d['cv_idx'], d['cv_fold_id']
        else:
            print(f"⚠️ Stale or un-fingerprinted split cache detected in {split_path.name}. Recomputing folds...")

    test_idx = np.where(data['is_lockbox'])[0]
    cv_idx = np.where(~data['is_lockbox'])[0]

    y_cv = data['y'][cv_idx].numpy()
    is_synth_cv = data['is_synth'][cv_idx]

    dopant_idx = [
        i for i, name in enumerate(data['names'])
        if name not in HOST_ELEMENTS and (y_cv[:, i] > 0).any()
    ]

    cv_fold_id = assign_group_folds(
        X=y_cv, group_id=data['group_id'][cv_idx], is_synthetic=is_synth_cv,
        dopant_idx=dopant_idx, elem_names=data['names'], n_splits=num_folds, seed=seed
    )

    if allow_cache:
        split_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            split_path,
            test_idx=test_idx,
            cv_idx=cv_idx,
            cv_fold_id=cv_fold_id,
            num_folds=num_folds,
            seed=seed,
            fingerprint=current_fp,
        )
    return test_idx, cv_idx, cv_fold_id


def compute_subset_mean_spectrum(X: torch.Tensor, indices: np.ndarray, chunk_size: int = 4096) -> torch.Tensor:
    """Computes column-wise mean spectrum over X[indices] in chunks without allocating a full slice copy."""
    idx_t = torch.as_tensor(indices, dtype=torch.long)
    acc = torch.zeros(X.shape[1], dtype=torch.float64)
    for c in torch.split(idx_t, chunk_size):
        acc += X[c].sum(dim=0, dtype=torch.float64)
    return (acc / max(len(indices), 1)).float()


def compute_global_mean_and_sst(X: torch.Tensor, indices: np.ndarray, chunk_size: int = 4096) -> tuple[float, float, int]:
    """Computes global scalar mean, total sum of squares (SST), and element count in chunks."""
    idx_t = torch.as_tensor(indices, dtype=torch.long)
    n_elem = int(len(indices)) * int(X.shape[1])

    total_sum = 0.0
    for c in torch.split(idx_t, chunk_size):
        total_sum += float(X[c].sum(dtype=torch.float64).item())
    x_mean = total_sum / max(n_elem, 1)

    sst = 0.0
    for c in torch.split(idx_t, chunk_size):
        sst += float((X[c].double() - x_mean).pow_(2).sum().item())
    return x_mean, sst, n_elem


def compute_chunked_sse_arrays(y_true: torch.Tensor | np.ndarray, y_pred: torch.Tensor | np.ndarray, chunk_size: int = 4096) -> float:
    """Computes float64 SSE between two same-shaped arrays in chunks with minimal temporary memory."""
    n = len(y_true)
    sse = 0.0
    for s in range(0, n, chunk_size):
        t_c = torch.as_tensor(y_true[s:s + chunk_size], dtype=torch.float64)
        p_c = torch.as_tensor(y_pred[s:s + chunk_size], dtype=torch.float64)
        sse += float((t_c - p_c).pow_(2).sum().item())
    return sse


def fold_range_check(Y_wt, fold_id, is_synth, names, dopants=('Ce', 'Gd', 'Sm', 'U'), logger=None):
    Y_np = Y_wt.cpu().numpy() if hasattr(Y_wt, 'cpu') else np.asarray(Y_wt)
    is_synth_np = is_synth.cpu().numpy() if hasattr(is_synth, 'cpu') else np.asarray(is_synth)
    fold_id_np = fold_id.cpu().numpy() if hasattr(fold_id, 'cpu') else np.asarray(fold_id)

    exp = (is_synth_np == 0)
    for k in np.unique(fold_id_np[fold_id_np >= 0]):
        val = (fold_id_np == k)
        tr = exp & ~val

        row = []
        for e in dopants:
            if e in names:
                e_idx = names.index(e)
                val_max = Y_np[val, e_idx].max() if np.any(val) else 0.0
                train_max = Y_np[tr, e_idx].max() if np.any(tr) else 0.0
                row.append(f"{e}: val {val_max:.3f} / train {train_max:.3f}")

        msg = f"[Range Check] Fold {k}: " + " | ".join(row)
        if logger:
            log(logger=logger, msg=msg)
        else:
            print(msg)


def worst_groups(true_wt, pred_wt, gid, names, dopants=('Ce', 'Gd', 'Sm', 'U'), top=5, logger=None):
    true_np = true_wt.cpu().numpy() if hasattr(true_wt, 'cpu') else np.asarray(true_wt)
    pred_np = pred_wt.cpu().numpy() if hasattr(pred_wt, 'cpu') else np.asarray(pred_wt)
    gid_np = gid.cpu().numpy() if hasattr(gid, 'cpu') else np.asarray(gid)

    j = [names.index(e) for e in dopants if e in names]
    err = ((np.log1p(pred_np[:, j]) - np.log1p(true_np[:, j])) ** 2).mean(axis=1)

    ug, inv = np.unique(gid_np, return_inverse=True)
    inv = inv.ravel()
    cnt = np.bincount(inv)
    ge = np.bincount(inv, weights=err) / cnt

    msg_lines = [f"=== Top {top} Worst Performing Composition Groups (wt%) ==="]
    for o in np.argsort(-ge)[:top]:
        m = (inv == o)
        t_val = np.round(true_np[m][0, j], 3)
        p_val = np.round(pred_np[m][:, j].mean(axis=0), 3)
        msg_lines.append(
            f"Group {ug[o]:<4} | Shots: {cnt[o]:<4} | Log-MSE: {ge[o]:.4f} | "
            f"True: {t_val} | Pred Mean: {p_val}"
        )

    full_msg = "\n".join(msg_lines)
    if logger:
        log(logger=logger, msg=full_msg)
    else:
        print(full_msg)


# =============================================================================
# Plotting & Metric Tables (Matching Forward & Reverse Benchmarks)
# =============================================================================
def plot_parity_grid(y_true_wt, y_pred_wt, target_names, title_tag="P3VAE Encoder", save_path=None):
    """Generates an NxM concentration parity grid plot matching Forward PLS & 1D CNN."""
    active_elem_idx = [j for j in range(y_true_wt.shape[1]) if y_true_wt[:, j].max() > 1e-5]
    n_active = len(active_elem_idx)
    cols = 4
    rows = int(np.ceil(n_active / cols)) if n_active > 0 else 1

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.8, rows * 3.4))
    axes_flat = np.atleast_1d(axes).flatten()

    for idx_plot, j in enumerate(active_elem_idx):
        ax = axes_flat[idx_plot]
        name = target_names[j]
        yt = y_true_wt[:, j]
        yp = y_pred_wt[:, j]

        r2 = r2_score(yt, yp)
        mae = mean_absolute_error(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))

        ax.scatter(yt, yp, alpha=0.35, s=10, color="#1f77b4", edgecolors="none")
        max_v = max(np.max(yt), np.max(yp), 1e-3)
        min_v = min(np.min(yt), np.min(yp))
        ax.plot([min_v, max_v], [min_v, max_v], "k--", alpha=0.7, lw=1)
        ax.set_title(f"{name}\n$R^2$: {r2:.3f} | RMSE: {rmse:.3f} wt%\nMAE: {mae:.3f} wt%", fontsize=9, fontweight="bold")
        ax.set_xlabel("True wt%", fontsize=8)
        ax.set_ylabel("Pred wt%", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)

    for idx_plot in range(n_active, len(axes_flat)):
        fig.delaxes(axes_flat[idx_plot])

    plt.suptitle(f"{title_tag} \n Concentration Parity Grid", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved Parity Plot to: {save_path}")
    plt.close()


def print_benchmark_table(y_true, y_pred_model, y_train_mean, title_tag, elem_names, model_label="P3VAE Encoder"):
    """Prints element-wise metrics comparing P3VAE Encoder against Dummy Baselines."""
    active_idx = [j for j in range(y_true.shape[1]) if y_true[:, j].max() > 1e-5]

    print("\n" + "=" * 95)
    print(f" {title_tag}")
    print("=" * 95)
    print(f"{'Element':<8} | {'Strategy':<18} | {'R^2':<8} | {'MAE (wt%)':<10} | {'RMSE (wt%)':<10} | {'Max True':<10}")
    print("-" * 95)

    for j in active_idx:
        name = elem_names[j]
        yt = y_true[:, j]
        yp = y_pred_model[:, j]

        pred_zero = np.zeros_like(yt)
        pred_mean = np.full_like(yt, y_train_mean[j])

        strategies = [
            ("Predict Zero", pred_zero),
            ("Predict Mean", pred_mean),
            (model_label, yp),
        ]

        for strat_name, y_hat in strategies:
            r2 = r2_score(yt, y_hat)
            mae = mean_absolute_error(yt, y_hat)
            rmse = np.sqrt(mean_squared_error(yt, y_hat))
            print(f"{name:<8} | {strat_name:<18} | {r2:8.3f} | {mae:10.4f} | {rmse:10.4f} | {yt.max():10.3f}")
        print("-" * 95)


def rmse_calcs(y_true, y_pred):
    active_cols = y_true.max(axis=0) > 1e-5
    if not np.any(active_cols):
        return float('inf')
    return float(np.nanmean([
        np.sqrt(mean_squared_error(y_true[:, col], y_pred[:, col]))
        for col in np.where(active_cols)[0]
    ]))


def r2_calcs(y_true, y_pred):
    active_cols = y_true.max(axis=0) > 1e-5
    if not np.any(active_cols):
        return 0.0
    return float(np.nanmean([
        r2_score(y_true[:, col], y_pred[:, col])
        for col in np.where(active_cols)[0]
    ]))


def select_representative_samples(
    y_concentrations: np.ndarray,
    elem_names: list[str],
    sample_nums: list[int] | None = None,
) -> tuple[list[int], list[str]]:
    """Selects the exact same 6 representative samples used in Reverse PLS & Reverse MLP."""
    n_samples = len(y_concentrations)
    dop_cols = [j for j, n in enumerate(elem_names) if n in DOPANT_NAMES]
    if not dop_cols:
        dop_cols = list(range(y_concentrations.shape[1]))

    if sample_nums is not None and len(sample_nums) > 0:
        sample_indices = [int(idx) for idx in sample_nums if 0 <= idx < n_samples]
        sample_tags = [f"Sample {idx}" for idx in sample_indices]
        return sample_indices, sample_tags

    sample_indices: list[int] = []
    sample_tags: list[str] = []
    Yd = y_concentrations[:, dop_cols]

    # 1. One MAX sample per dopant (restricted to pure single-dopant samples)
    for j in dop_cols:
        if y_concentrations[:, j].max() <= 0:
            continue
        other_dop_cols = [c for c in dop_cols if c != j]
        if other_dop_cols:
            other_sums = y_concentrations[:, other_dop_cols].sum(axis=1)
            pure_mask = (other_sums <= 1e-5) & (y_concentrations[:, j] > 0)
        else:
            pure_mask = y_concentrations[:, j] > 0

        pure_indices = np.where(pure_mask)[0]
        if len(pure_indices) > 0:
            best_idx = int(pure_indices[np.argmax(y_concentrations[pure_indices, j])])
            sample_tags.append(f"Max Pure {elem_names[j]}")
        else:
            best_idx = int(np.argmax(y_concentrations[:, j]))
            sample_tags.append(f"Max {elem_names[j]} (Mixed)")
        sample_indices.append(best_idx)

    # 2. Complex mixture (highest normalized concentration entropy)
    row_sums = Yd.sum(axis=1, keepdims=True)
    conc_norm = np.divide(Yd, row_sums, out=np.zeros_like(Yd, dtype=float), where=row_sums > 0)
    log_conc = np.zeros_like(conc_norm)
    mask = conc_norm > 0
    log_conc[mask] = np.log(conc_norm[mask])
    entropy = -np.sum(conc_norm * log_conc, axis=1)
    sample_indices.append(int(np.argmax(entropy)))
    sample_tags.append("Complex Mixture")

    # 3. Trace concentration (lowest positive total dopant concentration)
    total_conc = Yd.sum(axis=1)
    pos = np.where(total_conc > 0)[0]
    trace_idx = int(pos[np.argmin(total_conc[pos])]) if len(pos) else int(np.argmin(total_conc))
    sample_indices.append(trace_idx)
    sample_tags.append("Trace Concentration")

    return sample_indices, sample_tags


def plot_spectral_reconstruction_subset(
    x_true_sub: np.ndarray,
    x_pred_sub: np.ndarray,
    wavelengths: np.ndarray,
    y_conc_sub: np.ndarray,
    orig_sample_indices: list[int],
    sample_tags: list[str],
    elem_names: list[str],
    save_path: Path | None = None,
    model_name: str = "P3VAE Full Decoder",
):
    """Plots True vs Predicted spectra for pre-selected representative rows."""
    n_plots = len(orig_sample_indices)
    if n_plots == 0:
        print("⚠️ No valid sample indices available to plot.")
        return

    dop_cols = [j for j, n in enumerate(elem_names) if n in DOPANT_NAMES] or list(range(y_conc_sub.shape[1]))

    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2.5 * n_plots), sharex=True)
    axes_flat = np.atleast_1d(axes)

    for plot_idx, (orig_idx, tag) in enumerate(zip(orig_sample_indices, sample_tags)):
        ax = axes_flat[plot_idx]
        xt = x_true_sub[plot_idx]
        xp = x_pred_sub[plot_idx]

        conc_str = ", ".join(
            f"{elem_names[j]}: {y_conc_sub[plot_idx, j]:.2f}%"
            for j in dop_cols if y_conc_sub[plot_idx, j] > 0
        ) or "0.00%"

        r2 = r2_score(xt, xp)
        rmse = np.sqrt(mean_squared_error(xt, xp))

        ax.plot(wavelengths, xt, label="True Spectrum", color="black", alpha=0.7, lw=1.0)
        ax.plot(wavelengths, xp, label=f"{model_name} Pred", color="crimson", alpha=0.8, lw=0.9, linestyle="--")
        ax.set_title(f"[{tag}] Sample {orig_idx} ({conc_str}) | $R^2$: {r2:.4f} | RMSE: {rmse:.5f}", fontsize=9, fontweight="bold")
        ax.set_ylabel("Norm Intensity", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)
        if plot_idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes_flat[-1].set_xlabel("Wavelength (nm)", fontsize=9)
    plt.suptitle(f"{model_name} Spectral Reconstruction", fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved Spectral Overlay Plot to: {save_path}")
    plt.close()


def plot_physics_residual_breakdown_subset(
    x_true_sub: np.ndarray,
    x_recon_sub: np.ndarray,
    x_phys_sub: np.ndarray,
    x_res_sub: np.ndarray,
    wavelengths: np.ndarray,
    y_conc_sub: np.ndarray,
    orig_sample_indices: list[int],
    sample_tags: list[str],
    elem_names: list[str],
    save_path: Path | None = None,
    title_tag: str = "P3VAE OOF",
):
    """Plots Physics vs Matrix Residual decomposition for the Complex Mixture sample."""
    if not orig_sample_indices:
        return

    # Pick Complex Mixture (index -2 in auto-selection) or first sample
    pick = -2 if ("Complex Mixture" in sample_tags and len(orig_sample_indices) >= 2) else 0
    orig_idx = orig_sample_indices[pick]
    tag = sample_tags[pick]

    dop_cols = [j for j, n in enumerate(elem_names) if n in DOPANT_NAMES] or list(range(y_conc_sub.shape[1]))
    conc_str = ", ".join(
        f"{elem_names[j]}: {y_conc_sub[pick, j]:.2f}%"
        for j in dop_cols if y_conc_sub[pick, j] > 0
    ) or "0.00%"

    r2_full = r2_score(x_true_sub[pick], x_recon_sub[pick])
    r2_phys = r2_score(x_true_sub[pick], x_phys_sub[pick])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)

    ax1.plot(wavelengths, x_true_sub[pick], label="Measured Spectrum", color="black", alpha=0.75, lw=1.0)
    ax1.plot(wavelengths, x_recon_sub[pick], label=f"Full P3VAE Recon ($R^2$: {r2_full:.4f})", color="crimson", alpha=0.85, lw=0.95, linestyle="--")
    ax1.set_ylabel("Norm Intensity", fontsize=9)
    ax1.set_title(f"{title_tag} [{tag}] Sample {orig_idx} ({conc_str}) — Full Reconstruction", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2.plot(wavelengths, x_phys_sub[pick], label=f"Physics Decoder (NIST Lines, $R^2$: {r2_phys:.4f})", color="forestgreen", lw=1.0)
    ax2.plot(wavelengths, x_res_sub[pick], label="Matrix Residual Decoder", color="darkorange", lw=1.0)
    ax2.set_xlabel("Wavelength (nm)", fontsize=9)
    ax2.set_ylabel("Norm Intensity", fontsize=9)
    ax2.set_title("P3VAE Decoded Physics vs. Matrix Residual Component Breakdown", fontsize=10, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved Physics vs. Residual Breakdown Plot to: {save_path}")
    plt.close()


# =============================================================================
# Single-Fold Training & Memory-Safe Streaming Evaluation
# =============================================================================
def train_one_fold(
    X_cv: torch.Tensor,
    row_maxes_cv: np.ndarray,
    Y_cv_norm: torch.Tensor,
    is_synth_cv: np.ndarray,
    cv_fold_id: np.ndarray,
    fold: int,
    include_synthetic: bool,
    wavelengths: np.ndarray,
    vae_elem_names: list[str],
    keep_idx: list[int],
    dopant_indices_in_vae: list[int],
    elem_scale_wt_gpu: torch.Tensor,
    master_table_path: str | Path,
    save_path: Path,
    logger,
    device_obj: torch.device,
    element_weights: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    pos_in_exp: np.ndarray,
    plot_exp_lookup: dict[int, int],
    plot_buffers_cv: dict[str, np.ndarray],
    lr_patience: int = LR_PATIENCE,
    stop_patience: int = STOP_PATIENCE,
    use_compile: bool = False,
    X_test: torch.Tensor | None = None,
    lbx_recon_sum: np.ndarray | None = None,
    lbx_phys_sum: np.ndarray | None = None,
    lbx_conc_sum: np.ndarray | None = None,
    plot_lbx_indices: list[int] | None = None,
    plot_lbx_res_sum: np.ndarray | None = None,
) -> dict[str, Any]:
    torch._dynamo.reset()

    if include_synthetic:
        train_mask = (cv_fold_id != fold)
    else:
        train_mask = (cv_fold_id != fold) & (is_synth_cv == 0)
    val_mask = (cv_fold_id == fold)

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    pin = device_obj.type == 'cuda'

    # Fit SpectralScaler from precomputed 1D row maxes (0 bytes of spectrum slice copies)
    spec_scaler = SpectralScaler(mode='robust_max', q=99.9)
    spec_scaler.fit_from_row_maxes(row_maxes_cv[train_idx])
    scale_gpu = spec_scaler.scale_.to(device_obj)  # type: ignore
    log(logger=logger, msg=f"[F{fold + 1}/{NUM_FOLDS}] spec_scaler scale_ = {scale_gpu.item():.5f}")

    train_ds = LIBSSpectraDataset(X_cv, Y_cv_norm, is_synth_cv, train_idx)
    val_ds = LIBSSpectraDataset(X_cv, Y_cv_norm, is_synth_cv, val_idx)

    drop_last_train = len(train_ds) >= batch_size
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=drop_last_train,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=(NUM_WORKERS > 0),
    )

    log(logger=logger, msg=f"=== Fold {fold + 1}/{NUM_FOLDS}: train {len(train_ds)} | val {len(val_ds)} ===")

    wl_grid = torch.from_numpy(wavelengths).to(device_obj)
    raw_model = PhysicsInformedVAE(
        input_dim=X_cv.shape[1],
        n_elements=Y_cv_norm.shape[1],
        wl_grid=wl_grid,
        master_table_path=master_table_path,
        elem_names=vae_elem_names,
        keep_idx=keep_idx,
        n_broadening=1,
        dropout=DROPOUT,
    ).to(device_obj)

    model = cast(nn.Module, torch.compile(raw_model, mode='default')) if use_compile else raw_model

    criterion = PhysicsVAELoss(
        element_weights=element_weights,
        alpha_recon=1.0,
        beta_sup=20.0,
        gamma_kl=1e-4,
        use_log_conc=False,
    ).to(device_obj)

    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=lr_patience, min_lr=MIN_LR
    )
    use_amp = device_obj.type == 'cuda'
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)  # type: ignore
    early_stopper = EarlyStopping(patience=stop_patience)

    dop_idx_t = torch.tensor(dopant_indices_in_vae, dtype=torch.long, device=device_obj)
    dop_scales_gpu = elem_scale_wt_gpu[dop_idx_t]

    best_dop_mse, best_epoch = float('inf'), 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss = 0.0
        n_tr_seen = 0

        for batch in tqdm(train_loader, desc=f"F{fold + 1} E{epoch:03d}", mininterval=5, leave=False):
            spectra = batch['spectrum'].to(device_obj, non_blocking=True) / scale_gpu
            targets = batch['targets'].to(device_obj, non_blocking=True)
            batch_gpu = {'spectrum': spectra, 'targets': targets}

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_obj.type, enabled=use_amp, dtype=torch.float16):  # type: ignore
                out = model(spectra)
                ld = criterion(out, batch_gpu)

            if torch.isnan(ld['loss']):
                continue

            amp_scaler.scale(ld['loss']).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()

            bs_cur = spectra.size(0)
            tr_loss += ld['loss'].item() * bs_cur
            n_tr_seen += bs_cur

        model.eval()
        va_loss, va_sup, va_dop_mse, va_recon = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                spectra = batch['spectrum'].to(device_obj, non_blocking=True) / scale_gpu
                targets = batch['targets'].to(device_obj, non_blocking=True)
                batch_gpu = {'spectrum': spectra, 'targets': targets}

                with torch.amp.autocast(device_obj.type, enabled=use_amp, dtype=torch.float16):  # type: ignore
                    out = model(spectra)
                    ld = criterion(out, batch_gpu)

                bs_cur = spectra.size(0)
                va_loss += ld['loss'].item() * bs_cur
                va_sup += ld['sup_loss'].item() * bs_cur
                va_recon += ld['recon_loss'].item() * bs_cur

                pred_dop = out['concentrations'].index_select(1, dop_idx_t).float()
                true_dop = targets.index_select(1, dop_idx_t).float()
                va_dop_mse += F.mse_loss(pred_dop, true_dop, reduction='mean').item() * bs_cur

        tr_loss /= max(n_tr_seen, 1)
        va_loss /= len(val_ds)
        va_sup /= len(val_ds)
        va_dop_mse /= len(val_ds)
        va_recon /= len(val_ds)
        va_dop_rmse_wt = math.sqrt(va_dop_mse) * float(dop_scales_gpu.mean().item())

        log(
            logger=logger,
            msg=f"[F{fold + 1}] Epoch {epoch:03d}/{epochs} [{time.time() - t0:.1f}s] | "
                f"Train {tr_loss:.4f} | Val {va_loss:.4f} | Dopant RMSE {va_dop_rmse_wt:.4f} wt% | "
                f"Recon {va_recon:.4f} | lr {optimizer.param_groups[0]['lr']:.2e}",
        )

        # Schedule LR and checkpoint on Dopant MSE to align with reported OOF dopant metric
        scheduler.step(va_dop_mse)
        if va_dop_mse < best_dop_mse:
            best_dop_mse, best_epoch = va_dop_mse, epoch
            torch.save({
                'fold': fold,
                'epoch': epoch,
                'model_state_dict': raw_model.state_dict(),
                'scaler_state_dict': spec_scaler.state_dict(),
                'va_dop_mse': best_dop_mse,
                'va_sup': va_sup,
                'va_recon': va_recon,
            }, save_path)

        if early_stopper(va_dop_mse):
            log(logger=logger, msg=f"[F{fold + 1}] Early stop at epoch {epoch} (Best Dopant RMSE: {math.sqrt(best_dop_mse) * MAX_WT_PERCENT:.4f} wt%)")
            break

    # ---- Out-of-Fold Evaluation from BEST Checkpoint (Streaming SSE + Plot Rows) ----
    ckpt = torch.load(save_path, map_location=device_obj)
    raw_model.load_state_dict(ckpt['model_state_dict'])
    raw_model.eval()

    # Chunked training-fold mean spectrum for baseline SSE (0 large slice copies)
    mu_tr_gpu = compute_subset_mean_spectrum(X_cv, train_idx).to(device_obj)
    scale_val = float(spec_scaler.scale_.item())  # type: ignore

    sse_base = torch.zeros((), dtype=torch.float64, device=device_obj)
    sse_full = torch.zeros((), dtype=torch.float64, device=device_obj)
    sse_phys = torch.zeros((), dtype=torch.float64, device=device_obj)
    sse_trueconc_phys = torch.zeros((), dtype=torch.float64, device=device_obj)

    oof_conc_list = []
    offset = 0

    with torch.no_grad():
        for batch in val_loader:
            x_raw_gpu = batch['spectrum'].to(device_obj, non_blocking=True)
            targets_gpu = batch['targets'].to(device_obj, non_blocking=True)
            bs_cur = x_raw_gpu.size(0)

            out = raw_model(x_raw_gpu / scale_gpu)
            trueconc_phys_scaled = raw_model.physics_decoder(targets_gpu, out['broadening'])

            # Invert SpectralScaler and clip at 0.0 to match Reverse PLS & Reverse MLP
            recon_unscaled = torch.clamp(out['reconstructed_spectrum'].float() * scale_val, min=0.0)
            phys_unscaled = torch.clamp(out['ideal_physics_spectrum'].float() * scale_val, min=0.0)
            res_unscaled = out['matrix_residual'].float() * scale_val
            trueconc_phys_unscaled = torch.clamp(trueconc_phys_scaled.float() * scale_val, min=0.0)

            x_raw_f64 = x_raw_gpu.double()
            sse_base += (x_raw_f64 - mu_tr_gpu.double()).pow_(2).sum()
            sse_full += (x_raw_f64 - recon_unscaled.double()).pow_(2).sum()
            sse_phys += (x_raw_f64 - phys_unscaled.double()).pow_(2).sum()
            sse_trueconc_phys += (x_raw_f64 - trueconc_phys_unscaled.double()).pow_(2).sum()

            oof_conc_list.append(out['concentrations'].float().cpu().numpy())

            # Copy only the 6 pre-selected representative rows into small plotting buffers
            batch_val_positions = val_idx[offset:offset + bs_cur]
            batch_exp_positions = pos_in_exp[batch_val_positions]
            for local_i, exp_pos in enumerate(batch_exp_positions):
                plot_slot = plot_exp_lookup.get(int(exp_pos))
                if plot_slot is not None:
                    plot_buffers_cv['recon'][plot_slot] = recon_unscaled[local_i].cpu().numpy()
                    plot_buffers_cv['phys'][plot_slot] = phys_unscaled[local_i].cpu().numpy()
                    plot_buffers_cv['res'][plot_slot] = res_unscaled[local_i].cpu().numpy()
                    plot_buffers_cv['trueconc_phys'][plot_slot] = trueconc_phys_unscaled[local_i].cpu().numpy()

            offset += bs_cur

    res_dict: dict[str, Any] = dict(
        fold=fold,
        val_idx=val_idx,
        pred_conc=np.concatenate(oof_conc_list, axis=0),
        sse_base=float(sse_base.item()),
        sse_full=float(sse_full.item()),
        sse_phys=float(sse_phys.item()),
        sse_trueconc_phys=float(sse_trueconc_phys.item()),
        best_epoch=best_epoch,
        best_dop_mse=best_dop_mse,
    )

    # Optional Lockbox Inference (accumulated in-place into pre-allocated buffers)
    if X_test is not None and lbx_conc_sum is not None and lbx_recon_sum is not None and lbx_phys_sum is not None:
        lbx_conc_list = []
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                bx = X_test[i:i + batch_size].to(device_obj) / scale_gpu
                out = raw_model(bx)
                lbx_conc_list.append(out['concentrations'].float().cpu().numpy())
                lbx_recon_sum[i:i + batch_size] += (out['reconstructed_spectrum'].float() * scale_val).cpu().numpy()
                lbx_phys_sum[i:i + batch_size] += (out['ideal_physics_spectrum'].float() * scale_val).cpu().numpy()

            if plot_lbx_indices and plot_lbx_res_sum is not None:
                bx_plot = X_test[plot_lbx_indices].to(device_obj) / scale_gpu
                out_plot = raw_model(bx_plot)
                plot_lbx_res_sum += (out_plot['matrix_residual'].float() * scale_val).cpu().numpy()

        lbx_conc_sum += np.concatenate(lbx_conc_list, axis=0)

    del model, raw_model, optimizer, amp_scaler, train_loader, val_loader, mu_tr_gpu
    gc.collect()
    if device_obj.type == 'cuda':
        torch.cuda.empty_cache()

    return res_dict


# =============================================================================
# Main Execution Pipeline
# =============================================================================
def main(
    h5_file: Path | str = H5_FILE,
    split_dir: Path | str = SPLIT_DIR,
    output_dir: Path | str = OUTPUT_DIR,
    master_table_path: Path | str = MASTER_TABLE_PATH,
    include_synthetic: bool = False,
    seeds: list[int] = SEEDS,
    dataset_tag: str = '20261110b',
    target_names: list[str] | None = None,
    include_lbx: bool = True,
    sample_nums: list[int] | None = None,
    epochs: int = MAX_EPOCHS,
    folds: list[int] | None = None,
    max_samples: int | None = None,
    use_compile: bool = False,
):
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "exp-syn" if include_synthetic else "exp"
    print(f"=== Starting Memory-Safe P3VAE Encoder + Decoder Benchmark Pipeline [{DEVICE}] ===")
    print(f"Dataset: {dataset_tag} ({mode_tag}) | Folds: {NUM_FOLDS} | Seeds: {seeds}")
    t0_start = time.perf_counter()

    h5_file = str(h5_file)
    split_dir = Path(split_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    master_table_path = Path(master_table_path).resolve()

    log_path = LOG_DIR / f"p3vae700_{dataset_tag}_{mode_tag}_{TIMESTAMP}.txt"
    logger = get_worker_logger(log_path.stem)

    # 1. Load Data (Optional subsample for fast smoke testing without overwriting split cache)
    data = load_crossval_data(h5_file)
    if max_samples is not None and max_samples < len(data['X']):
        rng_sub = np.random.default_rng(seeds[0])
        sub_idx = np.sort(rng_sub.choice(len(data['X']), size=max_samples, replace=False))
        for k_arr in ('X', 'y', 'is_synth', 'is_lockbox', 'group_id'):
            data[k_arr] = data[k_arr][sub_idx]
        print(f"⚠️ Smoke-test subsample active: {len(data['X'])} rows (split cache writes disabled).")

    n_synth = int((data['is_synth'] == 1).sum())
    print(f"Loaded {len(data['X'])} total shots | Real: {len(data['X']) - n_synth} | Synthetic: {n_synth}")

    # 2. Element Mapping & Per-Element Concentration Scaling Check
    keep_elements = HOST_ELEMENTS.union(EXP_DOPANTS)
    keep_idx = [i for i, name in enumerate(data['names']) if name in keep_elements]
    vae_elem_names = [data['names'][i] for i in keep_idx]

    if target_names is None:
        target_names = TARGET_NAMES
    target_indices_in_vae = [vae_elem_names.index(name) for name in target_names if name in vae_elem_names]
    actual_target_names = [vae_elem_names[i] for i in target_indices_in_vae]

    raw_max_wt = data['y'][:, keep_idx].max(dim=0).values.numpy()
    elem_scale_wt = np.full(len(vae_elem_names), MAX_WT_PERCENT, dtype=np.float32)
    for idx_e, (el_name, max_val) in enumerate(zip(vae_elem_names, raw_max_wt)):
        if el_name in HOST_ELEMENTS and max_val > MAX_WT_PERCENT:
            elem_scale_wt[idx_e] = float(np.ceil(max_val))
            print(f"ℹ️ Host '{el_name}' max is {max_val:.2f} wt% (> {MAX_WT_PERCENT}); scaling by {elem_scale_wt[idx_e]:.1f} wt% to prevent [0,1] clamping.")

    elem_scale_wt_t = torch.from_numpy(elem_scale_wt)
    elem_scale_wt_gpu = elem_scale_wt_t.to(DEVICE)

    element_weights = torch.tensor(
        [ELEMENT_WEIGHTS_REFERENCE.get(n.encode() if isinstance(n, str) else n, 5.0) for n in vae_elem_names],
        dtype=torch.float32,
    )

    # 3. Prepare Splits Across Requested Seeds (Primary seed = seeds[0])
    seed_splits = {}
    for s in seeds:
        seed_splits[s] = build_splits_for_seed(
            data, split_dir, NUM_FOLDS, seed=s, allow_cache=(max_samples is None)
        )

    primary_seed = seeds[0]
    seed_everything(seed=primary_seed)
    test_idx, cv_idx, cv_fold_id = seed_splits[primary_seed]

    X_cv = data['X'][cv_idx]
    row_maxes_cv = X_cv.max(dim=1).values.numpy()
    Y_cv_norm = data['y'][cv_idx][:, keep_idx] / elem_scale_wt_t
    is_synth_cv = data['is_synth'][cv_idx]
    group_id_cv = data['group_id'][cv_idx]
    wavelengths = data['wavelengths']

    X_test = data['X'][test_idx] if include_lbx else None
    Y_test_wt = data['y'][test_idx][:, keep_idx][:, target_indices_in_vae].numpy() if include_lbx else None

    del data['X']
    gc.collect()

    fold_range_check(
        Y_wt=Y_cv_norm * elem_scale_wt_t,
        fold_id=cv_fold_id,
        is_synth=is_synth_cv,
        names=vae_elem_names,
        dopants=tuple(actual_target_names),
        logger=logger,
    )

    # 4. Pre-Select Representative Sample Rows Up Front (Memory-Safe Plotting)
    exp_rows = np.where(cv_fold_id >= 0)[0]
    n_exp = len(exp_rows)
    pos_in_exp = np.full(len(cv_idx), -1, dtype=np.int64)
    pos_in_exp[exp_rows] = np.arange(n_exp)

    true_cv_wt = (Y_cv_norm[exp_rows][:, target_indices_in_vae] * elem_scale_wt_t[target_indices_in_vae]).numpy()
    plot_exp_indices, plot_exp_tags = select_representative_samples(
        true_cv_wt, actual_target_names, sample_nums=sample_nums
    )
    plot_exp_lookup = {int(exp_idx): slot for slot, exp_idx in enumerate(plot_exp_indices)}

    n_plot_cv = len(plot_exp_indices)
    n_wl = X_cv.shape[1]
    n_vae_elem = len(keep_idx)

    plot_buffers_cv = {
        'recon': np.zeros((n_plot_cv, n_wl), dtype=np.float32),
        'phys': np.zeros((n_plot_cv, n_wl), dtype=np.float32),
        'res': np.zeros((n_plot_cv, n_wl), dtype=np.float32),
        'trueconc_phys': np.zeros((n_plot_cv, n_wl), dtype=np.float32),
    }

    # Global mean & SST for experimental OOF spectra (chunked, ~0 extra RAM)
    x_mean_exp, sst_exp, n_elem_exp = compute_global_mean_and_sst(X_cv, exp_rows)

    # Tiny concentration OOF buffer (n_exp x 7 float32 ~ 6.5 MB)
    oof_conc_exp = np.zeros((n_exp, n_vae_elem), dtype=np.float32)

    # Lockbox accumulators (only allocated when include_lbx=True)
    plot_lbx_indices: list[int] = []
    plot_lbx_tags: list[str] = []
    lbx_conc_sum = np.zeros((len(test_idx), n_vae_elem), dtype=np.float64) if include_lbx else None
    lbx_recon_sum = np.zeros((len(test_idx), n_wl), dtype=np.float32) if include_lbx else None
    lbx_phys_sum = np.zeros((len(test_idx), n_wl), dtype=np.float32) if include_lbx else None
    plot_lbx_res_sum = None
    if include_lbx and Y_test_wt is not None:
        plot_lbx_indices, plot_lbx_tags = select_representative_samples(
            Y_test_wt, actual_target_names, sample_nums=sample_nums
        )
        plot_lbx_res_sum = np.zeros((len(plot_lbx_indices), n_wl), dtype=np.float32)

    # 5. K-Fold Training & Streaming Evaluation
    run_folds = folds if folds is not None else list(range(NUM_FOLDS))
    tot_sse_base = 0.0
    tot_sse_full = 0.0
    tot_sse_phys = 0.0
    tot_sse_trueconc_phys = 0.0

    print(f"\n--- Training P3VAE across Folds {run_folds} [{'Syn+Exp' if include_synthetic else 'Exp Only'}] ---")
    for k in run_folds:
        fold_ckpt = TRAINED_MODEL_DIR / f"p3vae700_{dataset_tag}_{mode_tag}_{TIMESTAMP}_fold{k + 1}.pt"
        res = train_one_fold(
            X_cv=X_cv,
            row_maxes_cv=row_maxes_cv,
            Y_cv_norm=Y_cv_norm,
            is_synth_cv=is_synth_cv,
            cv_fold_id=cv_fold_id,
            fold=k,
            include_synthetic=include_synthetic,
            wavelengths=wavelengths,
            vae_elem_names=vae_elem_names,
            keep_idx=keep_idx,
            dopant_indices_in_vae=target_indices_in_vae,
            elem_scale_wt_gpu=elem_scale_wt_gpu,
            master_table_path=master_table_path,
            save_path=fold_ckpt,
            logger=logger,
            device_obj=DEVICE,
            element_weights=element_weights,
            epochs=epochs,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            pos_in_exp=pos_in_exp,
            plot_exp_lookup=plot_exp_lookup,
            plot_buffers_cv=plot_buffers_cv,
            lr_patience=LR_PATIENCE,
            stop_patience=STOP_PATIENCE,
            use_compile=use_compile,
            X_test=X_test,
            lbx_recon_sum=lbx_recon_sum,
            lbx_phys_sum=lbx_phys_sum,
            lbx_conc_sum=lbx_conc_sum,
            plot_lbx_indices=plot_lbx_indices,
            plot_lbx_res_sum=plot_lbx_res_sum,
        )

        dst = pos_in_exp[res['val_idx']]
        oof_conc_exp[dst] = res['pred_conc']
        tot_sse_base += res['sse_base']
        tot_sse_full += res['sse_full']
        tot_sse_phys += res['sse_phys']
        tot_sse_trueconc_phys += res['sse_trueconc_phys']

    # Subset mask if running partial folds (e.g., smoke test with folds=[0])
    eval_exp_mask = np.isin(cv_fold_id[exp_rows], run_folds)
    eval_n_elem = int(eval_exp_mask.sum()) * n_wl
    eval_sst = sst_exp if len(run_folds) == NUM_FOLDS else compute_global_mean_and_sst(X_cv, exp_rows[eval_exp_mask])[1]

    # =========================================================================
    # 6A. ENCODER EVALUATIONS (Spectra -> Concentrations, matching PLS & 1D CNN)
    # =========================================================================
    true_eval_wt = true_cv_wt[eval_exp_mask]
    pred_cv_wt = np.clip(
        oof_conc_exp[:, target_indices_in_vae] * elem_scale_wt[target_indices_in_vae], 0.0, None
    )
    pred_eval_wt = pred_cv_wt[eval_exp_mask]
    train_mean_cv = true_eval_wt.mean(axis=0)

    enc_oof_rmse = rmse_calcs(true_eval_wt, pred_eval_wt)
    enc_oof_r2 = r2_calcs(true_eval_wt, pred_eval_wt)
    print(f"\n✅ P3VAE Encoder OOF Summary | Mean Dopant RMSE: {enc_oof_rmse:.4f} wt% | Mean R^2: {enc_oof_r2:.4f}")

    print_benchmark_table(
        y_true=true_eval_wt,
        y_pred_model=pred_eval_wt,
        y_train_mean=train_mean_cv,
        title_tag=f"P3VAE Encoder OOF Benchmarks [{'Syn+Exp' if include_synthetic else 'Exp Only'}]",
        elem_names=actual_target_names,
        model_label="P3VAE Encoder",
    )

    worst_groups(
        true_wt=true_eval_wt,
        pred_wt=pred_eval_wt,
        gid=group_id_cv[exp_rows][eval_exp_mask],
        names=actual_target_names,
        dopants=tuple(actual_target_names),
        top=5,
        logger=logger,
    )

    oof_parity_path = output_dir / f"p3vae_oof_parity_{dataset_tag}_{TIMESTAMP}.png"
    plot_parity_grid(
        true_eval_wt,
        pred_eval_wt,
        actual_target_names,
        title_tag=f"P3VAE Encoder OOF ({dataset_tag}, {mode_tag})",
        save_path=oof_parity_path,
    )

    pred_test_wt = None
    if include_lbx and lbx_conc_sum is not None and Y_test_wt is not None:
        n_active_folds = len(run_folds)
        pred_test_wt = np.clip(
            (lbx_conc_sum / n_active_folds)[:, target_indices_in_vae] * elem_scale_wt[target_indices_in_vae],
            0.0,
            None,
        )
        print_benchmark_table(
            y_true=Y_test_wt,
            y_pred_model=pred_test_wt,
            y_train_mean=train_mean_cv,
            title_tag=f"P3VAE Encoder Lockbox Benchmarks (Ensemble of {n_active_folds} Folds)",
            elem_names=actual_target_names,
            model_label="P3VAE Encoder",
        )
        lbx_parity_path = output_dir / f"p3vae_lbx_parity_{dataset_tag}_{TIMESTAMP}.png"
        plot_parity_grid(
            Y_test_wt,
            pred_test_wt,
            actual_target_names,
            title_tag=f"P3VAE Encoder LOCKBOX ({dataset_tag}, {mode_tag})",
            save_path=lbx_parity_path,
        )

    # =========================================================================
    # 6B. DECODER EVALUATIONS (Streaming Metrics + Representative Plot Overlays)
    # =========================================================================
    base_rmse = math.sqrt(tot_sse_base / max(eval_n_elem, 1))
    base_r2 = 1.0 - (tot_sse_base / max(eval_sst, 1e-12))

    full_oof_rmse = math.sqrt(tot_sse_full / max(eval_n_elem, 1))
    full_oof_r2 = 1.0 - (tot_sse_full / max(eval_sst, 1e-12))

    phys_oof_rmse = math.sqrt(tot_sse_phys / max(eval_n_elem, 1))
    phys_oof_r2 = 1.0 - (tot_sse_phys / max(eval_sst, 1e-12))

    trueconc_phys_rmse = math.sqrt(tot_sse_trueconc_phys / max(eval_n_elem, 1))
    trueconc_phys_r2 = 1.0 - (tot_sse_trueconc_phys / max(eval_sst, 1e-12))

    print("\n" + "=" * 105)
    print(f" P3VAE Decoder OOF Spectral Reconstruction Benchmarks [{'Syn+Exp' if include_synthetic else 'Exp Only'}]")
    print(" Note: Full Decoder uses spectrum-encoded matrix residual (autoencoder upper bound);")
    print("       True-Conc Physics uses 7 modeled elements + spectrum-inferred broadening.")
    print("=" * 105)
    print(f"{'Decoder Pathway':<52} | {'Spectral RMSE':<15} | {'Overall R^2':<12}")
    print("-" * 105)
    print(f"{'Mean-Spectrum Baseline':<52} | {base_rmse:15.5f} | {base_r2:12.4f}")
    print(f"{'Physics-Only Decoder (Pred Conc -> NIST)':<52} | {phys_oof_rmse:15.5f} | {phys_oof_r2:12.4f}")
    print(f"{'Physics Decoder (True 7-Elem Conc + Pred Gamma)':<52} | {trueconc_phys_rmse:15.5f} | {trueconc_phys_r2:12.4f}")
    print(f"{'Full P3VAE Decoder (Physics + Matrix Residual)':<52} | {full_oof_rmse:15.5f} | {full_oof_r2:12.4f}")
    print("=" * 105)

    # Extract only the 6 ground-truth rows from X_cv for plotting & compact cache
    x_true_plot_cv = X_cv[exp_rows[plot_exp_indices]].numpy()
    y_conc_plot_cv = true_cv_wt[plot_exp_indices]

    plot_spectral_reconstruction_subset(
        x_true_sub=x_true_plot_cv,
        x_pred_sub=plot_buffers_cv['recon'],
        wavelengths=wavelengths,
        y_conc_sub=y_conc_plot_cv,
        orig_sample_indices=plot_exp_indices,
        sample_tags=plot_exp_tags,
        elem_names=actual_target_names,
        save_path=output_dir / f"p3vae_oof_spectra_full_{dataset_tag}_{TIMESTAMP}.png",
        model_name="P3VAE Full Decoder",
    )

    plot_spectral_reconstruction_subset(
        x_true_sub=x_true_plot_cv,
        x_pred_sub=plot_buffers_cv['phys'],
        wavelengths=wavelengths,
        y_conc_sub=y_conc_plot_cv,
        orig_sample_indices=plot_exp_indices,
        sample_tags=plot_exp_tags,
        elem_names=actual_target_names,
        save_path=output_dir / f"p3vae_oof_spectra_physics_{dataset_tag}_{TIMESTAMP}.png",
        model_name="P3VAE Physics Decoder",
    )

    plot_physics_residual_breakdown_subset(
        x_true_sub=x_true_plot_cv,
        x_recon_sub=plot_buffers_cv['recon'],
        x_phys_sub=plot_buffers_cv['phys'],
        x_res_sub=plot_buffers_cv['res'],
        wavelengths=wavelengths,
        y_conc_sub=y_conc_plot_cv,
        orig_sample_indices=plot_exp_indices,
        sample_tags=plot_exp_tags,
        elem_names=actual_target_names,
        save_path=output_dir / f"p3vae_oof_physics_vs_residual_{dataset_tag}_{TIMESTAMP}.png",
        title_tag=f"P3VAE OOF ({mode_tag})",
    )

    lbx_full_rmse, lbx_full_r2, lbx_phys_rmse, lbx_phys_r2 = None, None, None, None
    x_true_plot_lbx, x_pred_full_plot_lbx, x_pred_phys_plot_lbx, x_pred_res_plot_lbx = None, None, None, None
    if include_lbx and X_test is not None and lbx_recon_sum is not None and lbx_phys_sum is not None and Y_test_wt is not None:
        n_active_folds = len(run_folds)
        lbx_recon_sum /= n_active_folds
        np.clip(lbx_recon_sum, 0.0, None, out=lbx_recon_sum)

        lbx_phys_sum /= n_active_folds
        np.clip(lbx_phys_sum, 0.0, None, out=lbx_phys_sum)

        _, lbx_sst, lbx_n_elem = compute_global_mean_and_sst(X_test, np.arange(len(X_test)))
        lbx_sse_full = compute_chunked_sse_arrays(X_test, lbx_recon_sum)
        lbx_sse_phys = compute_chunked_sse_arrays(X_test, lbx_phys_sum)

        lbx_full_rmse = math.sqrt(lbx_sse_full / max(lbx_n_elem, 1))
        lbx_full_r2 = 1.0 - (lbx_sse_full / max(lbx_sst, 1e-12))
        lbx_phys_rmse = math.sqrt(lbx_sse_phys / max(lbx_n_elem, 1))
        lbx_phys_r2 = 1.0 - (lbx_sse_phys / max(lbx_sst, 1e-12))

        print(f"\n✅ Lockbox Full Decoder    | Spectral RMSE: {lbx_full_rmse:.5f} | Overall R^2: {lbx_full_r2:.4f}")
        print(f"✅ Lockbox Physics Decoder | Spectral RMSE: {lbx_phys_rmse:.5f} | Overall R^2: {lbx_phys_r2:.4f}")

        x_true_plot_lbx = X_test[plot_lbx_indices].numpy()
        x_pred_full_plot_lbx = lbx_recon_sum[plot_lbx_indices].copy()
        x_pred_phys_plot_lbx = lbx_phys_sum[plot_lbx_indices].copy()
        x_pred_res_plot_lbx = (plot_lbx_res_sum / n_active_folds) if plot_lbx_res_sum is not None else np.zeros_like(x_true_plot_lbx)
        y_conc_plot_lbx = Y_test_wt[plot_lbx_indices]

        # Immediately reclaim lockbox full spectral arrays
        del lbx_recon_sum, lbx_phys_sum
        gc.collect()

        plot_spectral_reconstruction_subset(
            x_true_sub=x_true_plot_lbx,
            x_pred_sub=x_pred_full_plot_lbx,
            wavelengths=wavelengths,
            y_conc_sub=y_conc_plot_lbx,
            orig_sample_indices=plot_lbx_indices,
            sample_tags=plot_lbx_tags,
            elem_names=actual_target_names,
            save_path=output_dir / f"p3vae_lbx_spectra_full_{dataset_tag}_{TIMESTAMP}.png",
            model_name="P3VAE Full Decoder",
        )
        plot_spectral_reconstruction_subset(
            x_true_sub=x_true_plot_lbx,
            x_pred_sub=x_pred_phys_plot_lbx,
            wavelengths=wavelengths,
            y_conc_sub=y_conc_plot_lbx,
            orig_sample_indices=plot_lbx_indices,
            sample_tags=plot_lbx_tags,
            elem_names=actual_target_names,
            save_path=output_dir / f"p3vae_lbx_spectra_physics_{dataset_tag}_{TIMESTAMP}.png",
            model_name="P3VAE Physics Decoder",
        )

    # =========================================================================
    # 7. Compact Unified Cache (<15 MB .npz — No Full 9 GB Spectral Arrays)
    # =========================================================================
    cache_path = output_dir / f"p3vae_predictions_{dataset_tag}.npz"
    save_dict: dict[str, Any] = dict(
        # Forward Encoder predictions (matches pls_predictions & cnn1d_predictions)
        y_true_cv=true_eval_wt,
        y_pred_cv=pred_eval_wt,
        target_names=np.array(actual_target_names),
        # Full 7-element physical predictions (Li, K, Cl, Ce, Gd, Sm, U)
        y_true_all_cv=(Y_cv_norm[exp_rows][eval_exp_mask] * elem_scale_wt_t).numpy(),
        y_pred_all_cv=np.clip(oof_conc_exp[eval_exp_mask] * elem_scale_wt, 0.0, None),
        vae_elem_names=np.array(vae_elem_names),
        # Decoder summary metrics (exact float64 across all evaluated OOF rows)
        dec_base_rmse=np.float64(base_rmse),
        dec_base_r2=np.float64(base_r2),
        dec_full_oof_rmse=np.float64(full_oof_rmse),
        dec_full_oof_r2=np.float64(full_oof_r2),
        dec_phys_oof_rmse=np.float64(phys_oof_rmse),
        dec_phys_oof_r2=np.float64(phys_oof_r2),
        dec_trueconc_phys_rmse=np.float64(trueconc_phys_rmse),
        dec_trueconc_phys_r2=np.float64(trueconc_phys_r2),
        # Representative sample spectra for downstream overlay comparisons
        plot_sample_indices_cv=np.array(plot_exp_indices, dtype=np.int64),
        plot_sample_tags_cv=np.array(plot_exp_tags),
        x_true_plot_cv=x_true_plot_cv,
        x_pred_plot_cv=plot_buffers_cv['recon'],
        x_pred_phys_plot_cv=plot_buffers_cv['phys'],
        x_pred_res_plot_cv=plot_buffers_cv['res'],
        x_pred_trueconc_phys_plot_cv=plot_buffers_cv['trueconc_phys'],
        y_conc_plot_cv=y_conc_plot_cv,
        wavelengths=wavelengths,
    )

    if include_lbx and Y_test_wt is not None and pred_test_wt is not None:
        save_dict["y_true_test"] = Y_test_wt
        save_dict["y_pred_test"] = pred_test_wt
        save_dict["dec_full_test_rmse"] = np.float64(lbx_full_rmse)
        save_dict["dec_full_test_r2"] = np.float64(lbx_full_r2)
        save_dict["dec_phys_test_rmse"] = np.float64(lbx_phys_rmse)
        save_dict["dec_phys_test_r2"] = np.float64(lbx_phys_r2)
        save_dict["plot_sample_indices_test"] = np.array(plot_lbx_indices, dtype=np.int64)
        save_dict["plot_sample_tags_test"] = np.array(plot_lbx_tags)
        save_dict["x_true_plot_test"] = x_true_plot_lbx
        save_dict["x_pred_plot_test"] = x_pred_full_plot_lbx
        save_dict["x_pred_phys_plot_test"] = x_pred_phys_plot_lbx
        save_dict["x_pred_res_plot_test"] = x_pred_res_plot_lbx
        save_dict["y_conc_plot_test"] = Y_test_wt[plot_lbx_indices]

    np.savez(cache_path, **save_dict)
    print(f"Cached compact P3VAE encoder + decoder results to: {cache_path}")
    print(f"P3VAE Pipeline completed in {(time.perf_counter() - t0_start) / 60:.2f} minutes.")

    del X_cv, X_test
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(
        h5_file="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/exp-syn_processed-data_20261110b.h5",
        split_dir="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/fold_splits/processed-data_20261110b",
        output_dir=Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/plots/P3VAE/processed-data_20261110b/exp"),
        master_table_path="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/master_line_table.csv",
        include_synthetic=False,
        seeds=SEEDS,
        dataset_tag='20261110b',
        target_names=['Ce', 'Gd', 'Sm', 'U'],
        include_lbx=False,
        sample_nums=None,
    )

    main(
        h5_file="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/exp-syn_processed-data_20261110b.h5",
        split_dir="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/fold_splits/processed-data_20261110b",
        output_dir=Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/plots/P3VAE/processed-data_20261110b/exp-syn"),
        master_table_path="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/master_line_table.csv",
        include_synthetic=True,
        seeds=SEEDS,
        dataset_tag='20261110b',
        target_names=['Ce', 'Gd', 'Sm', 'U'],
        include_lbx=False,
        sample_nums=None,
    )