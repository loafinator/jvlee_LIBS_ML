from __future__ import annotations

"""
jvlee_LIBS_ML > LIBS > p3VAE > p3vae_005_sfthresh_NonZeroStratifiedEvalMetrics.py
"""

# region Imports
import h5py
import sys
import time
import torch
import math
import gc
import random
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.cuda.amp as amp

from datetime import datetime
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error
from typing import Any, Dict, cast
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
    get_h5_ds,
    check_h5_compositions,
    evaluate_and_plot_kfold
)
# endregion

# region Global Variables
BATCH_SIZE = 128         # Prevents GPU OOM on local/cluster GPUs
CHUNK_SIZE = 128         # Keeps broadcast tensor overhead small
MAX_EPOCHS = 200         # Execution loop max epochs
MIN_LR = 1e-06

cpus_per_task = 8
NUM_WORKERS = cpus_per_task - 2

learning_rate = 1e-03
weight_decay = 1e-04
NUM_FOLDS = 5
DROPOUT = 0.15
THRESH = 2.8 

ELEMENT_WEIGHTS_REFERENCE = {
    b'Ba': 5.0, b'Ca': 5.0, b'Ce': 5.0, b'Cr': 5.0, b'Cs': 5.0,
    b'Fe': 5.0, b'Gd': 5.0, b'K':  0.2, b'La': 5.0, b'Li': 0.2,
    b'Mg': 5.0, b'Mn': 5.0, b'Nd': 5.0, b'Ni': 5.0, b'Sm': 5.0, 
    b'Sr': 5.0, b'U':  5.0, b'Y':  5.0, b'Cl': 0.2,
}

ELEMENT_WEIGHTS = torch.tensor([
    5.0, 5.0, 5.0, 5.0, 5.0,
    5.0, 5.0, 0.2, 5.0, 0.2,
    5.0, 5.0, 5.0, 5.0, 5.0,
    5.0, 5.0, 5.0, 0.2,
], dtype=torch.float32)

h5_file = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/max12_cts_noleak_crossval.h5"
master_table_path = Path('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/master_line_table.csv').resolve()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_save_path = Path(f'/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/p3VAE/trained_models/best_p3vae_model_sfthresh_NonZeroStratifiedEvalMetrics_{timestamp}.pt').resolve()
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
    def __init__(self, spectra, targets, is_synth, indices):
        self.spectra, self.targets, self.is_synth = spectra, targets, is_synth
        self.idx = np.asarray(indices)
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        return {'spectrum': self.spectra[j], 'targets': self.targets[j],
                'is_synthetic': self.is_synth[j]}


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

        target_dtype = concentrations.dtype
        grid_in = self.grid.to(dtype=target_dtype)
        centers_in = self.line_centers.to(dtype=target_dtype)

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

        # Restrict residual scale capacity to prevent physics decoder bypass
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

        # Smooth Softplus activation prevents hard ReLU zero-gradient traps
        concentrations = F.softplus(concs_raw)
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

        recon_loss = F.mse_loss(x_recon, x_target, reduction="mean")

        if self.use_log_conc:
            diff = torch.log1p(pred_conc) - torch.log1p(true_conc)
        else:
            diff = pred_conc - true_conc

        element_errors = diff**2

        if self.element_weights is not None:
            element_errors = element_errors * self.element_weights

        sup_loss = element_errors.mean()

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


class SpectralScaler:
    """Scales spectra to align magnitude with the physics decoder without data leakage."""
    def __init__(self, mode: str = "global_max", q: float = 99.9):
        self.mode = mode
        self.q = q
        self.scale_ = None

    def fit(self, y_train: torch.Tensor):
        if self.mode == "global_max":
            self.scale_ = torch.clamp(y_train.max(), min=1e-6)
        elif self.mode == "mean_max":
            self.scale_ = torch.clamp(y_train.max(dim=1).values.mean(), min=1e-6)
        elif self.mode == "robust_max":
            row_maxes = y_train.max(dim=1).values.numpy()
            scale_val = np.percentile(row_maxes, self.q)
            self.scale_ = torch.tensor(scale_val, dtype=torch.float32).clamp(min=1e-6)
        else:
            self.scale_ = torch.clamp(y_train.max(), min=1e-6)

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        if self.scale_ is None:
            raise RuntimeError("Scaler must be fitted before calling transform!")
        return y / self.scale_.to(y.device)

    def inverse_transform(self, y_scaled: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        scale_val = self.scale_.cpu().numpy() if isinstance(y_scaled, np.ndarray) else self.scale_.to(y_scaled.device)   # type: ignore
        return y_scaled * scale_val

    def state_dict(self) -> dict:
        return {"scale_": self.scale_, "mode": self.mode}

    def load_state_dict(self, state: dict):
        self.scale_ = state["scale_"]
        self.mode = state["mode"]


def assign_group_folds(X, group_id, is_synthetic, dopant_idx, elem_names=None,
                       n_splits=5, n_trials=2000, min_groups_per_dopant=1, seed=42, verbose=True):
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

    best_fog, best_score = None, None
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

    if best_fog is None or best_score is None:
        raise ValueError("Failed to assign fold groups: n_trials must be > 0.")

    if -best_score[0] < 1:
        raise ValueError(
            f"A dopant has no groups in at least one fold (min={-best_score[0]}). "
            "Lower n_splits, raise n_trials, or assign the rare groups by hand."
        )

    fold_id = np.full(len(X), -1, dtype=np.int32)
    fold_id[exp] = best_fog[inv]

    if verbose:
        names = elem_names or [str(i) for i in range(X.shape[1])]
        print(f"{n_groups} groups | min groups/dopant/fold={-best_score[0]} | "
              f"worst shot-fraction deviation={best_score[1]:.3f}")
        print(f"{'fold':<5}{'groups':>7}{'shots':>9}  " + "  ".join(f"{names[j]:>11}" for j in active_dopant_idx) + "  (groups/shots)")
        for k in range(n_splits):
            g = best_fog == k
            cells = [f"{int(present[g, j].sum())}/{int(shots[g & present[:, j]].sum())}" for j in range(len(active_dopant_idx))]
            print(f"{k:<5}{g.sum():>7}{shots[g].sum():>9}  " + "  ".join(f"{c:>11}" for c in cells))
    return fold_id


def best_threshold(t, p, grid=np.linspace(0, 0.15, 151)):
    pres = t > 0
    def score(tau):
        pz = np.where(p < tau, 0.0, p)
        return 0.5 * (np.abs(t[pres] - pz[pres]).mean() + np.abs(pz[~pres]).mean())
    return grid[np.argmin([score(tau) for tau in grid])]


def build_splits_once(data, split_path, num_folds, seed=42, min_groups_per_dopant=1):
    split_path = Path(split_path).with_suffix('.npz')
    if split_path.exists():
        d = np.load(split_path)
        assert int(d['num_folds']) == num_folds and int(d['seed']) == seed, \
            "cached split doesn't match requested settings"
        return d['test_idx'], d['cv_idx'], d['cv_fold_id']
    split_path.parent.mkdir(parents=True, exist_ok=True)
    test_idx, cv_idx, cv_fold_id = create_test_and_cv_folds(
        data=data, num_folds=num_folds,
        min_groups_per_dopant=min_groups_per_dopant, seed=seed)
    np.savez(split_path, test_idx=test_idx, cv_idx=cv_idx, cv_fold_id=cv_fold_id,
             num_folds=num_folds, seed=seed)
    return test_idx, cv_idx, cv_fold_id


def check_bad_row_compositions(h5_path: str, thresh: float = 2.5):
    with h5py.File(h5_path, 'r') as hf:
        y = hf['y'][:]   # type: ignore
        X = hf['X'][:]   # type: ignore
        is_synth = hf['is_synthetic'][:]   # type: ignore
        group_id = hf['group_id'][:]   # type: ignore
        names = [
            n.decode('utf-8') if isinstance(n, bytes) else str(n) 
            for n in hf['feature_cols'][:]   # type: ignore
        ]

    row_maxes = y.max(axis=1)   # type: ignore
    bad_mask = row_maxes > thresh
    bad_indices = np.where(bad_mask)[0]

    print(f"Found {len(bad_indices)} bad rows exceeding threshold > {thresh}\n")
    if len(bad_indices) == 0:
        return

    X_bad = X[bad_indices]   # type: ignore
    gid_bad = group_id[bad_indices]   # type: ignore
    synth_bad = is_synth[bad_indices]   # type: ignore

    print("=========================================================================")
    print(" Element Concentration Summary Across All Bad Rows")
    print("=========================================================================")
    for j, elem in enumerate(names):
        vals = X_bad[:, j]   # type: ignore
        non_zero = vals[vals > 0]   # type: ignore
        if len(non_zero) > 0:   # type: ignore
            print(
                f"Element {elem:<5} | Non-Zero Shots: {len(non_zero):<4}/{len(bad_indices)} | "   # type: ignore
                f"Min: {non_zero.min():.4f}% | Max: {non_zero.max():.4f}% | Mean: {non_zero.mean():.4f}%"   # type: ignore
            )
        else:
            print(f"Element {elem:<5} | 0 non-zero entries (Absent)")

    unique_groups = np.unique(gid_bad)   # type: ignore
    print("\n=========================================================================")
    print(f" Composition Breakdown by Group ID ({len(unique_groups)} unique group(s))")
    print("=========================================================================")
    for g in unique_groups:
        g_mask = (gid_bad == g)
        sample_x = X_bad[g_mask][0]   # type: ignore
        is_s = "Synthetic" if synth_bad[g_mask][0] == 1 else "Experimental"   # type: ignore
        
        non_zero_str = ", ".join([
            f"{names[j]}: {sample_x[j]:.3f} wt%"   # type: ignore
            for j in range(len(names)) if sample_x[j] > 0   # type: ignore
        ])
        print(f"Group {g:<3} [{is_s}] | Count: {np.sum(g_mask):<3} shots | Compositions: {non_zero_str}")


def clean_spectral_spikes(y, thresh=THRESH):
    return torch.clamp(y, max=thresh)


def create_test_and_cv_folds(
    data: Dict[str, Any], 
    num_folds: int = 5, 
    min_groups_per_dopant: int = 1, 
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    test_mask = data['is_lockbox']
    test_idx = np.where(test_mask)[0]

    cv_mask = ~data['is_lockbox']
    cv_idx = np.where(cv_mask)[0]

    X_cv = data['X'][cv_idx]
    X_cv_np = X_cv.numpy() if hasattr(X_cv, 'numpy') else X_cv
    is_synth_cv = data['is_synth'][cv_idx]
    is_synth_cv_np = is_synth_cv.numpy() if hasattr(is_synth_cv, 'numpy') else is_synth_cv

    host_elements = {'Li', 'K', 'Cl'}
    dopant_idx = [
        i for i, name in enumerate(data['names']) 
        if name not in host_elements and (X_cv_np[:, i] > 0).any()
    ]

    cv_fold_id = assign_group_folds(
        X=X_cv_np,
        group_id=data['group_id'][cv_idx],
        is_synthetic=is_synth_cv_np,
        dopant_idx=dopant_idx,
        elem_names=data['names'],
        n_splits=num_folds,
        min_groups_per_dopant=min_groups_per_dopant,
        seed=seed,
        verbose=True
    )

    return test_idx, cv_idx, cv_fold_id


def diagnose_spike_geometry(y, group_id, thresh=2.5):
    y_np = y.cpu().numpy() if hasattr(y, 'cpu') else np.asarray(y)
    gid_np = group_id.cpu().numpy() if hasattr(group_id, 'cpu') else np.asarray(group_id)

    row_max = y_np.max(axis=1)
    bad_indices = np.where(row_max > thresh)[0]

    print(f"=== Diagnosing {len(bad_indices)} Outlier Rows ===")
    
    spike_counts = []
    for idx in bad_indices:
        row = y_np[idx]
        num_channels_above_thresh = np.sum(row > thresh)
        spike_counts.append(num_channels_above_thresh)

    spike_counts = np.array(spike_counts)
    print(f"Channels exceeding {thresh} per bad row:")
    print(f"  Min channels:  {spike_counts.min()}")
    print(f"  Max channels:  {spike_counts.max()}")
    print(f"  Mean channels: {spike_counts.mean():.2f}")

    g30_normal = np.where((gid_np == 30) & (row_max <= thresh))[0]
    g30_bad = np.where((gid_np == 30) & (row_max > thresh))[0]

    if len(g30_normal) > 0 and len(g30_bad) > 0:
        norm_mean_intensity = y_np[g30_normal].mean()
        bad_mean_intensity = y_np[g30_bad].mean()
        
        print("\nGroup 30 Intensity Comparison:")
        print(f"  Normal shots mean intensity across all channels: {norm_mean_intensity:.5f}")
        print(f"  Bad shots mean intensity across all channels:    {bad_mean_intensity:.5f}")


def find_spike_rows(y, group_id, is_synth, thresh=3.0, chunk=20000):
    row_max = torch.empty(len(y))
    for s in range(0, len(y), chunk):
        row_max[s:s+chunk] = y[s:s+chunk].max(dim=1).values
    rm, syn = row_max.numpy(), np.asarray(is_synth) == 1
    for name, m in [('exp', ~syn), ('syn', syn)]:
        print(name, 'row-max percentiles 50/90/99/99.9/100:', np.round(np.percentile(rm[m], [50, 90, 99, 99.9, 100]), 3))
    bad = np.where(rm > thresh)[0]
    print(f"{len(bad)} rows with a pixel > {thresh}; groups:", np.unique(group_id[bad], return_counts=True))


def fold_range_check(X, fold_id, is_synth, names, dopants=('Ce', 'Gd', 'Sm', 'U'), logger=None):
    X_np = X.cpu().numpy() if hasattr(X, 'cpu') else np.asarray(X)
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
                val_max = X_np[val, e_idx].max() if np.any(val) else 0.0
                train_max = X_np[tr, e_idx].max() if np.any(tr) else 0.0
                row.append(f"{e}: val {val_max:.3f} / train {train_max:.3f}")

        msg = f"[Range Check] Fold {k}: " + " | ".join(row)
        if logger:
            log(logger=logger, msg=msg)
        else:
            print(msg)


def get_fold_ids(cv_data, num_folds, h5_path, dopants=('Ce', 'Gd', 'Sm', 'U'), seed=42):
    fold_path = Path(h5_path).with_name(f"{Path(h5_path).stem}_folds{num_folds}_seed{seed}.npy")
    if fold_path.exists():
        return np.load(fold_path)
    dop = [cv_data['names'].index(e) for e in dopants]
    fold_id = assign_group_folds(cv_data['X'].numpy(), cv_data['group_id'], cv_data['is_synth'].numpy(),
                                 dop, cv_data['names'], n_splits=num_folds, seed=seed)
    np.save(fold_path, fold_id)
    return fold_id


def group_mean(arr, gid):
    ug, inv = np.unique(gid, return_inverse=True); inv = inv.ravel()
    cnt = np.bincount(inv)
    return np.stack([np.bincount(inv, weights=arr[:, j]) / cnt for j in range(arr.shape[1])], 1), ug


def load_crossval_data(h5_path: str, spike_thresh: float = THRESH) -> Dict[str, Any]:
    with h5py.File(h5_path, 'r') as hf:
        y = torch.from_numpy(get_h5_ds(hf, 'y')[:]).float()
        X = torch.from_numpy(get_h5_ds(hf, 'X')[:]).float()
        is_synth = torch.from_numpy(get_h5_ds(hf, 'is_synthetic')[:]).long()
        is_lockbox = get_h5_ds(hf, 'is_lockbox')[:].astype(bool)
        group_id = get_h5_ds(hf, 'group_id')[:]
        wavelengths = get_h5_ds(hf, 'wavelengths')[:]
        names = [
            n.decode('utf-8') if isinstance(n, bytes) else str(n) 
            for n in get_h5_ds(hf, 'feature_cols')[:]
        ]

    row_maxes = y.max(dim=1).values
    clean_mask = (row_maxes <= spike_thresh).numpy()
    
    n_dropped = len(y) - np.sum(clean_mask)
    if n_dropped > 0:
        print(f"🧹 Data Prep: Dropped {n_dropped} corrupted saturation shots (max intensity > {spike_thresh}).")

    data = dict(
        X=X[clean_mask],
        y=y[clean_mask],
        is_synth=is_synth[clean_mask],
        is_lockbox=is_lockbox[clean_mask],
        group_id=group_id[clean_mask],
        wavelengths=wavelengths,
        names=names,
    )

    return data


def per_spectrum_norm(x, eps=1e-08):
    return x / (x.sum(dim=1, keepdim=True) + eps) * x.shape[1]


def pull_test_holdout(data, test_ratio=0.15, seed=42):
    is_synth = data['is_synth']
    if hasattr(is_synth, 'cpu'):
        is_synth = is_synth.cpu().numpy()

    group_id = data['group_id']
    if hasattr(group_id, 'cpu'):
        group_id = group_id.cpu().numpy()

    exp_mask = (is_synth == 0)
    exp_groups = np.unique(group_id[exp_mask])

    rng = np.random.default_rng(seed)
    test_groups = rng.choice(exp_groups, size=int(test_ratio * len(exp_groups)), replace=False)

    test_mask = np.isin(group_id, test_groups) & exp_mask
    cv_mask = ~np.isin(group_id, test_groups)

    test_idx = np.where(test_mask)[0]
    cv_idx = np.where(cv_mask)[0]

    return test_idx, cv_idx


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_fold(cv_data, fold_id, fold, master_table_path, save_path, logger, device_obj,
                   element_weights, epochs, batch_size, lr, weight_decay, lr_patience=5, stop_patience=10, use_compile=True):
    torch._dynamo.reset()
    train_idx = np.where(fold_id != fold)[0]
    val_idx = np.where(fold_id == fold)[0]
    pin = device_obj.type == 'cuda'
    
    train_y_slice = cv_data['y'][train_idx]
    spec_scaler = SpectralScaler(mode='robust_max', q=99.9)
    spec_scaler.fit(train_y_slice)
    scale_gpu = spec_scaler.scale_.to(device_obj)   # type: ignore
    log(logger=logger, msg=f'[F{fold}] spec_scaler scale_ = {scale_gpu.item():.4f}')
    
    sub = train_idx[::200]
    ys, sy = cv_data['y'][sub], cv_data['is_synth'][sub]
    log(logger=logger, msg=f"exp max/mean {ys[sy==0].max():.4g}/{ys[sy==0].mean():.4g} | "
                           f"syn max/mean {ys[sy==1].max():.4g}/{ys[sy==1].mean():.4g}")

    del train_y_slice
    gc.collect()

    train_ds = LIBSSpectraDataset(cv_data['y'], cv_data['X'], cv_data['is_synth'], train_idx)
    val_ds = LIBSSpectraDataset(cv_data['y'], cv_data['X'], cv_data['is_synth'], val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=pin, persistent_workers=(NUM_WORKERS > 0), drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=pin, persistent_workers=(NUM_WORKERS > 0)
    )
    
    log(logger=logger, msg=f"=== Fold {fold}: train {len(train_ds)} | val {len(val_ds)} "
                           f"({len(np.unique(cv_data['group_id'][val_idx]))} compositions) ===")

    wl_grid = torch.from_numpy(cv_data['wavelengths']).to(device_obj)
    raw_model = PhysicsInformedVAE(
        input_dim=cv_data['y'].shape[1], n_elements=cv_data['X'].shape[1],
        wl_grid=wl_grid, master_table_path=master_table_path, n_broadening=1
    ).to(device_obj)

    if use_compile:
        model = cast(nn.Module, torch.compile(raw_model, mode='default'))
    else:
        model = raw_model

    criterion = PhysicsVAELoss(
        element_weights=element_weights, alpha_recon=1.0, beta_sup=10.0,
        gamma_kl=1e-4, use_log_conc=True
    ).to(device_obj)

    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=lr_patience, min_lr=MIN_LR)
    use_amp = device_obj.type == 'cuda'
    
    amp_scaler = amp.GradScaler(enabled=use_amp)
    early_stopper = EarlyStopping(patience=stop_patience)
    best_val, best_epoch = float('inf'), 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss = 0.0
        for batch in tqdm(train_loader, desc=f"F{fold} E{epoch:02d}", mininterval=5, leave=False):
            spectra = batch['spectrum'].to(device_obj, non_blocking=True) / scale_gpu
            targets = batch['targets'].to(device_obj, non_blocking=True)
            batch_gpu = {'spectrum': spectra, 'targets': targets}

            optimizer.zero_grad()
            with amp.autocast(enabled=use_amp, dtype=torch.float16):
                out = model(spectra)
                ld = criterion(out, batch_gpu)

            if torch.isnan(ld['loss']):
                continue
            amp_scaler.scale(ld['loss']).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            tr_loss += ld['loss'].item() * spectra.size(0)

        model.eval()
        va_loss, va_sup = 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                spectra = batch['spectrum'].to(device_obj, non_blocking=True) / scale_gpu
                targets = batch['targets'].to(device_obj, non_blocking=True)
                batch_gpu = {'spectrum': spectra, 'targets': targets}

                with amp.autocast(enabled=use_amp, dtype=torch.float16):
                    ld = criterion(model(spectra), batch_gpu)
                n = spectra.size(0)
                va_loss += ld['loss'].item() * n
                va_sup += ld['sup_loss'].item() * n

        tr_loss /= len(train_ds); va_loss /= len(val_ds); va_sup /= len(val_ds)
        scheduler.step(va_loss)
        log(logger=logger, msg=f"[F{fold}] Epoch {epoch:02d}/{epochs} [{time.time()-t0:.1f}s] | "
                               f"Train {tr_loss:.4f} | Val {va_loss:.4f} | Conc MSE {va_sup:.4f}")

        if va_loss < best_val:
            best_val, best_epoch = va_loss, epoch
            torch.save({
                'epoch': epoch, 
                'model_state_dict': raw_model.state_dict(), 
                'scaler_state_dict': spec_scaler.state_dict(),
                'val_loss': best_val
            }, save_path)

        if early_stopper(va_loss):
            log(logger=logger, msg=f"[F{fold}] early stop at epoch {epoch}")
            break

    ckpt = torch.load(save_path, map_location=device_obj)
    raw_model.load_state_dict(ckpt['model_state_dict'])
    raw_model.eval()
    preds = []
    with torch.no_grad():
        for batch in val_loader:
            spectra = batch['spectrum'].to(device_obj, non_blocking=True) / scale_gpu
            preds.append(raw_model(spectra)['concentrations'].float().cpu().numpy())
    pred = np.concatenate(preds, axis=0)

    del model, raw_model, optimizer, amp_scaler, train_loader, val_loader
    gc.collect(); torch.cuda.empty_cache()
    return dict(fold=fold, val_idx=val_idx, pred=pred, best_epoch=best_epoch, best_val=best_val)


def train_kfold_physics_vae(
    h5_path, master_table_path, log_path=None, save_path=None,
    epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, lr=learning_rate,
    weight_decay=weight_decay, num_folds=NUM_FOLDS, folds=None,
    device=None, lr_patience=5, stop_patience=10, use_compile=True
):
    save_path = Path(save_path or model_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_path or f"/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/logs/p3vae_logs/kfold_{timestamp}.txt").resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = get_worker_logger(log_path.stem)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    data = load_crossval_data(h5_path)

    seed_everything(seed=42)
    split_path = Path('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/fold_splits') / f'max12_{num_folds}_folds_seed42.npz'
    test_idx, cv_idx, cv_fold_id = build_splits_once(data, split_path, num_folds, seed=42)

    test_idx_path = save_path.with_name(f"{save_path.stem}_test_idx.npy")
    np.save(test_idx_path, test_idx)
    log(logger=logger, msg='This is the sfthresh NonZeroStratifiedEvalMetrics experiment')
    log(logger=logger, msg=f"Saved test holdout indices ({len(test_idx)} samples) → {test_idx_path.name}")

    n_samples = len(data['X'])  

    cv_data = {}
    for key, val in data.items():
        if isinstance(val, (np.ndarray, torch.Tensor)) and val.ndim > 0 and val.shape[0] == n_samples:
            cv_data[key] = val[cv_idx]
        else:
            cv_data[key] = val

    fold_range_check(
        X=cv_data['X'],
        fold_id=cv_fold_id,
        is_synth=cv_data['is_synth'],
        names=cv_data['names'],
        dopants=('Ce', 'Gd', 'Sm', 'U'),
        logger=logger
    )

    del data
    gc.collect()

    element_weights = torch.tensor(
        [ELEMENT_WEIGHTS_REFERENCE[n.encode() if isinstance(n, str) else n] for n in cv_data['names']], 
        dtype=torch.float32
    )

    results = []
    for k in (folds if folds is not None else range(num_folds)):
        fold_ckpt = save_path.with_name(f"{save_path.stem}_fold{k}.pt")
        res = train_one_fold(
            cv_data, cv_fold_id, k, master_table_path, fold_ckpt, logger, device_obj,
            element_weights, epochs, batch_size, lr, weight_decay, lr_patience, stop_patience, use_compile=use_compile
        )
        np.savez(save_path.with_name(f"{save_path.stem}_oof_fold{k}.npz"), val_idx=res['val_idx'], pred=res['pred'])
        log(logger=logger, msg=f"[F{k}] done: best epoch {res['best_epoch']}, val {res['best_val']:.4f}")
        results.append((k, res['best_epoch'], res['best_val']))

    oof_preds = np.zeros_like(cv_data['X'].cpu().numpy() if hasattr(cv_data['X'], 'cpu') else cv_data['X'])
    evaluated_mask = np.zeros(len(cv_data['X']), dtype=bool)
    
    for k in range(num_folds):
        oof_path = save_path.with_name(f"{save_path.stem}_oof_fold{k}.npz")
        if oof_path.exists():
            oof_data = np.load(oof_path)
            val_idx_k = oof_data['val_idx']
            oof_preds[val_idx_k] = oof_data['pred']
            evaluated_mask[val_idx_k] = True

    if np.any(evaluated_mask):
        log(logger=logger, msg="=== Out-Of-Fold (OOF) Group Diagnostic ===")
        worst_groups(
            true=cv_data['X'][evaluated_mask],
            pred=oof_preds[evaluated_mask],
            gid=cv_data['group_id'][evaluated_mask],
            names=cv_data['names'],
            dopants=('Ce', 'Gd', 'Sm', 'U'),
            top=5,
            logger=logger
        )

    return results


def worst_groups(true, pred, gid, names, dopants=('Ce', 'Gd', 'Sm', 'U'), top=3, logger=None):
    true_np = true.cpu().numpy() if hasattr(true, 'cpu') else np.asarray(true)
    pred_np = pred.cpu().numpy() if hasattr(pred, 'cpu') else np.asarray(pred)
    gid_np = gid.cpu().numpy() if hasattr(gid, 'cpu') else np.asarray(gid)

    j = [names.index(e) for e in dopants if e in names]
    err = ((np.log1p(pred_np[:, j]) - np.log1p(true_np[:, j])) ** 2).mean(axis=1)

    ug, inv = np.unique(gid_np, return_inverse=True)
    inv = inv.ravel()
    cnt = np.bincount(inv)
    ge = np.bincount(inv, weights=err) / cnt

    msg_lines = [f"=== Top {top} Worst Performing Composition Groups ==="]
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


if __name__ == "__main__":
    print('Starting run')
    loop_time_start = time.perf_counter()
    k = 7

    # --------------------------------------------------
    # region Train Model
    # --------------------------------------------------
    results = train_kfold_physics_vae(
        h5_path=h5_file,
        master_table_path=master_table_path,
        log_path=f'/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/logs/p3vae_logs/005_sfthresh_NonZeroStratifiedEvalMetrics_{timestamp}_log.txt',
        save_path=model_save_path,
        num_folds=k,
        folds=None,
        device='cuda',
        lr_patience=4,
        stop_patience=20,
        use_compile=False
    )
    print("Training Results (fold, best_epoch, best_val_loss):")
    print(results)
    # endregion

    # --------------------------------------------------
    # region Evaluate Model
    # --------------------------------------------------
    model_prefix = str(model_save_path.with_suffix(""))
    evaluate_and_plot_kfold(
        h5_path=h5_file,
        master_table_path=master_table_path,
        model_prefix=model_prefix,
        model_cls=PhysicsInformedVAE,
        scaler_cls=SpectralScaler,
        dataset_cls=LIBSSpectraDataset,
        load_data_fn=load_crossval_data,
        num_folds=k,
        batch_size=BATCH_SIZE,
        output_dir="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/p3VAE/plots/kfold/sfthresh",
        device="cuda",
        model_tag=f'NonZeroStratifiedEvalMetrics_{timestamp}',
        max_wt_percent=12.0
    )
    # endregion

    loop_time_end = time.perf_counter()
    print(f"Run completed in {(loop_time_end - loop_time_start) / 60:.2f} minutes.")