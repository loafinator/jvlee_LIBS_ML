from __future__ import annotations

"""
jvlee_LIBS_ML > LIBS > p3VAE > p3vae_005.py
"""

# region Imports
import h5py
import sys
import time
import torch
import math
import gc
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
from typing import Any, Dict
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
NUM_FOLDS = 5

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


h5_file = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/cts_noleak_crossval.h5"
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
    def __init__(self, spectra, targets, is_synth, indices):   # tensors loaded ONCE outside
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


def assign_group_folds(X, group_id, is_synthetic, dopant_idx, elem_names=None,
                       n_splits=5, n_trials=2000, min_groups_per_dopant=3, seed=42, verbose=True):
    exp = np.where(is_synthetic == 0)[0]
    groups, inv = np.unique(group_id[exp], return_inverse=True)
    inv = inv.ravel()
    n_groups = len(groups)
    shots = np.bincount(inv, minlength=n_groups)

    first = np.zeros(n_groups, dtype=int)                 # one representative row per group
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    present = X[exp[first]][:, dopant_idx] > 0            # (n_groups, n_dopants)
    sig = (present.astype(int) * (1 << np.arange(len(dopant_idx)))).sum(axis=1)
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

    fold_id = np.full(len(X), -1, dtype=np.int32)
    fold_id[exp] = best_fog[inv]

    if verbose:
        names = elem_names or [str(i) for i in range(X.shape[1])]
        print(f"{n_groups} groups | min groups/dopant/fold={-best_score[0]} | "
              f"worst shot-fraction deviation={best_score[1]:.3f}")
        print(f"{'fold':<5}{'groups':>7}{'shots':>9}  " + "  ".join(f"{names[j]:>11}" for j in dopant_idx) + "  (groups/shots)")
        for k in range(n_splits):
            g = best_fog == k
            cells = [f"{int(present[g, j].sum())}/{int(shots[g & present[:, j]].sum())}" for j in range(len(dopant_idx))]
            print(f"{k:<5}{g.sum():>7}{shots[g].sum():>9}  " + "  ".join(f"{c:>11}" for c in cells))
    return fold_id

def _get_dataset(hf: h5py.File, key: str) -> h5py.Dataset:
    node = hf[key]
    assert isinstance(node, h5py.Dataset)
    return node

def load_crossval_data(h5_path: str) -> Dict[str, Any]:
    """Loads everything ONCE. y is ~15 GB, so never copy or slice it per fold."""
    with h5py.File(h5_path, 'r') as hf:
        data = dict(
            X=torch.from_numpy(_get_dataset(hf, 'X')[:]).float(),
            y=torch.from_numpy(_get_dataset(hf, 'y')[:]).float(),
            is_synth=torch.from_numpy(_get_dataset(hf, 'is_synthetic')[:]).long(),
            group_id=_get_dataset(hf, 'group_id')[:],
            wavelengths=_get_dataset(hf, 'wavelengths')[:],
            names=[
                n.decode('utf-8') if isinstance(n, bytes) else str(n) 
                for n in _get_dataset(hf, 'feature_cols')[:]
            ],
        )
    return data
 
 
def get_fold_ids(data, num_folds, h5_path, dopants=('Ce', 'Gd', 'Sm', 'U'), seed=42):
    fold_path = Path(h5_path).with_name(f"{Path(h5_path).stem}_folds{num_folds}_seed{seed}.npy")
    if fold_path.exists():
        return np.load(fold_path)                      # reuse -> identical folds on every run
    dop = [data['names'].index(e) for e in dopants]
    fold_id = assign_group_folds(data['X'].numpy(), data['group_id'], data['is_synth'].numpy(),
                                 dop, data['names'], n_splits=num_folds, seed=seed)
    np.save(fold_path, fold_id)
    return fold_id
 
 
def train_one_fold(data, fold_id, fold, master_table_path, save_path, logger, device_obj,
                   element_weights, epochs, batch_size, lr, weight_decay, patience=10, use_compile=True):
    torch._dynamo.reset()                              # fresh compile cache per fold (default limit is 8 recompiles)
    train_idx = np.where(fold_id != fold)[0]           # synthetic (-1) + the other folds
    val_idx = np.where(fold_id == fold)[0]
    pin = device_obj.type == 'cuda'
    train_ds = LIBSSpectraDataset(data['y'], data['X'], data['is_synth'], train_idx)
    val_ds = LIBSSpectraDataset(data['y'], data['X'], data['is_synth'], val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    log(logger=logger, msg=f"=== Fold {fold}: train {len(train_ds)} | val {len(val_ds)} "
                           f"({len(np.unique(data['group_id'][val_idx]))} compositions) ===")
 
    wl_grid = torch.from_numpy(data['wavelengths']).to(device_obj)
    raw_model = PhysicsInformedVAE(input_dim=data['y'].shape[1], n_elements=data['X'].shape[1],
                                   wl_grid=wl_grid, master_table_path=master_table_path,
                                   n_broadening=1).to(device_obj)
    model = torch.compile(raw_model, mode='reduce-overhead') if use_compile else raw_model
 
    criterion = PhysicsVAELoss(element_weights=element_weights, alpha_recon=1.0, beta_sup=10.0,
                               gamma_kl=1e-4, use_log_conc=True).to(device_obj)
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=MIN_LR)
    use_amp = device_obj.type == 'cuda'
    scaler = amp.GradScaler(enabled=use_amp)
    early_stopper = EarlyStopping(patience=patience)
    best_val, best_epoch = float('inf'), 0
 
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss = 0.0
        for batch in tqdm(train_loader, desc=f"F{fold} E{epoch:02d}", mininterval=60, leave=False):
            batch = {k: v.to(device_obj, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad()
            with amp.autocast(enabled=use_amp, dtype=torch.float16):
                out = model(batch['spectrum'])
                ld = criterion(out, batch)
            if torch.isnan(ld['loss']):
                continue
            scaler.scale(ld['loss']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            tr_loss += ld['loss'].item() * batch['spectrum'].size(0)
 
        model.eval()
        va_loss, va_sup = 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device_obj, non_blocking=True) for k, v in batch.items()}
                with amp.autocast(enabled=use_amp, dtype=torch.float16):
                    ld = criterion(model(batch['spectrum']), batch)
                n = batch['spectrum'].size(0)
                va_loss += ld['loss'].item() * n
                va_sup += ld['sup_loss'].item() * n
        tr_loss /= len(train_ds); va_loss /= len(val_ds); va_sup /= len(val_ds)
        scheduler.step(va_loss)
        log(logger=logger, msg=f"[F{fold}] Epoch {epoch:02d}/{epochs} [{time.time()-t0:.1f}s] | "
                               f"Train {tr_loss:.4f} | Val {va_loss:.4f} | Conc MSE {va_sup:.4f}")
        if va_loss < best_val:
            best_val, best_epoch = va_loss, epoch
            torch.save({'epoch': epoch, 'model_state_dict': raw_model.state_dict(), 'val_loss': best_val}, save_path)
        if early_stopper(va_loss):
            log(logger=logger, msg=f"[F{fold}] early stop at epoch {epoch}")
            break
 
    # ---- out-of-fold predictions from the BEST epoch (fp32, uncompiled) ----
    raw_model.load_state_dict(torch.load(save_path, map_location=device_obj)['model_state_dict'])
    raw_model.eval()
    preds = []
    with torch.no_grad():
        for batch in val_loader:
            preds.append(raw_model(batch['spectrum'].to(device_obj))['concentrations'].float().cpu().numpy())
    pred = np.concatenate(preds, axis=0)
 
    del model, raw_model, optimizer, scaler, train_loader, val_loader
    gc.collect(); torch.cuda.empty_cache()
    return dict(fold=fold, val_idx=val_idx, pred=pred, best_epoch=best_epoch, best_val=best_val)
 
 
def train_kfold_physics_vae(h5_path, master_table_path, log_path=None, save_path=None,
                            epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, lr=learning_rate,
                            weight_decay=weight_decay, num_folds=NUM_FOLDS, folds=None,
                            device=None, use_compile=True):
    save_path = Path(save_path or model_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_path or f"/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/logs/p3vae_logs/kfold_{timestamp}.txt").resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = get_worker_logger(log_path.stem)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
 
    data = load_crossval_data(h5_path)
    fold_id = get_fold_ids(data, num_folds, h5_path)
    # element weights built from the file's own column order, so they can't silently misalign
    element_weights = torch.tensor([ELEMENT_WEIGHTS_REFERENCE[n.encode()] for n in data['names']], dtype=torch.float32)
 
    results = []
    for k in (folds if folds is not None else range(num_folds)):
        fold_ckpt = save_path.with_name(f"{save_path.stem}_fold{k}.pt")
        res = train_one_fold(data, fold_id, k, master_table_path, fold_ckpt, logger, device_obj,
                             element_weights, epochs, batch_size, lr, weight_decay, use_compile=use_compile)
        np.savez(save_path.with_name(f"{save_path.stem}_oof_fold{k}.npz"), val_idx=res['val_idx'], pred=res['pred'])
        log(logger=logger, msg=f"[F{k}] done: best epoch {res['best_epoch']}, val {res['best_val']:.4f}")
        results.append((k, res['best_epoch'], res['best_val']))
    return results
 
 
def evaluate_oof(h5_path, oof_prefix, num_folds, output_dir, min_true_max=1e-6):
    """Pools per-fold out-of-fold predictions and reports shot-level and per-composition metrics.
    Only reads X and group_id from the h5 (never the 15 GB spectra)."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, 'r') as hf:
        X = hf['X'][:]
        gid = hf['group_id'][:]
        names = [n.decode('utf-8') if isinstance(n, bytes) else str(n) for n in hf['feature_cols'][:]]
 
    pred = np.full(X.shape, np.nan, dtype=np.float32)
    fold = np.full(len(X), -1, dtype=np.int32)
    for k in range(num_folds):
        p = Path(f"{oof_prefix}_oof_fold{k}.npz")
        if p.exists():
            d = np.load(p)
            pred[d['val_idx']] = d['pred']
            fold[d['val_idx']] = k
    m = fold >= 0
    print(f"Pooled OOF: {m.sum()} shots from folds {sorted(set(fold[m].tolist()))}")
 
    t, p_ = X[m], pred[m]
    ug, inv = np.unique(gid[m], return_inverse=True)
    inv = inv.ravel()
    cnt = np.bincount(inv)
    gm_t = np.stack([np.bincount(inv, weights=t[:, j]) / cnt for j in range(t.shape[1])], axis=1)
    gm_p = np.stack([np.bincount(inv, weights=p_[:, j]) / cnt for j in range(t.shape[1])], axis=1)
 
    def r2(a, b):
        ss_tot = np.sum((a - a.mean()) ** 2)
        return np.nan if ss_tot < 1e-12 else 1.0 - np.sum((a - b) ** 2) / ss_tot
 
    active = [j for j in range(t.shape[1]) if t[:, j].max() > min_true_max]   # skip elements never present
    print(f"{len(ug)} compositions pooled. Units are the scaled units stored in X.")
    print(f"{'elem':<5}{'shotMAE':>9}{'compMAE':>9}{'compR2':>9}{'shotR2':>9}")
    for j in active:
        print(f"{names[j]:<5}{np.abs(t[:, j]-p_[:, j]).mean():>9.4f}{np.abs(gm_t[:, j]-gm_p[:, j]).mean():>9.4f}"
              f"{r2(gm_t[:, j], gm_p[:, j]):>9.3f}{r2(t[:, j], p_[:, j]):>9.3f}")
 
    fig, axes = plt.subplots(1, len(active), figsize=(3.4 * len(active), 3.4), squeeze=False)
    for ax, j in zip(axes[0], active):
        ax.scatter(t[::50, j], p_[::50, j], s=3, alpha=0.15, color="gray")          # shots (thinned)
        ax.scatter(gm_t[:, j], gm_p[:, j], s=20, color="crimson")                   # one dot per composition
        lim = [0, max(gm_t[:, j].max(), gm_p[:, j].max()) * 1.05]
        ax.plot(lim, lim, "k--", lw=1); ax.set_title(names[j]); ax.set_xlabel("true"); ax.set_ylabel("pred")
    plt.tight_layout(); plt.savefig(output_dir / "oof_parity_by_composition.png", dpi=200); plt.close()
    return dict(pred=pred, fold=fold, gm_true=gm_t, gm_pred=gm_p, groups=ug)

if __name__ == "__main__":
    print('Starting run')
    loop_time_start = time.perf_counter()

    # --------------------------------------------------
    # Train Model
    # --------------------------------------------------
    results = train_kfold_physics_vae(
        h5_path=h5_file,
        master_table_path=master_table_path,
        log_path='/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/logs/p3vae_logs/005_log.txt',
        save_path=model_save_path,
        folds=[0],              # smoke test first, then None for all five
        device='cuda',
    )
    print(results)              # (fold, best_epoch, best_val_loss)

    # --------------------------------------------------
    # Evaluate Model
    # --------------------------------------------------
    evaluate_oof(
        h5_path=h5_file,
        oof_prefix=str(model_save_path.with_suffix('').with_name(model_save_path.stem)),
        num_folds=NUM_FOLDS,
        output_dir='/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/p3VAE/plots',
    )


    loop_time_end = time.perf_counter()
    print(f"Run completed in {(loop_time_end - loop_time_start) / 60:.2f} minutes.")