from __future__ import annotations

import sys
import time
import h5py
import torch
import hashlib
import copy
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Any, Dict

# Add project root setup matching your directory layout
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from setup import add_project_root_to_path
add_project_root_to_path(parent_generation=1)
from utils import get_h5_ds


# =============================================================================
# Global Configuration & File Paths
# =============================================================================
H5_FILE = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/exp-syn_processed-data_20261110b.h5"
SPLIT_DIR = Path('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/fold_splits/processed-data_20261110b')
OUTPUT_DIR = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/spec_to_conc/MLP_reverse/plots/processed-data_20261110b")

NUM_FOLDS = 7
SEEDS = [42]
MAX_WT_PERCENT = 12.0
SEED = 42
HOST_ELEMENTS = {'Li', 'K', 'Cl'}
TARGET_NAMES = ['Ce', 'Gd', 'Sm', 'U']
DOPANT_NAMES = {'Ce', 'Gd', 'Sm', 'U'}

# Reverse MLP Hyperparameters
EPOCHS = 150
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIMS = [64, 256, 1024, 4096]
DROPOUT = 0.1

PATIENCE = 15
MIN_DELTA = 1e-06
LOG_EVERY = 1

# Speed options
USE_AMP = True
USE_COMPILE = False
GPU_DATA_BUDGET_FRAC = 0.6

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAINED_MODEL_DIR = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/spec_to_conc/MLP_reverse/trained_models")
TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# PyTorch Reverse MLP Architecture (Concentrations -> Spectra)
# =============================================================================
class PyTorchReverseMLP(nn.Module):
    """Expanding MLP for LIBS reverse spectral generation (y -> X)."""
    def __init__(
        self,
        in_features: int = 4,
        out_features: int = 10000,
        hidden_dims: list[int] = [64, 256, 1024, 4096],
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = in_features

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        # Final linear projection to 10,000 spectral wavelength channels
        layers.append(nn.Linear(prev_dim, out_features))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.normal_(last.weight, std=0.001)

    @torch.no_grad()
    def set_output_bias(self, target_mean_spectrum: torch.Tensor):
        """Sets final layer bias to the dataset's mean spectrum for rapid convergence."""
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            last.bias.copy_(target_mean_spectrum.to(last.bias.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# Helper & Data Functions
# =============================================================================
def compute_dataset_fingerprint(data: dict) -> str:
    """Computes a fast cryptographic hash of targets, groups, and lockbox flags."""
    h = hashlib.blake2b(digest_size=16)
    h.update(data['y'].numpy().tobytes())
    h.update(np.ascontiguousarray(data['group_id']).tobytes())
    h.update(np.ascontiguousarray(data['is_lockbox']).tobytes())
    return h.hexdigest()

def load_crossval_data(h5_path: str) -> dict:
    """Loads pre-cleaned dataset as float32 tensors stored in System CPU RAM."""
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
            raise KeyError("HDF5 file missing 'is_lockbox'.")
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

def assign_group_folds(X, group_id, is_synthetic, dopant_idx, elem_names=None,
                       n_splits=5, n_trials=2000, min_groups_per_dopant=1, seed=42):
    """Stratified group fold assignment matching original P3VAE script."""
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
    fold_id[exp] = best_fog[inv]  # type: ignore
    return fold_id

def build_splits_for_seed(data, split_dir, num_folds, seed):
    """Loads existing pre-computed CV splits for a given seed or creates them if absent or stale."""
    split_dir = Path(split_dir)
    split_path = split_dir / f'max12_{num_folds}_folds_seed{seed}.npz'
    current_fp = compute_dataset_fingerprint(data)
    
    if split_path.exists():
        d = np.load(split_path)
        cached_fp = str(d['fingerprint']) if 'fingerprint' in d else None
        if cached_fp == current_fp and d.get('num_folds', None) == num_folds:
            return d['test_idx'], d['cv_idx'], d['cv_fold_id']
        else:
            print(f"⚠️ Stale or un-fingerprinted split cache detected in {split_path.name}. Recomputing folds...")

    split_dir.mkdir(parents=True, exist_ok=True)
    test_idx = np.where(data['is_lockbox'])[0]
    cv_idx = np.where(~data['is_lockbox'])[0]

    y_cv = data['y'][cv_idx].numpy()
    is_synth_cv = data['is_synth'][cv_idx]

    host_elements = {'Li', 'K', 'Cl'}
    dopant_idx = [
        i for i, name in enumerate(data['names']) 
        if name not in host_elements and (y_cv[:, i] > 0).any()
    ]

    cv_fold_id = assign_group_folds(
        X=y_cv, group_id=data['group_id'][cv_idx], is_synthetic=is_synth_cv,
        dopant_idx=dopant_idx, elem_names=data['names'], n_splits=num_folds, seed=seed
    )

    np.savez(
        split_path, test_idx=test_idx, cv_idx=cv_idx, cv_fold_id=cv_fold_id,
        num_folds=num_folds, seed=seed, fingerprint=current_fp
    )
    return test_idx, cv_idx, cv_fold_id

def plot_spectral_reconstruction(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    wavelengths: np.ndarray,
    y_concentrations: np.ndarray,
    elem_names: list[str],
    save_path: Path | None = None,
    sample_nums: list[int] | None = None,
    model_name: str = "Reverse MLP"
):
    """Plots True vs Predicted spectra across wavelength grid for sample spectra."""
    sample_indices: list[int] = []
    sample_tags: list[str] = []

    # Identify dopant columns (fallback to non-host elements if DOPANT_NAMES isn't in scope)
    if 'DOPANT_NAMES' in globals():
        dop_cols = [j for j, n in enumerate(elem_names) if n in globals()['DOPANT_NAMES']]
    else:
        dop_cols = [j for j, n in enumerate(elem_names) if n not in {'Li', 'K', 'Cl'}]

    if not dop_cols:
        dop_cols = list(range(y_concentrations.shape[1]))

    Yd = y_concentrations[:, dop_cols]

    if sample_nums is not None and len(sample_nums) > 0:
        # Explicit sample numbers supplied
        sample_indices = [int(idx) for idx in sample_nums if 0 <= idx < len(x_true)]
        sample_tags = [f"Sample {idx}" for idx in sample_indices]
    else:
        # Auto-select representative samples (RESTRICTED TO PURE SINGLE-DOPANT SAMPLES)
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

        # Complex mixture sample (highest normalized concentration entropy)
        row_sums = Yd.sum(axis=1, keepdims=True)
        conc_norm = np.divide(Yd, row_sums, out=np.zeros_like(Yd, dtype=float), where=row_sums > 0)
        log_conc = np.zeros_like(conc_norm)
        mask = conc_norm > 0
        log_conc[mask] = np.log(conc_norm[mask])
        entropy = -np.sum(conc_norm * log_conc, axis=1)
        sample_indices.append(int(np.argmax(entropy)))
        sample_tags.append("Complex Mixture")

        # Trace concentration sample
        total_conc = Yd.sum(axis=1)
        pos = np.where(total_conc > 0)[0]
        trace_idx = int(pos[np.argmin(total_conc[pos])]) if len(pos) else int(np.argmin(total_conc))
        sample_indices.append(trace_idx)
        sample_tags.append("Trace Concentration")

    n_plots = len(sample_indices)
    if n_plots == 0:
        print("⚠️ No valid sample indices available to plot.")
        return

    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2.5 * n_plots), sharex=True)
    axes_flat = np.atleast_1d(axes)

    for plot_idx, (sample_idx, tag) in enumerate(zip(sample_indices, sample_tags)):
        ax = axes_flat[plot_idx]
        xt = x_true[sample_idx]
        xp = x_pred[sample_idx]
        conc_str = ", ".join(
            f"{elem_names[j]}: {y_concentrations[sample_idx, j]:.2f}%"
            for j in dop_cols if y_concentrations[sample_idx, j] > 0
        )
        if not conc_str:
            conc_str = "0.00%"

        r2 = r2_score(xt, xp)
        rmse = np.sqrt(mean_squared_error(xt, xp))

        ax.plot(wavelengths, xt, label="True Spectrum", color="black", alpha=0.7, lw=1.0)
        ax.plot(wavelengths, xp, label=f"{model_name} Pred", color="crimson", alpha=0.8, lw=0.9, linestyle="--")
        ax.set_title(f"[{tag}] Sample {sample_idx} ({conc_str}) | $R^2$: {r2:.4f} | RMSE: {rmse:.5f}", fontsize=9, fontweight="bold")
        ax.set_ylabel("Norm Intensity", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)
        if plot_idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes_flat[-1].set_xlabel("Wavelength (nm)", fontsize=9)
    plt.suptitle(f"{model_name} Spectral Reconstruction ($Y \\rightarrow X$)", fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved Reverse Spectral Overlay Plot to: {save_path}")
    plt.close()


# =============================================================================
# Fast GPU-resident Training Utilities
# =============================================================================
def stash(X: torch.Tensor, name: str) -> torch.Tensor:
    if DEVICE.type != "cuda":
        return X
    X = X.to(torch.float16)
    nbytes = X.numel() * X.element_size()
    free, _ = torch.cuda.mem_get_info()
    if nbytes < GPU_DATA_BUDGET_FRAC * free:
        print(f"  [{name}] {nbytes / 1e9:.2f} GB (fp16) -> GPU")
        return X.to(DEVICE)
    print(f"  [{name}] {nbytes / 1e9:.2f} GB -> pinned CPU memory")
    return X.pin_memory()

def gather_x(Xs: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    bx = Xs[idx]
    if bx.device != DEVICE:
        bx = bx.to(DEVICE, non_blocking=True)
    return bx.float()

def amp_setup():
    if not (USE_AMP and DEVICE.type == "cuda"):
        return None, None
    if torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16, None
    return torch.float16, torch.amp.GradScaler("cuda")  # type: ignore

@torch.no_grad()
def predict(model: nn.Module, Ys: torch.Tensor, idx: torch.Tensor, bs: int = 1024) -> torch.Tensor:
    model.eval()
    amp_dtype, _ = amp_setup()
    out = []
    for i in range(0, idx.numel(), bs):
        by = gather_x(Ys, idx[i:i + bs])
        with torch.autocast(DEVICE.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out.append(model(by).float())
    return torch.cat(out)

def fit(
    model: nn.Module, Ys: torch.Tensor, Xs_target: torch.Tensor, tr_idx: torch.Tensor, va_idx: torch.Tensor,
    epochs: int, seed: int, verbose: bool = True, enable_early_stopping: bool = True, log_every: int = LOG_EVERY
):
    n = tr_idx.numel()
    steps = n // BATCH_SIZE
    if steps == 0:
        raise ValueError(f"Only {n} training rows for BATCH_SIZE={BATCH_SIZE}.")

    g = torch.Generator(device=tr_idx.device)
    g.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    
    amp_dtype, scaler = amp_setup()
    hist = dict(epoch=[], train=[], val=[], stopped_epoch=epochs)

    best_val_loss = float("inf")
    best_model_weights = None
    patience_counter = 0
    t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        t_ep = time.perf_counter()
        model.train()
        perm = tr_idx[torch.randperm(n, device=tr_idx.device, generator=g)]
        running = torch.zeros((), device=DEVICE)

        for s in range(steps):
            idx = perm[s * BATCH_SIZE:(s + 1) * BATCH_SIZE]
            by = gather_x(Ys, idx)
            bx_target = gather_x(Xs_target, idx)

            with torch.autocast(DEVICE.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                pred_x = model(by)
            loss = F.mse_loss(pred_x.float(), bx_target)

            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            running += loss.detach()

        if ep == 1 or ep % log_every == 0 or ep == epochs:
            tr_loss = (running / steps).item()
            va_pred_x = predict(model, Ys, va_idx)
            va_target_x = gather_x(Xs_target, va_idx)
            va_loss = F.mse_loss(va_pred_x, va_target_x).item()

            if enable_early_stopping:
                sched.step(va_loss)

            hist["epoch"].append(ep) # type: ignore
            hist["train"].append(tr_loss) # type: ignore
            hist["val"].append(va_loss) # type: ignore

            if verbose:
                print(f"    ep {ep:3d}/{epochs} | train Spectral MSE {tr_loss:.6f} | held-out MSE {va_loss:.6f} | lr {opt.param_groups[0]['lr']:.2e}")

            if enable_early_stopping:
                if va_loss < (best_val_loss - MIN_DELTA):
                    best_val_loss = va_loss
                    best_model_weights = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= PATIENCE:
                    if verbose:
                        print(f"    --> Early stopping triggered at epoch {ep}. Best Val Spectral MSE: {best_val_loss:.6f}")
                    hist["stopped_epoch"] = ep  # type: ignore
                    break

    if enable_early_stopping and best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    dt = time.perf_counter() - t0
    hist["seconds"] = dt  # type: ignore
    hist["samples_per_sec"] = len(hist["epoch"]) * steps * BATCH_SIZE / dt # type: ignore
    return hist

def new_model(in_dim: int, out_dim: int, init_bias_spectrum: torch.Tensor) -> nn.Module:
    model = PyTorchReverseMLP(
        in_features=in_dim, out_features=out_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT
    ).to(DEVICE)
    model.set_output_bias(init_bias_spectrum)
    if USE_COMPILE:
        model = torch.compile(model)
    return model  # type: ignore

def fold_indices(fold_id_t: torch.Tensor, k: int):
    tr_idx = torch.nonzero(fold_id_t != k).squeeze(1)
    va_idx = torch.nonzero(fold_id_t == k).squeeze(1)
    return tr_idx, va_idx


# =============================================================================
# Main Execution Pipeline
# =============================================================================
def main(
        h5_file: Path | str = H5_FILE,
        split_dir: Path | str = SPLIT_DIR,
        output_dir: Path | str = OUTPUT_DIR,
        include_synthetic: bool = False,
        seeds: list[int] = SEEDS,
        dataset_tag: str = '20261110b',
        target_names: list[str] | None = None,
        include_lbx: bool = False,
        sample_nums: list[int] | None = [352,43999,21552,19216,0,57626]
):
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"=== Starting Fast Reverse MLP Pipeline (Concentrations -> Spectra) [{DEVICE}] ===")
    print(f"Dataset: {dataset_tag} | K-Fold: {NUM_FOLDS} Folds x {len(seeds)} Seeds")
    t0_start = time.perf_counter()

    h5_file = str(h5_file)
    split_dir = Path(split_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    data = load_crossval_data(h5_file)
    n_synth = (data['is_synth'] == 1).sum()
    print(f"Loaded {len(data['X'])} total shots | Real: {len(data['X']) - n_synth} | Synthetic: {n_synth}")

    seed_splits = {}
    for seed in seeds:
        test_idx, cv_idx, cv_fold_id = build_splits_for_seed(data, split_dir, NUM_FOLDS, seed=seed)
        seed_splits[seed] = (test_idx, cv_idx, cv_fold_id)

    test_idx = seed_splits[seeds[0]][0]
    cv_idx = seed_splits[seeds[0]][1]

    if target_names is None:
        target_names = TARGET_NAMES
    target_indices = [data['names'].index(name) for name in target_names if name in data['names']]
    actual_target_names = [data['names'][i] for i in target_indices]
    
    in_dim = len(target_indices)      # Input: 4 concentrations
    out_dim = data['X'].shape[1]      # Target: 10,000 wavelengths

    # Standardize inputs (y normalized to [0, 1]) and spectra targets (X)
    Y_cv_cpu = data['y'][cv_idx][:, target_indices] / MAX_WT_PERCENT
    X_cv_cpu = data['X'][cv_idx]
    
    Y_test_cpu = data['y'][test_idx][:, target_indices] / MAX_WT_PERCENT if include_lbx else None
    X_test_cpu = data['X'][test_idx] if include_lbx else None
    del data['X']

    # 2. Stage Data to GPU
    Ys_cv = stash(Y_cv_cpu, "CV Concentrations")
    Xs_cv = stash(X_cv_cpu, "CV Spectra")
    fold_t = torch.from_numpy(seed_splits[seeds[0]][2].astype(np.int64)).to(Ys_cv.device)

    Ys_test, Xs_test, te_idx = None, None, None
    if include_lbx:
        Ys_test = stash(Y_test_cpu, "Lockbox Concentrations")  # type: ignore
        Xs_test = stash(X_test_cpu, "Lockbox Spectra")          # type: ignore
        te_idx = torch.arange(len(test_idx), device=Ys_test.device)

    del X_cv_cpu, X_test_cpu

    # 3. K-Fold Training Loop
    oof_preds_np = np.zeros((len(cv_idx), out_dim), dtype=np.float32)
    lockbox_preds = torch.zeros((NUM_FOLDS, len(test_idx), out_dim), device="cpu")

    print("\n--- Training Reverse MLP across K Folds ---")
    for k in range(NUM_FOLDS):
        tr_idx, va_idx = fold_indices(fold_t, k)
        
        if not include_synthetic:
            tr_exp_mask = torch.from_numpy(
                data['is_synth'][cv_idx][tr_idx.cpu().numpy()] == 0
            ).to(tr_idx.device)
            tr_idx = tr_idx[tr_exp_mask]

        print(f"Fold {k + 1}/{NUM_FOLDS} | train rows {tr_idx.numel()} | held-out rows {va_idx.numel()}")

        torch.manual_seed(SEED + k)
        model = new_model(in_dim, out_dim, gather_x(Xs_cv, tr_idx).mean(dim=0))
        hist = fit(model, Ys_cv, Xs_cv, tr_idx, va_idx, EPOCHS, seed=SEED + k, enable_early_stopping=True)

        oof = predict(model, Ys_cv, va_idx).clamp(min=0).cpu().numpy()
        oof_preds_np[va_idx.cpu().numpy()] = oof
        if include_lbx:
            lockbox_preds[k] = predict(model, Ys_test, te_idx).clamp(min=0).cpu()  # type: ignore

        torch.save({
            'fold': k,
            'model_state_dict': model.state_dict(),
            'in_dim': in_dim,
            'out_dim': out_dim,
            'stopped_epoch': hist["stopped_epoch"],
            'best_val_mse': min(hist["val"]), # type: ignore
        }, TRAINED_MODEL_DIR / f"mlp_reverse_{dataset_tag}_{TIMESTAMP}_fold{k + 1}.pt")

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # 4. Evaluate & Print Performance
    exp_mask = seed_splits[seeds[0]][2] >= 0
    x_true_cv = gather_x(Xs_cv, torch.from_numpy(np.where(exp_mask)[0]).to(Xs_cv.device)).cpu().numpy()
    x_pred_cv = oof_preds_np[exp_mask]
    y_conc_cv = data['y'][cv_idx][exp_mask][:, target_indices].numpy()

    oof_rmse = np.sqrt(mean_squared_error(x_true_cv, x_pred_cv))
    oof_r2 = r2_score(x_true_cv.ravel(), x_pred_cv.ravel())
    print(f"\n✅ OOF Reverse MLP Performance | Spectral RMSE: {oof_rmse:.5f} | Overall R^2: {oof_r2:.4f}")

    if include_lbx:
        lockbox_pred_avg = lockbox_preds.mean(dim=0).clamp(min=0).cpu().numpy()
        x_true_test = gather_x(Xs_test, te_idx).cpu().numpy()  # type: ignore
        y_conc_test = (Ys_test.cpu().numpy()) * MAX_WT_PERCENT  # type: ignore

        lockbox_rmse = np.sqrt(mean_squared_error(x_true_test, lockbox_pred_avg))
        lockbox_r2 = r2_score(x_true_test.ravel(), lockbox_pred_avg.ravel())
        print(f"✅ Lockbox Reverse MLP Performance | Spectral RMSE: {lockbox_rmse:.5f} | Overall R^2: {lockbox_r2:.4f}")

    # 5. Save Spectral Overlay Plots
    oof_plot_path = output_dir / f"gpu_mlp_reverse_oof_spectra_{dataset_tag}_{TIMESTAMP}.png"
    lockbox_plot_path = output_dir / f"gpu_mlp_reverse_lockbox_spectra_{dataset_tag}_{TIMESTAMP}.png"

    plot_spectral_reconstruction(
        x_true=x_true_cv,
        x_pred=x_pred_cv,
        wavelengths=data['wavelengths'],
        y_concentrations=y_conc_cv,
        elem_names=actual_target_names,
        save_path=oof_plot_path,
        sample_nums=sample_nums
    )

    if include_lbx:
        plot_spectral_reconstruction(
            x_true=x_true_test,  # type: ignore
            x_pred=lockbox_pred_avg,  # type: ignore
            wavelengths=data['wavelengths'],
            y_concentrations=y_conc_test,  # type: ignore
            elem_names=actual_target_names,
            save_path=lockbox_plot_path,
            sample_nums=sample_nums
        )

    cache_path = output_dir / f"mlp_reverse_prediction_{dataset_tag}.npz"
    save_dict = dict(
        x_true_cv=x_true_cv,
        x_pred_cv=x_pred_cv,
        y_conc_cv=y_conc_cv,
        wavelengths=data['wavelengths'],
        target_names=np.array(actual_target_names)
    )
    if include_lbx:
        save_dict["x_true_test"] = x_true_test        # type: ignore
        save_dict["x_pred_test"] = lockbox_pred_avg   # type: ignore
        save_dict["y_conc_test"] = y_conc_test        # type: ignore

    np.savez(cache_path, **save_dict)
    print(f"Cached reverse MLP predictions to: {cache_path}")

    print(f"Reverse MLP Pipeline completed in {(time.perf_counter() - t0_start):.2f} seconds.")


if __name__ == "__main__":
    main(
        h5_file="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/exp-syn_processed-data_20261110b.h5",
        split_dir="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/fold_splits/processed-data_20261110b",
        output_dir=Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/plots/MLP/processed-data_20261110b/exp"),
        include_synthetic=False,
        seeds=SEEDS,
        dataset_tag='20261110b',
        target_names=['Ce', 'Gd', 'Sm', 'U'],
        include_lbx=False,
        sample_nums=None
    )

    main(
        h5_file="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/exp-syn_processed-data_20261110b.h5",
        split_dir="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/fold_splits/processed-data_20261110b",
        output_dir=Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/plots/MLP/processed-data_20261110b/exp-syn"),
        include_synthetic=True,
        seeds=SEEDS,
        dataset_tag='20261110b',
        target_names=['Ce', 'Gd', 'Sm', 'U'],
        include_lbx=False,
        sample_nums=None
    )
