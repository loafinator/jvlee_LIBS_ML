from __future__ import annotations

import sys
import time
import h5py
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Any, Dict

# Add project root setup matching directory layout
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from setup import add_project_root_to_path
add_project_root_to_path(parent_generation=1)
from utils import get_h5_ds


# =============================================================================
# Global Configuration & File Paths
# =============================================================================
H5_FILE = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/max12_cts_noleak_crossval.h5"
SPLIT_DIR = Path('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/fold_splits_NoExtraDopants')
OUTPUT_DIR = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/conc_to_spec/MLP/cts_plots")

NUM_FOLDS = 7
THRESH = 2.8
SEED = 42
HOST_ELEMENTS = {'Li', 'K', 'Cl'}
EXP_DOPANTS = {'Ce', 'Gd', 'Sm', 'U'}

# CTS MLP Hyperparameters
EPOCHS = 80
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIMS = (512, 2048, 5120)  # Expanding architecture for 10k output reconstruction
DROPOUT = 0.1

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Concentration-to-Spectra (CTS) PyTorch MLP Architecture
# =============================================================================
class PyTorchCTSMLP(nn.Module):
    """Deep MLP for mapping elemental concentrations (7 features) to LIBS spectra (10,000 bins)."""
    def __init__(
        self, 
        in_features: int = 7, 
        out_wavelengths: int = 10000, 
        hidden_dims: tuple[int, ...] = (512, 2048, 5120), 
        dropout: float = 0.1
    ):
        super().__init__()
        layers = []
        in_dim = in_features

        # Expanding hidden layers
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim

        # Final projection to 10,000 spectral wavelength channels
        layers.append(nn.Linear(in_dim, out_wavelengths))
        # Non-negative output activation (LIBS intensities >= 0)
        layers.append(nn.ReLU())
        
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# Helper & Data Functions
# =============================================================================
def load_crossval_data(
    h5_path: str, 
    spike_thresh: float = THRESH,
    exp_dopants: set = EXP_DOPANTS,
    host_elements: set = HOST_ELEMENTS
) -> Dict[str, Any]:
    """Loads dataset into CPU RAM, cleans saturation spikes, filters unmodeled 
    synthetic dopants, and slices target columns to active elements."""
    
    keep_elements = host_elements.union(exp_dopants)

    with h5py.File(h5_path, 'r') as hf:
        y = get_h5_ds(hf, 'y')[:].astype(np.float32)
        X = get_h5_ds(hf, 'X')[:].astype(np.float32)
        is_synth = get_h5_ds(hf, 'is_synthetic')[:]
        is_lockbox = get_h5_ds(hf, 'is_lockbox')[:].astype(bool) if 'is_lockbox' in hf else None
        group_id = get_h5_ds(hf, 'group_id')[:]
        wavelengths = get_h5_ds(hf, 'wavelengths')[:]
        names = [
            n.decode('utf-8') if isinstance(n, bytes) else str(n) 
            for n in get_h5_ds(hf, 'feature_cols')[:]
        ]

    # 1. Identify active columns to keep vs. extra synthetic dopants to drop
    keep_idx = [i for i, name in enumerate(names) if name in keep_elements]
    extra_dopant_idx = [i for i, name in enumerate(names) if name not in keep_elements]

    # 2. Row Filter A: Drop corrupted saturation shots
    row_maxes = y.max(axis=1)
    clean_spike_mask = row_maxes <= spike_thresh

    # 3. Row Filter B: Drop synthetic samples containing non-experimental dopants
    has_extra_dopants = (X[:, extra_dopant_idx] > 0).any(axis=1)
    valid_synth_mask = (is_synth == 0) | (~has_extra_dopants)

    # Combine row filters
    final_row_mask = clean_spike_mask & valid_synth_mask

    # 4. Slice concentrations X (columns) and filter samples (rows)
    X_clean = X[final_row_mask][:, keep_idx]
    y_clean = y[final_row_mask]
    is_synth_clean = is_synth[final_row_mask]
    group_id_clean = group_id[final_row_mask]
    names_clean = [names[i] for i in keep_idx]

    # 5. L2 Normalization on cleaned spectra
    y_tensor = torch.from_numpy(y_clean)
    l2_norms = torch.norm(y_tensor, p=2, dim=1, keepdim=True).clamp(min=1e-08)
    y_tensor = y_tensor / l2_norms

    # Reporting
    n_dropped_spikes = len(y) - np.sum(clean_spike_mask)
    n_dropped_synth = np.sum(clean_spike_mask & ~valid_synth_mask)
    if n_dropped_spikes > 0 or n_dropped_synth > 0:
        print(f"Data loading summary:")
        print(f"  - Dropped {n_dropped_spikes} corrupted saturation shots (max > {spike_thresh}).")
        print(f"  - Filtered {n_dropped_synth} synthetic shots containing non-experimental dopants.")
        print(f"  - Retained {len(y_tensor)} total shots across {len(names_clean)} features: {names_clean}")

    out_dict = dict(
        X=torch.from_numpy(X_clean),
        y=y_tensor,
        is_synth=torch.from_numpy(is_synth_clean).long(),
        group_id=group_id_clean,
        wavelengths=wavelengths,
        names=names_clean,
    )

    if is_lockbox is not None:
        out_dict['is_lockbox'] = is_lockbox[final_row_mask]

    return out_dict


def assign_group_folds(X, group_id, is_synthetic, dopant_idx, elem_names=None,
                       n_splits=5, n_trials=2000, min_groups_per_dopant=1, seed=42):
    """Stratified group fold assignment matching original script logic."""
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


def build_splits_once(data, split_path, num_folds, seed=42):
    """Loads existing pre-computed CV splits or creates them if absent."""
    split_path = Path(split_path).with_suffix('.npz')
    if split_path.exists():
        print(f"Loading cached splits from: {split_path}")
        d = np.load(split_path)
        return d['test_idx'], d['cv_idx'], d['cv_fold_id']

    split_path.parent.mkdir(parents=True, exist_ok=True)
    test_idx = np.where(data['is_lockbox'])[0] if 'is_lockbox' in data else np.array([], dtype=int)
    cv_idx = np.where(~data['is_lockbox'])[0] if 'is_lockbox' in data else np.arange(len(data['X']), dtype=int)

    X_cv = data['X'][cv_idx].numpy()
    is_synth_cv = data['is_synth'][cv_idx]

    host_elements = HOST_ELEMENTS
    dopant_idx = [
        i for i, name in enumerate(data['names']) 
        if name not in host_elements and (X_cv[:, i] > 0).any()
    ]

    cv_fold_id = assign_group_folds(
        X=X_cv, group_id=data['group_id'][cv_idx], is_synthetic=is_synth_cv,
        dopant_idx=dopant_idx, elem_names=data['names'], n_splits=num_folds, seed=seed
    )

    np.savez(split_path, test_idx=test_idx, cv_idx=cv_idx, cv_fold_id=cv_fold_id,
             num_folds=num_folds, seed=seed)
    return test_idx, cv_idx, cv_fold_id


def plot_spectral_overlay_sample(
    wavelengths: np.ndarray, 
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    sample_idx: int = 0, 
    save_path: Path | None = None
):
    """Plots true vs predicted spectrum overlay and residual reconstruction curve."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Upper panel: Spectrum overlay
    ax1.plot(wavelengths, y_true[sample_idx], label="Measured / Target Spectrum", color="black", alpha=0.8, lw=1)
    ax1.plot(wavelengths, y_pred[sample_idx], label="MLP Predicted Spectrum", color="crimson", alpha=0.8, lw=1)
    ax1.set_ylabel("Normalized Intensity (a.u.)", fontsize=10)
    ax1.set_title(f"Concentration-to-Spectra MLP Forward Prediction — Sample #{sample_idx}", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Lower panel: Residual difference
    residual = y_true[sample_idx] - y_pred[sample_idx]
    ax2.plot(wavelengths, residual, label="Residual Error (True - Pred)", color="darkblue", lw=0.8)
    ax2.axhline(0, color="gray", linestyle="--", lw=1)
    ax2.set_xlabel("Wavelength (nm)", fontsize=10)
    ax2.set_ylabel("Error (a.u.)", fontsize=10)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved Spectral Overlay Plot to: {save_path}")
    plt.close()


# =============================================================================
# Main Execution Pipeline
# =============================================================================
def main():
    print(f"=== Starting GPU-Accelerated CTS MLP Forward Pipeline [{DEVICE}] ===")
    t0_start = time.perf_counter()

    # 1. Load Data into CPU RAM
    data = load_crossval_data(H5_FILE)
    split_path = SPLIT_DIR / f'max12_{NUM_FOLDS}_folds_seed{SEED}.npz'
    test_idx, cv_idx, cv_fold_id = build_splits_once(data, split_path, NUM_FOLDS, seed=SEED)

    # INPUTS = Concentrations X (All 7 elements: Li, K, Cl + Ce, Gd, Sm, U)
    # OUTPUTS = Spectra y (10,000 channels)
    X_cv_cpu = data['X'][cv_idx]
    Y_cv_cpu = data['y'][cv_idx]
    X_test_cpu = data['X'][test_idx]
    Y_test_cpu = data['y'][test_idx]

    input_dim = X_cv_cpu.shape[1]        # 7 elemental concentration features
    output_channels = Y_cv_cpu.shape[1]  # 10,000 spectral wavelength bins

    print(f"Dataset Loaded | CV Pool: {len(cv_idx)} samples | Lockbox Test Set: {len(test_idx)} samples")
    print(f"Inputs (Concentrations): {data['names']} | Target Output: {output_channels} Spectral Channels")

    # Containers for Out-Of-Fold and Lockbox predictions
    oof_preds_np = np.zeros((len(cv_idx), output_channels), dtype=np.float32)
    lockbox_preds_gpu = torch.zeros((NUM_FOLDS, len(test_idx), output_channels), device=DEVICE)

    # 2. K-Fold Cross-Validation Training Loop
    print("\n--- Training Deep CTS MLP across K Folds ---")
    
    for k in range(NUM_FOLDS):
        t_fold_start = time.perf_counter()
        train_mask = cv_fold_id != k
        val_mask = cv_fold_id == k

        # DataLoader Setup (X = Concentrations, Y = Spectra)
        train_ds = TensorDataset(X_cv_cpu[train_mask], Y_cv_cpu[train_mask])
        val_ds = TensorDataset(X_cv_cpu[val_mask], Y_cv_cpu[val_mask])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

        # Fresh CTS MLP Instance
        model = PyTorchCTSMLP(
            in_features=input_dim, 
            out_wavelengths=output_channels, 
            hidden_dims=HIDDEN_DIMS, 
            dropout=DROPOUT
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        best_model_weights: dict[str, Any] | None = None

        for epoch in range(1, EPOCHS + 1):
            model.train()
            train_loss = 0.0

            for bx, by in train_loader:
                bx, by = bx.to(DEVICE, non_blocking=True), by.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()
                pred_spectra = model(bx)
                loss = criterion(pred_spectra, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * bx.size(0)

            train_loss /= len(train_ds)

            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(DEVICE, non_blocking=True), by.to(DEVICE, non_blocking=True)
                    pred_spectra = model(bx)
                    val_loss += criterion(pred_spectra, by).item() * bx.size(0)

            val_loss /= len(val_ds)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = {k_w: v_w.cpu().clone() for k_w, v_w in model.state_dict().items()}

        # Restore best model parameters
        if best_model_weights is not None:
            model.load_state_dict(best_model_weights)
        else:
            print("Warning: best_model_weights was never updated during training.")
        model.eval()

        # Generate Out-Of-Fold Spectral Predictions
        fold_oof_preds = []
        with torch.no_grad():
            for bx, _ in val_loader:
                bx = bx.to(DEVICE, non_blocking=True)
                fold_oof_preds.append(model(bx).cpu())
        oof_preds_np[val_mask] = torch.cat(fold_oof_preds, dim=0).numpy()

        # Generate Lockbox Test Predictions for this fold model
        X_test_gpu = X_test_cpu.to(DEVICE)
        with torch.no_grad():
            lockbox_preds_gpu[k] = model(X_test_gpu)

        dt = time.perf_counter() - t_fold_start
        print(f"Fold {k + 1}/{NUM_FOLDS} Complete | Best Val Spectral MSE: {best_val_loss:.6e} | Time: {dt:.2f}s")

        del model, optimizer, scheduler, X_test_gpu
        torch.cuda.empty_cache()

    # 3. Final Lockbox Prediction Ensembling
    lockbox_pred_avg = lockbox_preds_gpu.mean(dim=0).cpu().numpy()

    # 4. Compute Spectral Metrics (Experimental Only)
    exp_mask = cv_fold_id >= 0
    true_cv_spectra = Y_cv_cpu[exp_mask].numpy()
    pred_cv_spectra = oof_preds_np[exp_mask]

    true_test_spectra = Y_test_cpu.numpy()
    pred_test_spectra = lockbox_pred_avg

    # Spectral Reconstruction Performance
    oof_mse = mean_squared_error(true_cv_spectra, pred_cv_spectra)
    oof_mae = mean_absolute_error(true_cv_spectra, pred_cv_spectra)
    
    test_mse = mean_squared_error(true_test_spectra, pred_test_spectra)
    test_mae = mean_absolute_error(true_test_spectra, pred_test_spectra)

    print("\n" + "=" * 80)
    print(" Concentration-to-Spectra (CTS) MLP Performance Summary")
    print("=" * 80)
    print(f" Out-Of-Fold (OOF)  | Spectral MSE: {oof_mse:.6e} | Spectral MAE: {oof_mae:.6f} a.u.")
    print(f" Lockbox Test Set   | Spectral MSE: {test_mse:.6e} | Spectral MAE: {test_mae:.6f} a.u.")
    print("=" * 80 + "\n")

    # 5. Plot Sample Reconstruction Overlays
    sample_idx = np.random.randint(0, len(true_test_spectra))
    overlay_path = OUTPUT_DIR / f"cts_mlp_sample_{sample_idx}_reconstruction_{TIMESTAMP}.png"
    plot_spectral_overlay_sample(
        wavelengths=data['wavelengths'], 
        y_true=true_test_spectra, 
        y_pred=pred_test_spectra, 
        sample_idx=sample_idx, 
        save_path=overlay_path
    )

    print(f"CTS Pipeline completed in {(time.perf_counter() - t0_start):.2f} seconds.")


if __name__ == "__main__":
    main()