from __future__ import annotations

import sys
import time
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Any, Dict

# Project Setup & Path Corrections
project_root = Path(__file__).resolve().parents[4]  # Points to .../LIBS/
p3vae_006_dir = project_root / "LIBS" / "p3VAE" / "006"        # Points to .../LIBS/p3VAE/006/

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(p3vae_006_dir))

from setup import add_project_root_to_path
add_project_root_to_path(parent_generation=1)
from utils import get_h5_ds

# Import Model Architectures & Datasets
from p3vae_006 import PhysicsInformedVAE, SpectralScaler, LIBSSpectraDataset    # type: ignore


# =============================================================================
# Global Configuration & Model Checkpoint Paths
# =============================================================================
H5_FILE = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/max12_cts_noleak_crossval.h5"
SPLIT_DIR = Path('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/fold_splits_NoExtraDopants')
MASTER_TABLE_PATH = Path('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/master_line_table.csv').resolve()
OUTPUT_DIR = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/poster_plots")

# -----------------------------------------------------------------------------
# TOGGLE FLAGS FOR TROUBLESHOOTING
# -----------------------------------------------------------------------------
ENABLE_P3VAE = True   # Set to True when P3VAE training finishes; False for troubleshooting PLS & CNN

NUM_FOLDS = 7
THRESH = 2.8
MAX_WT_PERCENT = 12.0
SEED = 42
HOST_ELEMENTS = {'Li', 'K', 'Cl'}
TARGET_DOPANTS = ['Ce', 'Gd', 'Sm', 'U']  # The 4 rows in our 4x3 poster grid

# Model Checkpoint Prefixes
P3VAE_MODEL_PREFIX = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/p3VAE/trained_models/best_p3vae6_model_20261024_162246"
CNN_MODEL_PREFIX = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/spec_to_conc/CNN_1D/trained_models/cnn1d_20261024_180000"

PLS_COMPONENTS = 15
EVAL_MODE = "lockbox"  # Options: 'lockbox' (Ensemble Test Set) or 'oof' (Out-Of-Fold Validation)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Helper Utilities & Fold Checkpoint Offset Detection
# =============================================================================
def detect_fold_offset(prefix: str, num_folds: int = NUM_FOLDS) -> int:
    """
    Detects whether checkpoints follow 0-indexed (_fold0) or 1-indexed (_fold1) naming convention,
    and asserts that all expected folds exist in a contiguous sequence.
    """
    p = Path(prefix)
    matching_files = list(p.parent.glob(p.name + "_fold*.pt"))
    if not matching_files:
        raise FileNotFoundError(f"No checkpoint files matching '{p.name}_fold*.pt' in directory {p.parent}")
    
    found = sorted(int(f.stem.rsplit("fold", 1)[1]) for f in matching_files)
    assert len(found) == num_folds, (
        f"Expected {num_folds} fold checkpoints for '{p.name}', but found {len(found)}: {found}"
    )
    expected_sequence = list(range(found[0], found[0] + num_folds))
    assert found == expected_sequence, (
        f"Fold checkpoint numbering for '{p.name}' is non-contiguous. Found {found}, expected {expected_sequence}"
    )
    return found[0]


# =============================================================================
# 1D CNN Architecture (Matching 1DCNN_benchmark.py)
# =============================================================================
class PyTorch1DCNN(nn.Module):
    def __init__(self, in_channels: int = 1, out_features: int = 4, base_filters: int = 32, dropout: float = 0.2, pool_bins: int = 16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(base_filters, base_filters * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base_filters * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(base_filters * 2, base_filters * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(base_filters * 4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(base_filters * 4, base_filters * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(base_filters * 8),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(pool_bins),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filters * 8 * pool_bins, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.fc(self.features(x))


# =============================================================================
# PyTorch PLS Implementation (In-Place Standardization Memory Fix)
# =============================================================================
class PyTorchPLS:
    def __init__(self, n_components: int = 15):
        self.n_components = n_components
        self.mean_X, self.std_X, self.mean_Y, self.B = None, None, None, None

    def fit(self, X: torch.Tensor, Y: torch.Tensor, inplace: bool = False):
        self.mean_X = X.mean(dim=0, keepdim=True)
        self.std_X = X.std(dim=0, keepdim=True).clamp(min=1e-6)
        self.mean_Y = Y.mean(dim=0, keepdim=True)

        X_std = X if inplace else X.clone()
        X_std.sub_(self.mean_X).div_(self.std_X)
        Y0 = Y - self.mean_Y

        S = torch.matmul(X_std.T, Y0)
        P, M = X.shape[1], Y.shape[1]

        R = torch.zeros((P, self.n_components), dtype=X.dtype, device=X.device)
        Q = torch.zeros((M, self.n_components), dtype=X.dtype, device=X.device)
        V = torch.zeros((P, self.n_components), dtype=X.dtype, device=X.device)

        for a in range(self.n_components):
            U, _, _ = torch.linalg.svd(S, full_matrices=False)
            r_a = U[:, 0]
            t_a = torch.matmul(X_std, r_a)
            t_a = t_a - t_a.mean()
            norm_t = torch.linalg.norm(t_a)
            t_a, r_a = t_a / norm_t, r_a / norm_t

            p_a = torch.matmul(X_std.T, t_a)
            q_a = torch.matmul(Y0.T, t_a)

            v_a = p_a.clone()
            if a > 0:
                V_prev = V[:, :a]
                v_a = v_a - torch.matmul(V_prev, torch.matmul(V_prev.T, p_a))
            v_a = v_a / torch.linalg.norm(v_a)

            S = S - v_a.unsqueeze(1) @ (v_a.unsqueeze(0) @ S)
            R[:, a], Q[:, a], V[:, a] = r_a, q_a, v_a

        self.B = torch.matmul(R, Q.T)
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        X0 = (X - self.mean_X) / self.std_X
        return torch.matmul(X0, self.B) + self.mean_Y   # type: ignore


# =============================================================================
# Data Loading & Batched Inference Utilities
# =============================================================================
def load_crossval_data(h5_path: str, spike_thresh: float = THRESH) -> Dict[str, Any]:
    keep_elements = HOST_ELEMENTS.union(set(TARGET_DOPANTS))

    with h5py.File(h5_path, 'r') as hf:
        y_ds = get_h5_ds(hf, 'y')
        y_full = y_ds[:] if y_ds.dtype == np.float32 else y_ds[:].astype(np.float32)
        X_full = get_h5_ds(hf, 'X')[:].astype(np.float32)
        is_synth = get_h5_ds(hf, 'is_synthetic')[:]
        is_lockbox = get_h5_ds(hf, 'is_lockbox')[:].astype(bool) if 'is_lockbox' in hf else None
        group_id = get_h5_ds(hf, 'group_id')[:]
        wavelengths = get_h5_ds(hf, 'wavelengths')[:]
        names = [n.decode('utf-8') if isinstance(n, bytes) else str(n) for n in get_h5_ds(hf, 'feature_cols')[:]]

    keep_idx = [i for i, name in enumerate(names) if name in keep_elements]
    extra_dopant_idx = [i for i, name in enumerate(names) if name not in keep_elements]

    clean_spike_mask = y_full.max(axis=1) <= spike_thresh
    has_extra_dopants = (X_full[:, extra_dopant_idx] > 0).any(axis=1)
    valid_synth_mask = (is_synth == 0) | (~has_extra_dopants)
    final_row_mask = clean_spike_mask & valid_synth_mask

    # Slicing creates masked copies; free unmasked raw arrays immediately
    y = torch.from_numpy(y_full[final_row_mask])
    del y_full

    X = torch.from_numpy(X_full[final_row_mask][:, keep_idx])
    del X_full

    print(f"📊 Dataset Loaded | Masked y shape: {y.shape} | Base Memory: {y.element_size() * y.nelement() / 1e9:.2f} GB")

    return dict(
        X=X,
        y=y,
        is_synth=torch.from_numpy(is_synth[final_row_mask]).long(),
        group_id=group_id[final_row_mask],
        wavelengths=wavelengths,
        names=[names[i] for i in keep_idx],
        keep_idx=keep_idx,
        is_lockbox=is_lockbox[final_row_mask] if is_lockbox is not None else None
    )


def predict_batched(model: nn.Module, X: torch.Tensor, norms: torch.Tensor, idx: np.ndarray | torch.Tensor, scale: float, bs: int = 256) -> np.ndarray:
    """
    Batched inference for 1D CNN with L2 normalization and sqrt(P) RMS scaling.
    """
    out = []
    idx_t = torch.as_tensor(idx, dtype=torch.long)
    with torch.no_grad():
        for i in range(0, len(idx_t), bs):
            batch_idx = idx_t[i:i + bs]
            xb = (X[batch_idx] / norms[batch_idx]) * scale
            xb = xb.to(DEVICE)
            out.append(model(xb).clamp(min=0.0).cpu())
    return torch.cat(out, dim=0).numpy()


def get_model_predictions(data, cv_fold_id, test_idx, cv_idx):
    dopant_indices = [i for i, name in enumerate(data['names']) if name in TARGET_DOPANTS]
    n_samples = len(test_idx) if EVAL_MODE == "lockbox" else len(cv_idx)
    
    y_true = (data['X'][test_idx][:, dopant_indices].numpy() if EVAL_MODE == "lockbox" 
              else data['X'][cv_idx][:, dopant_indices].numpy()) * MAX_WT_PERCENT

    preds = {
        'PLS': np.zeros((n_samples, len(dopant_indices)), dtype=np.float32),
        '1D CNN': np.zeros((n_samples, len(dopant_indices)), dtype=np.float32),
        'P3VAE': np.zeros((n_samples, len(dopant_indices)), dtype=np.float32)
    }

    # Precalculate L2 norms for spectral normalization
    norms = torch.norm(data['y'], p=2, dim=1, keepdim=True).clamp(min=1e-8)

    # -------------------------------------------------------------------------
    # 1. Memory-Efficient PyTorch PLS Predictions
    # -------------------------------------------------------------------------
    print("Generating GPU PLS predictions...")

    def rows(idx):
        idx_t = torch.as_tensor(idx, dtype=torch.long)
        x = data['y'][idx_t]      # Fancy indexing already creates a copy!
        x.div_(norms[idx_t])       # In-place L2 normalization
        return x.to(DEVICE)

    Y_all = data['X'][:, dopant_indices].to(DEVICE)
    cv_idx_t = torch.as_tensor(cv_idx, dtype=torch.long)

    if EVAL_MODE == "lockbox":
        X_te = rows(test_idx)
        pls_folds = torch.zeros((NUM_FOLDS, len(test_idx), len(dopant_indices)), device=DEVICE)
        for k in range(NUM_FOLDS):
            tr_idx = cv_idx_t[cv_fold_id != k]
            pls = PyTorchPLS(n_components=PLS_COMPONENTS).fit(rows(tr_idx), Y_all[tr_idx], inplace=True)
            pls_folds[k] = pls.predict(X_te)
        preds['PLS'] = torch.clamp(pls_folds.mean(dim=0), min=0.0).cpu().numpy() * MAX_WT_PERCENT
    else:
        pls_oof = torch.zeros((len(cv_idx), len(dopant_indices)), device=DEVICE)
        for k in range(NUM_FOLDS):
            tr_idx = cv_idx_t[cv_fold_id != k]
            va_mask = cv_fold_id == k
            va_idx = cv_idx_t[va_mask]
            pls = PyTorchPLS(n_components=PLS_COMPONENTS).fit(rows(tr_idx), Y_all[tr_idx], inplace=True)
            pls_oof[torch.from_numpy(va_mask)] = pls.predict(rows(va_idx))
        preds['PLS'] = torch.clamp(pls_oof, min=0.0).cpu().numpy() * MAX_WT_PERCENT

    # -------------------------------------------------------------------------
    # 2. Batched 1D CNN Model Predictions
    # -------------------------------------------------------------------------
    print("Generating 1D CNN predictions...")
    scale = data['y'].shape[1] ** 0.5
    cnn_offset = detect_fold_offset(CNN_MODEL_PREFIX)
    
    cnn_folds = []
    for k in range(NUM_FOLDS):
        ckpt_path = Path(f"{CNN_MODEL_PREFIX}_fold{k + cnn_offset}.pt")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        cnn = PyTorch1DCNN(in_channels=1, out_features=len(dopant_indices)).to(DEVICE)
        cnn.load_state_dict(ckpt['model_state_dict'])
        cnn.eval()

        eval_idx = test_idx if EVAL_MODE == "lockbox" else cv_idx[cv_fold_id == k]
        pred_k = predict_batched(cnn, data['y'], norms, eval_idx, scale)

        if EVAL_MODE == "lockbox":
            cnn_folds.append(pred_k)
        else:
            preds['1D CNN'][cv_fold_id == k] = pred_k * MAX_WT_PERCENT

    if EVAL_MODE == "lockbox" and cnn_folds:
        preds['1D CNN'] = np.mean(cnn_folds, axis=0) * MAX_WT_PERCENT

    # -------------------------------------------------------------------------
    # 3. P3VAE Model Predictions (Toggleable)
    # -------------------------------------------------------------------------
    if ENABLE_P3VAE:
        print("Generating P3VAE predictions...")
        p3vae_offset = detect_fold_offset(P3VAE_MODEL_PREFIX)
        p3vae_folds = []
        for k in range(NUM_FOLDS):
            ckpt_path = Path(f"{P3VAE_MODEL_PREFIX}_fold{k + p3vae_offset}.pt")
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            
            p3vae = PhysicsInformedVAE(
                input_dim=data['y'].shape[1],
                n_elements=data['X'].shape[1],
                wl_grid=torch.from_numpy(data['wavelengths']).to(DEVICE),
                master_table_path=MASTER_TABLE_PATH,
                elem_names=data['names'],
                keep_idx=data.get('keep_idx'),
                n_broadening=1
            ).to(DEVICE)
            p3vae.load_state_dict(ckpt['model_state_dict'])
            p3vae.eval()

            scaler = SpectralScaler()
            scaler.load_state_dict(ckpt['scaler_state_dict'])

            eval_idx = test_idx if EVAL_MODE == "lockbox" else cv_idx[cv_fold_id == k]
            spec_scaled = scaler.transform(data['y'][eval_idx])
            ds = LIBSSpectraDataset(spec_scaled, data['X'][eval_idx], data['is_synth'][eval_idx], np.arange(len(eval_idx)))
            loader = DataLoader(ds, batch_size=256, shuffle=False)

            fold_preds = []
            with torch.no_grad():
                for batch in loader:
                    out = p3vae(batch['spectrum'].to(DEVICE))
                    fold_preds.append(out['concentrations'][:, dopant_indices].cpu().numpy())
            
            pred_k = np.concatenate(fold_preds, axis=0)
            if EVAL_MODE == "lockbox":
                p3vae_folds.append(pred_k)
            else:
                preds['P3VAE'][cv_fold_id == k] = pred_k * MAX_WT_PERCENT

        if EVAL_MODE == "lockbox" and p3vae_folds:
            preds['P3VAE'] = np.mean(p3vae_folds, axis=0) * MAX_WT_PERCENT
    else:
        print("⚠️ ENABLE_P3VAE is set to False. Skipping P3VAE inference for troubleshooting.")

    return y_true, preds


# =============================================================================
# Main Plotting Routine: Poster 4x3 Parity Grid
# =============================================================================
def main():
    print("=== Generating Master Poster 4x3 Parity Grid Plot ===")
    data = load_crossval_data(H5_FILE)
    split_path = SPLIT_DIR / f'max12_{NUM_FOLDS}_folds_seed42.npz'
    d_split = np.load(split_path.with_suffix('.npz'))
    test_idx, cv_idx, cv_fold_id = d_split['test_idx'], d_split['cv_idx'], d_split['cv_fold_id']

    y_true, model_preds = get_model_predictions(data, cv_fold_id, test_idx, cv_idx)
    models = ['PLS', '1D CNN', 'P3VAE']

    # Create 4 Rows x 3 Columns Figure
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(10.5, 13.5), dpi=300)

    for row_idx, elem_name in enumerate(TARGET_DOPANTS):
        yt = y_true[:, row_idx]
        
        # Lock x/y limits identically per row based on maximum true or predicted value
        active_preds = [model_preds[m][:, row_idx].max() for m in models if m != 'P3VAE' or ENABLE_P3VAE]
        row_max_pred = max(active_preds) if active_preds else yt.max()
        max_bound = max(yt.max(), row_max_pred) * 1.05
        min_bound = 0.0

        for col_idx, model_name in enumerate(models):
            ax = axes[row_idx, col_idx]
            yp = model_preds[model_name][:, row_idx]

            ax.set_xlim(min_bound, max_bound)
            ax.set_ylim(min_bound, max_bound)
            ax.grid(True, linestyle=":", alpha=0.5)

            if model_name == 'P3VAE' and not ENABLE_P3VAE:
                # Placeholder overlay while P3VAE is actively training
                ax.text(
                    0.5, 0.5, "P3VAE Training\nin Progress...",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=11, fontweight="bold", color="#777777",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#cccccc")
                )
                ax.plot([min_bound, max_bound], [min_bound, max_bound], "k--", alpha=0.3, lw=1.0)
                ax.set_title("Training in Progress", fontsize=9.5, color="#777777")
            else:
                # Compute quantitative metrics
                r2 = r2_score(yt, yp)
                rmse = np.sqrt(mean_squared_error(yt, yp))
                mae = mean_absolute_error(yt, yp)

                # Scatter Plot & Identity Line
                ax.scatter(yt, yp, alpha=0.35, s=12, color="#1f77b4" if col_idx != 1 else "#d62728", edgecolors="none")
                ax.plot([min_bound, max_bound], [min_bound, max_bound], "k--", alpha=0.7, lw=1.2)
                ax.set_title(f"$R^2$: {r2:.3f} | RMSE: {rmse:.3f} wt%\nMAE: {mae:.3f} wt%", fontsize=9.5)

            # Column Headers (Top Row Only)
            if row_idx == 0:
                ax.set_title(f"{model_name}\n" + ax.get_title(), fontsize=12, fontweight="bold", pad=8)

            # Row Labels (Left Column Only)
            if col_idx == 0:
                ax.set_ylabel(f"{elem_name}\nPred wt%", fontsize=11, fontweight="bold")
            else:
                ax.set_ylabel("Pred wt%", fontsize=9)

            # X-axis Labels (Bottom Row Only)
            if row_idx == 3:
                ax.set_xlabel("True wt%", fontsize=11, fontweight="bold")

    plt.suptitle(f"Quantitative Performance Comparison — {EVAL_MODE.upper()} Set", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    save_path = OUTPUT_DIR / f"poster_4x3_parity_grid_{EVAL_MODE}_{TIMESTAMP}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Successfully saved poster 4x3 parity figure to: {save_path}")


if __name__ == "__main__":
    main()