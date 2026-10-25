from __future__ import annotations

import sys
import time
import h5py
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, Dataset, get_worker_info
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Any, Dict

# Add project root setup matching your directory layout
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from setup import add_project_root_to_path
add_project_root_to_path(parent_generation=1)
from utils import get_h5_ds

# Import model architecture and scaler from your P3VAE script
from p3vae_006 import PhysicsInformedVAE, SpectralScaler, LIBSSpectraDataset


# =============================================================================
# Global Configuration & File Paths (Matched to 1D CNN Benchmark)
# =============================================================================
H5_FILE = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/max12_cts_noleak_crossval.h5"
SPLIT_DIR = Path('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/fold_splits_NoExtraDopants')
MASTER_TABLE_PATH = Path('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/master_line_table.csv').resolve()
OUTPUT_DIR = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/p3VAE/plots/encoder_benchmarks")

NUM_FOLDS = 7
THRESH = 2.8
MAX_WT_PERCENT = 12.0
SEED = 42
HOST_ELEMENTS = {'Li', 'K', 'Cl'}
EXP_DOPANTS = {'Ce', 'Gd', 'Sm', 'U'}

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Helper Data Functions (Identical Slicing to 1D CNN)
# =============================================================================
def load_crossval_data(
    h5_path: str, 
    spike_thresh: float = THRESH,
    exp_dopants: set = EXP_DOPANTS,
    host_elements: set = HOST_ELEMENTS
) -> Dict[str, Any]:
    """Loads dataset, cleans saturation spikes, filters unmodeled synthetic 
    dopants, and slices target columns to active elements."""
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

    keep_idx = [i for i, name in enumerate(names) if name in keep_elements]
    extra_dopant_idx = [i for i, name in enumerate(names) if name not in keep_elements]

    row_maxes = y.max(axis=1)
    clean_spike_mask = row_maxes <= spike_thresh

    has_extra_dopants = (X[:, extra_dopant_idx] > 0).any(axis=1)
    valid_synth_mask = (is_synth == 0) | (~has_extra_dopants)

    final_row_mask = clean_spike_mask & valid_synth_mask

    out_dict = dict(
        X=torch.from_numpy(X[final_row_mask][:, keep_idx]),
        y=torch.from_numpy(y[final_row_mask]),
        is_synth=torch.from_numpy(is_synth[final_row_mask]).long(),
        group_id=group_id[final_row_mask],
        wavelengths=wavelengths,
        names=[names[i] for i in keep_idx],
    )

    if is_lockbox is not None:
        out_dict['is_lockbox'] = is_lockbox[final_row_mask]

    return out_dict


def build_splits_once(data, split_path, num_folds):
    """Loads existing pre-computed CV splits matching 1D CNN script."""
    split_path = Path(split_path).with_suffix('.npz')
    if split_path.exists():
        print(f"Loading cached splits from: {split_path}")
        d = np.load(split_path)
        return d['test_idx'], d['cv_idx'], d['cv_fold_id']
    else:
        raise FileNotFoundError(f"Split file not found at {split_path}. Run 1D CNN script first to generate canonical splits.")


# =============================================================================
# Metric & Plotting Utilities (Matched to 1D CNN Layout)
# =============================================================================
def print_benchmark_table(y_true, y_pred_model, y_train_mean, title_tag, elem_names):
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
            ("P3VAE Encoder", yp)
        ]
        
        for strat_name, y_hat in strategies:
            r2 = r2_score(yt, y_hat)
            mae = mean_absolute_error(yt, y_hat)
            rmse = np.sqrt(mean_squared_error(yt, y_hat))
            print(f"{name:<8} | {strat_name:<18} | {r2:8.3f} | {mae:10.4f} | {rmse:10.4f} | {yt.max():10.3f}")
        print("-" * 95)


def plot_parity_grid(y_true_wt, y_pred_wt, target_names, title_tag="P3VAE Encoder Benchmark", save_path=None):
    """Generates an NxM parity grid plot matching 1D CNN visualization."""
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

    plt.suptitle(f"{title_tag} - Concentration Parity Grid", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved Parity Plot to: {save_path}")
    plt.close()


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================
def evaluate_p3vae_encoder(model_prefix: str | Path):
    print(f"=== Evaluating P3VAE Encoder Benchmark [{DEVICE}] ===")
    
    # 1. Load data and canonical splits
    data = load_crossval_data(H5_FILE)
    split_path = SPLIT_DIR / f'max12_{NUM_FOLDS}_folds_seed{SEED}.npz'
    test_idx, cv_idx, cv_fold_id = build_splits_once(data, split_path, NUM_FOLDS)
    has_test = len(test_idx) > 0

    dopant_indices = [i for i, name in enumerate(data['names']) if name not in HOST_ELEMENTS]
    target_names = [data['names'][i] for i in dopant_indices]
    
    wl_grid = torch.from_numpy(data["wavelengths"]).to(DEVICE)
    input_dim = data["y"].shape[-1]
    n_elements = data["X"].shape[-1]
    
    Y_cv_np = data['X'][cv_idx][:, dopant_indices].numpy()
    Y_test_np = data['X'][test_idx][:, dopant_indices].numpy() if has_test else None

    oof_preds_np = np.zeros((len(cv_idx), len(dopant_indices)), dtype=np.float32)
    lockbox_preds = torch.zeros((NUM_FOLDS, len(test_idx), len(dopant_indices)), device=DEVICE) if has_test else None

    loaded_folds = 0

    # 2. Iterate through K-Fold Checkpoints
    for k in range(NUM_FOLDS):
        ckpt_path = Path(f"{model_prefix}_fold{k}.pt")
        if not ckpt_path.exists():
            print(f"⚠️ Warning: Checkpoint for fold {k} not found ({ckpt_path.name}). Skipping.")
            continue

        print(f"Running inference for Fold {k + 1}/{NUM_FOLDS}...")
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)

        model = PhysicsInformedVAE(
            input_dim=input_dim,
            n_elements=n_elements,
            wl_grid=wl_grid,
            master_table_path=MASTER_TABLE_PATH,
            n_broadening=1,
        ).to(DEVICE)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.eval()

        scaler = SpectralScaler()
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

        va_idx = np.where(cv_fold_id == k)[0]

        # Evaluate OOF Validation Fold
        va_spectra_scaled = scaler.transform(data["y"][cv_idx[va_idx]])
        va_ds = LIBSSpectraDataset(va_spectra_scaled, data["X"][cv_idx[va_idx]], data["is_synth"][cv_idx[va_idx]], np.arange(len(va_idx)))
        va_loader = DataLoader(va_ds, batch_size=256, shuffle=False)

        fold_preds = []
        with torch.no_grad():
            for batch in va_loader:
                spectra = batch["spectrum"].to(DEVICE)
                out = model(spectra)
                fold_preds.append(out["concentrations"][:, dopant_indices].cpu().numpy())

        oof_preds_np[va_idx] = np.concatenate(fold_preds, axis=0)

        # Evaluate Lockbox Test Set
        if has_test:
            te_spectra_scaled = scaler.transform(data["y"][test_idx])
            te_ds = LIBSSpectraDataset(te_spectra_scaled, data["X"][test_idx], data["is_synth"][test_idx], np.arange(len(test_idx)))
            te_loader = DataLoader(te_ds, batch_size=256, shuffle=False)

            te_preds = []
            with torch.no_grad():
                for batch in te_loader:
                    spectra = batch["spectrum"].to(DEVICE)
                    out = model(spectra)
                    te_preds.append(out["concentrations"][:, dopant_indices])

            lockbox_preds[k] = torch.cat(te_preds, dim=0)   # type: ignore

        loaded_folds += 1

    if loaded_folds == 0:
        raise RuntimeError("No model checkpoints were loaded!")

    # 3. Unscale Concentrations to wt%
    exp_mask = cv_fold_id >= 0
    true_cv_wt = Y_cv_np[exp_mask] * MAX_WT_PERCENT
    pred_cv_wt = oof_preds_np[exp_mask] * MAX_WT_PERCENT
    train_mean_cv = true_cv_wt.mean(axis=0)

    # 4. Print Benchmark Tables
    print_benchmark_table(
        y_true=true_cv_wt, 
        y_pred_model=pred_cv_wt, 
        y_train_mean=train_mean_cv,
        title_tag=f"P3VAE Encoder OOF Benchmarks ({loaded_folds} Folds)",
        elem_names=target_names,
    )

    if has_test and lockbox_preds is not None:
        lockbox_pred_avg = lockbox_preds.mean(dim=0).cpu().numpy()
        true_test_wt = Y_test_np * MAX_WT_PERCENT   # type: ignore
        pred_test_wt = lockbox_pred_avg * MAX_WT_PERCENT

        print_benchmark_table(
            y_true=true_test_wt, 
            y_pred_model=pred_test_wt, 
            y_train_mean=train_mean_cv,
            title_tag=f"P3VAE Encoder Lockbox Benchmarks (Ensemble of {loaded_folds} Folds)",
            elem_names=target_names,
        )

    # 5. Save Parity Plots
    plot_parity_grid(
        true_cv_wt, pred_cv_wt, target_names,
        title_tag="P3VAE Encoder OOF",
        save_path=OUTPUT_DIR / f"p3vae_encoder_oof_parity_{TIMESTAMP}.png"
    )

    if has_test:
        plot_parity_grid(
            true_test_wt, pred_test_wt, target_names,   # type: ignore
            title_tag="P3VAE Encoder Lockbox Ensemble",
            save_path=OUTPUT_DIR / f"p3vae_encoder_lockbox_parity_{TIMESTAMP}.png"
        )


if __name__ == "__main__":
    # Pass your trained model prefix here (without the _foldK.pt suffix)
    MODEL_PREFIX = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/p3VAE/trained_models/best_p3vae6_model_20261024_211302"
    evaluate_p3vae_encoder(MODEL_PREFIX)