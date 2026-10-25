from __future__ import annotations

import sys
import time
import h5py
import torch
import torch.nn as nn
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

# Import model architecture, scaler, and dataset from your P3VAE script
from p3vae_006 import PhysicsInformedVAE, SpectralScaler, LIBSSpectraDataset


# =============================================================================
# Global Configuration & File Paths (Matched to MLP & 1D CNN Benchmarks)
# =============================================================================
H5_FILE = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/max12_cts_noleak_crossval.h5"
SPLIT_DIR = Path('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/fold_splits_NoExtraDopants')
MASTER_TABLE_PATH = Path('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/master_line_table.csv').resolve()
OUTPUT_DIR = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/p3VAE/plots/decoder_benchmarks")

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
# Helper Data Functions (Identical Slicing to Benchmark Scripts)
# =============================================================================
def load_crossval_data(
    h5_path: str, 
    spike_thresh: float = THRESH,
    exp_dopants: set = EXP_DOPANTS,
    host_elements: set = HOST_ELEMENTS
) -> Dict[str, Any]:
    """Loads dataset into RAM, cleans saturation spikes, filters unmodeled 
    synthetic dopants, and retains matching target features."""
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
    """Loads existing pre-computed CV splits matching standard benchmarks."""
    split_path = Path(split_path).with_suffix('.npz')
    if split_path.exists():
        print(f"Loading cached splits from: {split_path}")
        d = np.load(split_path)
        return d['test_idx'], d['cv_idx'], d['cv_fold_id']
    else:
        raise FileNotFoundError(f"Split file not found at {split_path}. Run benchmark scripts first to generate canonical splits.")


# =============================================================================
# Benchmark Metrics & Tables (Chunked for Memory Safety)
# =============================================================================
def _chunk_stats(y_true, y_pred, cols, chunk=2048):
    """Computes SSE, SAE, SST, and channel counts without memory overflow."""
    n = y_true.shape[0]
    sse = sae = 0.0
    s1 = s2 = None
    for i in range(0, n, chunk):
        t = y_true[i:i + chunk][:, cols].astype(np.float64)
        p = (y_pred[i:i + chunk][:, cols].astype(np.float64) if y_pred.ndim == 2
             else np.broadcast_to(y_pred[cols].astype(np.float64), t.shape))
        d = t - p
        sse += float(np.einsum('ij,ij->', d, d))
        sae += float(np.abs(d).sum())
        if s1 is None: 
            s1, s2 = t.sum(0), np.einsum('ij,ij->j', t, t)
        else: 
            s1 += t.sum(0); s2 += np.einsum('ij,ij->j', t, t)
    sst = float((s2 - s1 ** 2 / n).sum())   # type: ignore
    return sse, sae, sst, n * len(cols)


def print_benchmark_table(y_true, y_pred_full, y_train_mean, title_tag, y_pred_phys=None):
    """Prints element-wise & regional spectral metrics matching MLP table format."""
    mean_spec = y_true.mean(axis=0, dtype=np.float64)
    top = mean_spec >= np.percentile(mean_spec, 90)
    regions = [("Full Spectrum", np.arange(y_true.shape[1])),
               ("Peak Channels (Top 10%)", np.where(top)[0]),
               ("Continuum / Background", np.where(~top)[0])]
    
    zero_vec = np.zeros(y_true.shape[1], dtype=np.float32)

    strategies = [
        ("Predict Zero", zero_vec),
        ("Predict Mean", y_train_mean),
    ]
    if y_pred_phys is not None:
        strategies.append(("P3VAE Physics Line", y_pred_phys))
    strategies.append(("P3VAE Full Recon", y_pred_full))

    print("\n" + "=" * 95)
    print(f" {title_tag}")
    print("=" * 95)
    print(f"{'Spectral Region':<25} | {'Strategy':<20} | {'Global R^2':<10} | {'MAE':<10} | {'RMSE':<10}")
    print("-" * 95)
    
    for reg_name, cols in regions:
        for strat_name, y_hat in strategies:
            sse, sae, sst, cnt = _chunk_stats(y_true, y_hat, cols)
            r2 = 1.0 - sse / sst if sst > 1e-8 else 0.0
            print(f"{reg_name:<25} | {strat_name:<20} | {r2:10.4f} | {sae / cnt:10.4f} | {np.sqrt(sse / cnt):10.4f}")
        print("-" * 95)


# =============================================================================
# Plotting Utilities (Parity Grids & Spectral Residual Overlays)
# =============================================================================
def plot_parity_grid(y_true, y_pred, wavelengths=None, title_tag="P3VAE Decoder Benchmark", save_path=None):
    """Generates intensity parity plots across top spectral emission peak bins."""
    mean_spec = y_true.mean(axis=0)
    top_peak_indices = np.argsort(mean_spec)[-12:][::-1]

    cols = 4
    rows = int(np.ceil(len(top_peak_indices) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.8, rows * 3.4))
    axes_flat = np.atleast_1d(axes).flatten()

    for idx_plot, ch in enumerate(top_peak_indices):
        ax = axes_flat[idx_plot]
        wl_str = f"{wavelengths[ch]:.2f} nm" if wavelengths is not None else f"Bin {ch}"
        yt = y_true[:, ch]
        yp = y_pred[:, ch]

        r2 = r2_score(yt, yp) if (yt.max() - yt.min() > 1e-6) else 0.0
        mae = mean_absolute_error(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))

        ax.scatter(yt, yp, alpha=0.35, s=10, color="#1f77b4", edgecolors="none")
        max_v = max(np.max(yt), np.max(yp), 1e-3)
        min_v = min(np.min(yt), np.min(yp))
        ax.plot([min_v, max_v], [min_v, max_v], "k--", alpha=0.7, lw=1)
        ax.set_title(f"Peak @ {wl_str}\n$R^2$: {r2:.3f} | RMSE: {rmse:.3f}\nMAE: {mae:.3f}", fontsize=9, fontweight="bold")
        ax.set_xlabel("True Intensity", fontsize=8)
        ax.set_ylabel("Pred Intensity", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)

    for idx_plot in range(len(top_peak_indices), len(axes_flat)):
        fig.delaxes(axes_flat[idx_plot])

    plt.suptitle(f"{title_tag} - Top Peak Intensity Parity Grid", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved Parity Plot to: {save_path}")
    plt.close()


def plot_sample_spectra(
    y_true,
    y_pred_full,
    wavelengths,
    y_pred_phys=None,
    sample_indices=None,
    X=None,
    feature_names=None,
    title_tag="P3VAE Decoder Benchmark",
    save_path=None
):
    """Plots true vs predicted spectral overlays alongside stacked residual panels."""
    if sample_indices is None:
        sample_indices = [0, len(y_true) - 1]

    n_samples = len(sample_indices)

    fig, axes = plt.subplots(
        2, n_samples, 
        figsize=(6.5 * n_samples, 5.5), 
        sharex='col', 
        gridspec_kw={'height_ratios': [3.5, 1]}
    )

    for i, idx in enumerate(sample_indices):
        ax_main = axes[0, i] if n_samples > 1 else axes[0]
        ax_res = axes[1, i] if n_samples > 1 else axes[1]

        yt = y_true[idx]
        yp_full = y_pred_full[idx] if y_pred_full.ndim == 2 else np.mean(y_pred_full[:, idx, :], axis=0)
        residual = yt - yp_full

        # --- TOP PANEL ---
        ax_main.plot(wavelengths, yt, label="Measured Spectrum", color="black", lw=1.0, alpha=0.75)
        ax_main.plot(wavelengths, yp_full, label="P3VAE Full Recon", color="crimson", lw=0.9, ls="--", alpha=0.9)

        if y_pred_phys is not None:
            yp_phys = y_pred_phys[idx] if y_pred_phys.ndim == 2 else np.mean(y_pred_phys[:, idx, :], axis=0)
            ax_main.plot(wavelengths, yp_phys, label="Physics Lines (NIST)", color="forestgreen", lw=0.8, alpha=0.8)

        # R^2 calculation for sample
        ss_res = np.sum(residual ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2) + 1e-8
        r2 = 1.0 - (ss_res / ss_tot)

        ax_main.set_title(f"Sample #{idx} — Spectral $R^2$: {r2:.4f}", fontsize=11, fontweight="bold")
        ax_main.set_ylabel("Intensity (a.u.)", fontsize=9)
        ax_main.grid(True, linestyle=":", alpha=0.4)

        # Composition Inset Box
        if X is not None and feature_names is not None:
            sample_conc = (X[idx].cpu().numpy() if hasattr(X[idx], "cpu") else X[idx]) * MAX_WT_PERCENT
            comp_items = [
                f"{name}: {val:.2f} wt%" 
                for name, val in zip(feature_names, sample_conc) 
                if val > 1e-4
            ]
            comp_text = "Composition:\n" + ("\n".join(comp_items) if comp_items else "Pure Host Matrix")
            
            ax_main.text(
                0.02, 0.95, comp_text,
                transform=ax_main.transAxes,
                fontsize=8,
                family="monospace",
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="white",
                    edgecolor="#333333",
                    alpha=0.85,
                    linewidth=0.8
                )
            )

        ax_main.legend(loc="upper right", fontsize=8, frameon=True)

        # --- BOTTOM PANEL: Residuals ---
        ax_res.plot(wavelengths, residual, color="#b25395", lw=0.75, label="Residual")
        ax_res.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        ax_res.fill_between(wavelengths, 0, residual, color="#b25395", alpha=0.25)

        ax_res.set_xlabel("Wavelength (nm)", fontsize=9)
        ax_res.set_ylabel("Residual", fontsize=9, color="#b25395")
        ax_res.grid(True, linestyle=":", alpha=0.4)

        max_abs = max(abs(np.min(residual)), abs(np.max(residual))) * 1.1
        if max_abs > 0:
            ax_res.set_ylim(-max_abs, max_abs)

    plt.suptitle(f"{title_tag} — Measured vs. Reconstructed Spectra", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved Sample Overlay Plot to: {save_path}")
    plt.close()


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================
def evaluate_p3vae_decoder(model_prefix: str | Path):
    print(f"=== Running Standalone P3VAE Decoder Evaluation [{DEVICE}] ===")
    
    # 1. Load data and canonical splits
    data = load_crossval_data(H5_FILE)
    split_path = SPLIT_DIR / f'max12_{NUM_FOLDS}_folds_seed{SEED}.npz'
    test_idx, cv_idx, cv_fold_id = build_splits_once(data, split_path, NUM_FOLDS)
    has_test = len(test_idx) > 0

    dopant_indices = [i for i, name in enumerate(data['names']) if name not in HOST_ELEMENTS]
    feature_names = [data['names'][i] for i in dopant_indices]

    wl_grid = torch.from_numpy(data["wavelengths"]).to(DEVICE)
    input_dim = data["y"].shape[-1]
    n_elements = data["X"].shape[-1]
    wavelengths = data['wavelengths']

    true_cv_spec = data['y'][cv_idx].numpy()
    true_test_spec = data['y'][test_idx].numpy() if has_test else None

    # Pre-allocate output arrays
    oof_recon_np = np.zeros_like(true_cv_spec, dtype=np.float32)
    oof_phys_np = np.zeros_like(true_cv_spec, dtype=np.float32)

    lockbox_recon_folds = np.zeros((NUM_FOLDS, len(test_idx), input_dim), dtype=np.float32) if has_test else None
    lockbox_phys_folds = np.zeros((NUM_FOLDS, len(test_idx), input_dim), dtype=np.float32) if has_test else None

    loaded_folds = 0

    # 2. Process Checkpoints across Folds
    for k in range(NUM_FOLDS):
        ckpt_path = Path(f"{model_prefix}_fold{k}.pt")
        if not ckpt_path.exists():
            print(f"⚠️ Warning: Missing checkpoint {ckpt_path.name}. Skipping fold {k}.")
            continue

        print(f"Processing Fold {k + 1}/{NUM_FOLDS}...")
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

        fold_recon, fold_phys = [], []
        with torch.no_grad():
            for batch in va_loader:
                spectra = batch["spectrum"].to(DEVICE)
                out = model(spectra)
                
                # Unscale back to raw intensity units
                fold_recon.append(scaler.inverse_transform(out["reconstructed_spectrum"].cpu().numpy()))
                fold_phys.append(scaler.inverse_transform(out["ideal_physics_spectrum"].cpu().numpy()))

        oof_recon_np[va_idx] = np.concatenate(fold_recon, axis=0)
        oof_phys_np[va_idx] = np.concatenate(fold_phys, axis=0)

        # Evaluate Lockbox Test Set
        if has_test and lockbox_recon_folds is not None and lockbox_phys_folds is not None:
            te_spectra_scaled = scaler.transform(data["y"][test_idx])
            te_ds = LIBSSpectraDataset(te_spectra_scaled, data["X"][test_idx], data["is_synth"][test_idx], np.arange(len(test_idx)))
            te_loader = DataLoader(te_ds, batch_size=256, shuffle=False)

            te_recon, te_phys = [], []
            with torch.no_grad():
                for batch in te_loader:
                    spectra = batch["spectrum"].to(DEVICE)
                    out = model(spectra)
                    te_recon.append(scaler.inverse_transform(out["reconstructed_spectrum"].cpu().numpy()))
                    te_phys.append(scaler.inverse_transform(out["ideal_physics_spectrum"].cpu().numpy()))

            lockbox_recon_folds[k] = np.concatenate(te_recon, axis=0)
            lockbox_phys_folds[k] = np.concatenate(te_phys, axis=0)

        loaded_folds += 1

    if loaded_folds == 0:
        raise RuntimeError("No model checkpoints were loaded!")

    # 3. Print Regional Benchmark Metrics
    exp_mask = cv_fold_id >= 0
    true_cv_eval = true_cv_spec[exp_mask]
    recon_cv_eval = oof_recon_np[exp_mask]
    phys_cv_eval = oof_phys_np[exp_mask]
    train_mean_cv = true_cv_eval.mean(axis=0)

    print_benchmark_table(
        y_true=true_cv_eval,
        y_pred_full=recon_cv_eval,
        y_train_mean=train_mean_cv,
        title_tag=f"P3VAE Decoder OOF Benchmarks ({loaded_folds} Folds)",
        y_pred_phys=phys_cv_eval
    )

    if has_test and lockbox_recon_folds is not None and lockbox_phys_folds is not None:
        lockbox_recon_avg = lockbox_recon_folds.mean(axis=0)
        lockbox_phys_avg = lockbox_phys_folds.mean(axis=0)

        print_benchmark_table(
            y_true=true_test_spec,
            y_pred_full=lockbox_recon_avg,
            y_train_mean=train_mean_cv,
            title_tag=f"P3VAE Decoder Lockbox Benchmarks (Ensemble of {loaded_folds} Folds)",
            y_pred_phys=lockbox_phys_avg
        )

    # 4. Generate Plots
    X_cv_cpu = data['X'][cv_idx]
    X_test_cpu = data['X'][test_idx] if has_test else None

    # Pick Best vs. Worst sample indices based on sample-wise R^2
    sample_r2s = np.array([
        1.0 - (np.sum((true_cv_eval[i] - recon_cv_eval[i]) ** 2) / (np.sum((true_cv_eval[i] - np.mean(true_cv_eval[i])) ** 2) + 1e-8))
        for i in range(len(true_cv_eval))
    ])
    best_idx = int(np.argmax(sample_r2s))
    worst_idx = int(np.argmin(sample_r2s))

    plot_parity_grid(
        true_cv_eval, recon_cv_eval, wavelengths=wavelengths,
        title_tag="P3VAE Decoder OOF Peak Parity",
        save_path=OUTPUT_DIR / f"p3vae_decoder_oof_parity_{TIMESTAMP}.png"
    )

    plot_sample_spectra(
        y_true=true_cv_eval,
        y_pred_full=recon_cv_eval,
        y_pred_phys=phys_cv_eval,
        wavelengths=wavelengths,
        sample_indices=[best_idx, worst_idx],
        X=X_cv_cpu,
        feature_names=data['names'],
        title_tag="P3VAE Decoder OOF Reconstruction",
        save_path=OUTPUT_DIR / f"p3vae_decoder_oof_samples_{TIMESTAMP}.png"
    )

    if has_test and lockbox_recon_folds is not None:
        lockbox_recon_avg = lockbox_recon_folds.mean(axis=0)
        lockbox_phys_avg = lockbox_phys_folds.mean(axis=0)  # type: ignore

        test_r2s = np.array([
            1.0 - (np.sum((true_test_spec[i] - lockbox_recon_avg[i]) ** 2) / (np.sum((true_test_spec[i] - np.mean(true_test_spec[i])) ** 2) + 1e-8))  # type: ignore
            for i in range(len(true_test_spec))  # type: ignore
        ])
        test_best_idx = int(np.argmax(test_r2s))
        test_worst_idx = int(np.argmin(test_r2s))

        plot_parity_grid(
            true_test_spec, lockbox_recon_avg, wavelengths=wavelengths,
            title_tag="P3VAE Decoder Lockbox Peak Parity",
            save_path=OUTPUT_DIR / f"p3vae_decoder_lockbox_parity_{TIMESTAMP}.png"
        )

        plot_sample_spectra(
            y_true=true_test_spec,
            y_pred_full=lockbox_recon_folds,
            y_pred_phys=lockbox_phys_folds,
            wavelengths=wavelengths,
            sample_indices=[test_best_idx, test_worst_idx],
            X=X_test_cpu,
            feature_names=data['names'],
            title_tag="P3VAE Decoder Lockbox Ensemble",
            save_path=OUTPUT_DIR / f"p3vae_decoder_lockbox_samples_{TIMESTAMP}.png"
        )

    print("Decoder evaluation complete! All figures and benchmark metrics generated safely.")


if __name__ == "__main__":
    # Pass your trained model prefix here (without the _foldK.pt suffix)
    MODEL_PREFIX = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/p3VAE/trained_models/best_p3vae6_model_20261024_162246"
    evaluate_p3vae_decoder(MODEL_PREFIX)