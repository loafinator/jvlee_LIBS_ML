from __future__ import annotations

import sys
import gc
import resource
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

class TargetedSamples:
    """Tracks specific sample indices (e.g. Complex & Trace) during streaming evaluation."""
    def __init__(self, target_indices: list[int]):
        self.target_indices = target_indices
        self.stored = {}

    def update(self, labels, t, rec, phy, x_rows) -> None:
        for idx in self.target_indices:
            if idx in labels:
                pos = np.where(labels == idx)[0][0]
                self.stored[idx] = (
                    int(labels[pos]),
                    t[pos].copy(),
                    rec[pos].copy(),
                    phy[pos].copy(),
                    np.array(x_rows[pos])
                )

    def as_plot_args(self):
        ordered = [self.stored[idx] for idx in self.target_indices if idx in self.stored]
        return dict(
            y_true=np.stack([item[1] for item in ordered]),
            y_pred_full=np.stack([item[2] for item in ordered]),
            y_pred_phys=np.stack([item[3] for item in ordered]),
            X=np.stack([item[4] for item in ordered]),
            sample_indices=list(range(len(ordered))),
            sample_labels=[item[0] for item in ordered],
        )

def load_crossval_data(
    h5_path: str, 
    spike_thresh: float = THRESH,
    exp_dopants: set = EXP_DOPANTS,
    host_elements: set = HOST_ELEMENTS,
    read_chunk: int = 4096,
) -> Dict[str, Any]:
    """Loads dataset into RAM, cleans saturation spikes, filters unmodeled 
    synthetic dopants, and retains matching target features.

    Memory-lean: the big spectra array is streamed from disk in chunks (two
    passes: spike mask, then gather) so the full unmasked array -- and any
    float64 -> float32 conversion copy of it -- never exists in RAM."""
    keep_elements = host_elements.union(exp_dopants)

    with h5py.File(h5_path, 'r') as hf:
        X_full = get_h5_ds(hf, 'X')[:].astype(np.float32)
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

        y_ds = get_h5_ds(hf, 'y')
        n_rows, n_bins = y_ds.shape

        # Pass 1: saturation-spike mask, chunk by chunk
        clean_spike_mask = np.empty(n_rows, dtype=bool)
        for i in range(0, n_rows, read_chunk):
            clean_spike_mask[i:i + read_chunk] = y_ds[i:i + read_chunk].max(axis=1) <= spike_thresh

        has_extra_dopants = (X_full[:, extra_dopant_idx] > 0).any(axis=1)
        valid_synth_mask = (is_synth == 0) | (~has_extra_dopants)
        final_row_mask = clean_spike_mask & valid_synth_mask

        # Pass 2: gather surviving rows straight into a float32 array
        y = np.empty((int(final_row_mask.sum()), n_bins), dtype=np.float32)
        pos = 0
        for i in range(0, n_rows, read_chunk):
            sel = final_row_mask[i:i + read_chunk]
            if sel.any():
                blk = y_ds[i:i + read_chunk][sel]
                y[pos:pos + len(blk)] = blk
                pos += len(blk)

    out_dict = dict(
        X=torch.from_numpy(X_full[final_row_mask][:, keep_idx]),
        y=torch.from_numpy(y),
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
def _chunk_stats(y_true, y_pred, cols, chunk=1024):
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


def plot_parity_peaks(yt_peaks, yp_peaks, peak_cols, wavelengths=None,
                      title_tag="P3VAE Decoder Benchmark", save_path=None):
    """Same figure as plot_parity_grid, but takes only the (N, n_peaks) columns
    that are plotted instead of the full N x P arrays."""
    cols = 4
    rows = int(np.ceil(len(peak_cols) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.8, rows * 3.4))
    axes_flat = np.atleast_1d(axes).flatten()

    for j, ch in enumerate(peak_cols):
        ax = axes_flat[j]
        wl_str = f"{wavelengths[ch]:.2f} nm" if wavelengths is not None else f"Bin {ch}"
        yt, yp = yt_peaks[:, j], yp_peaks[:, j]

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

    for j in range(len(peak_cols), len(axes_flat)):
        fig.delaxes(axes_flat[j])

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
    save_path=None,
    sample_labels=None
):
    """Plots true vs predicted spectral overlays alongside stacked residual panels."""
    if sample_indices is None:
        sample_indices = [0, len(y_true) - 1]

    n_samples = len(sample_indices)

    fig, axes = plt.subplots(
        2, n_samples, 
        figsize=(5.0 * n_samples, 4), 
        sharex='col', 
        gridspec_kw={'height_ratios': [2.5, 1]}
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

        ax_main.set_title(f"Sample #{sample_labels[i] if sample_labels is not None else idx} — Spectral $R^2$: {r2:.4f}", fontsize=11, fontweight="bold")
        ax_main.set_ylabel("Intensity (a.u.)", fontsize=9)
        ax_main.grid(True, linestyle=":", alpha=0.4)

        # Composition Inset Box
        if X is not None and feature_names is not None:
            host_comps = HOST_ELEMENTS
            sample_conc = (X[idx].cpu().numpy() if hasattr(X[idx], "cpu") else X[idx]) * MAX_WT_PERCENT
            comp_items = [
                f"{name}: {val:.2f} wt%" 
                for name, val in zip(feature_names, sample_conc) 
                if val > 1e-4 and name not in host_comps
            ]
            comp_text = "Composition:\n" + ("\n".join(comp_items) if comp_items else "Pure Host Matrix")
            
            ax_main.text(
                0.02, 0.05, comp_text,
                transform=ax_main.transAxes,
                horizontalalignment = "left",
                verticalalignment = "bottom",
                fontsize=10,
                family="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="white",
                    edgecolor="#333333",
                    alpha=0.85,
                    linewidth=0.8
                )
            )

        ax_main.legend(loc="upper left", fontsize=9, frameon=True)

        # --- BOTTOM PANEL: Residuals ---
        ax_res.plot(wavelengths, residual, color="#b25395", lw=0.75, label="Residual")
        ax_res.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        ax_res.fill_between(wavelengths, 0, residual, color="#b25395", alpha=0.25)

        ax_res.set_xlabel("Wavelength (nm)", fontsize=9)
        ax_res.set_ylabel("Residual", fontsize=9)
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


# =============================================================================
# Streaming evaluation machinery (O(batch) memory instead of O(N x P) outputs)
# =============================================================================
LOCKBOX_BLOCK_GB = 1.0        # RAM for EACH of the two lockbox ensemble accumulators
MODEL_CACHE_LIMIT_GB = 2.0    # cache all fold models only if they fit in this
N_PEAKS = 12


def log_mem(tag: str) -> None:
    """Prints peak RSS so you can see which stage actually blows up."""
    print(f"[mem] {tag}: peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB", flush=True)


def per_sample_r2(y_true: np.ndarray, y_pred: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """Vectorized, chunked per-spectrum R^2 (same formula as before)."""
    out = np.empty(len(y_true), dtype=np.float64)
    for i in range(0, len(y_true), chunk):
        t = y_true[i:i + chunk].astype(np.float64)
        p = y_pred[i:i + chunk].astype(np.float64)
        ss_res = ((t - p) ** 2).sum(axis=1)
        ss_tot = ((t - t.mean(axis=1, keepdims=True)) ** 2).sum(axis=1) + 1e-8
        out[i:i + chunk] = 1.0 - ss_res / ss_tot
    return out


def rows_mean(y_np: np.ndarray, rows: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """float64 mean spectrum over y_np[rows], gathered chunk by chunk."""
    s = np.zeros(y_np.shape[1], dtype=np.float64)
    for i in range(0, len(rows), chunk):
        s += y_np[rows[i:i + chunk]].sum(axis=0, dtype=np.float64)
    return s / len(rows)


class RegionStats:
    """Streams the benchmark table: accumulates SSE / SAE per (region, strategy)
    and the SST ingredients per region, one batch at a time. Produces exactly
    the numbers print_benchmark_table() would, without holding any prediction array."""
    STRATS = ["Predict Zero", "Predict Mean", "P3VAE Physics Line", "P3VAE Full Recon"]

    def __init__(self, mean_spec: np.ndarray, train_mean: np.ndarray):
        top = mean_spec >= np.percentile(mean_spec, 90)
        self.regions = [
            ("Full Spectrum", slice(None), len(mean_spec)),
            ("Peak Channels (Top 10%)", np.where(top)[0], int(top.sum())),
            ("Continuum / Background", np.where(~top)[0], int((~top).sum())),
        ]
        self.train_mean = train_mean.astype(np.float64)
        self.n = 0
        self.s1 = [np.zeros(c, np.float64) for _, _, c in self.regions]
        self.s2 = [np.zeros(c, np.float64) for _, _, c in self.regions]
        self.sse = {(r, s): 0.0 for r in range(len(self.regions)) for s in self.STRATS}
        self.sae = dict(self.sse)

    def update(self, t: np.ndarray, rec: np.ndarray, phy: np.ndarray) -> None:
        t = t.astype(np.float64)
        rec = rec.astype(np.float64)
        phy = phy.astype(np.float64)
        self.n += t.shape[0]
        for ri, (_, cols, _) in enumerate(self.regions):
            tc = t[:, cols]
            self.s1[ri] += tc.sum(0)
            self.s2[ri] += np.einsum('ij,ij->j', tc, tc)
            preds = {
                "Predict Zero": None,
                "Predict Mean": self.train_mean[cols],
                "P3VAE Physics Line": phy[:, cols],
                "P3VAE Full Recon": rec[:, cols],
            }
            for name, p in preds.items():
                d = tc if p is None else tc - p
                self.sse[(ri, name)] += float(np.einsum('ij,ij->', d, d))
                self.sae[(ri, name)] += float(np.abs(d).sum())

    def print_table(self, title_tag: str) -> None:
        print("\n" + "=" * 95)
        print(f" {title_tag}")
        print("=" * 95)
        print(f"{'Spectral Region':<25} | {'Strategy':<20} | {'Global R^2':<10} | {'MAE':<10} | {'RMSE':<10}")
        print("-" * 95)
        for ri, (reg_name, _, ncols) in enumerate(self.regions):
            sst = float((self.s2[ri] - self.s1[ri] ** 2 / self.n).sum())
            cnt = self.n * ncols
            for name in self.STRATS:
                sse, sae = self.sse[(ri, name)], self.sae[(ri, name)]
                r2 = 1.0 - sse / sst if sst > 1e-8 else 0.0
                print(f"{reg_name:<25} | {name:<20} | {r2:10.4f} | {sae / cnt:10.4f} | {np.sqrt(sse / cnt):10.4f}")
            print("-" * 95)


class BestWorst:
    """Tracks the best / worst spectrum (by per-sample R^2 of the full recon)
    while streaming, keeping only those two rows."""
    def __init__(self):
        self.best = (-np.inf, None)
        self.worst = (np.inf, None)

    def update(self, labels, t, rec, phy, x_rows) -> None:
        r2 = per_sample_r2(t, rec)
        i, j = int(np.argmax(r2)), int(np.argmin(r2))
        if r2[i] > self.best[0]:
            self.best = (r2[i], (int(labels[i]), t[i].copy(), rec[i].copy(), phy[i].copy(), np.array(x_rows[i])))
        if r2[j] < self.worst[0]:
            self.worst = (r2[j], (int(labels[j]), t[j].copy(), rec[j].copy(), phy[j].copy(), np.array(x_rows[j])))

    def as_plot_args(self):
        b, w = self.best[1], self.worst[1]
        return dict(
            y_true=np.stack([b[1], w[1]]),          # type: ignore
            y_pred_full=np.stack([b[2], w[2]]),     # type: ignore
            y_pred_phys=np.stack([b[3], w[3]]),     # type: ignore
            X=np.stack([b[4], w[4]]),               # type: ignore
            sample_indices=[0, 1],
            sample_labels=[b[0], w[0]],             # type: ignore
        )


def load_fold(model_prefix, k, input_dim, n_elements, wl_grid):
    ckpt_path = Path(f"{model_prefix}_fold{k}.pt")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
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
    del checkpoint
    return model, scaler


class FoldModels:
    """Loads fold models on demand; keeps them cached if they are small enough
    (so the lockbox pass doesn't rebuild 7 models per block)."""
    def __init__(self, prefix, input_dim, n_elements, wl_grid):
        self.args = (prefix, input_dim, n_elements, wl_grid)
        self.cache, self.use_cache = {}, None

    def get(self, k):
        if k in self.cache:
            return self.cache[k]
        model, scaler = load_fold(self.args[0], k, *self.args[1:])
        if self.use_cache is None:
            nbytes = sum(t.numel() * t.element_size() for t in list(model.parameters()) + list(model.buffers()))
            self.use_cache = nbytes * NUM_FOLDS <= MODEL_CACHE_LIMIT_GB * 1e9
            print(f"[models] one model = {nbytes / 1e6:.0f} MB -> {'caching all folds' if self.use_cache else 'reloading per use'}")
        if self.use_cache:
            self.cache[k] = (model, scaler)
        return model, scaler

    def release(self):
        if not self.use_cache:
            gc.collect()


@torch.inference_mode()
def infer_batch(model, scaler, spec_np: np.ndarray):
    """One batch -> un-scaled (reconstruction, physics-only) as float32 numpy."""
    xb = scaler.transform(torch.from_numpy(spec_np))
    xb = torch.as_tensor(xb, dtype=torch.float32).to(DEVICE)
    out = model(xb)
    rec = scaler.inverse_transform(out["reconstructed_spectrum"].cpu().numpy())
    phy = scaler.inverse_transform(out["ideal_physics_spectrum"].cpu().numpy())
    return rec, phy


def evaluate_p3vae_decoder(model_prefix: str | Path, batch_size: int = 64):
    print(f"=== Running Standalone P3VAE Decoder Evaluation [{DEVICE}] ===")

    # 1. Load data and canonical splits
    data = load_crossval_data(H5_FILE)
    log_mem("after load")
    split_path = SPLIT_DIR / f'max12_{NUM_FOLDS}_folds_seed{SEED}.npz'
    test_idx, cv_idx, cv_fold_id = build_splits_once(data, split_path, NUM_FOLDS)
    test_idx, cv_idx = np.asarray(test_idx), np.asarray(cv_idx)
    has_test = len(test_idx) > 0

    y_np = data['y'].numpy()
    X_np = data['X'].numpy()
    input_dim = y_np.shape[1]
    n_elements = X_np.shape[1]
    wavelengths = data['wavelengths']
    wl_grid = torch.from_numpy(wavelengths).to(DEVICE)

    # 2. Identify OOF experimental rows and select Complex & Trace indices
    eval_pos_all = np.flatnonzero(cv_fold_id >= 0)             # positions within cv_idx
    eval_rank = np.full(len(cv_idx), -1, dtype=np.int64)
    eval_rank[eval_pos_all] = np.arange(len(eval_pos_all))

    dopant_indices = [i for i, name in enumerate(data['names']) if name not in HOST_ELEMENTS]
    X_cv_exp = X_np[cv_idx[eval_pos_all]][:, dopant_indices]

    dopant_sums = X_cv_exp.sum(axis=1)
    active_counts = (X_cv_exp > 1e-4).sum(axis=1)

    # Complex Sample: Highest number of active dopants, then highest concentration
    max_active = np.max(active_counts)
    complex_candidates = np.where(active_counts == max_active)[0]
    complex_idx = int(complex_candidates[np.argmax(dopant_sums[complex_candidates])])

    # Trace Sample: Lowest non-zero dopant concentration
    nonzero_mask = dopant_sums > 1e-4
    trace_candidates = np.where(nonzero_mask)[0]
    trace_idx = int(trace_candidates[np.argmin(dopant_sums[trace_candidates])])

    print(f"\n--- Targeted Sample Selection (OOF Array Indexing) ---")
    print(f"  - Complex Sample Index : #{complex_idx} ({active_counts[complex_idx]} dopants, total = {dopant_sums[complex_idx] * MAX_WT_PERCENT:.2f} wt%)")
    print(f"  - Trace Sample Index   : #{trace_idx} ({active_counts[trace_idx]} dopants, total = {dopant_sums[trace_idx] * MAX_WT_PERCENT:.2f} wt%)")

    available = []
    for k in range(NUM_FOLDS):
        if Path(f"{model_prefix}_fold{k}.pt").exists():
            available.append(k)
        else:
            print(f"⚠️ Warning: Missing checkpoint {Path(f'{model_prefix}_fold{k}.pt').name}. Skipping fold {k}.")
    if not available:
        raise RuntimeError("No model checkpoints were loaded!")
    fm = FoldModels(model_prefix, input_dim, n_elements, wl_grid)

    # 3. OOF evaluation pass
    mean_cv64 = rows_mean(y_np, cv_idx[eval_pos_all])
    train_mean_cv = mean_cv64.astype(np.float32)
    peak_cols_cv = np.argsort(train_mean_cv)[-N_PEAKS:][::-1]

    stats_cv = RegionStats(mean_cv64, train_mean_cv)
    bw_cv = BestWorst()
    targeted_cv = TargetedSamples([complex_idx, trace_idx])
    yt_pk, yp_pk = [], []

    for k in available:
        print(f"Processing OOF Fold {k + 1}/{NUM_FOLDS}...")
        model, scaler = fm.get(k)
        va_pos = np.where(cv_fold_id == k)[0]
        for i in range(0, len(va_pos), batch_size):
            pos = va_pos[i:i + batch_size]
            rows = cv_idx[pos]
            t = y_np[rows]
            rec, phy = infer_batch(model, scaler, t)
            stats_cv.update(t, rec, phy)
            bw_cv.update(eval_rank[pos], t, rec, phy, X_np[rows])
            targeted_cv.update(eval_rank[pos], t, rec, phy, X_np[rows])
            yt_pk.append(t[:, peak_cols_cv])
            yp_pk.append(rec[:, peak_cols_cv])
        del model, scaler
        fm.release()
        log_mem(f"after OOF fold {k}")

    # Output Tables & Figures
    stats_cv.print_table(f"P3VAE Decoder OOF Benchmarks ({len(available)} Folds)")
    
    # Complex vs Trace Overlay Plot
    plot_sample_spectra(
        wavelengths=wavelengths, feature_names=data['names'],
        title_tag="P3VAE Decoder OOF (Complex vs. Trace)",
        save_path=OUTPUT_DIR / f"p3vae_decoder_complex_vs_trace_{TIMESTAMP}.png",
        **targeted_cv.as_plot_args()
    )

    # Best vs Worst Overlay Plot
    plot_sample_spectra(
        wavelengths=wavelengths, feature_names=data['names'],
        title_tag="P3VAE Decoder OOF (Best vs. Worst)",
        save_path=OUTPUT_DIR / f"p3vae_decoder_best_vs_worst_{TIMESTAMP}.png",
        **bw_cv.as_plot_args()
    )

    del yt_pk, yp_pk, stats_cv, bw_cv, targeted_cv
    gc.collect()

    # 4. Lockbox evaluation pass (if present)
    if has_test:
        mean_te64 = rows_mean(y_np, test_idx)
        peak_cols_te = np.argsort(mean_te64.astype(np.float32))[-N_PEAKS:][::-1]
        stats_te, bw_te = RegionStats(mean_te64, train_mean_cv), BestWorst()
        yt_pk, yp_pk = [], []
        block_rows = max(batch_size, int(LOCKBOX_BLOCK_GB * 1e9 // (input_dim * 4)))
        n_blocks = int(np.ceil(len(test_idx) / block_rows))

        for b, a in enumerate(range(0, len(test_idx), block_rows)):
            print(f"Lockbox block {b + 1}/{n_blocks} ({len(available)} folds)...")
            rows_blk = test_idx[a:a + block_rows]
            t_blk = y_np[rows_blk]
            rec_sum, phy_sum = np.zeros_like(t_blk), np.zeros_like(t_blk)

            for k in available:
                model, scaler = fm.get(k)
                for i in range(0, len(rows_blk), batch_size):
                    rec, phy = infer_batch(model, scaler, t_blk[i:i + batch_size])
                    rec_sum[i:i + len(rec)] += rec
                    phy_sum[i:i + len(phy)] += phy
                del model, scaler
                fm.release()

            rec_sum /= len(available)
            phy_sum /= len(available)

            for i in range(0, len(rows_blk), 1024):
                sl = slice(i, i + 1024)
                stats_te.update(t_blk[sl], rec_sum[sl], phy_sum[sl])
                bw_te.update(np.arange(a + i, a + i + len(t_blk[sl])), t_blk[sl], rec_sum[sl], phy_sum[sl],
                             X_np[rows_blk[sl]])
                yt_pk.append(t_blk[sl][:, peak_cols_te])
                yp_pk.append(rec_sum[sl][:, peak_cols_te])
            del t_blk, rec_sum, phy_sum
            gc.collect()

        stats_te.print_table(f"P3VAE Decoder Lockbox Benchmarks (Ensemble of {len(available)} Folds)")
        plot_sample_spectra(
            wavelengths=wavelengths, feature_names=data['names'],
            title_tag="P3VAE Decoder Lockbox Ensemble",
            save_path=OUTPUT_DIR / f"p3vae_decoder_lockbox_samples_{TIMESTAMP}.png",
            **bw_te.as_plot_args()
        )

    print("Decoder evaluation complete! All figures and benchmark metrics generated safely.")

if __name__ == "__main__":
    # Pass your trained model prefix here (without the _foldK.pt suffix)
    MODEL_PREFIX = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/p3VAE/trained_models/best_p3vae6_model_20261024_211302"
    evaluate_p3vae_decoder(MODEL_PREFIX)