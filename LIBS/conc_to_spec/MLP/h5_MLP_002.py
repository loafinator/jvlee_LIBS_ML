from __future__ import annotations

import sys
import time
import h5py
import torch
import copy
import gc
import hashlib

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Any, Dict

# Add project root setup matching your directory layout
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from setup import add_project_root_to_path
add_project_root_to_path(parent_generation=1)
from utils import get_h5_ds


# =============================================================================
# Global Configuration & File Paths
# =============================================================================
H5_FILE = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/max12_cts_noleak_crossval.h5"
SPLIT_DIR = Path('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/fold_splits_NoExtraDopants')
OUTPUT_DIR = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/conc_to_spec/MLP/h5_plots")
TRAINED_MODEL_DIR = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/conc_to_spec/MLP/trained_models")
NPZ_PATH = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/conc_to_spec/MLP/h5_plots/mlp_spectral_predictions_20261024_135220.npz")
OUTPUT_SAVE_PATH = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/conc_to_spec/MLP/h5_plots/updated_sample_plot.png")

NUM_FOLDS = 7
THRESH = 2.8
MAX_WT_PERCENT = 12.0
SEED = 42
HOST_ELEMENTS = {'Li', 'K', 'Cl'}
EXP_DOPANTS = {'Ce', 'Gd', 'Sm', 'U'}

# MLP Hyperparameters (Concentration -> Spectrum)
EPOCHS = 150
BATCH_SIZE = 256
LEARNING_RATE = 1e-3          # peak LR of the one-cycle schedule
WEIGHT_DECAY = 1e-4
HIDDEN_DIMS = [256, 512, 1024, 1024]
DROPOUT = 0.2

# Early stopping config
PATIENCE = 15
MIN_DELTA = 1e-06
LOG_EVERY = 1

# Speed options
USE_AMP = True                # bf16 autocast on Ampere+, fp16 + GradScaler on older GPUs
USE_COMPILE = False           # torch.compile
GPU_DATA_BUDGET_FRAC = 0.6    # keep dataset on-GPU if it fits

# Diagnostics
RUN_SANITY_CHECKS = True      # data stats + overfit-a-tiny-subset test
RUN_SHUFFLE_CONTROL = True    # train on shuffled targets; held-out R^2 must collapse to ~0
# LOG_EVERY = 5                 # epochs between held-out loss evaluations

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# PyTorch MLP Architecture (Concentration -> Spectra)
# =============================================================================
class PyTorchMLP(nn.Module):
    """Multi-Layer Perceptron for forward LIBS spectral synthesis.

    Inputs:  Concentrations (dopant weight fractions)
    Outputs: High-dimensional LIBS spectrum (wavelength intensity bins)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dims: list[int] = HIDDEN_DIMS,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        layers = []
        prev_dim = in_features
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

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
        nn.init.normal_(last.weight, std=0.01)

    @torch.no_grad()
    def set_output_bias(self, target_mean: torch.Tensor):
        """Set output layer bias to the target mean spectrum, so epoch 0 matches 
        the 'predict the mean' baseline."""
        self.net[-1].bias.copy_(target_mean.to(self.net[-1].bias.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# Helper & Data Functions
# =============================================================================
def load_crossval_data(
    h5_path: str, 
    spike_thresh: float = THRESH,
    exp_dopants: set = EXP_DOPANTS,
    host_elements: set = HOST_ELEMENTS,
    load_y: bool = True
) -> Dict[str, Any]:
    """Loads dataset features into RAM. Stream-checks saturation spikes so 
    the full multi-gigabyte spectra matrix is never loaded unless load_y=True.
    """
    keep_elements = host_elements.union(exp_dopants)

    with h5py.File(h5_path, 'r') as hf:
        X = get_h5_ds(hf, 'X')[:].astype(np.float32)
        is_synth = get_h5_ds(hf, 'is_synthetic')[:]
        is_lockbox = get_h5_ds(hf, 'is_lockbox')[:].astype(bool) if 'is_lockbox' in hf else None
        group_id = get_h5_ds(hf, 'group_id')[:]
        wavelengths = get_h5_ds(hf, 'wavelengths')[:]
        names = [
            n.decode('utf-8') if isinstance(n, bytes) else str(n) 
            for n in get_h5_ds(hf, 'feature_cols')[:]
        ]

        # Stream row max calculations in chunks to save RAM
        y_ds = get_h5_ds(hf, 'y')
        n_rows = y_ds.shape[0]
        row_maxes = np.empty(n_rows, dtype=np.float32)
        for i in range(0, n_rows, 4096):
            row_maxes[i:i + 4096] = y_ds[i:i + 4096].max(axis=1)

        y_raw = y_ds[:].astype(np.float32) if load_y else None

    keep_idx = [i for i, name in enumerate(names) if name in keep_elements]
    extra_dopant_idx = [i for i, name in enumerate(names) if name not in keep_elements]

    clean_spike_mask = row_maxes <= spike_thresh
    has_extra_dopants = (X[:, extra_dopant_idx] > 0).any(axis=1)
    valid_synth_mask = (is_synth == 0) | (~has_extra_dopants)

    final_row_mask = clean_spike_mask & valid_synth_mask

    X_clean = X[final_row_mask][:, keep_idx]
    is_synth_clean = is_synth[final_row_mask]
    group_id_clean = group_id[final_row_mask]
    names_clean = [names[i] for i in keep_idx]

    out_dict = dict(
        X=torch.from_numpy(X_clean),
        is_synth=torch.from_numpy(is_synth_clean).long(),
        group_id=group_id_clean,
        wavelengths=wavelengths,
        names=names_clean,
    )

    if load_y and y_raw is not None:
        y_clean = y_raw[final_row_mask]
        y_tensor = torch.from_numpy(y_clean)
        l2_norms = torch.norm(y_tensor, p=2, dim=1, keepdim=True).clamp(min=1e-08)
        out_dict['y'] = y_tensor / l2_norms

    if is_lockbox is not None:
        out_dict['is_lockbox'] = is_lockbox[final_row_mask]

    return out_dict

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
    fold_id[exp] = best_fog[inv]
    return fold_id


def build_splits_once(data, split_path, num_folds, seed=42):
    """Loads existing pre-computed CV splits or creates them if absent."""
    split_path = Path(split_path).with_suffix('.npz')
    if split_path.exists():
        print(f"Loading cached splits from: {split_path}")
        d = np.load(split_path)
        test_idx, cv_idx, cv_fold_id = d['test_idx'], d['cv_idx'], d['cv_fold_id']
        n_rows = len(data['X'])
        if len(cv_idx) + len(test_idx) != n_rows or len(cv_fold_id) != len(cv_idx):
            raise ValueError(
                f"Cached split {split_path} covers {len(cv_idx) + len(test_idx)} rows but the filtered "
                f"data has {n_rows}. Delete the cache so it is rebuilt for the current filters."
            )
        return test_idx, cv_idx, cv_fold_id

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


def r2_calcs(y_true, y_pred):
    """Calculates overall R^2 score across high-dimensional spectral prediction."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2)
    return float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-8 else 0.0


def plot_parity_grid(y_true, y_pred, wavelengths=None, title_tag="MLP Benchmark", save_path=None):
    """Generates a grid of intensity parity plots across top spectral emission peak bins."""
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
    fold_preds,
    wavelengths,
    sample_indices=None,
    X=None,
    feature_names=None,
    title_tag="MLP Benchmark",
    save_path=None
):
    """Plots true vs ensemble predicted spectral overlays (top row) side-by-side
    with separate stacked residual panels underneath (bottom row).
    """
    if sample_indices is None:
        sample_indices = np.linspace(0, len(y_true) - 1, 2, dtype=int)

    n_samples = len(sample_indices)
    has_folds = (fold_preds.ndim == 3)

    fig, axes = plt.subplots(
        2, n_samples, 
        figsize=(6.5 * n_samples, 3.5), 
        sharex='col', 
        gridspec_kw={'height_ratios': [2.5, 1]}
    )

    for i, idx in enumerate(sample_indices):
        ax_main = axes[0, i] if n_samples > 1 else axes[0]
        ax_res = axes[1, i] if n_samples > 1 else axes[1]

        if has_folds:
            sample_folds = fold_preds[:, idx, :]
            y_mean = np.mean(sample_folds, axis=0)
            y_std = np.std(sample_folds, axis=0)
        else:
            y_mean = fold_preds[idx]
            y_std = None

        residual = y_true[idx] - y_mean

        # --- TOP PANEL: Actual vs Predicted ---
        ax_main.plot(wavelengths, y_true[idx], label="Actual", color="#1f77b4", lw=1.0, alpha=0.85)
        ax_main.plot(wavelengths, y_mean, label="Predicted", color="#d62728", lw=0.9, ls="--", alpha=0.9)

        if y_std is not None:
            ax_main.fill_between(
                wavelengths, y_mean - y_std, y_mean + y_std,
                color="#d62728", alpha=0.20, label="Uncertainty (±1σ)"
            )

        # Performance Metrics
        ss_res = np.sum((y_true[idx] - y_mean) ** 2)
        ss_tot = np.sum((y_true[idx] - np.mean(y_true[idx])) ** 2) + 1e-8
        r2 = 1.0 - (ss_res / ss_tot)

        # tag_label = f"Sample #{idx}"
        # ax_main.set_title(f"{tag_label} (Sample #{idx}) — $R^2$: {r2:.4f}", fontsize=11, fontweight="bold")
        ax_main.set_title(f"Sample #{idx} — $R^2$: {r2:.4f}", fontsize=11, fontweight="bold")
        ax_main.set_ylabel("Intensity", fontsize=10)
        ax_main.tick_params(axis="both", labelsize=10)  # Increased tick label size

        ax_main.grid(True, linestyle=":", alpha=0.4)

        # Composition Inset
        if X is not None and feature_names is not None:
            sample_conc = X[idx].cpu().numpy() if hasattr(X[idx], "cpu") else X[idx]
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

        ax_res.set_xlabel("Wavelength (nm)", fontsize=10)
        ax_res.set_ylabel("Residual", fontsize=10, color="black")
        
        # Increased size to 10 and changed labelcolor from "#b25395" to "black"
        ax_res.tick_params(axis="both", labelcolor="black", labelsize=10)
        ax_res.grid(True, linestyle=":", alpha=0.4)

        # Symmetrical Y-limits centered on zero
        max_abs = max(abs(np.min(residual)), abs(np.max(residual))) * 1.1
        if max_abs > 0:
            ax_res.set_ylim(-max_abs, max_abs)

    plt.suptitle(f"{title_tag} — Measured vs. Reconstructed Spectra", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved Sample Spectra Plot to: {save_path}")
    plt.close()

def just_plotting(sample_indices: list[int] | None = None):
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"Could not find saved predictions file at: {NPZ_PATH}")

    print(f"Loading pre-computed predictions from: {NPZ_PATH.name}...")
    saved_data = np.load(NPZ_PATH)

    wavelengths = saved_data["wavelengths"]
    true_spectra = saved_data["true_cv_spectra"]
    pred_spectra = saved_data["pred_cv_spectra"]

    # Pass load_y=False so RAM usage stays under ~20 MB
    print("Loading composition features and fold splits...")
    data = load_crossval_data(H5_FILE, load_y=False)
    split_path = SPLIT_DIR / f'max12_{NUM_FOLDS}_folds_seed{SEED}.npz'
    _, cv_idx, cv_fold_id = build_splits_once(data, split_path, NUM_FOLDS, seed=SEED)

    dopant_indices = [i for i, name in enumerate(data['names']) if name not in HOST_ELEMENTS]
    feature_names = [data['names'][i] for i in dopant_indices]

    # Align concentration rows with true_cv_spectra (filtered by exp_mask)
    exp_mask = cv_fold_id >= 0
    X_cv_exp = data['X'][cv_idx][exp_mask][:, dopant_indices]

    n_samples_avail = len(true_spectra)

    if sample_indices is None:
        sample_r2s = [
            1.0 - (np.sum((true_spectra[i] - pred_spectra[i]) ** 2) / 
                  (np.sum((true_spectra[i] - np.mean(true_spectra[i])) ** 2) + 1e-8))
            for i in range(n_samples_avail)
        ]
        sample_indices = [int(np.argmax(sample_r2s)), int(np.argmin(sample_r2s))]
    else:
        # Clamp or wrap indices to ensure they fall within valid array bounds
        sample_indices = [idx % n_samples_avail for idx in sample_indices]

    print(f"Plotting sample indices: {sample_indices} (out of {n_samples_avail} available CV samples)")

    plot_sample_spectra(
        y_true=true_spectra,
        fold_preds=pred_spectra,
        wavelengths=wavelengths,
        sample_indices=sample_indices,
        X=X_cv_exp,
        feature_names=feature_names,
        title_tag="MLP OOF (Complex vs. Trace)",
        save_path=OUTPUT_SAVE_PATH
    )    
def _chunk_stats(y_true, y_pred, cols, chunk=2048):
    """SSE, SAE, SST(about per-wavelength mean) and element count over the columns `cols` (index array).
    y_pred may be [N, n_wl] or a single [n_wl] vector broadcast over rows. Never copies the full array."""
    n = y_true.shape[0]
    sse = sae = 0.0
    s1 = s2 = None
    for i in range(0, n, chunk):
        t = y_true[i:i + chunk][:, cols].astype(np.float64)
        p = (y_pred[i:i + chunk][:, cols].astype(np.float64) if y_pred.ndim == 2
             else np.broadcast_to(y_pred[cols].astype(np.float64), t.shape))
        d = t - p
        sse += float(np.einsum('ij,ij->', d, d)); sae += float(np.abs(d).sum())
        if s1 is None: s1, s2 = t.sum(0), np.einsum('ij,ij->j', t, t)
        else: s1 += t.sum(0); s2 += np.einsum('ij,ij->j', t, t)
    sst = float((s2 - s1 ** 2 / n).sum())   # type: ignore
    return sse, sae, sst, n * len(cols)
 
def print_benchmark_table(y_true, y_pred_model, y_train_mean, title_tag, wavelengths=None):
    mean_spec = y_true.mean(axis=0, dtype=np.float64)
    top = mean_spec >= np.percentile(mean_spec, 90)
    regions = [("Full Spectrum", np.arange(y_true.shape[1])),
               ("Peak Channels (Top 10%)", np.where(top)[0]),
               ("Continuum / Background", np.where(~top)[0])]
    zero_vec = np.zeros(y_true.shape[1], dtype=np.float32)
    print("\n" + "=" * 95); print(f" {title_tag}"); print("=" * 95)
    print(f"{'Spectral Region':<25} | {'Strategy':<18} | {'Global R^2':<10} | {'MAE':<10} | {'RMSE':<10}")
    print("-" * 95)
    for reg_name, cols in regions:
        for strat_name, y_hat in [("Predict Zero", zero_vec), ("Predict Mean", y_train_mean),
                                  ("MLP Model", y_pred_model)]:
            sse, sae, sst, cnt = _chunk_stats(y_true, y_hat, cols)
            r2 = 1.0 - sse / sst if sst > 1e-8 else 0.0
            print(f"{reg_name:<25} | {strat_name:<18} | {r2:10.4f} | {sae / cnt:10.4f} | {np.sqrt(sse / cnt):10.4f}")
        print("-" * 95)
 

# =============================================================================
# Fast GPU-resident training utilities
# =============================================================================
def stash(X: torch.Tensor, name: str) -> torch.Tensor:
    """Park array where batches can be gathered cheaply: GPU or pinned CPU memory."""
    if DEVICE.type != "cuda":
        return X
    X = X.to(torch.float16)
    nbytes = X.numel() * X.element_size()
    free, _ = torch.cuda.mem_get_info()
    if nbytes < GPU_DATA_BUDGET_FRAC * free:
        print(f"  [{name}] {nbytes / 1e9:.2f} GB (fp16) -> GPU")
        return X.to(DEVICE)
    print(f"  [{name}] {nbytes / 1e9:.2f} GB does not fit on GPU -> pinned CPU memory")
    return X.pin_memory()


def gather_x(Xs: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    bx = Xs[idx]
    if bx.device != DEVICE:
        bx = bx.to(DEVICE, non_blocking=True)
    return bx.float()


def amp_setup():
    if not (USE_AMP and DEVICE.type == "cuda"):
        return None, None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, None
    return torch.float16, torch.amp.GradScaler("cuda")  # type: ignore


@torch.no_grad()
def predict(model: nn.Module, Xs: torch.Tensor, idx: torch.Tensor, bs: int = 1024) -> torch.Tensor:
    model.eval()
    amp_dtype, _ = amp_setup()
    out = []
    for i in range(0, idx.numel(), bs):
        bx = gather_x(Xs, idx[i:i + bs])
        with torch.autocast(DEVICE.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out.append(model(bx).float())
    return torch.cat(out)


def fit(model, Xs, Y_fit, Y_eval, tr_idx, va_idx, epochs, seed, verbose=True):
    """Training loop with validation checking and early stopping with weight restoration."""
    n = tr_idx.numel()
    steps = n // BATCH_SIZE
    if steps == 0:
        raise ValueError(f"Only {n} training rows for BATCH_SIZE={BATCH_SIZE}.")

    g = torch.Generator(device=tr_idx.device)
    g.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=5
    )
    amp_dtype, scaler = amp_setup()
    hist = dict(epoch=[], train=[], val=[], stopped_epoch=epochs)
    
    # Early Stopping state tracking
    best_val_loss = float("inf")
    best_model_weights = None
    patience_counter = 0
    
    t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        model.train()
        perm = tr_idx[torch.randperm(n, device=tr_idx.device, generator=g)]
        running = torch.zeros((), device=DEVICE)

        for s in range(steps):
            idx = perm[s * BATCH_SIZE:(s + 1) * BATCH_SIZE]
            bx = gather_x(Xs, idx)
            by = Y_fit[idx.to(Y_fit.device)]
            with torch.autocast(DEVICE.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                pred = model(bx)
            loss = F.mse_loss(pred.float(), by)

            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            # sched.step()
            running += loss.detach()

        # Evaluate Validation Metrics
        if ep == 1 or ep % LOG_EVERY == 0 or ep == epochs:
            tr_loss = (running / steps).item()
            va_pred = predict(model, Xs, va_idx)
            va_loss = F.mse_loss(va_pred, Y_eval[va_idx.to(Y_eval.device)]).item()

            sched.step(va_loss)
            
            hist["epoch"].append(ep)
            hist["train"].append(tr_loss)
            hist["val"].append(va_loss)

            if verbose:
                print(f"    ep {ep:3d}/{epochs} | train MSE {tr_loss:.6f} (dropout on) | "
                      f"held-out MSE {va_loss:.6f} | lr {sched.get_last_lr()[0]:.2e}")

            # Early Stopping Evaluation
            if va_loss < (best_val_loss - MIN_DELTA):
                best_val_loss = va_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                if verbose:
                    print(f"    --> Early stopping triggered at epoch {ep}. Best Val MSE: {best_val_loss:.6f}")
                hist["stopped_epoch"] = ep  # type: ignore
                break

    # Restore best validation checkpoint weights
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        if verbose:
            print(f"    --> Restored best model weights from epoch {hist['epoch'][hist['val'].index(best_val_loss)]}")

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    hist["seconds"] = dt  # type: ignore
    hist["samples_per_sec"] = len(hist["epoch"]) * steps * BATCH_SIZE / dt
    return hist


def new_model(in_dim: int, out_dim: int, init_bias: torch.Tensor) -> nn.Module:
    model = PyTorchMLP(
        in_features=in_dim, out_features=out_dim, hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
    ).to(DEVICE)
    model.set_output_bias(init_bias)
    if USE_COMPILE:
        model = torch.compile(model)
    return model  # type: ignore


def fold_indices(fold_id_t: torch.Tensor, k: int):
    tr_idx = torch.nonzero(fold_id_t != k).squeeze(1)
    va_idx = torch.nonzero(fold_id_t == k).squeeze(1)
    return tr_idx, va_idx


def mean_baseline_mse(Y: torch.Tensor, tr_idx: torch.Tensor, va_idx: torch.Tensor) -> float:
    """Held-out MSE of the constant 'predict the training mean' baseline."""
    mu = Y[tr_idx.to(Y.device)].mean(0, keepdim=True)
    return F.mse_loss(mu.expand(va_idx.numel(), -1), Y[va_idx.to(Y.device)]).item()


# =============================================================================
# Diagnostics
# =============================================================================
def data_report(X_cv_cpu, Y_cv_np, fold_id, target_names):
    print("\n" + "=" * 80)
    print(" SANITY CHECK 1: inputs and targets")
    print("=" * 80)
    print(f"Concentrations (Inputs): {tuple(X_cv_cpu.shape)} | finite: {bool(torch.isfinite(X_cv_cpu).all())} | "
          f"min/median/max: {X_cv_cpu.min():.3f}/{X_cv_cpu.median():.3f}/{X_cv_cpu.max():.3f}")
    
    sub = torch.from_numpy(Y_cv_np[:: max(1, len(Y_cv_np) // 2000)])
    rms = sub.pow(2).mean(1).sqrt()
    print(f"Spectra (Targets): {Y_cv_np.shape} | finite: {bool(np.isfinite(Y_cv_np).all())} | "
          f"per-spectrum RMS min/median/max: {rms.min():.3f}/{rms.median():.3f}/{rms.max():.3f} (expect ~1.0)")


def overfit_test(Xs, Y, fold_id_t, in_dim, out_dim, seed):
    print("\n" + "=" * 80)
    print(" SANITY CHECK 2: can the model memorise 512 training samples?")
    print("=" * 80)
    g = torch.Generator().manual_seed(seed)
    cand = torch.nonzero(fold_id_t >= 0).squeeze(1)
    pick = cand[torch.randperm(cand.numel(), generator=g)[:512].to(cand.device)]
    Yp = Y[pick.to(Y.device)]
    base = ((Yp - Yp.mean(0)) ** 2).mean().item()
    model = new_model(in_dim, out_dim, Yp.mean(0))
    fit(model, Xs, Y, Y, pick, pick, epochs=150, seed=seed, verbose=False)
    final = F.mse_loss(predict(model, Xs, pick), Yp).item()
    ok = final < 0.2 * base
    print(f"Predict-mean MSE on these rows: {base:.6f} | MSE after 150 epochs: {final:.6f} "
          f"({100 * final / base:.1f}% of baseline)")
    print("PASS" if ok else "FAIL: cannot fit 512 samples -> bug in data/targets/optimiser.")
    del model
    return ok


def shuffle_control(Xs, Y, fold_id_t, in_dim, out_dim, seed):
    print("\n" + "=" * 80)
    print(" SANITY CHECK 3: shuffled-label control (fold 0)")
    print("=" * 80)
    tr_idx, va_idx = fold_indices(fold_id_t, 0)
    g = torch.Generator().manual_seed(seed + 999)
    Y_shuf = Y.clone()
    perm = torch.randperm(tr_idx.numel(), generator=g).to(tr_idx.device)
    Y_shuf[tr_idx.to(Y.device)] = Y[tr_idx[perm].to(Y.device)]

    model = new_model(in_dim, out_dim, Y_shuf[tr_idx.to(Y.device)].mean(0))
    fit(model, Xs, Y_shuf, Y, tr_idx, va_idx, epochs=EPOCHS, seed=seed, verbose=False)
    pred = predict(model, Xs, va_idx).cpu().numpy()
    true = Y[va_idx.to(Y.device)].cpu().numpy()
    r2 = r2_calcs(true, pred)
    print(f"  Held-out Global R^2 with shuffled targets: {r2:8.3f}")
    ok = bool(r2 < 0.1)
    print("PASS: nothing to learn -> nothing learned (no leakage)" if ok else
          "FAIL: model 'learns' from shuffled targets -> leakage detected")
    del model
    return ok


def plot_learning_curves(histories, baselines, save_path):
    n = len(histories)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.4 * rows), squeeze=False)
    for k, (h, b) in enumerate(zip(histories, baselines)):
        ax = axes.flat[k]
        ax.plot(h["epoch"], h["train"], "o-", ms=3, label="train (dropout on)")
        ax.plot(h["epoch"], h["val"], "s-", ms=3, label="held-out fold")
        ax.axhline(b, color="gray", ls="--", label="predict-mean baseline")
        ax.set_yscale("log"); ax.set_title(f"Fold {k + 1}", fontsize=10, fontweight="bold")
        ax.set_xlabel("epoch"); ax.set_ylabel("MSE"); ax.grid(True, ls=":", alpha=0.6)
        if k == 0:
            ax.legend(fontsize=7)
    for k in range(n, rows * cols):
        fig.delaxes(axes.flat[k])
    plt.suptitle("Learning curves (held-out loss must fall well below the dashed baseline)", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved learning curves to: {save_path}")


def print_learning_verdict(y_true, y_pred, tag):
    print(f"\n--- Learning verdict: {tag} ---")
    global_r2 = r2_calcs(y_true, y_pred)
    if global_r2 <= 0.05:
        status = "NOT LEARNING (no better than predicting the mean spectrum)"
    elif global_r2 < 0.5:
        status = "weak"
    else:
        status = "learning"
    print(f"  Global Spectrum R^2 = {global_r2:7.3f}  -> {status}")


# =============================================================================
# Memory-lean evaluation of saved checkpoints
#
# Peak RAM is a few hundred MB to ~1 GB regardless of dataset size. Nothing of shape
# [N_rows, n_wavelengths] is ever held in memory:
#   * spectra are filtered / L2-normalised / scaled by sqrt(n_wl) while STREAMING out of the h5 file
#     into a float32 disk cache, which is then read back through np.memmap in small chunks;
#   * predictions are made chunk-by-chunk (each CV row is predicted by the model of its own fold,
#     the lockbox by the mean of all fold models) and reduced to per-wavelength sufficient statistics;
#   * the tables, verdicts and plots are produced from those statistics.
# =============================================================================
CACHE_DIR = OUTPUT_DIR.parent / "cache"     # ~ (n_clean_rows * n_wavelengths * 4) bytes, built once
STREAM_CHUNK = 1024                          # rows per chunk (~80 MB per float64 temporary at 10k wavelengths)
N_PARITY_PEAKS = 12                          # must match the 12 channels plot_parity_grid draws


def _cache_tag(h5_path, spike_thresh, exp_dopants, host_elements) -> str:
    st = Path(h5_path).stat()
    key = (f"{Path(h5_path).resolve()}|{st.st_size}|{int(st.st_mtime)}|{spike_thresh}|"
           f"{sorted(exp_dopants)}|{sorted(host_elements)}")
    return f"{Path(h5_path).stem}_{hashlib.md5(key.encode()).hexdigest()[:10]}"


def load_clean_spectra_lean(
    h5_path: str,
    cache_dir: Path = CACHE_DIR,
    spike_thresh: float = THRESH,
    exp_dopants: set = EXP_DOPANTS,
    host_elements: set = HOST_ELEMENTS,
    chunk: int = 4096,
):
    """Drop-in for `load_crossval_data(...)` + `* sqrt(n_wavelengths)` that never loads the spectra.

    Applies exactly the same row filters, L2 normalisation and RMS-1 scaling, but streams the h5 'y'
    dataset in row chunks into a float32 file (built once, then reused). Returns
    (data, Y): `data` has the same keys as load_crossval_data minus 'y'; `Y` is a read-only
    np.memmap of shape [n_clean_rows, n_wavelengths] already in RMS-1 units."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = _cache_tag(h5_path, spike_thresh, exp_dopants, host_elements)
    bin_path = cache_dir / f"{tag}_spectra_f32.bin"
    meta_path = cache_dir / f"{tag}_meta.npz"
    keep_elements = host_elements.union(exp_dopants)

    with h5py.File(h5_path, 'r') as hf:
        X = get_h5_ds(hf, 'X')[:].astype(np.float32)
        is_synth = get_h5_ds(hf, 'is_synthetic')[:]
        is_lockbox = get_h5_ds(hf, 'is_lockbox')[:].astype(bool) if 'is_lockbox' in hf else None
        group_id = get_h5_ds(hf, 'group_id')[:]
        wavelengths = get_h5_ds(hf, 'wavelengths')[:]
        names = [
            n.decode('utf-8') if isinstance(n, bytes) else str(n)
            for n in get_h5_ds(hf, 'feature_cols')[:]
        ]
        ds_y = get_h5_ds(hf, 'y')
        n_total, n_wl = ds_y.shape

        keep_idx = [i for i, name in enumerate(names) if name in keep_elements]
        extra_dopant_idx = [i for i, name in enumerate(names) if name not in keep_elements]
        has_extra_dopants = (X[:, extra_dopant_idx] > 0).any(axis=1)
        valid_synth_mask = (is_synth == 0) | (~has_extra_dopants)

        if bin_path.exists() and meta_path.exists():
            meta = np.load(meta_path)
            final_row_mask = meta['final_row_mask']
            n_dropped_spikes = int(meta['n_dropped_spikes'])
            n_dropped_synth = int(meta['n_dropped_synth'])
            print(f"Using cached clean spectra: {bin_path}")
        else:
            print("Building clean-spectra cache (one-time streaming pass over the h5 file)...")
            scale = n_wl ** 0.5
            clean_spike_mask = np.zeros(n_total, dtype=bool)
            final_row_mask = np.zeros(n_total, dtype=bool)
            tmp_path = bin_path.with_suffix('.tmp')
            with open(tmp_path, 'wb') as f:
                for a in range(0, n_total, chunk):
                    b = min(a + chunk, n_total)
                    yc = ds_y[a:b].astype(np.float32)
                    clean = yc.max(axis=1) <= spike_thresh
                    sel = clean & valid_synth_mask[a:b]
                    clean_spike_mask[a:b] = clean
                    final_row_mask[a:b] = sel
                    if sel.any():
                        ys = yc[sel]
                        norms = np.maximum(np.linalg.norm(ys, axis=1, keepdims=True), 1e-8)
                        ((ys / norms) * scale).astype(np.float32).tofile(f)
            n_dropped_spikes = n_total - int(clean_spike_mask.sum())
            n_dropped_synth = int((clean_spike_mask & ~valid_synth_mask).sum())
            tmp_path.replace(bin_path)
            np.savez(meta_path, final_row_mask=final_row_mask,
                     n_dropped_spikes=n_dropped_spikes, n_dropped_synth=n_dropped_synth)

    n_rows = int(final_row_mask.sum())
    expected = n_rows * n_wl * 4
    if bin_path.stat().st_size != expected:
        raise ValueError(f"Cache {bin_path} has {bin_path.stat().st_size} bytes, expected {expected}. "
                         f"Delete it (and its _meta.npz) so it is rebuilt.")

    names_clean = [names[i] for i in keep_idx]
    if n_dropped_spikes > 0 or n_dropped_synth > 0:
        print(f"Data loading summary:")
        print(f"  - Dropped {n_dropped_spikes} corrupted saturation shots (max > {spike_thresh}).")
        print(f"  - Filtered {n_dropped_synth} synthetic shots containing non-experimental dopants.")
        print(f"  - Retained {n_rows} total shots across {len(names_clean)} features: {names_clean}")

    data = dict(
        X=torch.from_numpy(X[final_row_mask][:, keep_idx]),
        is_synth=torch.from_numpy(is_synth[final_row_mask]).long(),
        group_id=group_id[final_row_mask],
        wavelengths=wavelengths,
        names=names_clean,
    )
    if is_lockbox is not None:
        data['is_lockbox'] = is_lockbox[final_row_mask]

    Y = np.memmap(bin_path, dtype=np.float32, mode='r', shape=(n_rows, n_wl))
    return data, Y


def stream_spectral_stats(Y, rows, predict_fn, conc, folds=None, train_mean=None,
                          n_peaks: int = N_PARITY_PEAKS, chunk: int = STREAM_CHUNK):
    """Two chunked passes over Y[rows]; returns per-wavelength sufficient statistics.

    rows       : indices into Y (the rows being evaluated)
    predict_fn : (conc_chunk, folds_chunk_or_None) -> predicted spectra, numpy [chunk, n_wl]
    conc       : model inputs aligned with `rows`
    folds      : per-row fold id (each row is predicted by its own fold's model) or None (ensemble)
    train_mean : spectrum used by the 'Predict Mean' strategy (default: the mean of these rows)
    """
    rows = np.asarray(rows)
    n, n_wl = len(rows), Y.shape[1]

    # Pass 1: truth only (mean spectrum -> region masks, Predict-Zero and Predict-Mean baselines)
    s1, s2, sabs = np.zeros(n_wl), np.zeros(n_wl), np.zeros(n_wl)
    for a in range(0, n, chunk):
        t = np.asarray(Y[rows[a:a + chunk]], dtype=np.float64)
        s1 += t.sum(0)
        s2 += np.einsum('ij,ij->j', t, t)
        sabs += np.abs(t).sum(0)
    mean_spec = s1 / n
    mu = mean_spec if train_mean is None else np.asarray(train_mean, dtype=np.float64)
    top_cols = np.argsort(mean_spec)[-n_peaks:][::-1]

    # Pass 2: truth + predictions
    sse_m, sae_m = np.zeros(n_wl), np.zeros(n_wl)
    sse_mu, sae_mu = np.zeros(n_wl), np.zeros(n_wl)
    row_r2 = np.empty(n)
    yt_top = np.empty((n, len(top_cols)), dtype=np.float32)
    yp_top = np.empty((n, len(top_cols)), dtype=np.float32)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        t = np.asarray(Y[rows[a:b]], dtype=np.float64)
        p = np.asarray(predict_fn(conc[a:b], None if folds is None else folds[a:b]), dtype=np.float64)
        d = t - p
        sse_m += np.einsum('ij,ij->j', d, d)
        sae_m += np.abs(d).sum(0)
        dm = t - mu
        sse_mu += np.einsum('ij,ij->j', dm, dm)
        sae_mu += np.abs(dm).sum(0)
        ss_res = np.einsum('ij,ij->i', d, d)
        tc = t - t.mean(1, keepdims=True)
        ss_tot = np.einsum('ij,ij->i', tc, tc) + 1e-08
        row_r2[a:b] = 1.0 - ss_res / ss_tot
        yt_top[a:b] = t[:, top_cols]
        yp_top[a:b] = p[:, top_cols]

    return dict(n=n, s1=s1, s2=s2, sabs=sabs, mean_spec=mean_spec, mu=mu, top_cols=top_cols,
                sse_m=sse_m, sae_m=sae_m, sse_mu=sse_mu, sae_mu=sae_mu,
                row_r2=row_r2, yt_top=yt_top, yp_top=yp_top)


def print_table_from_stats(st, title_tag):
    """Same table as print_benchmark_table, computed from stream_spectral_stats() output."""
    mean_spec = st['mean_spec']
    top = mean_spec >= np.percentile(mean_spec, 90)
    regions = [("Full Spectrum", np.arange(len(mean_spec))),
               ("Peak Channels (Top 10%)", np.where(top)[0]),
               ("Continuum / Background", np.where(~top)[0])]
    sst_w = st['s2'] - st['s1'] ** 2 / st['n']
    strategies = [("Predict Zero", st['s2'], st['sabs']),
                  ("Predict Mean", st['sse_mu'], st['sae_mu']),
                  ("MLP Model", st['sse_m'], st['sae_m'])]

    print("\n" + "=" * 95); print(f" {title_tag}"); print("=" * 95)
    print(f"{'Spectral Region':<25} | {'Strategy':<18} | {'Global R^2':<10} | {'MAE':<10} | {'RMSE':<10}")
    print("-" * 95)
    for reg_name, cols in regions:
        sst = float(sst_w[cols].sum())
        cnt = st['n'] * len(cols)
        for strat_name, sse_w, sae_w in strategies:
            sse, sae = float(sse_w[cols].sum()), float(sae_w[cols].sum())
            r2 = 1.0 - sse / sst if sst > 1e-8 else 0.0
            print(f"{reg_name:<25} | {strat_name:<18} | {r2:10.4f} | {sae / cnt:10.4f} | {np.sqrt(sse / cnt):10.4f}")
        print("-" * 95)


def print_verdict_from_stats(st, tag):
    """Same read-out as print_learning_verdict, from stats."""
    sst = float((st['s2'] - st['s1'] ** 2 / st['n']).sum())
    global_r2 = float(1.0 - st['sse_m'].sum() / sst) if sst > 1e-8 else 0.0
    print(f"\n--- Learning verdict: {tag} ---")
    if global_r2 <= 0.05:
        status = "NOT LEARNING (no better than predicting the mean spectrum)"
    elif global_r2 < 0.5:
        status = "weak"
    else:
        status = "learning"
    print(f"  Global Spectrum R^2 = {global_r2:7.3f}  -> {status}")


def make_predict_fn(models, n_wl):
    @torch.no_grad()
    def fn(conc_np, folds_np):
        x = torch.from_numpy(np.ascontiguousarray(conc_np, dtype=np.float32)).to(DEVICE)
        if folds_np is None:                      # ensemble mean over all fold models
            out = torch.zeros(len(x), n_wl, device=DEVICE)
            for m in models:
                out += m(x).float()
            out /= len(models)
        else:                                     # each row -> the model of its own fold
            out = torch.empty(len(x), n_wl, device=DEVICE)
            for k in np.unique(folds_np):
                sel = torch.from_numpy(folds_np == k).to(DEVICE)
                out[sel] = models[int(k)](x[sel]).float()
        return out.cpu().numpy()
    return fn


@torch.no_grad()
def per_model_preds(models, conc_row):
    """[K, n_wl] predictions of every fold model for one sample (for the +-1 sigma band)."""
    x = torch.from_numpy(np.ascontiguousarray(conc_row[None], dtype=np.float32)).to(DEVICE)
    return np.stack([m(x).float().cpu().numpy()[0] for m in models])


class _Rows:
    """Duck-typed stand-in for a big [N, n_wl] array when plot_sample_spectra only touches a few rows."""
    def __init__(self, d, ndim=2):
        self.d, self.ndim = d, ndim

    def __getitem__(self, key):
        if isinstance(key, tuple):      # fold_preds[:, idx, :] on the 3-D [K, N, n_wl] layout
            key = key[1]
        return self.d[int(key)]


class _Cols:
    """Duck-typed stand-in for a big [N, n_wl] array when plot_parity_grid only touches 12 columns."""
    def __init__(self, mean_spec, cols):
        self.mean_spec, self.cols = mean_spec, cols

    def mean(self, axis=0):
        return self.mean_spec

    def __getitem__(self, key):
        return self.cols[int(key[1])]


def _parity_views(st):
    names = [int(c) for c in st['top_cols']]
    yt = _Cols(st['mean_spec'], {c: st['yt_top'][:, j] for j, c in enumerate(names)})
    yp = _Cols(st['mean_spec'], {c: st['yp_top'][:, j] for j, c in enumerate(names)})
    return yt, yp


def evaluate_saved_models(model_prefix: str | Path):
    print(f"=== Running Standalone Evaluation from Saved Checkpoints [{DEVICE}] ===")
    prefix_str = str(model_prefix)

    # 1. Metadata + memory-mapped spectra
    data, Y = load_clean_spectra_lean(H5_FILE)
    split_path = SPLIT_DIR / f'max12_{NUM_FOLDS}_folds_seed{SEED}.npz'
    test_idx, cv_idx, cv_fold_id = build_splits_once(data, split_path, NUM_FOLDS, seed=SEED)
    has_test = len(test_idx) > 0

    dopant_indices = [i for i, name in enumerate(data['names']) if name not in HOST_ELEMENTS]
    feature_names = [data['names'][i] for i in dopant_indices]
    input_dim = len(dopant_indices)
    output_dim = Y.shape[1]
    wavelengths = data['wavelengths']

    X_cv_np = data['X'][cv_idx][:, dopant_indices].numpy().astype(np.float32)   # type: ignore
    X_test_np = data['X'][test_idx][:, dopant_indices].numpy().astype(np.float32) if has_test else None   # type: ignore
    del data
    gc.collect()

    # 2. Load all fold models (~50 MB each)
    models = []
    for k in range(NUM_FOLDS):
        ckpt_path = Path(f"{prefix_str}_fold{k + 1}.pt")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        print(f"Loading checkpoint: {ckpt_path.name}")
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        model = PyTorchMLP(in_features=input_dim, out_features=output_dim).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
    predict_fn = make_predict_fn(models, output_dim)

    # 3. Streamed evaluation: OOF
    exp_pos = np.where(cv_fold_id >= 0)[0]
    cv_rows = cv_idx[exp_pos]
    print("Streaming OOF evaluation (constant memory)...")
    st_cv = stream_spectral_stats(Y, cv_rows, predict_fn, X_cv_np[exp_pos], folds=cv_fold_id[exp_pos])
    print_table_from_stats(st_cv, "MLP OOF Benchmarks (Loaded Checkpoints)")
    print_verdict_from_stats(st_cv, "OOF")

    st_te = None
    if has_test:
        print("Streaming lockbox evaluation (constant memory)...")
        st_te = stream_spectral_stats(Y, test_idx, predict_fn, X_test_np, folds=None, train_mean=st_cv['mean_spec'])
        print_table_from_stats(st_te, f"MLP Lockbox Benchmarks (Ensemble of {NUM_FOLDS} Checkpoints)")
        print_verdict_from_stats(st_te, "Lockbox")

    # 4. Extract arrays and export .npz file
    true_cv_spec = np.asarray(Y[cv_rows])
    pred_cv_spec = predict_fn(X_cv_np[exp_pos], cv_fold_id[exp_pos])
    
    true_test_spec = np.asarray(Y[test_idx]) if has_test else None
    pred_test_spec = predict_fn(X_test_np, None) if has_test else None

    # Derive timestamp tag from prefix or current time
    prefix_path = Path(prefix_str)
    tag = prefix_path.name.replace("mlp_", "")
    save_npz_path = OUTPUT_DIR / f"mlp_spectral_predictions_{tag}.npz"

    np.savez(
        save_npz_path,
        wavelengths=wavelengths,
        true_cv_spectra=true_cv_spec,
        pred_cv_spectra=pred_cv_spec,
        true_lockbox_spectra=true_test_spec,   # type: ignore
        pred_lockbox_spectra=pred_test_spec,   # type: ignore
    )
    print(f"✅ Exported predictions array to: {save_npz_path}")

    # 5. Figures
    yt_par, yp_par = _parity_views(st_cv)
    plot_parity_grid(
        yt_par, yp_par,
        wavelengths=wavelengths,
        title_tag="MLP CONCENTRATION TO SPECTRA OOF",
        save_path=OUTPUT_DIR / "eval_oof_parity.png"
    )
    picks = [int(np.argmax(st_cv['row_r2'])), int(np.argmin(st_cv['row_r2']))]
    plot_sample_spectra(
        y_true=_Rows({p: true_cv_spec[p] for p in picks}),
        fold_preds=_Rows({p: pred_cv_spec[p] for p in picks}),
        wavelengths=wavelengths,
        sample_indices=picks,
        X=_Rows({p: X_cv_np[exp_pos[p]] for p in picks}),
        feature_names=feature_names,
        title_tag="MLP OOF",
        save_path=OUTPUT_DIR / f"eval_oof_samples_{TIMESTAMP}.png"
    )

    print("Evaluation complete! Prediction array and figures generated safely.")
#=============================================================================
# Main Execution Pipeline
# =============================================================================
def main():
    amp_dtype, _ = amp_setup()
    print(f"=== Fast MLP Benchmark (Concentration -> Spectrum) [{DEVICE}] | amp={amp_dtype} | "
          f"compile={USE_COMPILE} ===")
    t0_start = time.perf_counter()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True

    # 1. Load data and splits (CPU)
    data = load_crossval_data(H5_FILE)
    split_path = SPLIT_DIR / f'max12_{NUM_FOLDS}_folds_seed{SEED}.npz'
    test_idx, cv_idx, cv_fold_id = build_splits_once(data, split_path, NUM_FOLDS, seed=SEED)
    has_test = len(test_idx) > 0

    # Task reversal: Inputs = Concentrations (data['X']), Targets = Spectra (data['y'])
    dopant_indices = [i for i, name in enumerate(data['names']) if name not in HOST_ELEMENTS]
    feature_names = [data['names'][i] for i in dopant_indices]
    input_dim = len(dopant_indices)
    output_dim = data['y'].shape[1]  # n_wavelengths
    wavelengths = data['wavelengths']

    # Inputs: Concentration vectors
    X_cv_cpu = data['X'][cv_idx][:, dopant_indices]
    X_test_cpu = data['X'][test_idx][:, dopant_indices] if has_test else None

    # Targets: Spectra scaled by sqrt(n_wavelengths) so RMS ~ 1
    scale = output_dim ** 0.5
    Y_cv_cpu = data['y'][cv_idx] * scale
    Y_test_cpu = data['y'][test_idx] * scale if has_test else None

    Y_cv_np = Y_cv_cpu.numpy()
    Y_test_np = Y_test_cpu.numpy() if has_test else None    # type: ignore

    print(f"Dataset Loaded | CV Pool: {len(cv_idx)} samples | Lockbox Test Set: {len(test_idx)} samples")
    print(f"Input Concentration Features: {feature_names} ({input_dim} dims) -> Target Spectrum ({output_dim} bins)")

    # 2. Stage to GPU memory
    print("Staging data:")
    Xs_cv = stash(X_cv_cpu, "CV concentrations")
    Y_cv = Y_cv_cpu.to(DEVICE)
    fold_t = torch.from_numpy(cv_fold_id.astype(np.int64)).to(Xs_cv.device)
    if has_test:
        Xs_test = stash(X_test_cpu, "Lockbox concentrations")  # type: ignore
        te_idx = torch.arange(len(test_idx), device=Xs_test.device)

    # 3. Sanity checks
    if RUN_SANITY_CHECKS:
        data_report(X_cv_cpu, Y_cv_np, cv_fold_id, feature_names)
        overfit_test(Xs_cv, Y_cv, fold_t, input_dim, output_dim, SEED)
    if RUN_SHUFFLE_CONTROL:
        shuffle_control(Xs_cv, Y_cv, fold_t, input_dim, output_dim, SEED)

    # 4. K-fold training
    oof_preds_np = np.zeros((len(cv_idx), output_dim), dtype=np.float32)
    lockbox_preds = torch.zeros((NUM_FOLDS, len(test_idx), output_dim), device=DEVICE)
    histories, baselines = [], []

    print("\n--- Training MLP (Concentration -> Spectra) across K Folds ---")
    for k in range(NUM_FOLDS):
        ckpt_dir = TRAINED_MODEL_DIR
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        tr_idx, va_idx = fold_indices(fold_t, k)
        base_mse = mean_baseline_mse(Y_cv, tr_idx, va_idx)
        print(f"Fold {k + 1}/{NUM_FOLDS} | train rows {tr_idx.numel()} | held-out rows {va_idx.numel()} | "
              f"predict-mean held-out MSE {base_mse:.6f}")

        # Inside main() K-fold loop:
        torch.manual_seed(SEED + k)
        model = new_model(input_dim, output_dim, Y_cv[tr_idx.to(DEVICE)].mean(0))
        hist = fit(model, Xs_cv, Y_cv, Y_cv, tr_idx, va_idx, EPOCHS, seed=SEED + k)
        histories.append(hist)
        baselines.append(base_mse)

        # OOF and Lockbox predictions use the best restored weights
        oof = predict(model, Xs_cv, va_idx).cpu().numpy()
        oof_preds_np[va_idx.cpu().numpy()] = oof
        if has_test:
            lockbox_preds[k] = predict(model, Xs_test, te_idx)  # type: ignore

        torch.save({
            'fold': k,
            'model_state_dict': model.state_dict(),  # Saves best validation weights
            'input_dim': input_dim,
            'output_dim': output_dim,
            'stopped_epoch': hist["stopped_epoch"],
            'best_val_mse': min(hist["val"]),
        }, ckpt_dir / f"mlp_{TIMESTAMP}_fold{k + 1}.pt")

        best_mse = min(hist["val"])
        ratio = best_mse / base_mse
        print(f"Fold {k + 1}/{NUM_FOLDS} Complete | Best held-out MSE {best_mse:.6f} "
              f"({100 * ratio:.1f}% of baseline) stopped at epoch {hist['stopped_epoch']} | "
              f"{hist['seconds']:.1f}s ({hist['samples_per_sec']:,.0f} samples/s)")

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    plot_learning_curves(histories, baselines, OUTPUT_DIR / f"mlp_conc_to_spec_learning_curves_{TIMESTAMP}.png")

    # TODO: This is where I keep running into an OOM "Killed"

    # 5. Evaluate and Report Metrics
    exp_mask = cv_fold_id >= 0
    true_cv_spec = Y_cv_np[exp_mask]
    pred_cv_spec = oof_preds_np[exp_mask]
    train_mean_cv = true_cv_spec.mean(axis=0)

    print_benchmark_table(
        y_true=true_cv_spec, y_pred_model=pred_cv_spec, y_train_mean=train_mean_cv,
        title_tag=f"MLP OOF Benchmarks (Concentration -> Spectra, HIDDEN_DIMS={HIDDEN_DIMS})",
        wavelengths=wavelengths,
    )
    print_learning_verdict(true_cv_spec, pred_cv_spec, "OOF")

    if has_test:
        lockbox_pred_avg = lockbox_preds.mean(dim=0).cpu().numpy()
        true_test_spec = Y_test_np
        pred_test_spec = lockbox_pred_avg

        print_benchmark_table(
            y_true=true_test_spec, y_pred_model=pred_test_spec, y_train_mean=train_mean_cv,
            title_tag=f"MLP Lockbox Benchmarks (Concentration -> Spectra, Ensemble of {NUM_FOLDS} Folds)",
            wavelengths=wavelengths,
        )
        print_learning_verdict(true_test_spec, pred_test_spec, "Lockbox")

    # 6. Parity Plots and Sample Overlay Diagnostics
    plot_parity_grid(
        true_cv_spec, pred_cv_spec, wavelengths=wavelengths,
        title_tag="MLP CONCENTRATION TO SPECTRA OOF",
        save_path=OUTPUT_DIR / f"gpu_mlp_conc_to_spec_oof_parity_{TIMESTAMP}.png"
    )
    plot_sample_spectra(
        y_true=true_cv_spec, 
        fold_preds=pred_cv_spec, 
        wavelengths=wavelengths,
        X=X_cv_cpu,
        feature_names=feature_names,
        title_tag="MLP CONCENTRATION TO SPECTRA OOF",
        save_path=OUTPUT_DIR / "eval_oof_samples.png"
    )

    if has_test:
        plot_parity_grid(
            true_test_spec, lockbox_pred_avg, wavelengths=wavelengths,   # type: ignore
            title_tag="MLP LOCKBOX ENSEMBLE",
            save_path=OUTPUT_DIR / "eval_lockbox_parity.png"
        )
        plot_sample_spectra(
            y_true=true_test_spec,    # type: ignore
            fold_preds=lockbox_preds.cpu().numpy(),    # type: ignore
            wavelengths=wavelengths,
            X=X_test_cpu,
            feature_names=feature_names,
            title_tag="MLP Lockbox Ensemble",
            save_path=OUTPUT_DIR / "eval_lockbox_samples.png"
        )

    total_train = sum(h["seconds"] for h in histories)
    print(f"Training time {total_train:.1f}s | total pipeline {(time.perf_counter() - t0_start):.1f}s.")

    # Save true vs predicted arrays for custom poster figures
    np.savez(
        OUTPUT_DIR / f"mlp_spectral_predictions_{TIMESTAMP}.npz",
        wavelengths=wavelengths,
        true_cv_spectra=true_cv_spec,
        pred_cv_spectra=pred_cv_spec,
        true_lockbox_spectra=true_test_spec if has_test else None,  # type: ignore
        pred_lockbox_spectra=pred_test_spec if has_test else None,  # type: ignore
    )
    print(f"Saved raw prediction arrays for poster plotting to {OUTPUT_DIR}")


if __name__ == "__main__":
    # main()
    # evaluate_saved_models(model_prefix='/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/conc_to_spec/MLP/trained_models/mlp_20261024_135220')
    just_plotting(sample_indices=[70568, 18937])