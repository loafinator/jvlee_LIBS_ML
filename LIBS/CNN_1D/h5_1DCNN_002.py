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
OUTPUT_DIR = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/spec_to_conc/CNN_1D/h5_plots/processed-data_20261110b")

NUM_FOLDS = 7
SEEDS = [42]
MAX_WT_PERCENT = 12.0
SEED = 42
HOST_ELEMENTS = {'Li', 'K', 'Cl'}
TARGET_NAMES = ['Ce', 'Gd', 'Sm', 'U']

# 1D CNN Hyperparameters
EPOCHS = 150
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BASE_FILTERS = 32
DROPOUT = 0.2
POOL_BINS = 1

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

TRAINED_MODEL_DIR = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/spec_to_conc/CNN_1D/trained_models")
TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Fast GPU SIMPLS Algorithm (In-Place Memory Optimized)
# =============================================================================
class PyTorch1DCNN(nn.Module):
    """1D CNN for LIBS spectral concentration regression matching paper spec."""
    def __init__(
        self,
        in_channels: int = 1,
        out_features: int = 4,
        base_filters: int = 32,
        dropout: float = 0.2,
        pool_bins: int = 1,  # Set to 1 to produce [B, 256, 1] as shown in table
    ):
        super().__init__()

        self.features = nn.Sequential(
            # Layer 1: Out [B, 32, 2500]
            nn.Conv1d(in_channels, base_filters, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Layer 2: Out [B, 64, 625]
            nn.Conv1d(base_filters, base_filters * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base_filters * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Layer 3: Out [B, 128, 156]
            nn.Conv1d(base_filters * 2, base_filters * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(base_filters * 4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Layer 4: Out [B, 256, 39]
            nn.Conv1d(base_filters * 4, base_filters * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(base_filters * 8),
            nn.ReLU(),
            # Pool: Out [B, 256, 1]
            nn.AdaptiveAvgPool1d(pool_bins),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            # Regression Layer: [B, 256] -> [B, 64]
            nn.Linear(base_filters * 8 * pool_bins, base_filters * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Output Layer: [B, 64] -> [B, 4]
            nn.Linear(base_filters * 2, out_features),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        last = self.fc[-1]
        nn.init.normal_(last.weight, std=0.01)

    @torch.no_grad()
    def set_output_bias(self, target_mean: torch.Tensor):
        self.fc[-1].bias.copy_(target_mean.to(self.fc[-1].bias.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.fc(self.features(x))

# =============================================================================
# Helper & Data Functions
# =============================================================================

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

def build_splits_for_seed(data, split_dir, num_folds, seed):
    """Loads existing pre-computed CV splits for a given seed or creates them if absent or stale."""
    split_dir = Path(split_dir)
    split_path = split_dir / f'max12_{num_folds}_folds_seed{seed}.npz'
    
    # Compute fingerprint for the current in-memory dataset
    current_fp = compute_dataset_fingerprint(data)
    
    if split_path.exists():
        d = np.load(split_path)
        cached_fp = str(d['fingerprint']) if 'fingerprint' in d else None
        
        # Verify hash match and fold count before reusing cache
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

    # Save splits along with current dataset fingerprint
    np.savez(
        split_path, 
        test_idx=test_idx, 
        cv_idx=cv_idx, 
        cv_fold_id=cv_fold_id,
        num_folds=num_folds, 
        seed=seed,
        fingerprint=current_fp
    )
    return test_idx, cv_idx, cv_fold_id

def compute_dataset_fingerprint(data: dict) -> str:
    """Computes a fast cryptographic hash of targets, groups, and lockbox flags."""
    h = hashlib.blake2b(digest_size=16)
    
    # Hash y array, group IDs, and lockbox mask
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
            is_synth = hf['meta/is_synthetic'][:] # type: ignore
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

    # Convert directly to PyTorch tensor (already filtered in data prep)
    X_tensor = torch.from_numpy(spectra_raw)

    # L2 Normalization on Spectra ONLY
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

def plot_parity_grid(y_true_wt, y_pred_wt, target_names, title_tag="GPU PLS Benchmark", save_path=None):
    """Generates an NxM parity grid plot matching P3VAE visualization."""
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

def print_benchmark_table(y_true, y_pred_model, y_train_mean, title_tag, elem_names):
    """
    Prints element-wise metrics comparing PLS against Dummy Baselines.
    """
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
        
        # Baselines
        pred_zero = np.zeros_like(yt)
        pred_mean = np.full_like(yt, y_train_mean[j])
        
        strategies = [
            ("Predict Zero", pred_zero),
            ("Predict Mean", pred_mean),
            ("PLS Model", yp)
        ]
        
        for strat_name, y_hat in strategies:
            r2 = r2_score(yt, y_hat)
            mae = mean_absolute_error(yt, y_hat)
            rmse = np.sqrt(mean_squared_error(yt, y_hat))
            print(f"{name:<8} | {strat_name:<18} | {r2:8.3f} | {mae:10.4f} | {rmse:10.4f} | {yt.max():10.3f}")
        print("-" * 95)

def rmse_calcs(y_true, y_pred):
    """Calculates mean RMSE across active target elements."""
    active_cols = y_true.max(axis=0) > 1e-5
    if not np.any(active_cols):
        return float('inf')
    rmse_scores = []
    for col in np.where(active_cols)[0]:
        rmse = np.sqrt(mean_squared_error(y_true[:, col], y_pred[:, col]))
        rmse_scores.append(rmse)
    return float(np.nanmean(rmse_scores))

def r2_calcs(y_true, y_pred):
    """Calculates mean R^2 score across active target elements."""
    active_cols = y_true.max(axis=0) > 1e-5
    if not np.any(active_cols):
        return 0.0
    r2_scores = []
    for col in np.where(active_cols)[0]:
        r2 = r2_score(y_true[:, col], y_pred[:, col])
        r2_scores.append(r2)
    return float(np.nanmean(r2_scores))


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
    return torch.float16, torch.amp.GradScaler("cuda") # type: ignore

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

def fit(
    model, Xs, Y_fit, Y_eval, tr_idx, va_idx, epochs, seed, verbose=True, 
    enable_early_stopping=True, log_every=LOG_EVERY
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
            running += loss.detach()

        if ep == 1 or ep % log_every == 0 or ep == epochs:
            tr_loss = (running / steps).item()
            va_pred = predict(model, Xs, va_idx)
            va_loss = F.mse_loss(va_pred, Y_eval[va_idx.to(Y_eval.device)]).item()

            if enable_early_stopping:
                sched.step(va_loss)

            hist["epoch"].append(ep)
            hist["train"].append(tr_loss)
            hist["val"].append(va_loss)

            if verbose:
                t_tot = time.perf_counter() - t_ep
                print(f"    ep {ep:3d}/{epochs} | train MSE {tr_loss:.6f} | held-out MSE {va_loss:.6f} | lr {opt.param_groups[0]['lr']:.2e}")

            if enable_early_stopping:
                if va_loss < (best_val_loss - MIN_DELTA):
                    best_val_loss = va_loss
                    best_model_weights = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= PATIENCE:
                    if verbose:
                        print(f"    --> Early stopping triggered at epoch {ep}. Best Val MSE: {best_val_loss:.6f}")
                    hist["stopped_epoch"] = ep # type: ignore
                    break

    if enable_early_stopping and best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    dt = time.perf_counter() - t0
    hist["seconds"] = dt # type: ignore
    hist["samples_per_sec"] = len(hist["epoch"]) * steps * BATCH_SIZE / dt
    return hist

def new_model(out_dim: int, init_bias: torch.Tensor) -> nn.Module:
    model = PyTorch1DCNN(
        in_channels=1, out_features=out_dim, base_filters=BASE_FILTERS,
        dropout=DROPOUT, pool_bins=POOL_BINS,
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
    mu = Y[tr_idx.to(Y.device)].mean(0, keepdim=True)
    return F.mse_loss(mu.expand(va_idx.numel(), -1), Y[va_idx.to(Y.device)]).item()

# =============================================================================
# Main Execution Pipeline
# =============================================================================
def main(
        h5_file: Path | str = H5_FILE,
        split_dir: Path | str = SPLIT_DIR,
        output_dir: Path | str = OUTPUT_DIR,
        include_synthetic: bool = False,
        seeds: list[int] = SEEDS,
        include_lbx: bool = False
):
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_tag = Path(h5_file).stem
    print(f"=== Starting Fast 1D CNN Benchmark Pipeline [{DEVICE}] ===")
    print(f"Dataset: {dataset_tag} | K-Fold: {NUM_FOLDS} Folds x {len(seeds)} Seeds")
    t0_start = time.perf_counter()

    h5_file = str(h5_file)
    split_dir = Path(split_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data & Prepare Multi-Seed Splits
    data = load_crossval_data(h5_file)
    n_synth = (data['is_synth'] == 1).sum()
    print(f"Loaded {len(data['X'])} total shots | Real: {len(data['X']) - n_synth} | Synthetic: {n_synth}")

    seed_splits = {}
    for seed in seeds:
        test_idx, cv_idx, cv_fold_id = build_splits_for_seed(data, split_dir, NUM_FOLDS, seed=seed)
        seed_splits[seed] = (test_idx, cv_idx, cv_fold_id)

    test_idx = seed_splits[seeds[0]][0]
    cv_idx = seed_splits[seeds[0]][1]

    # Target Slicing
    target_indices = [data['names'].index(name) for name in TARGET_NAMES if name in data['names']]
    actual_target_names = [data['names'][i] for i in target_indices]
    output_dim = len(target_indices)

    # Inputs = Spectra (X), Targets = Concentrations (y)
    X_cv_cpu = data['X'][cv_idx]
    X_test_cpu = data['X'][test_idx] if include_lbx else None

    # Scale spectra RMS for BatchNorm stability
    scale = X_cv_cpu.shape[1] ** 0.5
    X_cv_cpu.mul_(scale)
    if include_lbx:
        X_test_cpu.mul_(scale) # type: ignore

    # Normalize targets to [0, 1] range for network training
    Y_cv_np = (data['y'][cv_idx][:, target_indices] / MAX_WT_PERCENT).numpy()
    Y_test_np = (data['y'][test_idx][:, target_indices] / MAX_WT_PERCENT).numpy() if include_lbx else None
    del data['X']

    # 2. Stage Data to GPU
    Xs_cv = stash(X_cv_cpu, "CV spectra")
    Y_cv = torch.from_numpy(Y_cv_np).to(DEVICE)
    fold_t = torch.from_numpy(seed_splits[seeds[0]][2].astype(np.int64)).to(Xs_cv.device)
    Xs_test, te_idx = None, None
    if include_lbx:
        Xs_test = stash(X_test_cpu, "Lockbox spectra") # type: ignore
        te_idx = torch.arange(len(test_idx), device=Xs_test.device)

    del X_cv_cpu, X_test_cpu

    # 3. K-Fold Training Loop
    oof_preds_np = np.zeros((len(cv_idx), output_dim), dtype=np.float32)
    lockbox_preds = torch.zeros((NUM_FOLDS, len(test_idx), output_dim), device="cpu")

    print("\n--- Training 1D CNN across K Folds ---")
    for k in range(NUM_FOLDS):
        tr_idx, va_idx = fold_indices(fold_t, k)
        
        # Mask synthetic rows if include_synthetic=False
        if not include_synthetic:
            tr_exp_mask = torch.from_numpy(
                data['is_synth'][cv_idx][tr_idx.cpu().numpy()] == 0
            ).to(tr_idx.device)
            tr_idx = tr_idx[tr_exp_mask]

        base_mse = mean_baseline_mse(Y_cv, tr_idx, va_idx)
        print(f"Fold {k + 1}/{NUM_FOLDS} | train rows {tr_idx.numel()} | held-out rows {va_idx.numel()} | baseline MSE {base_mse:.6f}")

        torch.manual_seed(SEED + k)
        model = new_model(output_dim, Y_cv[tr_idx.to(DEVICE)].mean(0))
        hist = fit(model, Xs_cv, Y_cv, Y_cv, tr_idx, va_idx, EPOCHS, seed=SEED + k, enable_early_stopping=True)

        oof = predict(model, Xs_cv, va_idx).clamp(min=0).cpu().numpy()
        oof_preds_np[va_idx.cpu().numpy()] = oof
        if include_lbx:
            lockbox_preds[k] = predict(model, Xs_test, te_idx).clamp(min=0).cpu() # type: ignore

        torch.save({
            'fold': k,
            'model_state_dict': model.state_dict(),
            'output_dim': output_dim,
            'stopped_epoch': hist["stopped_epoch"],
            'best_val_mse': min(hist["val"]),
        }, TRAINED_MODEL_DIR / f"cnn1d_{dataset_tag}_{TIMESTAMP}_fold{k + 1}.pt")

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # 4. Scale Back to wt% and Print Benchmark Tables
    exp_mask = seed_splits[seeds[0]][2] >= 0
    true_cv_wt = Y_cv_np[exp_mask] * MAX_WT_PERCENT
    pred_cv_wt = oof_preds_np[exp_mask] * MAX_WT_PERCENT
    train_mean_cv = true_cv_wt.mean(axis=0)

    print_benchmark_table(
        y_true=true_cv_wt,
        y_pred_model=pred_cv_wt,
        y_train_mean=train_mean_cv,
        title_tag=f"1D CNN OOF Benchmarks [{'Syn+Exp' if include_synthetic else 'Exp Only'}]",
        elem_names=actual_target_names
    )

    if include_lbx:
        lockbox_pred_avg = lockbox_preds.mean(dim=0).clamp(min=0).cpu().numpy()
        true_test_wt = Y_test_np * MAX_WT_PERCENT # type: ignore
        pred_test_wt = lockbox_pred_avg * MAX_WT_PERCENT

        print_benchmark_table(
            y_true=true_test_wt,
            y_pred_model=pred_test_wt,
            y_train_mean=train_mean_cv,
            title_tag=f"1D CNN Lockbox Benchmarks (Ensemble of {NUM_FOLDS} Folds)",
            elem_names=actual_target_names
        )

    # 5. Save Parity Plots
    oof_parity_path = output_dir / f"gpu_cnn1d_oof_parity_{dataset_tag}_{TIMESTAMP}.png"
    lockbox_parity_path = output_dir / f"gpu_cnn1d_lockbox_parity_{dataset_tag}_{TIMESTAMP}.png"

    plot_parity_grid(true_cv_wt, pred_cv_wt, actual_target_names, title_tag=f"1D CNN OOF ({dataset_tag})", save_path=oof_parity_path)
    if include_lbx:
        plot_parity_grid(true_test_wt, pred_test_wt, actual_target_names, title_tag=f"1D CNN LOCKBOX ({dataset_tag})", save_path=lockbox_parity_path) # type: ignore

    cache_path = output_dir / f"cnn1d_predictions_{dataset_tag}.npz"
    save_dict = dict(
        y_true_cv=true_cv_wt,          # True wt% (OOF)
        y_pred_cv=pred_cv_wt,          # Predicted wt% (OOF)
        target_names=np.array(actual_target_names)
    )
    if include_lbx:
        save_dict["y_true_test"] = true_test_wt  # type: ignore
        save_dict["y_pred_test"] = pred_test_wt  # type: ignore

    np.savez(cache_path, **save_dict)
    print(f"Cached forward predictions to: {cache_path}")

    print(f"1D CNN Pipeline completed in {(time.perf_counter() - t0_start):.2f} seconds.")


if __name__ == "__main__":
    main(
        h5_file="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/exp-syn_processed-data_20261110b.h5",
        split_dir="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/fold_splits/processed-data_20261110b",
        output_dir=Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/plots/CNN_1D/processed-data_20261110b/exp"),
        include_synthetic=False,
        seeds=SEEDS,
        include_lbx=False
    )

    main(
        h5_file="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/exp-syn_processed-data_20261110b.h5",
        split_dir="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/fold_splits/processed-data_20261110b",
        output_dir=Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/plots/CNN_1D/processed-data_20261110b/exp-syn"),
        include_synthetic=True,
        seeds=SEEDS,
        include_lbx=False
    )