from __future__ import annotations

import sys
import time
import h5py
import torch
import copy
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
OUTPUT_DIR = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/spec_to_conc/CNN_1D/h5_plots")

NUM_FOLDS = 7
THRESH = 2.8
MAX_WT_PERCENT = 12.0
SEED = 42
HOST_ELEMENTS = {'Li', 'K', 'Cl'}
EXP_DOPANTS = {'Ce', 'Gd', 'Sm', 'U'}

# 1D CNN Hyperparameters
EPOCHS = 150
BATCH_SIZE = 256
LEARNING_RATE = 1e-3          # peak LR of the one-cycle schedule
WEIGHT_DECAY = 1e-4
BASE_FILTERS = 32
DROPOUT = 0.2
POOL_BINS = 16                # 1 reproduces the original global-average-pool head
                              # (which throws away *where* in the spectrum a feature sits)

PATIENCE = 15
MIN_DELTA = 1e-06
LOG_EVERY = 1

# Speed options
USE_AMP = True                # bf16 autocast on Ampere+, fp16 + GradScaler on older GPUs
USE_COMPILE = False           # torch.compile; can add ~1 min compile per fold, try it if epochs are long
GPU_DATA_BUDGET_FRAC = 0.6    # keep the dataset on-GPU only if it fits in this fraction of free VRAM

# Diagnostics ("is it actually learning?")
RUN_SANITY_CHECKS = False      # data stats + overfit-a-tiny-subset test
RUN_SHUFFLE_CONTROL = False    # train on shuffled labels; held-out R^2 must collapse to ~0
# LOG_EVERY = 5                 # epochs between held-out loss evaluations (logging only, never used for selection)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAINED_MODEL_DIR = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/spec_to_conc/CNN_1D/trained_models")
TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# PyTorch 1D CNN Architecture
# =============================================================================
class PyTorch1DCNN(nn.Module):
    """1D CNN for LIBS spectral regression.

    Differences from the original:
      * Final pooling is AdaptiveAvgPool1d(pool_bins) + flatten instead of global average
        pooling, so the head knows *which part of the spectrum* each feature came from.
        pool_bins=1 gives back the original behaviour.
      * Output layer starts small, with bias = mean training target, so epoch 0 of the
        loss curve equals the "predict the mean" baseline and any improvement is real.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_features: int = 4,
        base_filters: int = 32,
        dropout: float = 0.2,
        pool_bins: int = 16,
    ):
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
            nn.AdaptiveAvgPool1d(pool_bins),   # -> [Batch, base_filters * 8, pool_bins]
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filters * 8 * pool_bins, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_features),
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

    # 4. Slice targets X (columns) and filter samples (rows) simultaneously
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
        # The cached indices point into the *filtered* arrays. If THRESH / EXP_DOPANTS changed since the
        # cache was written, they silently misalign with the data -> fail loudly instead.
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
    """Calculates mean R^2 score across active target elements."""
    active_cols = y_true.max(axis=0) > 1e-5
    if not np.any(active_cols):
        return 0.0
    r2_scores = []
    for col in np.where(active_cols)[0]:
        r2 = r2_score(y_true[:, col], y_pred[:, col])
        r2_scores.append(r2)
    return float(np.mean(r2_scores))


def plot_parity_grid(y_true_wt, y_pred_wt, target_names, title_tag="1D CNN Benchmark", save_path=None):
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

        ax.scatter(yt, yp, alpha=0.35, s=10, color="#d62728", edgecolors="none")
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


def print_benchmark_table(y_true, y_pred_model, y_train_mean, title_tag, elem_names):
    """Prints element-wise metrics comparing 1D CNN against Dummy Baselines."""
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
            ("1D CNN Model", yp)
        ]
        
        for strat_name, y_hat in strategies:
            r2 = r2_score(yt, y_hat)
            mae = mean_absolute_error(yt, y_hat)
            rmse = np.sqrt(mean_squared_error(yt, y_hat))
            print(f"{name:<8} | {strat_name:<18} | {r2:8.3f} | {mae:10.4f} | {rmse:10.4f} | {yt.max():10.3f}")
        print("-" * 95)




# =============================================================================
# Fast GPU-resident training utilities
# =============================================================================
def stash(X: torch.Tensor, name: str) -> torch.Tensor:
    """Park a big array where batches can be gathered cheaply: on the GPU (as fp16, which is
    plenty for unit-RMS spectra) if it fits, otherwise in pinned CPU memory."""
    if DEVICE.type != "cuda":
        return X
    X = X.to(torch.float16)
    nbytes = X.numel() * X.element_size()
    free, _ = torch.cuda.mem_get_info()
    if nbytes < GPU_DATA_BUDGET_FRAC * free:
        print(f"  [{name}] {nbytes / 1e9:.2f} GB (fp16) -> GPU")
        return X.to(DEVICE)
    print(f"  [{name}] {nbytes / 1e9:.2f} GB does not fit on the GPU -> pinned CPU memory (slower)")
    return X.pin_memory()


def gather_x(Xs: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    bx = Xs[idx]
    if bx.device != DEVICE:
        bx = bx.to(DEVICE, non_blocking=True)
    return bx.float()


def amp_setup():
    if not (USE_AMP and DEVICE.type == "cuda"):
        return None, None
    if torch.cuda.get_device_capability()[0] >= 8:      # Ampere+ has native bf16
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


def fit(
    model, 
    Xs, 
    Y_fit, 
    Y_eval, 
    tr_idx, 
    va_idx, 
    epochs, 
    seed, 
    verbose=True, 
    enable_early_stopping=True,
    log_every=LOG_EVERY
):
    """Training loop with ReduceLROnPlateau, validation tracking, and early stopping with weight restoration."""
    n = tr_idx.numel()
    steps = n // BATCH_SIZE
    if steps == 0:
        raise ValueError(f"Only {n} training rows for BATCH_SIZE={BATCH_SIZE}.")

    g = torch.Generator(device=tr_idx.device)
    g.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Plateau scheduler stepping once per epoch on val loss
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=5
    )
    
    amp_dtype, scaler = amp_setup()
    hist = dict(epoch=[], train=[], val=[], stopped_epoch=epochs)

    # Early stopping state tracking
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
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        t_train = time.perf_counter() - t_ep

        # Evaluate Validation Metrics
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
                if DEVICE.type == "cuda": torch.cuda.synchronize()
                t_tot = time.perf_counter() - t_ep
                print(f"---- train {t_train:.1f}s | val {t_tot - t_train}s | {steps*BATCH_SIZE/t_train:,.0f} samples/s (train)----")
                print(f"    ep {ep:3d}/{epochs} | train MSE {tr_loss:.6f} (dropout on) | "
                      f"held-out MSE {va_loss:.6f} | lr {opt.param_groups[0]['lr']:.2e}")

            # Early Stopping Evaluation
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
                    hist["stopped_epoch"] = ep  # type: ignore
                    break

    # Restore best model state dict
    if enable_early_stopping and best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        best_ep = hist['epoch'][hist['val'].index(best_val_loss)]
        if verbose:
            print(f"    --> Restored best model weights from epoch {best_ep}")

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    hist["seconds"] = dt  # type: ignore
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
    return model    # type: ignore


def fold_indices(fold_id_t: torch.Tensor, k: int):
    tr_idx = torch.nonzero(fold_id_t != k).squeeze(1)
    va_idx = torch.nonzero(fold_id_t == k).squeeze(1)
    return tr_idx, va_idx


def mean_baseline_mse(Y: torch.Tensor, tr_idx: torch.Tensor, va_idx: torch.Tensor) -> float:
    """Held-out MSE of the constant 'predict the training mean' model: the number any real model must beat."""
    mu = Y[tr_idx.to(Y.device)].mean(0, keepdim=True)
    return F.mse_loss(mu.expand(va_idx.numel(), -1), Y[va_idx.to(Y.device)]).item()


# =============================================================================
# "Is it actually learning?" diagnostics
# =============================================================================
def per_element_r2(y_true, y_pred):
    return np.array([
        r2_score(y_true[:, j], y_pred[:, j]) if y_true[:, j].max() > 1e-5 else np.nan
        for j in range(y_true.shape[1])
    ])


def data_report(X_cv_cpu, Y_cv_np, fold_id, target_names):
    print("\n" + "=" * 80)
    print(" SANITY CHECK 1: inputs and targets")
    print("=" * 80)
    sub = X_cv_cpu[:: max(1, len(X_cv_cpu) // 2000)]
    rms = sub.pow(2).mean(1).sqrt()
    print(f"Spectra: {tuple(X_cv_cpu.shape)} | finite: {bool(torch.isfinite(X_cv_cpu).all())} | "
          f"per-spectrum RMS min/median/max: {rms.min():.3f}/{rms.median():.3f}/{rms.max():.3f} (expect ~1.0)")
    print(f"Targets: finite: {bool(np.isfinite(Y_cv_np).all())} | range [{Y_cv_np.min():.4f}, {Y_cv_np.max():.4f}] "
          f"(expect within [0, 1] = wt%/{MAX_WT_PERCENT:g})")
    exp = fold_id >= 0
    print(f"Rows: {exp.sum()} experimental (CV folds) + {(~exp).sum()} synthetic (train-only)")
    print(f"{'Element':<8} | {'% rows >0 (exp)':>15} | {'mean wt%':>9} | {'max wt%':>8} | {'predict-mean MSE (loss units)':>30}")
    for j, name in enumerate(target_names):
        y = Y_cv_np[exp, j]
        print(f"{name:<8} | {100 * (y > 0).mean():14.1f}% | {y.mean() * MAX_WT_PERCENT:9.3f} | "
              f"{y.max() * MAX_WT_PERCENT:8.3f} | {y.var():30.6f}")
    if (Y_cv_np[exp].max(axis=0) <= 1e-5).any():
        print("WARNING: at least one target is all zeros on experimental rows.")


def overfit_test(Xs, Y, fold_id_t, out_dim, seed):
    """A healthy model + optimiser + data pipeline must be able to memorise 512 samples."""
    print("\n" + "=" * 80)
    print(" SANITY CHECK 2: can the model memorise 512 training samples?")
    print("=" * 80)
    g = torch.Generator().manual_seed(seed)
    cand = torch.nonzero(fold_id_t >= 0).squeeze(1)
    pick = cand[torch.randperm(cand.numel(), generator=g)[:512].to(cand.device)]
    Yp = Y[pick.to(Y.device)]
    base = ((Yp - Yp.mean(0)) ** 2).mean().item()
    model = new_model(out_dim, Yp.mean(0))
    fit(model, Xs, Y, Y, pick, pick, epochs=150, seed=seed, verbose=False, enable_early_stopping=False)
    final = F.mse_loss(predict(model, Xs, pick), Yp).item()
    ok = final < 0.2 * base
    print(f"Predict-mean MSE on these rows: {base:.6f} | MSE after 300 steps: {final:.6f} "
          f"({100 * final / base:.1f}% of baseline)")
    print("PASS" if ok else "FAIL: cannot even fit 512 samples -> bug in data/targets/optimiser, "
          "or too few steps for this dataset.")
    del model
    return ok


# --- 1) In fit(): make the logging interval a parameter instead of the global LOG_EVERY ---------
#     signature:   def fit(model, Xs, Y_fit, Y_eval, tr_idx, va_idx, epochs, seed, verbose=True,
#                          enable_early_stopping=True, log_every=LOG_EVERY):
#     and in the body change the one line
#         if ep == 1 or ep % LOG_EVERY == 0 or ep == epochs:
#     to
#         if ep == 1 or ep % log_every == 0 or ep == epochs:

# --- 2) Replace shuffle_control with this (add the two constants next to RUN_SHUFFLE_CONTROL) ---
SHUFFLE_EPOCHS = 15           # a shuffled-label model has nothing to learn; 150 epochs buys nothing
SHUFFLE_MAX_TRAIN = 30000     # subsample the training rows (held-out fold is still evaluated in full)


def shuffle_control(Xs, Y, fold_id_t, out_dim, target_names, seed):
    """Train on shuffled labels. Held-out R^2 must be ~0 (or negative). If it is clearly positive,
    something is leaking and every other number is suspect."""
    print("\n" + "=" * 80)
    print(" SANITY CHECK 3: shuffled-label control (fold 0)")
    print("=" * 80)
    tr_idx, va_idx = fold_indices(fold_id_t, 0)
    g = torch.Generator().manual_seed(seed + 999)
    if tr_idx.numel() > SHUFFLE_MAX_TRAIN:
        keep = torch.randperm(tr_idx.numel(), generator=g)[:SHUFFLE_MAX_TRAIN].to(tr_idx.device)
        tr_idx = tr_idx[keep]
    print(f"  training {SHUFFLE_EPOCHS} epochs on {tr_idx.numel()} shuffled rows "
          f"(held-out: {va_idx.numel()} rows)...", flush=True)

    Y_shuf = Y.clone()
    perm = torch.randperm(tr_idx.numel(), generator=g).to(tr_idx.device)
    Y_shuf[tr_idx.to(Y.device)] = Y[tr_idx[perm].to(Y.device)]

    model = new_model(out_dim, Y_shuf[tr_idx.to(Y.device)].mean(0))
    fit(model, Xs, Y_shuf, Y, tr_idx, va_idx, epochs=SHUFFLE_EPOCHS, seed=seed, verbose=True,
        enable_early_stopping=False, log_every=5)
    pred = predict(model, Xs, va_idx).clamp(min=0).cpu().numpy()
    true = Y[va_idx.to(Y.device)].cpu().numpy()
    r2 = per_element_r2(true, pred)
    for n, r in zip(target_names, r2):
        print(f"  {n:<5} held-out R^2 with shuffled labels: {r:8.3f}")
    ok = bool(np.all(np.nan_to_num(r2, nan=0.0) < 0.1))
    print("PASS: nothing to learn -> nothing learned (no leakage)" if ok else
          "FAIL: model 'learns' from shuffled labels -> leakage between train and held-out rows")
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


def print_learning_verdict(y_true_wt, y_pred_wt, target_names, tag):
    """Heuristic read-out: does the model beat 'predict the mean' by a meaningful margin?"""
    print(f"\n--- Learning verdict: {tag} ---")
    r2 = per_element_r2(y_true_wt, y_pred_wt)
    for n, r in zip(target_names, r2):
        if np.isnan(r):
            status = "n/a (element absent in this set)"
        elif r <= 0.05:
            status = "NOT LEARNING (no better than predicting the mean)"
        elif r < 0.5:
            status = "weak"
        else:
            status = "learning"
        print(f"  {n:<5} R^2 = {r:7.3f}  -> {status}")


# =============================================================================
# Main Execution Pipeline
# =============================================================================
def main():
    amp_dtype, _ = amp_setup()
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()} | capability {torch.cuda.get_device_capability()} | amp={amp_dtype}")
    print(f"=== Fast 1D CNN Benchmark [{DEVICE}] | amp={amp_dtype} | pool_bins={POOL_BINS} | "
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

    # Inputs = spectra (data['y']), targets = dopant concentrations (data['X'])
    dopant_indices = [i for i, name in enumerate(data['names']) if name not in HOST_ELEMENTS]
    target_names = [data['names'][i] for i in dopant_indices]
    output_dim = len(dopant_indices)

    # Spectra are L2-normalised (norm 1 => values ~1/sqrt(n_wavelengths)). Multiply by sqrt(n) so each
    # spectrum has RMS 1: same information, but BatchNorm's eps no longer competes with tiny activations.
    X_cv_cpu = data['y'][cv_idx]
    X_test_cpu = data['y'][test_idx] if has_test else None
    scale = X_cv_cpu.shape[1] ** 0.5
    X_cv_cpu.mul_(scale)
    if has_test:
        X_test_cpu.mul_(scale)    # type: ignore
    Y_cv_np = data['X'][cv_idx][:, dopant_indices].numpy()
    Y_test_np = data['X'][test_idx][:, dopant_indices].numpy() if has_test else None
    del data['y']

    print(f"Dataset Loaded | CV Pool: {len(cv_idx)} samples | Lockbox Test Set: {len(test_idx)} samples")
    print(f"Target Elements: {target_names}")

    # 2. Everything to the GPU once. No DataLoader, no per-batch host->device copies.
    print("Staging data:")
    Xs_cv = stash(X_cv_cpu, "CV spectra")
    Y_cv = torch.from_numpy(Y_cv_np).to(DEVICE)
    fold_t = torch.from_numpy(cv_fold_id.astype(np.int64)).to(Xs_cv.device)
    if has_test:
        Xs_test = stash(X_test_cpu, "Lockbox spectra")    # type: ignore
        te_idx = torch.arange(len(test_idx), device=Xs_test.device)

    # 3. Sanity checks
    if RUN_SANITY_CHECKS:
        data_report(X_cv_cpu, Y_cv_np, cv_fold_id, target_names)
        overfit_test(Xs_cv, Y_cv, fold_t, output_dim, SEED)
    if RUN_SHUFFLE_CONTROL:
        shuffle_control(Xs_cv, Y_cv, fold_t, output_dim, target_names, SEED)
    del X_cv_cpu, X_test_cpu

    # 4. K-fold training
    oof_preds_np = np.zeros((len(cv_idx), output_dim), dtype=np.float32)
    # Store ensemble predictions in CPU memory to prevent VRAM OOM
    lockbox_preds = torch.zeros((NUM_FOLDS, len(test_idx), output_dim), device="cpu")
    histories, baselines = [], []

    print("\n--- Training 1D CNN across K Folds ---")
    for k in range(NUM_FOLDS):
        tr_idx, va_idx = fold_indices(fold_t, k)
        base_mse = mean_baseline_mse(Y_cv, tr_idx, va_idx)
        print(f"Fold {k + 1}/{NUM_FOLDS} | train rows {tr_idx.numel()} | held-out rows {va_idx.numel()} | "
              f"predict-mean held-out MSE {base_mse:.6f}")

        torch.manual_seed(SEED + k)
        model = new_model(output_dim, Y_cv[tr_idx.to(DEVICE)].mean(0))
        hist = fit(model, Xs_cv, Y_cv, Y_cv, tr_idx, va_idx, EPOCHS, seed=SEED + k, enable_early_stopping=True)
        histories.append(hist)
        baselines.append(base_mse)

        # OOF & Lockbox predictions generated using the best restored weights
        oof = predict(model, Xs_cv, va_idx).clamp(min=0).cpu().numpy()
        oof_preds_np[va_idx.cpu().numpy()] = oof
        if has_test:
            lockbox_preds[k] = predict(model, Xs_test, te_idx).clamp(min=0).cpu()   # type: ignore

        # Save Checkpoint with metadata
        torch.save({
            'fold': k,
            'model_state_dict': model.state_dict(),
            'output_dim': output_dim,
            'stopped_epoch': hist["stopped_epoch"],
            'best_val_mse': min(hist["val"]),
        }, TRAINED_MODEL_DIR / f"cnn1d_{TIMESTAMP}_fold{k + 1}.pt")

        best_mse = min(hist["val"])
        ratio = best_mse / base_mse
        print(f"Fold {k + 1}/{NUM_FOLDS} Complete | Best held-out MSE {best_mse:.6f} "
              f"({100 * ratio:.1f}% of baseline) stopped at epoch {hist['stopped_epoch']} | "
              f"{hist['seconds']:.1f}s ({hist['samples_per_sec']:,.0f} samples/s)")
        
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    plot_learning_curves(histories, baselines, OUTPUT_DIR / f"cnn_1d_learning_curves_{TIMESTAMP}.png")

    # 5. Scale back to physical wt% and report metrics
    exp_mask = cv_fold_id >= 0
    true_cv_wt = Y_cv_np[exp_mask] * MAX_WT_PERCENT
    pred_cv_wt = oof_preds_np[exp_mask] * MAX_WT_PERCENT
    train_mean_cv = true_cv_wt.mean(axis=0)

    print_benchmark_table(
        y_true=true_cv_wt, y_pred_model=pred_cv_wt, y_train_mean=train_mean_cv,
        title_tag=f"1D CNN OOF Benchmarks (BASE_FILTERS={BASE_FILTERS}, POOL_BINS={POOL_BINS})",
        elem_names=target_names,
    )
    print_learning_verdict(true_cv_wt, pred_cv_wt, target_names, "OOF")

    if has_test:
        lockbox_pred_avg = lockbox_preds.mean(dim=0).clamp(min=0).cpu().numpy()
        true_test_wt = Y_test_np * MAX_WT_PERCENT    # type: ignore
        pred_test_wt = lockbox_pred_avg * MAX_WT_PERCENT

        print_benchmark_table(
            y_true=true_test_wt, y_pred_model=pred_test_wt, y_train_mean=train_mean_cv,
            title_tag=f"1D CNN Lockbox Benchmarks (Ensemble of {NUM_FOLDS} Folds)",
            elem_names=target_names,
        )
        print_learning_verdict(true_test_wt, pred_test_wt, target_names, "Lockbox")

    active_elem_idx = [j for j in range(true_cv_wt.shape[1]) if true_cv_wt[:, j].max() > 1e-5]

    def summary(title, yt_all, yp_all):
        print("\n" + "=" * 80)
        print(f" {title}")
        print("=" * 80)
        for j in active_elem_idx:
            yt, yp = yt_all[:, j], yp_all[:, j]
            if yt.max() <= 1e-5:
                continue
            print(f"Element {target_names[j]:<5} | R^2: {r2_score(yt, yp):7.3f} | "
                  f"MAE: {mean_absolute_error(yt, yp):.4f} wt% | "
                  f"RMSE: {np.sqrt(mean_squared_error(yt, yp)):.4f} wt% | Max True: {yt.max():.3f} wt%")

    summary("1D CNN Benchmark - Out-Of-Fold (OOF) Performance", true_cv_wt, pred_cv_wt)
    if has_test:
        summary(f"1D CNN Benchmark - Lockbox Test Set Performance (Ensemble of {NUM_FOLDS} Folds)",
                true_test_wt, pred_test_wt)    # type: ignore
        print("=" * 80 + "\n")

    # 6. Parity Plots
    plot_parity_grid(true_cv_wt, pred_cv_wt, target_names,
                     title_tag="1D CNN L2 SYNTHETIC DATA INCLUDED OOF",
                     save_path=OUTPUT_DIR / f"gpu_cnn_1d_L2_synth_oof_parity_{TIMESTAMP}.png")
    if has_test:
        plot_parity_grid(true_test_wt, pred_test_wt, target_names,    # type: ignore
                         title_tag="1D CNN L2 SYNTHETIC DATA INCLUDED LOCKBOX ENSEMBLE",
                         save_path=OUTPUT_DIR / f"gpu_cnn_1d_L2_synth_lockbox_parity_{TIMESTAMP}.png")

    total_train = sum(h["seconds"] for h in histories)
    print(f"Training time {total_train:.1f}s | total pipeline {(time.perf_counter() - t0_start):.1f}s.")


if __name__ == "__main__":
    main()