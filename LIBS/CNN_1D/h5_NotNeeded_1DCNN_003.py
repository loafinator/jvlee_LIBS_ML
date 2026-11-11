from __future__ import annotations

import sys
import gc
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from setup import add_project_root_to_path
add_project_root_to_path(parent_generation=1)
from utils import get_h5_ds

# ==============================================================================
# Global Configuration & Constants
# ==============================================================================
H5_FILE = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/max12_cts_noleak_crossval.h5")
SPLIT_DIR = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/fold_splits_NoExtraDopants")
OUTPUT_DIR = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/conc_to_spec/CNN_1D/h5_plots")

HOST_ELEMENTS = {"Li", "K", "Cl"}
EXP_DOPANTS = {"Ce", "Gd", "Sm", "U"}

SEED = 42
NUM_FOLDS = 7
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
OVERFIT_EPOCHS = 500

BASE_FILTERS = 32
DROPOUT = 0.1
UP_STAGES = 4

USE_AMP = True
USE_COMPILE = False
CLAMP_NONNEGATIVE = True
LOG_EVERY = 10
N_BANDS = 5
N_EXAMPLE_SPECTRA = 6
RUN_SANITY_CHECKS = True
RUN_SHUFFLE_CONTROL = True

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_DATA_BUDGET_FRAC = 0.8
MAX_WT_PERCENT = 10.0


# ==============================================================================
# Model Architecture
# ==============================================================================
class PyTorch1DCNN(nn.Module):
    """
    1D CNN decoder for LIBS spectral prediction.

    Concentration vector -> small MLP -> [C, L0] feature map -> UP_STAGES x (upsample x2, Conv1d, BN, ReLU)
    -> Conv1d to a single channel -> crop to the true number of wavelengths.

      * Upsample + Conv1d (rather than ConvTranspose1d) avoids checkerboard artifacts.
      * The output is `conv_output + baseline`, where `baseline` is a learnable per-wavelength
        vector initialised to the mean training spectrum and the final conv starts with tiny weights.
        Epoch 0 of the loss curve therefore equals the "predict the mean spectrum" baseline and any
        improvement is real.
    """
    def __init__(
        self,
        in_features: int,
        n_wavelengths: int,
        base_filters: int = 32,
        dropout: float = 0.1,
        up_stages: int = 4
    ):
        super().__init__()
        self.in_features = in_features
        self.n_wavelengths = n_wavelengths
        self.base_filters = base_filters
        self.dropout = dropout
        self.up_stages = up_stages

        self.start_len = int(np.ceil(n_wavelengths / (2 ** up_stages)))
        self.start_ch = base_filters * (2 ** up_stages)

        self.fc = nn.Sequential(
            nn.Linear(in_features, self.start_ch * self.start_len),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        layers = []
        for i in range(up_stages):
            in_c = base_filters * (2 ** (up_stages - i))
            out_c = base_filters * (2 ** (up_stages - i - 1))
            layers.extend([
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv1d(in_c, out_c, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_c),
                nn.ReLU()
            ])
        self.up = nn.Sequential(*layers)
        self.out = nn.Conv1d(base_filters, 1, kernel_size=5, padding=2)
        self.baseline = nn.Parameter(torch.zeros(n_wavelengths))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.out.weight, std=1e-4)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def set_output_bias(self, target_means: torch.Tensor):
        self.baseline.data.copy_(target_means.to(self.baseline.device))

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        h = self.fc(c)
        h = h.view(h.size(0), self.start_ch, self.start_len)
        h = self.up(h)
        h = self.out(h)
        h = h[:, 0, :self.n_wavelengths]
        return h + self.baseline


# ==============================================================================
# Data Loading & Stratified Splitting
# ==============================================================================
def load_crossval_data(
    h5_path: str | Path = H5_FILE,
    spike_thresh: float = 10.0,
    exp_dopants: set[str] = EXP_DOPANTS,
    host_elements: set[str] = HOST_ELEMENTS,
    keep_elements: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Loads dataset into CPU RAM, cleans saturation spikes, filters unmodeled
    synthetic dopants, and slices target columns to active elements.
    """
    with h5py.File(h5_path, "r") as hf:
        X = get_h5_ds(hf, "X")[:].astype(np.float32)
        y = get_h5_ds(hf, "y")[:].astype(np.float32)
        is_synth = get_h5_ds(hf, "is_synthetic")[:].astype(bool)
        group_id = get_h5_ds(hf, "group_id")[:]
        wavelengths = get_h5_ds(hf, "wavelengths")[:]
        
        raw_names = get_h5_ds(hf, "feature_cols")[:]
        names = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in raw_names]

    # Filter out columns with corrupted saturation spikes
    row_maxes = y.max(axis=1)
    clean_spike_mask = row_maxes <= spike_thresh

    # Identify extra dopants in synthetic rows
    keep_elements = keep_elements or list(host_elements | exp_dopants)
    keep_idx = [i for i, name in enumerate(names) if name in keep_elements]
    extra_dopant_idx = [i for i, name in enumerate(names) if name not in keep_elements]

    if extra_dopant_idx:
        has_extra_dopants = (X[:, extra_dopant_idx] > 0).any(axis=1)
    else:
        has_extra_dopants = np.zeros(len(X), dtype=bool)

    valid_synth_mask = ~has_extra_dopants | ~is_synth
    final_row_mask = clean_spike_mask & valid_synth_mask

    X_clean = X[final_row_mask][:, keep_idx]
    y_clean = y[final_row_mask]
    is_synth_clean = is_synth[final_row_mask]
    group_id_clean = group_id[final_row_mask]
    names_clean = [names[i] for i in keep_idx]

    # L2 normalize spectra targets
    y_tensor = torch.from_numpy(y_clean)
    l2_norms = torch.norm(y_tensor, p=2, dim=1, keepdim=True).clamp(min=1e-8)
    y_tensor = y_tensor / l2_norms

    n_dropped_spikes = int((~clean_spike_mask).sum())
    n_dropped_synth = int((~valid_synth_mask & clean_spike_mask).sum())

    print("Data loading summary:")
    print(f"  - Dropped {n_dropped_spikes} corrupted saturation shots (max > {spike_thresh}).")
    print(f"  - Filtered {n_dropped_synth} synthetic shots containing non-experimental dopants.")
    print(f"  - Retained {len(X_clean)} total shots across {len(names_clean)} features: {names_clean}")

    return {
        "X": X_clean,
        "y": y_tensor,
        "is_synth": is_synth_clean,
        "group_id": group_id_clean,
        "wavelengths": wavelengths,
        "names": names_clean
    }


def assign_group_folds(
    is_synth: np.ndarray,
    group_id: np.ndarray,
    X: np.ndarray,
    dopant_idx: List[int],
    elem_names: List[str],
    n_splits: int = 7,
    n_trials: int = 1000,
    min_groups_per_dopant: int = 1,
    seed: int = 42
) -> np.ndarray:
    """Stratified group fold assignment matching original P3VAE script."""
    exp = ~is_synth
    groups, inv = np.unique(group_id[exp], return_inverse=True)
    n_groups = len(groups)
    shots = np.bincount(inv)

    first = np.empty(n_groups, dtype=int)
    first[inv] = np.arange(len(inv))[::-1]
    present = (X[exp][first][:, dopant_idx] > 0).astype(int)

    rng = np.random.default_rng(seed)
    best_score = float("inf")
    best_fog = None

    for _ in range(n_trials):
        fog = rng.integers(0, n_splits, size=n_groups)
        imbalance = 0.0
        for k in range(n_splits):
            members = (fog == k)
            if members.sum() < min_groups_per_dopant:
                imbalance += 1e6
            for d in range(present.shape[1]):
                if present[members, d].sum() < min_groups_per_dopant:
                    imbalance += 1e4

        if imbalance < best_score:
            best_score = imbalance
            best_fog = fog.copy()

    if best_fog is None or best_score >= 1e5:
        raise ValueError("Failed to assign fold groups: n_trials must be > 0.")

    fold_ids = np.full(len(is_synth), -1, dtype=np.int32)
    fold_ids[exp] = best_fog[inv]
    return fold_ids


def build_splits_once(data: Dict[str, Any], split_path: Path, num_folds: int = 7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads existing pre-computed CV splits or creates them if absent."""
    split_file = split_path.with_suffix(".npz")
    if split_file.exists():
        print(f"Loading cached splits from: {split_file}")
        arrs = np.load(split_file)
        test_idx, cv_idx, cv_fold_id = arrs["test_idx"], arrs["cv_idx"], arrs["cv_fold_id"]
        if len(test_idx) + len(cv_idx) == len(data["X"]):
            return test_idx, cv_idx, cv_fold_id
        print("Cached split covers mismatched rows; rebuilding for current filters.")

    split_path.parent.mkdir(parents=True, exist_ok=True)
    exp_mask = ~data["is_synth"]
    exp_groups = np.unique(data["group_id"][exp_mask])

    rng = np.random.default_rng(SEED)
    n_test_groups = int(np.ceil(0.15 * len(exp_groups)))
    test_groups = rng.choice(exp_groups, size=n_test_groups, replace=False)

    test_idx = np.where(np.isin(data["group_id"], test_groups) & exp_mask)[0]
    cv_idx = np.where(~np.isin(data["group_id"], test_groups))[0]

    dopant_idx = [i for i, name in enumerate(data["names"]) if name in EXP_DOPANTS]
    cv_fold_id = assign_group_folds(
        is_synth=data["is_synth"][cv_idx],
        group_id=data["group_id"][cv_idx],
        X=data["X"][cv_idx],
        dopant_idx=dopant_idx,
        elem_names=data["names"],
        n_splits=num_folds,
        seed=SEED
    )

    np.savez(split_file, test_idx=test_idx, cv_idx=cv_idx, cv_fold_id=cv_fold_id)
    return test_idx, cv_idx, cv_fold_id


# ==============================================================================
# Helper & Diagnostic Utilities
# ==============================================================================
def make_regions(wavelengths: np.ndarray, n_bands: int = 5) -> List[Tuple[str, slice]]:
    """('All', whole spectrum) followed by n_bands equal-width wavelength bands."""
    n_wl = len(wavelengths)
    regions = [("All", slice(0, n_wl))]
    chunks = np.array_split(np.arange(n_wl), n_bands)
    for chunk in chunks:
        a, b = chunk[0], chunk[-1]
        name = f"{wavelengths[a]:.0f}-{wavelengths[b]:.0f} nm"
        regions.append((name, slice(a, b + 1)))
    return regions


def region_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cols: slice
) -> Tuple[float, float, float, float]:
    """R^2, MAE, RMSE and max-true over `cols`."""
    yt, yp = y_true[:, cols], y_pred[:, cols]
    if yt.ndim == 1:
        yt = yt[None, :]
    if yp.ndim == 1:
        yp = yp[None, :]

    sse = np.sum((yt - yp) ** 2)
    sae = np.sum(np.abs(yt - yp))
    cnt = yt.size

    mean_t = np.mean(yt, axis=0, keepdims=True)
    ss_tot = np.sum((yt - mean_t) ** 2)

    r2 = 1.0 - (sse / ss_tot) if ss_tot > 0 else float("nan")
    mae = sae / cnt
    rmse = np.sqrt(sse / cnt)
    mx = float(np.max(yt))
    return r2, mae, rmse, mx


def pick_examples(Xc: np.ndarray, is_synth: np.ndarray, n_examples: int = 6) -> List[int]:
    """Highest-concentration sample of each present dopant, topped up with random samples."""
    exp = ~is_synth
    exp_indices = np.where(exp)[0]
    picks = []

    # Pick peak shot per dopant column
    for j in range(Xc.shape[1]):
        col = Xc[exp, j]
        if (col > 0).any():
            best_local = np.argmax(col)
            picks.append(exp_indices[best_local])

    picks = list(dict.fromkeys(picks))
    if len(picks) < n_examples:
        pool = np.setdiff1d(exp_indices, picks)
        rng = np.random.default_rng(SEED)
        extra = rng.choice(pool, size=n_examples - len(picks), replace=False)
        picks.extend(extra.tolist())

    return picks[:n_examples]


def plot_spectra_grid(
    wl: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    input_names: List[str],
    title_tag: str,
    save_path: Path
):
    """Grid of true vs predicted spectra for representative samples."""
    n_samples = len(y_true)
    cols = 2
    rows = int(np.ceil(n_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx in range(n_samples):
        ax = axes_flat[idx]
        yt, yp = y_true[idx], y_pred[idx]

        r2, mae, rmse, _ = region_metrics(yt[None, :], yp[None, :], slice(None))

        ax.plot(wl, yt, label="True", color="#1f77b4", lw=1.2)
        ax.plot(wl, yp, label="Pred", color="#d62728", lw=1.0, alpha=0.85)
        ax.set_title(f"Sample {idx+1} | R2: {r2:.3f} | RMSE: {rmse:.4f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Wavelength (nm)", fontsize=8)
        ax.set_ylabel("Intensity (a.u.)", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=8)

    for idx in range(n_samples, len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    fig.suptitle(f"{title_tag} - Spectral Overlay Grid", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Spectra Plot to: {save_path}")


def print_benchmark_table(
    y_true_model: np.ndarray,
    y_train_mean: np.ndarray,
    cols: slice,
    wavelengths: np.ndarray
):
    """Band-wise metrics comparing 1D CNN against Dummy Baselines."""
    zero_spec = np.zeros_like(y_train_mean)
    mean_spec = np.tile(y_train_mean, (len(y_true_model), 1))

    print("\n===============================================================================================")
    print(" Band          | Strategy      | R2       | MAE (a.u.) | RMSE (a.u.) | Max True")
    print("-----------------------------------------------------------------------------------------------")

    strategies = [
        ("Predict Zero", zero_spec),
        ("Predict Mean", mean_spec),
        ("1D CNN Model", y_true_model)
    ]

    for band_name, band_slice in make_regions(wavelengths, N_BANDS):
        for strat_name, y_hat in strategies:
            r2, mae, rmse, mx = region_metrics(y_true_model, y_hat, band_slice)
            print(f" {band_name:<13} | {strat_name:<13} | {r2:8.3f} | {mae:10.4f} | {rmse:11.4f} | {mx:10.3f}")
        print("-----------------------------------------------------------------------------------------------")


# ==============================================================================
# Model Evaluation & Inference Infrastructure
# ==============================================================================
def stash(X: torch.Tensor) -> torch.Tensor:
    """Park array in GPU RAM or pinned memory."""
    if DEVICE.type == "cuda":
        nbytes = X.numel() * 2
        free, _ = torch.cuda.mem_get_info()
        if nbytes < free * GPU_DATA_BUDGET_FRAC:
            return X.to(DEVICE, dtype=torch.float16)
    return X.pin_memory()


def gather_spec(Ys: torch.Tensor, idx: np.ndarray, by: int = 128) -> torch.Tensor:
    batch = Ys[idx]
    if batch.device != DEVICE:
        batch = batch.to(DEVICE, non_blocking=True)
    return batch.float()


def amp_setup() -> Tuple[Any, Optional[torch.cuda.amp.GradScaler]]:
    if USE_AMP and DEVICE.type == "cuda":
        is_bf16 = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if is_bf16 else torch.float16
        scaler = None if is_bf16 else torch.cuda.amp.GradScaler()
        return dtype, scaler
    return torch.float32, None


def post(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(min=0.0) if CLAMP_NONNEGATIVE else x


def predict(model: nn.Module, Xc: torch.Tensor, bs: int = 128) -> np.ndarray:
    model.eval()
    amp_dtype, _ = amp_setup()
    preds = []
    n = len(Xc)

    for i in range(0, n, bs):
        batch_c = Xc[i:i + bs].to(DEVICE)
        with torch.cuda.amp.autocast(enabled=(amp_dtype != torch.float32), dtype=amp_dtype):
            with torch.no_grad():
                out = post(model(batch_c))
        preds.append(out.cpu().numpy())

    return np.vstack(preds)


def eval_mse(model: nn.Module, Xc: torch.Tensor, Ys: torch.Tensor, idx: np.ndarray, bs: int = 128) -> float:
    """Streaming MSE over the rows in `idx`."""
    model.eval()
    amp_dtype, _ = amp_setup()
    total_sse = 0.0
    n = len(idx)

    for i in range(0, n, bs):
        batch_idx = idx[i:i + bs]
        batch_c = Xc[batch_idx].to(DEVICE)
        batch_y = gather_spec(Ys, batch_idx, bs)

        with torch.cuda.amp.autocast(enabled=(amp_dtype != torch.float32), dtype=amp_dtype):
            with torch.no_grad():
                pred = post(model(batch_c))
                sse = F.mse_loss(pred, batch_y, reduction="sum").item()
                total_sse += sse

    return total_sse / (n * Ys.shape[1])


def fit(
    model: nn.Module,
    Xc_fit: torch.Tensor,
    Xc_eval: torch.Tensor,
    Ys: torch.Tensor,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    epochs: int = EPOCHS,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """Fixed-length training with OneCycleLR schedule."""
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    steps_per_epoch = int(np.ceil(len(tr_idx) / BATCH_SIZE))
    total_steps = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LEARNING_RATE, total_steps=total_steps, pct_start=0.3
    )

    amp_dtype, scaler = amp_setup()
    hist = {"train": [], "val": [], "lr": []}
    t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(tr_idx)
        running_sse = 0.0

        for i in range(0, len(tr_idx), BATCH_SIZE):
            batch_idx = perm[i:i + BATCH_SIZE]
            c_batch = Xc_fit[batch_idx].to(DEVICE)
            y_batch = gather_spec(Ys, batch_idx, BATCH_SIZE)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(amp_dtype != torch.float32), dtype=amp_dtype):
                pred = post(model(c_batch))
                loss = F.mse_loss(pred, y_batch)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            sched.step()
            running_sse += loss.item() * len(batch_idx) * Ys.shape[1]

        tr_mse = running_sse / (len(tr_idx) * Ys.shape[1])
        va_mse = eval_mse(model, Xc_eval, Ys, va_idx, BATCH_SIZE)

        hist["train"].append(tr_mse)
        hist["val"].append(va_mse)
        hist["lr"].append(sched.get_last_lr()[0])

        if verbose and (ep % LOG_EVERY == 0 or ep == epochs):
            dt = time.perf_counter() - t0
            sps = (len(tr_idx) * ep) / dt
            print(f"    ep {ep:3d}/{epochs} | train MSE {tr_mse:.6f} | held-out MSE {va_mse:.6f} | lr {hist['lr'][-1]:.2e} | {sps:.0f} samples/s")

    return hist


def new_model(in_dim: int, n_wl: int, init_bias: Optional[torch.Tensor] = None) -> nn.Module:
    m = PyTorch1DCNN(
        in_features=in_dim,
        n_wavelengths=n_wl,
        base_filters=BASE_FILTERS,
        dropout=DROPOUT,
        up_stages=UP_STAGES
    )
    if init_bias is not None:
        m.set_output_bias(init_bias)
    if USE_COMPILE:
        m = torch.compile(m)
    return m    # type: ignore


def fold_indices(fold_ids: np.ndarray, fold: int) -> Tuple[np.ndarray, np.ndarray]:
    tr = np.where(fold_ids != fold)[0]
    va = np.where(fold_ids == fold)[0]
    return tr, va


def mean_spectrum(Ys: torch.Tensor, idx: np.ndarray) -> torch.Tensor:
    """Mean spectrum of the rows in `idx`."""
    total = torch.zeros(Ys.shape[1], device=DEVICE)
    for i in range(0, len(idx), BATCH_SIZE):
        batch_idx = idx[i:i + BATCH_SIZE]
        total += gather_spec(Ys, batch_idx, BATCH_SIZE).sum(dim=0)
    return total / len(idx)


def mean_baseline_mse(Ys: torch.Tensor, tr_idx: np.ndarray, va_idx: np.ndarray) -> float:
    mu = mean_spectrum(Ys, tr_idx)
    total_sse = 0.0
    for i in range(0, len(va_idx), BATCH_SIZE):
        batch_idx = va_idx[i:i + BATCH_SIZE]
        y_batch = gather_spec(Ys, batch_idx, BATCH_SIZE)
        total_sse += F.mse_loss(y_batch, mu.expand_as(y_batch), reduction="sum").item()
    return total_sse / (len(va_idx) * Ys.shape[1])


# ==============================================================================
# Sanity Checks
# ==============================================================================
def data_report(Xc_np: np.ndarray, Y_cv_cpu: torch.Tensor, is_synth: np.ndarray, feature_names: List[str]):
    print("\n================================================================================")
    print(" SANITY CHECK 1: inputs and targets")
    print("================================================================================")
    
    rms = torch.norm(Y_cv_cpu, p=2, dim=1) / math.sqrt(Y_cv_cpu.shape[1])
    print(f"Spectra: finite: {torch.isfinite(Y_cv_cpu).all().item()} | per-spectrum RMS min/median/max: {rms.min().item():.4f}/{rms.median().item():.4f}/{rms.max().item():.4f}")
    print(f"Inputs: finite: {np.isfinite(Xc_np).all()} | range [{Xc_np.min():.4f}, {Xc_np.max():.4f}]")
    
    exp_mask = ~is_synth
    print(f"Rows: {exp_mask.sum()} experimental + {is_synth.sum()} synthetic")
    
    base_mse = mean_baseline_mse(Y_cv_cpu, np.where(exp_mask)[0], np.where(exp_mask)[0])
    print(f"Spectra predict-mean MSE (experimental rows): {base_mse:.6f}")


def overfit_test(Xc: torch.Tensor, Ys: torch.Tensor, is_synth: np.ndarray, in_dim: int, n_wl: int):
    print("\n================================================================================")
    print(" SANITY CHECK 2: can the model memorise 512 training samples?")
    print("================================================================================")
    
    exp_idx = np.where(~is_synth)[0][:512]
    mu = mean_spectrum(Ys, exp_idx)
    base_mse = mean_baseline_mse(Ys, exp_idx, exp_idx)

    model = new_model(in_dim, n_wl, mu)
    hist = fit(model, Xc, Xc, Ys, exp_idx, exp_idx, epochs=OVERFIT_EPOCHS, verbose=False)
    
    final_mse = hist["val"][-1]
    pct = (1.0 - final_mse / base_mse) * 100.0
    status = "PASS" if pct >= 80.0 else "FAIL"

    print(f"Predict-mean MSE on 512 samples: {base_mse:.6f}")
    print(f"Final MSE after {OVERFIT_EPOCHS} steps: {final_mse:.6f} ({pct:.1f}% improvement) -> {status}")


def shuffle_control(Xc: torch.Tensor, Ys: torch.Tensor, is_synth: np.ndarray, in_dim: int, n_wl: int, fold_ids: np.ndarray):
    print("\n================================================================================")
    print(" SANITY CHECK 3: shuffled-label control (fold 0)")
    print("================================================================================")
    
    tr_idx, va_idx = fold_indices(fold_ids, 0)
    perm = np.random.permutation(len(tr_idx))
    Xc_shuffled = Xc.clone()
    Xc_shuffled[tr_idx] = Xc_shuffled[tr_idx[perm]]

    mu = mean_spectrum(Ys, tr_idx)
    model = new_model(in_dim, n_wl, mu)
    hist = fit(model, Xc_shuffled, Xc, Ys, tr_idx, va_idx, epochs=EPOCHS, verbose=False)

    base_mse = mean_baseline_mse(Ys, tr_idx, va_idx)
    final_mse = hist["val"][-1]
    r2 = 1.0 - (final_mse / base_mse)
    status = "PASS" if r2 <= 0.05 else "FAIL"

    print(f"Predict-mean held-out MSE: {base_mse:.6f}")
    print(f"Shuffled-label held-out MSE: {final_mse:.6f} (R2 = {r2:.3f}) -> {status}")


def plot_learning_curves(histories: List[Dict[str, List[float]]], baselines: List[float], save_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for k, (h, b) in enumerate(zip(histories, baselines)):
        ax.plot(h["val"], label=f"Fold {k} Held-out MSE", alpha=0.8)
        ax.axhline(b, color="gray", linestyle="--", alpha=0.5)

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("1D CNN Learning Curves across CV Folds")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved learning curves to: {save_path}")


def print_learning_verdict(histories: List[Dict[str, List[float]]], baselines: List[float]):
    print("\n--- Learning verdict ---")
    for k, (h, b) in enumerate(zip(histories, baselines)):
        final_val = h["val"][-1]
        r2 = 1.0 - (final_val / b) if b > 0 else float("nan")
        status = "LEARNING" if r2 > 0.1 else "NOT LEARNING"
        print(f"Fold {k}: R2 = {r2:7.3f} -> {status}")


# ==============================================================================
# Main Orchestration Script
# ==============================================================================
def main():
    t0_start = time.perf_counter()
    print(f"=== Fast 1D CNN Benchmark [{TIMESTAMP}] | amp={USE_AMP} | up_stages={UP_STAGES} | compile={USE_COMPILE} ===")

    data = load_crossval_data()
    test_idx, cv_idx, cv_fold_id = build_splits_once(data, SPLIT_DIR / f"max12_folds_seed{SEED}.npz", NUM_FOLDS)

    print(f"Dataset Loaded | CV Pool: {len(cv_idx)} samples | Lockbox Test Set: {len(test_idx)} samples")

    Xc_np = data["X"]
    Y_cpu = data["y"]
    is_synth = data["is_synth"]
    feature_names = data["names"]
    wavelengths = data["wavelengths"]

    in_dim = Xc_np.shape[1]
    n_wl = Y_cpu.shape[1]

    # Stash data tensors
    Xc_tensor = torch.from_numpy(Xc_np).float()
    Ys_stashed = stash(Y_cpu)

    if RUN_SANITY_CHECKS:
        data_report(Xc_np, Y_cpu, is_synth, feature_names)
        overfit_test(Xc_tensor, Ys_stashed, is_synth, in_dim, n_wl)
        if RUN_SHUFFLE_CONTROL:
            shuffle_control(Xc_tensor, Ys_stashed, is_synth, in_dim, n_wl, cv_fold_id)

    print("\n--- Training 1D CNN across K Folds ---")
    
    oof_preds = np.zeros((len(cv_idx), n_wl), dtype=np.float32)
    lockbox_sum = np.zeros((len(test_idx), n_wl), dtype=np.float32)

    histories = []
    baselines = []

    for fold in range(NUM_FOLDS):
        tr_fold_idx, va_fold_idx = fold_indices(cv_fold_id, fold)
        
        tr_global_idx = cv_idx[tr_fold_idx]
        va_global_idx = cv_idx[va_fold_idx]

        mu = mean_spectrum(Ys_stashed, tr_global_idx)
        base_mse = mean_baseline_mse(Ys_stashed, tr_global_idx, va_global_idx)
        baselines.append(base_mse)

        print(f"\nFold {fold+1}/{NUM_FOLDS} | train rows {len(tr_global_idx)} | held-out rows {len(va_global_idx)} | predict-mean held-out MSE {base_mse:.6f}")

        # Train model
        model = new_model(in_dim, n_wl, mu)
        hist = fit(model, Xc_tensor, Xc_tensor, Ys_stashed, tr_global_idx, va_global_idx, epochs=EPOCHS, verbose=True)
        histories.append(hist)

        # SAVE MODEL WEIGHTS FOR THIS FOLD
        model_save_path = OUTPUT_DIR / f"cnn_1d_model_fold{fold}_{TIMESTAMP}.pt"
        torch.save({
            "fold": fold,
            "model_state_dict": model.state_dict(),
            "in_dim": in_dim,
            "n_wl": n_wl,
            "baseline": model.baseline.data.cpu()
        }, model_save_path)
        print(f"    Saved Fold {fold+1} model to: {model_save_path.name}")

        val_preds = predict(model, Xc_tensor[va_global_idx])
        oof_preds[va_fold_idx] = val_preds

        test_preds = predict(model, Xc_tensor[test_idx])
        lockbox_sum += test_preds

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    plot_learning_curves(histories, baselines, OUTPUT_DIR / f"cnn_1d_learning_curves_{TIMESTAMP}.png")
    print_learning_verdict(histories, baselines)

    # Evaluate OOF Predictions
    Y_cv_np = Y_cpu[cv_idx].numpy()
    print_benchmark_table(oof_preds, Y_cv_np.mean(axis=0), slice(None), wavelengths)

    # Evaluate Lockbox Predictions
    lockbox_avg = lockbox_sum / NUM_FOLDS
    Y_test_np = Y_cpu[test_idx].numpy()
    print_benchmark_table(lockbox_avg, Y_cv_np.mean(axis=0), slice(None), wavelengths)

    # Plot sample spectral grid
    sample_picks = pick_examples(Xc_np[test_idx], is_synth[test_idx], N_EXAMPLE_SPECTRA)
    plot_spectra_grid(
        wavelengths,
        Y_test_np[sample_picks],
        lockbox_avg[sample_picks],
        feature_names,
        "1D CNN Lockbox Ensemble",
        OUTPUT_DIR / f"gpu_cnn_1d_L2_synth_lockbox_spectra_{TIMESTAMP}.png"
    )

    t_total = time.perf_counter() - t0_start
    print(f"\nTraining time completed in {t_total:.1f}s.")


if __name__ == "__main__":
    main()