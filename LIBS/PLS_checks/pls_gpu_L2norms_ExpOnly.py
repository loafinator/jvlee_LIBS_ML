from __future__ import annotations

import sys
import time
import h5py
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
H5_FILE = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/max12_cts_noleak_crossval.h5"
SPLIT_DIR = Path('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/fold_splits')
OUTPUT_DIR = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/PLS_checks/plots/pls_benchmark")

NUM_FOLDS = 7
THRESH = 2.8
MAX_WT_PERCENT = 12.0
SEED = 42
HOST_ELEMENTS = {'Li', 'K', 'Cl'}
EXP_DOPANTS = {'Ce', 'Gd', 'Sm', 'U'}

# Candidate range for PLS components
N_COMPONENTS_GRID = [2, 5, 10, 15, 20, 25, 30, 40, 50]

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Fast GPU SIMPLS Algorithm (In-Place Memory Optimized)
# =============================================================================
class PyTorchPLS:
    """Fast GPU-accelerated PLS2 (SIMPLS Algorithm) in PyTorch."""
    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.mean_X: torch.Tensor | None = None
        self.std_X: torch.Tensor | None = None
        self.mean_Y: torch.Tensor | None = None
        self.B: torch.Tensor | None = None

    def fit(self, X: torch.Tensor, Y: torch.Tensor):
        N, P = X.shape
        M = Y.shape[1]

        # Standardize X and center Y
        self.mean_X = X.mean(dim=0, keepdim=True)
        self.std_X = X.std(dim=0, keepdim=True).clamp(min=1e-6)
        self.mean_Y = Y.mean(dim=0, keepdim=True)

        # In-place standardization on GPU tensor to save 13 GB VRAM
        X.sub_(self.mean_X).div_(self.std_X)
        Y0 = Y - self.mean_Y

        # Initial Cross-Covariance Matrix S [P x M]
        S = torch.matmul(X.T, Y0)

        R = torch.zeros((P, self.n_components), dtype=X.dtype, device=X.device)
        Q = torch.zeros((M, self.n_components), dtype=X.dtype, device=X.device)
        V = torch.zeros((P, self.n_components), dtype=X.dtype, device=X.device)

        for a in range(self.n_components):
            # Compute dominant left singular vector of S via GPU SVD
            U, _, _ = torch.linalg.svd(S, full_matrices=False)
            r_a = U[:, 0]

            # Compute score t_a and normalize
            t_a = torch.matmul(X, r_a)
            t_a = t_a - t_a.mean()
            norm_t = torch.linalg.norm(t_a)
            t_a = t_a / norm_t
            r_a = r_a / norm_t

            # Compute loadings
            p_a = torch.matmul(X.T, t_a)
            q_a = torch.matmul(Y0.T, t_a)

            # Deflation vector v_a (orthogonalized against previous V)
            v_a = p_a.clone()
            if a > 0:
                V_prev = V[:, :a]
                v_a = v_a - torch.matmul(V_prev, torch.matmul(V_prev.T, p_a))
            v_a = v_a / torch.linalg.norm(v_a)

            # Deflate cross-covariance matrix
            S = S - v_a.unsqueeze(1) @ (v_a.unsqueeze(0) @ S)

            R[:, a] = r_a
            Q[:, a] = q_a
            V[:, a] = v_a

        # Compute final regression weight matrix B
        self.B = torch.matmul(R, Q.T)
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if self.B is None or self.mean_X is None or self.std_X is None or self.mean_Y is None:
            raise RuntimeError("Model must be fitted via `.fit()` before calling `.predict()`!")

        X0 = (X - self.mean_X) / self.std_X
        pred = torch.matmul(X0, self.B) + self.mean_Y
        return pred


# =============================================================================
# Helper & Data Functions
# =============================================================================
def load_crossval_data(h5_path: str, spike_thresh: float = THRESH) -> dict:
    """Loads dataset as float32 tensors stored in System CPU RAM."""
    with h5py.File(h5_path, 'r') as hf:
        y = get_h5_ds(hf, 'y')[:].astype(np.float32)
        X = get_h5_ds(hf, 'X')[:].astype(np.float32)
        is_synth = get_h5_ds(hf, 'is_synthetic')[:]
        is_lockbox = get_h5_ds(hf, 'is_lockbox')[:].astype(bool)
        group_id = get_h5_ds(hf, 'group_id')[:]
        wavelengths = get_h5_ds(hf, 'wavelengths')[:]
        names = [
            n.decode('utf-8') if isinstance(n, bytes) else str(n) 
            for n in get_h5_ds(hf, 'feature_cols')[:]
        ]


    row_maxes = y.max(axis=1)
    clean_mask = row_maxes <= spike_thresh

    y_tensor = torch.from_numpy(y[clean_mask])

    # L2 Normalization
    l2_norms = torch.norm(y_tensor, p=2, dim=1, keepdim=True).clamp(min=1e-08)
    y_tensor = y_tensor / l2_norms
    
    n_dropped = len(y) - np.sum(clean_mask)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} corrupted saturation shots (max intensity > {spike_thresh}).")

    return dict(
        X=torch.from_numpy(X[clean_mask]),
        y=y_tensor,
        is_synth=is_synth[clean_mask],
        is_lockbox=is_lockbox[clean_mask],
        group_id=group_id[clean_mask],
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
    test_idx = np.where(data['is_lockbox'])[0]
    cv_idx = np.where(~data['is_lockbox'])[0]

    X_cv = data['X'][cv_idx].numpy()
    is_synth_cv = data['is_synth'][cv_idx]

    host_elements = {'Li', 'K', 'Cl'}
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
    Prints element-wise metrics comparing PLS against Dummy Baselines:
    - Predict Zero (y_pred = 0)
    - Predict Mean (y_pred = train_mean)
    - Active Model (y_pred = y_pred_model)
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
# =============================================================================
# Main Execution Pipeline
# =============================================================================
def main():
    print(f"=== Starting GPU-Accelerated PLS Benchmark Pipeline [{DEVICE}] ===")
    t0_start = time.perf_counter()

    # 1. Load Data into CPU System RAM
    data = load_crossval_data(H5_FILE)
    split_path = SPLIT_DIR / f'max12_{NUM_FOLDS}_folds_seed{SEED}.npz'
    test_idx, cv_idx, cv_fold_id = build_splits_once(data, split_path, NUM_FOLDS, seed=SEED)

    # Store matrices in CPU System RAM
    X_cv_cpu = data['y'][cv_idx]
    Y_cv_cpu = data['X'][cv_idx]
    X_test_cpu = data['y'][test_idx]
    Y_test_cpu = data['X'][test_idx]
    # Sliced target matrices
    dopant_indices = [i for i, name in enumerate(data['names']) if name in EXP_DOPANTS]
    Y_cv_dopants = Y_cv_cpu[:, dopant_indices]
    Y_test_dopants = Y_test_cpu[:, dopant_indices]
    Y_cv_cpu = Y_cv_dopants
    Y_test_cpu = Y_test_dopants
    target_names = [data['names'][i] for i in dopant_indices]
    # target_names = data['names']

    print(f"Dataset Stored in CPU RAM | CV Pool: {len(cv_idx)} samples | Lockbox Test Set: {len(test_idx)} samples")

    # 2. Hyperparameter Grid Search over `n_components`
    print("\n--- Tuning PLS Components via GPU Cross-Validation ---")
    best_n_comp = N_COMPONENTS_GRID[0]
    best_overall_r2 = -float('inf')
    oof_by_ncomp = {}

    for n_comp in N_COMPONENTS_GRID:
        t_grid_start = time.perf_counter()
        oof_preds_gpu = torch.zeros((len(cv_idx), Y_cv_cpu.shape[1]), device=DEVICE)

        for k in range(NUM_FOLDS):
            train_mask = cv_fold_id != k
            val_mask = cv_fold_id == k

            # Transfer ONLY active fold to GPU
            X_tr_gpu = X_cv_cpu[train_mask].to(DEVICE)
            Y_tr_gpu = Y_cv_cpu[train_mask].to(DEVICE)
            X_va_gpu = X_cv_cpu[val_mask].to(DEVICE)

            pls = PyTorchPLS(n_components=n_comp)
            pls.fit(X_tr_gpu, Y_tr_gpu)

            oof_preds_gpu[val_mask] = pls.predict(X_va_gpu)

            # Reclaim GPU VRAM immediately after fold evaluation
            del X_tr_gpu, Y_tr_gpu, X_va_gpu, pls
            torch.cuda.empty_cache()

        oof_preds_np = torch.clamp(oof_preds_gpu, min=0.0).cpu().numpy()
        oof_by_ncomp[n_comp] = oof_preds_np

        exp_mask = cv_fold_id >= 0
        mean_r2 = r2_calcs(Y_cv_cpu.numpy()[exp_mask] * MAX_WT_PERCENT, oof_preds_np[exp_mask] * MAX_WT_PERCENT)
        dt = time.perf_counter() - t_grid_start
        print(f"n_components = {n_comp:2d} | OOF Mean R^2: {mean_r2:7.4f} | Elapsed: {dt:.2f}s")

        if mean_r2 > best_overall_r2:
            best_overall_r2 = mean_r2
            best_n_comp = n_comp

    print(f"\nOptimal PLS Components: {best_n_comp} (Mean OOF R^2: {best_overall_r2:.4f})")

    # 3. Final Lockbox Evaluation & Ensembling
    print("\n--- Running Final Lockbox Ensemble Evaluation on GPU ---")
    lockbox_preds_gpu = torch.zeros((NUM_FOLDS, len(test_idx), Y_test_cpu.shape[1]), device=DEVICE)
    X_test_gpu = X_test_cpu.to(DEVICE)
    is_synth_cv = data['is_synth'][cv_idx]

    for k in range(NUM_FOLDS):
        train_mask = (cv_fold_id != k) & (is_synth_cv == 0)
        X_tr_gpu = X_cv_cpu[train_mask].to(DEVICE)
        Y_tr_gpu = Y_cv_cpu[train_mask].to(DEVICE)

        pls = PyTorchPLS(n_components=best_n_comp)
        pls.fit(X_tr_gpu, Y_tr_gpu)

        lockbox_preds_gpu[k] = pls.predict(X_test_gpu)

        del X_tr_gpu, Y_tr_gpu, pls
        torch.cuda.empty_cache()

    lockbox_pred_avg = torch.clamp(lockbox_preds_gpu.mean(dim=0), min=0.0).cpu().numpy()

    # 4. Scale back to physical wt% and report metrics
    exp_mask = cv_fold_id >= 0
    true_cv_wt = Y_cv_cpu[exp_mask].numpy() * MAX_WT_PERCENT
    pred_cv_wt = oof_by_ncomp[best_n_comp][exp_mask] * MAX_WT_PERCENT
    train_mean_cv = true_cv_wt.mean(axis=0)

    true_test_wt = Y_test_cpu.numpy() * MAX_WT_PERCENT
    pred_test_wt = lockbox_pred_avg * MAX_WT_PERCENT
    train_mean_full = (Y_cv_cpu[exp_mask].numpy() * MAX_WT_PERCENT).mean(axis=0)

    # Report OOF with Baselines
    print_benchmark_table(
        y_true=true_cv_wt,
        y_pred_model=pred_cv_wt,
        y_train_mean=train_mean_cv,
        title_tag=f"GPU PLS OOF Benchmarks (n_components={best_n_comp}) [Exp Only]",
        elem_names=target_names
    )

    # Report Lockbox with Baselines
    print_benchmark_table(
        y_true=true_test_wt,
        y_pred_model=pred_test_wt,
        y_train_mean=train_mean_full,
        title_tag=f"GPU PLS Lockbox Benchmarks (Ensemble of {NUM_FOLDS} Folds)",
        elem_names=target_names
    )

    active_elem_idx = [j for j in range(true_test_wt.shape[1]) if true_test_wt[:, j].max() > 1e-5]

    print("\n" + "=" * 80)
    print(f" GPU PLS Benchmark - Out-Of-Fold (OOF) Performance (n_components={best_n_comp})")
    print("=" * 80)
    for j in active_elem_idx:
        name = target_names[j]
        yt = true_cv_wt[:, j]
        yp = pred_cv_wt[:, j]
        r2 = r2_score(yt, yp)
        mae = mean_absolute_error(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        print(f"Element {name:<5} | R^2: {r2:7.3f} | MAE: {mae:.4f} wt% | RMSE: {rmse:.4f} wt% | Max True: {yt.max():.3f} wt%")

    print("\n" + "=" * 80)
    print(f" GPU PLS Benchmark - Lockbox Test Set Performance (Ensemble of {NUM_FOLDS} Folds)")
    print("=" * 80)
    for j in active_elem_idx:
        name = target_names[j]
        yt = true_test_wt[:, j]
        yp = pred_test_wt[:, j]
        r2 = r2_score(yt, yp)
        mae = mean_absolute_error(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        print(f"Element {name:<5} | R^2: {r2:7.3f} | MAE: {mae:.4f} wt% | RMSE: {rmse:.4f} wt% | Max True: {yt.max():.3f} wt%")
    print("=" * 80 + "\n")

    # 5. Parity Plots
    oof_parity_path = OUTPUT_DIR / f"gpu_pls_L2_expr_oof_parity_{TIMESTAMP}.png"
    lockbox_parity_path = OUTPUT_DIR / f"gpu_pls_L2_expr_lockbox_parity_{TIMESTAMP}.png"

    plot_parity_grid(true_cv_wt, pred_cv_wt, target_names, 
                    #  title_tag=f"PLS L2 EXPERIMENTAL OOF (n_comp={best_n_comp})", save_path=oof_parity_path)
                     title_tag=f"PLS L2 EXPERIMENTAL OOF", save_path=oof_parity_path)
    plot_parity_grid(true_test_wt, pred_test_wt, target_names, 
                    #  title_tag=f"PLS L2 EXPERIMENTAL LOCKBOX ENSEMBLE (n_comp={best_n_comp})", save_path=lockbox_parity_path)
                     title_tag=f"PLS L2 EXPERIMENTAL LOCKBOX ENSEMBLE", save_path=lockbox_parity_path)

    print(f"GPU Pipeline completed in {(time.perf_counter() - t0_start):.2f} seconds.")


if __name__ == "__main__":
    main()