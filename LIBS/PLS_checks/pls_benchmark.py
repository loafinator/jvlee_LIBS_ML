from __future__ import annotations

import sys
import time
import h5py
import torch
import hashlib

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
SEEDS = [42]    # for non pls up seed count to like 7: [42,43,44,45,46,47,48]
THRESH = 2.8
MAX_WT_PERCENT = 12.0
SEED = 42
HOST_ELEMENTS = {'Li', 'K', 'Cl'}
EXP_DOPANTS = {'Ce', 'Gd', 'Sm', 'U'}

# Candidate range for PLS components
# N_COMPONENTS_GRID = [1,2,5,10,20,30,40,50,60,70,80,90,100]
# N_COMPONENTS_GRID = [20,25,28,29,30,31,32,35,40]
# N_COMPONENTS_GRID = [50,55,58,59,60,61,62,65,70]
N_COMPONENTS_GRID = [70,75,78,79,80,81,82,85,90]

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

        # In-place standardization on GPU tensor to save VRAM
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
# Main Execution Pipeline
# =============================================================================
def main(
        h5_file: Path | str = H5_FILE,
        split_dir: Path | str = SPLIT_DIR,
        output_dir: Path | str = OUTPUT_DIR,
        include_synthetic: bool = False,
        seeds: list[int] = SEEDS,
):
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_tag = Path(h5_file).stem
    print(f"=== Starting GPU-Accelerated PLS Benchmark Pipeline [{DEVICE}] ===")
    print(f"Dataset: {dataset_tag} | Repeated K-Fold: {NUM_FOLDS} Folds x {len(seeds)} Seeds ({len(seeds)*NUM_FOLDS} models per grid point)")
    t0_start = time.perf_counter()

    h5_file = str(h5_file)
    split_dir = Path(split_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data & Prepare Multi-Seed Splits
    data = load_crossval_data(h5_file)
    n_synth = (data['is_synth'] == 1).sum()
    print(f"Loaded {len(data['X'])} total shots | Real: {len(data['X']) - n_synth} | Synthetic: {n_synth}")

    if include_synthetic and n_synth == 0:
        print("⚠️ WARNING: 'include_synthetic=True' was requested, but 0 synthetic shots were found in HDF5!")
    
    # Load fold IDs for all seeds
    seed_splits = {}
    for seed in seeds:
        test_idx, cv_idx, cv_fold_id = build_splits_for_seed(data, split_dir, NUM_FOLDS, seed=seed)
        seed_splits[seed] = (test_idx, cv_idx, cv_fold_id)

    # Note: test_idx & cv_idx are fixed across seeds (only fold partitioning inside cv_idx varies)
    test_idx = seed_splits[seeds[0]][0]
    cv_idx = seed_splits[seeds[0]][1]

    # X_cv_cpu = data['X'][cv_idx]
    # Y_cv_cpu = data['y'][cv_idx]
    # X_test_cpu = data['X'][test_idx]
    # Y_test_cpu = data['y'][test_idx]
    # is_synth_cv = data['is_synth'][cv_idx]

    # Move CV pool directly to GPU memory once to avoid CPU slicing OOMs
    X_cv_gpu = data['X'][cv_idx].to(DEVICE)
    Y_cv_gpu = data['y'][cv_idx].to(DEVICE)
    X_test_gpu = data['X'][test_idx].to(DEVICE)
    Y_test_gpu = data['y'][test_idx].to(DEVICE)
    is_synth_cv_gpu = torch.from_numpy(data['is_synth'][cv_idx]).to(DEVICE)

    # Target slicing
    dopant_indices = [i for i, name in enumerate(data['names']) if name in EXP_DOPANTS or name in {'Ce', 'Gd', 'Sm', 'U'}]
    if dopant_indices:
        Y_cv_gpu = Y_cv_gpu[:, dopant_indices]
        Y_test_gpu = Y_test_gpu[:, dopant_indices]
        target_names = [data['names'][i] for i in dopant_indices]
    else:
        target_names = data['names']

    # print(f"Dataset Stored in CPU RAM | CV Pool: {len(cv_idx)} samples | Lockbox Test Set: {len(test_idx)} samples")

    # 2. Hyperparameter Grid Search (Ensembled OOF across N Seeds)
    print("\n--- Tuning PLS Components via Multi-Seed GPU Cross-Validation ---")
    best_n_comp = N_COMPONENTS_GRID[0]
    best_overall_rmse = float('inf')
    best_overall_r2 = -float('inf')
    best_oof_preds_np = None

    for n_comp in N_COMPONENTS_GRID:
        t_grid_start = time.perf_counter()
        
        # Tensor to accumulate OOF predictions across all seeds
        oof_sum_gpu = torch.zeros((len(cv_idx), Y_cv_gpu.shape[1]), device=DEVICE)

        for seed in seeds:
            _, _, cv_fold_id = seed_splits[seed]
            seed_oof_gpu = torch.zeros((len(cv_idx), Y_cv_gpu.shape[1]), device=DEVICE)

            for k in range(NUM_FOLDS):
                if include_synthetic:
                    train_mask = torch.from_numpy(cv_fold_id != k).to(DEVICE)
                else:
                    train_mask = torch.from_numpy((cv_fold_id != k) & (data['is_synth'][cv_idx] == 0)).to(DEVICE)
                val_mask = torch.from_numpy(cv_fold_id == k).to(DEVICE)

                # GPU-to-GPU slicing (0 ms overhead, 0 CPU RAM used)
                X_tr_gpu = X_cv_gpu[train_mask]
                Y_tr_gpu = Y_cv_gpu[train_mask]
                X_va_gpu = X_cv_gpu[val_mask]

                pls = PyTorchPLS(n_components=n_comp)
                pls.fit(X_tr_gpu.clone(), Y_tr_gpu)

                seed_oof_gpu[val_mask] = pls.predict(X_va_gpu)

                del X_tr_gpu, Y_tr_gpu, X_va_gpu, pls
                torch.cuda.empty_cache()

            oof_sum_gpu += seed_oof_gpu

        # Multi-seed ensembled OOF prediction (Leak-free average across N seeds)
        oof_ens_gpu = oof_sum_gpu / len(seeds)
        oof_ens_np = torch.clamp(oof_ens_gpu, min=0.0).cpu().numpy()

        # Evaluate ensembled OOF RMSE and R^2 on experimental validation samples
        exp_mask = seed_splits[seeds[0]][2] >= 0
        y_true_exp = Y_cv_gpu.cpu().numpy()[exp_mask]
        y_pred_exp = oof_ens_np[exp_mask]

        mean_rmse = rmse_calcs(y_true_exp, y_pred_exp)
        mean_r2 = r2_calcs(y_true_exp, y_pred_exp)
        dt = time.perf_counter() - t_grid_start

        print(
            f"n_components = {n_comp:2d} | "
            f"Ensembled OOF Mean RMSE: {mean_rmse:7.4f} wt% (R^2: {mean_r2:7.4f}) | "
            f"Elapsed: {dt:.2f}s"
        )

        # Minimize RMSE as primary selection criterion
        if best_oof_preds_np is None or mean_rmse < best_overall_rmse:
            best_overall_rmse = mean_rmse
            best_overall_r2 = mean_r2
            best_n_comp = n_comp
            best_oof_preds_np = oof_ens_np

    print(
        f"\nOptimal PLS Components: {best_n_comp} "
        f"(Ensembled OOF Mean RMSE: {best_overall_rmse:.4f} wt% | R^2: {best_overall_r2:.4f})"
    )

    if best_oof_preds_np is None:
        raise RuntimeError("No valid out-of-fold predictions were captured during hyperparamater tuning.")
    # 3. Final Lockbox Evaluation (Ensemble of N_seeds x K_folds models)
    total_models = len(seeds) * NUM_FOLDS
    print(f"\n--- Running Final Lockbox Ensemble Evaluation on GPU ({total_models} total models) ---")
    
    lockbox_sum_gpu = torch.zeros((len(test_idx), Y_test_gpu.shape[1]), device=DEVICE)
    X_test_gpu = X_test_gpu.to(DEVICE)

    for seed in seeds:
        _, _, cv_fold_id = seed_splits[seed]

        for k in range(NUM_FOLDS):
            if include_synthetic:
                train_mask = torch.from_numpy(cv_fold_id != k).to(DEVICE)
            else:
                train_mask = torch.from_numpy((cv_fold_id != k) & (data['is_synth'][cv_idx] == 0)).to(DEVICE)

            X_tr_gpu = X_cv_gpu[train_mask]
            Y_tr_gpu = Y_cv_gpu[train_mask]

            pls = PyTorchPLS(n_components=best_n_comp)
            pls.fit(X_tr_gpu.clone(), Y_tr_gpu)

            lockbox_sum_gpu += pls.predict(X_test_gpu)

            del X_tr_gpu, Y_tr_gpu, pls
            torch.cuda.empty_cache()

    lockbox_pred_avg = torch.clamp(lockbox_sum_gpu / total_models, min=0.0).cpu().numpy()

    # 4. Reporting
    exp_mask = seed_splits[seeds[0]][2] >= 0
    true_cv_wt = Y_cv_gpu[exp_mask].cpu().numpy()
    # pred_cv_wt = oof_ens_by_ncomp[best_n_comp][exp_mask]
    pred_cv_wt = best_oof_preds_np[exp_mask]
    train_mean_cv = true_cv_wt.mean(axis=0)

    true_test_wt = Y_test_gpu.cpu().numpy()
    pred_test_wt = lockbox_pred_avg
    train_mean_full = true_cv_wt.mean(axis=0)

    print_benchmark_table(
        y_true=true_cv_wt,
        y_pred_model=pred_cv_wt,
        y_train_mean=train_mean_cv,
        title_tag=f"GPU PLS Ensembled OOF Benchmarks (n_components={best_n_comp}, {len(seeds)} Seeds) [{'Syn+Exp' if include_synthetic else 'Exp Only'}]",
        elem_names=target_names
    )

    print_benchmark_table(
        y_true=true_test_wt,
        y_pred_model=pred_test_wt,
        y_train_mean=train_mean_full,
        title_tag=f"GPU PLS Lockbox Benchmarks (Ensemble of {total_models} Models)",
        elem_names=target_names
    )

    # 5. Parity Plots with Unique Dataset & Timestamp Tagging
    oof_parity_path = output_dir / f"gpu_pls_oof_parity_{dataset_tag}_{TIMESTAMP}.png"
    lockbox_parity_path = output_dir / f"gpu_pls_lockbox_parity_{dataset_tag}_{TIMESTAMP}.png"

    plot_parity_grid(true_cv_wt, pred_cv_wt, target_names, title_tag=f"PLS ENSEMBLED OOF ({len(seeds)} Seeds)", save_path=oof_parity_path)
    plot_parity_grid(true_test_wt, pred_test_wt, target_names, title_tag=f"PLS LOCKBOX ENSEMBLE ({total_models} Models)", save_path=lockbox_parity_path)

    print(f"GPU Pipeline completed in {(time.perf_counter() - t0_start):.2f} seconds.")


if __name__ == "__main__":
    # main(
    #     h5_file="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/recleaned/exp-syn_TD14_TEN.h5",
    #     split_dir="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/fold_splits/exp_TD14_TEN",
    #     output_dir=OUTPUT_DIR,
    #     include_synthetic=False,
    #     seeds=SEEDS,
    # )
    
    # main(
    #     h5_file="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/recleaned/exp-syn_TDN_TEN.h5",
    #     split_dir="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/fold_splits/exp_TDN_TEN",
    #     output_dir=OUTPUT_DIR,
    #     include_synthetic=False,
    #     seeds=SEEDS,
    # )
    
    # main(
    #     h5_file="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/recleaned/exp-syn_TD14_TEN.h5",
    #     split_dir="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/fold_splits/exp-syn_TD14_TEN",
    #     output_dir=OUTPUT_DIR,
    #     include_synthetic=True,
    #     seeds=SEEDS,
    # )
    
    # main(
    #     h5_file="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/recleaned/exp-syn_TDN_TEN.h5",
    #     split_dir="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/fold_splits/exp-syn_TDN_TEN",
    #     output_dir=OUTPUT_DIR,
    #     include_synthetic=True,
    #     seeds=SEEDS,
    # )
    
    main(
        h5_file="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/recleaned/processed-data_20261106.h5",
        split_dir="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/fold_splits/processed-data_20261106",
        output_dir=Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/PLS_checks/plots/pls_benchmark/processed-data_20261106/exp"),
        include_synthetic=False,
        seeds=SEEDS,
    )
    
    main(
        h5_file="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/recleaned/exp-syn_processed-data_20261106.h5",
        split_dir="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/fold_splits/processed-data_20261106",
        output_dir=Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/PLS_checks/plots/pls_benchmark/processed-data_20261106/exp-syn"),
        include_synthetic=True,
        seeds=SEEDS,
    )