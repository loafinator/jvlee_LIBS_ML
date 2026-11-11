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
H5_FILE = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/exp-syn_processed-data_20261110b.h5"
SPLIT_DIR = Path('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/fold_splits/processed-data_20261110b')
OUTPUT_DIR = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/PLS_checks/plots/pls_reverse_benchmark")

NUM_FOLDS = 7
SEEDS = [42]    # for non pls up seed count to like 7: [42,43,44,45,46,47,48]
THRESH = 2.8
MAX_WT_PERCENT = 12.0
SEED = 42
HOST_ELEMENTS = {'Li', 'K', 'Cl'}
EXP_DOPANTS = {'Ce', 'Gd', 'Sm', 'U'}

# Candidate range for PLS components
# Component number capped by number of target input elements (e.g. 4 dopants)
N_COMPONENTS_GRID = [1,2,3,4]

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

def plot_spectral_reconstruction(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    wavelengths: np.ndarray,
    y_concentrations: np.ndarray,
    elem_names: list[str],
    n_samples: int = 4,
    save_path: Path | None = None
):
    """Plots True vs Predicted spectra across wavelength grid for sample spectra."""
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 2.5 * n_samples), sharex=True)
    axes_flat = np.atleast_1d(axes)

    # Pick representative samples across dataset
    sample_indices = np.linspace(0, len(x_true) - 1, n_samples, dtype=int)

    for plot_idx, sample_idx in enumerate(sample_indices):
        ax = axes_flat[plot_idx]
        xt = x_true[sample_idx]
        xp = x_pred[sample_idx]
        conc_str = ", ".join([f"{name}: {y_concentrations[sample_idx, j]:.2f}%" 
                              for j, name in enumerate(elem_names) if y_concentrations[sample_idx, j] > 0])

        r2 = r2_score(xt, xp)
        rmse = np.sqrt(mean_squared_error(xt, xp))

        ax.plot(wavelengths, xt, label="True Spectrum", color="black", alpha=0.7, lw=1.0)
        ax.plot(wavelengths, xp, label="PLS Reverse Pred", color="crimson", alpha=0.8, lw=0.9, linestyle="--")
        ax.set_title(f"Sample {sample_idx} ({conc_str}) | $R^2$: {r2:.4f} | RMSE: {rmse:.5f}", fontsize=9, fontweight="bold")
        ax.set_ylabel("Norm Intensity", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)
        if plot_idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes_flat[-1].set_xlabel("Wavelength (nm)", fontsize=9)
    plt.suptitle("Reverse PLS Spectral Reconstruction ($Y \\rightarrow X$)", fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved Reverse Spectral Overlay Plot to: {save_path}")
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
        dataset_tag: str = '20261110b'
):
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"=== Starting GPU Reverse PLS Pipeline (Concentrations -> Spectra) [{DEVICE}] ===")
    t0_start = time.perf_counter()

    h5_file = str(h5_file)
    split_dir = Path(split_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    data = load_crossval_data(h5_file)
    
    seed_splits = {}
    for seed in seeds:
        test_idx, cv_idx, cv_fold_id = build_splits_for_seed(data, split_dir, NUM_FOLDS, seed=seed)
        seed_splits[seed] = (test_idx, cv_idx, cv_fold_id)

    test_idx = seed_splits[seeds[0]][0]
    cv_idx = seed_splits[seeds[0]][1]

    # Move arrays to GPU
    X_cv_gpu = data['X'][cv_idx].to(DEVICE)          # Target: Spectra (N x P)
    Y_cv_gpu = data['y'][cv_idx].to(DEVICE)          # Input: Concentrations (N x M)
    X_test_gpu = data['X'][test_idx].to(DEVICE)
    Y_test_gpu = data['y'][test_idx].to(DEVICE)

    # Select active dopant columns
    dopant_indices = [i for i, name in enumerate(data['names']) if name in EXP_DOPANTS or name in {'Ce', 'Gd', 'Sm', 'U'}]
    if dopant_indices:
        Y_cv_gpu = Y_cv_gpu[:, dopant_indices]
        Y_test_gpu = Y_test_gpu[:, dopant_indices]
        target_names = [data['names'][i] for i in dopant_indices]
    else:
        target_names = data['names']

    # ---- one-time setup (after the dopant column selection) ----
    P = X_cv_gpu.shape[1]
    fold_ref = seed_splits[seeds[0]][2]
    exp_rows = np.where(fold_ref >= 0)[0]               # positions within cv_idx
    n_exp = len(exp_rows)
    pos_in_exp = np.full(len(cv_idx), -1, dtype=np.int64)
    pos_in_exp[exp_rows] = np.arange(n_exp)

    exp_rows_gpu = torch.from_numpy(exp_rows).to(DEVICE)
    X_true_exp = X_cv_gpu[exp_rows_gpu]                 # (n_exp, P), stays on GPU
    Y_exp = Y_cv_gpu[exp_rows_gpu]

    x_mean = X_true_exp.mean(dtype=torch.float64).item()
    sst = ((X_true_exp - x_mean) ** 2).sum(dtype=torch.float64)   # constant across grid
    n_elem = X_true_exp.numel()

    is_synth_cv = data['is_synth'][cv_idx]

    # free host copy of spectra now that they're on GPU
    data['X'] = None
    
    print("\n--- Tuning Reverse PLS Components via Multi-Seed GPU Cross-Validation ---")
    best_oof_gpu = None
    best_overall_rmse = float('inf')
    n_comp_grid = [c for c in N_COMPONENTS_GRID if c <= Y_cv_gpu.shape[1]]
    best_n_comp = n_comp_grid[0]

    for n_comp in n_comp_grid:
        t_grid_start = time.perf_counter()
        oof_sum = torch.zeros((n_exp, P), device=DEVICE)

        for seed in seeds:
            cv_fold_id = seed_splits[seed][2]
            for k in range(NUM_FOLDS):
                if include_synthetic:
                    train_np = cv_fold_id != k
                else:
                    train_np = (cv_fold_id != k) & (is_synth_cv == 0)
                val_np = cv_fold_id == k

                train_mask = torch.from_numpy(train_np).to(DEVICE)
                val_dst = torch.from_numpy(pos_in_exp[val_np]).to(DEVICE)   # rows in exp-only buffer
                val_mask = torch.from_numpy(val_np).to(DEVICE)

                pls = PyTorchPLS(n_components=n_comp)
                pls.fit(Y_cv_gpu[train_mask].clone(), X_cv_gpu[train_mask])
                oof_sum[val_dst] += pls.predict(Y_cv_gpu[val_mask])
                del pls

        oof = torch.clamp(oof_sum / len(seeds), min=0.0)
        del oof_sum

        sse = ((X_true_exp - oof) ** 2).sum(dtype=torch.float64)
        mean_rmse = torch.sqrt(sse / n_elem).item()
        mean_r2 = (1.0 - sse / sst).item()
        dt = time.perf_counter() - t_grid_start
        print(f"n_components = {n_comp:2d} | Spectral OOF RMSE: {mean_rmse:7.5f} | Overall R^2: {mean_r2:7.4f} | Elapsed: {dt:.2f}s")

        if best_oof_gpu is None or mean_rmse < best_overall_rmse:
            best_overall_rmse, best_n_comp = mean_rmse, n_comp
            best_oof_gpu = oof          # keep on GPU, no host copy
        else:
            del oof

    # 3. Lockbox Reverse Prediction Ensemble
    total_models = len(seeds) * NUM_FOLDS
    print(f"\n--- Running Lockbox Reverse Evaluation on GPU ({total_models} models, n_components={best_n_comp}) ---")
    lockbox_sum_gpu = torch.zeros((len(test_idx), X_test_gpu.shape[1]), device=DEVICE)

    for seed in seeds:
        _, _, cv_fold_id = seed_splits[seed]
        for k in range(NUM_FOLDS):
            if include_synthetic:
                train_mask = torch.from_numpy(cv_fold_id != k).to(DEVICE)
            else:
                train_mask = torch.from_numpy((cv_fold_id != k) & (is_synth_cv == 0)).to(DEVICE)

            X_tr_gpu = X_cv_gpu[train_mask]
            Y_tr_gpu = Y_cv_gpu[train_mask]

            pls = PyTorchPLS(n_components=best_n_comp)
            pls.fit(Y_tr_gpu.clone(), X_tr_gpu)

            lockbox_sum_gpu += pls.predict(Y_test_gpu)

            del X_tr_gpu, Y_tr_gpu, pls
            torch.cuda.empty_cache()

    lockbox_pred_avg = torch.clamp(lockbox_sum_gpu / total_models, min=0.0).cpu().numpy()

    # Calculate & print lockbox metrics to terminal
    x_test_true = X_test_gpu.cpu().numpy()
    lockbox_rmse = np.sqrt(mean_squared_error(x_test_true, lockbox_pred_avg))
    lockbox_r2 = r2_score(x_test_true.ravel(), lockbox_pred_avg.ravel())
    print(f"\n✅ Lockbox Reverse PLS Performance | Spectral RMSE: {lockbox_rmse:.5f} | Overall R^2: {lockbox_r2:.4f}")

    # 4. Save Reverse Spectral Overlay Plots
    oof_plot_path = output_dir / f"reverse_pls_oof_spectra_{dataset_tag}_{TIMESTAMP}.png"
    lockbox_plot_path = output_dir / f"reverse_pls_lockbox_spectra_{dataset_tag}_{TIMESTAMP}.png"

    plot_spectral_reconstruction(
        x_true=X_true_exp.cpu().numpy(),
        x_pred=best_oof_gpu.cpu().numpy(), # type: ignore
        wavelengths=data['wavelengths'],
        y_concentrations=Y_exp.cpu().numpy(),
        elem_names=target_names,
        save_path=oof_plot_path,
    )

    plot_spectral_reconstruction(
        x_true=x_test_true,
        x_pred=lockbox_pred_avg,
        wavelengths=data['wavelengths'],
        y_concentrations=Y_test_gpu.cpu().numpy(),
        elem_names=target_names,
        save_path=lockbox_plot_path,
    )

    print(f"Reverse PLS Pipeline completed in {(time.perf_counter() - t0_start):.2f} seconds.")

    del X_cv_gpu, Y_cv_gpu, X_test_gpu, Y_test_gpu, X_true_exp, best_oof_gpu
    torch.cuda.empty_cache()


if __name__ == "__main__":
    """
    Did you run the final combo in NIST_data.py to get the synthetic data set already?
    """
    
    main(
        h5_file="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/exp-syn_processed-data_20261110b.h5",
        split_dir="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/fold_splits/processed-data_20261110b",
        output_dir=Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/PLS_checks/plots/pls_reverse_benchmark/processed-data_20261110b/exp"),
        include_synthetic=False,
        seeds=SEEDS,
        dataset_tag='20261110b'
    )
    
    main(
        h5_file="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/exp-syn_processed-data_20261110b.h5",
        split_dir="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/fold_splits/processed-data_20261110b",
        output_dir=Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/PLS_checks/plots/pls_reverse_benchmark/processed-data_20261110b/exp-syn"),
        include_synthetic=True,
        seeds=SEEDS,
        dataset_tag='20261110b'
    )