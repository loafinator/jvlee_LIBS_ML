from __future__ import annotations

import sys
import time
import h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from pathlib import Path
from datetime import datetime
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Add project root setup matching your directory layout
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from setup import add_project_root_to_path
    add_project_root_to_path(parent_generation=1)
    from utils import get_h5_ds
except ImportError:
    pass

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

# Candidate range for PLS components
N_COMPONENTS_GRID = [2, 5, 10, 15, 20, 25, 30, 40, 50]

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Helper & Data Functions (Matching P3VAE Pipeline)
# =============================================================================
def load_crossval_data(h5_path: str, spike_thresh: float = THRESH) -> dict:
    """Loads dataset and filters corrupted saturation shots (matching P3VAE)."""
    with h5py.File(h5_path, 'r') as hf:
        y = hf['y'][:].astype(np.float32)           # Spectral intensities [N, Wavelengths]    # type: ignore
        X = hf['X'][:].astype(np.float32)           # Target concentrations [N, Elements]    # type: ignore
        is_synth = hf['is_synthetic'][:]    # type: ignore
        is_lockbox = hf['is_lockbox'][:].astype(bool)    # type: ignore
        group_id = hf['group_id'][:]    # type: ignore
        wavelengths = hf['wavelengths'][:]    # type: ignore
        names = [
            n.decode('utf-8') if isinstance(n, bytes) else str(n) 
            for n in hf['feature_cols'][:]    # type: ignore
        ]

    row_maxes = y.max(axis=1)    # type: ignore
    clean_mask = row_maxes <= spike_thresh
    
    n_dropped = len(y) - np.sum(clean_mask)    # type: ignore
    if n_dropped > 0:
        print(f"Dropped {n_dropped} corrupted saturation shots (max intensity > {spike_thresh}).")

    return dict(
        X=X[clean_mask],    # type: ignore
        y=y[clean_mask],    # type: ignore
        is_synth=is_synth[clean_mask],    # type: ignore
        is_lockbox=is_lockbox[clean_mask],    # type: ignore
        group_id=group_id[clean_mask],    # type: ignore
        wavelengths=wavelengths,
        names=names,
    )


def assign_group_folds(X, group_id, is_synthetic, dopant_idx, elem_names=None,
                       n_splits=5, n_trials=2000, min_groups_per_dopant=1, seed=42):
    """Stratified group fold assignment matching original p3vae script."""
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

    best_fog, best_score = None, None
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

    fold_id = np.full(len(X), -1, dtype=np.int32)
    fold_id[exp] = best_fog[inv]    # type: ignore
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

    X_cv = data['X'][cv_idx]
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


# =============================================================================
# Plotting & Diagnostics
# =============================================================================
def plot_parity_grid(y_true_wt, y_pred_wt, target_names, title_tag="PLS Benchmark", save_path=None):
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

    plt.suptitle(f"{title_tag} - Concentration Parity Grid", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved Parity Plot to: {save_path}")
    plt.close()


# =============================================================================
# Main Execution Pipeline
# =============================================================================
def _evaluate_single_fold(n_comp, k, cv_fold_id, X_cv_spec, Y_cv_conc):
    """Helper function to run PLS on a single fold across workers."""
    train_mask = (cv_fold_id != k)
    val_mask = (cv_fold_id == k)

    X_tr, Y_tr = X_cv_spec[train_mask], Y_cv_conc[train_mask]
    X_va = X_cv_spec[val_mask]

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)

    pls = PLSRegression(n_components=n_comp, scale=False)
    pls.fit(X_tr_scaled, Y_tr)

    val_pred = pls.predict(X_va_scaled)
    return n_comp, k, val_mask, val_pred


def main():
    print("=== Starting Parallel PLS Benchmark Pipeline ===")
    t0_start = time.perf_counter()

    # Set worker pool size from SLURM task allocation
    N_JOBS = 2

    # 1. Load Data
    data = load_crossval_data(H5_FILE)
    split_path = SPLIT_DIR / f'max12_{NUM_FOLDS}_folds_seed{SEED}.npz'
    test_idx, cv_idx, cv_fold_id = build_splits_once(data, split_path, NUM_FOLDS, seed=SEED)

    X_cv_spec = data['y'][cv_idx]
    Y_cv_conc = data['X'][cv_idx]
    X_test_spec = data['y'][test_idx]
    Y_test_conc = data['X'][test_idx]
    target_names = data['names']

    print(f"Dataset Loaded | CV Pool: {len(cv_idx)} samples | Lockbox Test Set: {len(test_idx)} samples")

    # 2. Parallel Hyperparameter Grid Search
    print(f"\n--- Tuning PLS Components via Parallel CV ({N_JOBS} Workers) ---")
    
    # Generate all (n_components, fold) tasks
    tasks = [(n_comp, k) for n_comp in N_COMPONENTS_GRID for k in range(NUM_FOLDS)]
    
    # Run all 63 tasks in parallel across 8 CPUs
    cv_results = Parallel(n_jobs=N_JOBS)(
        delayed(_evaluate_single_fold)(n_comp, k, cv_fold_id, X_cv_spec, Y_cv_conc)
        for n_comp, k in tasks
    )

    # Reconstruct OOF predictions per n_component option
    oof_by_ncomp = {n_comp: np.zeros_like(Y_cv_conc) for n_comp in N_COMPONENTS_GRID}
    for n_comp, k, val_mask, val_pred in cv_results:    # type: ignore
        oof_by_ncomp[n_comp][val_mask] = val_pred

    # Identify optimal n_components
    best_n_comp = N_COMPONENTS_GRID[0]
    best_overall_r2 = -float('inf')

    for n_comp in N_COMPONENTS_GRID:
        oof_clipped = np.clip(oof_by_ncomp[n_comp], a_min=0.0, a_max=None)
        mean_r2 = r2_calcs(Y_cv_conc * MAX_WT_PERCENT, oof_clipped * MAX_WT_PERCENT)
        print(f"n_components = {n_comp:2d} | OOF Mean R^2: {mean_r2:7.4f}")

        if mean_r2 > best_overall_r2:
            best_overall_r2 = mean_r2
            best_n_comp = n_comp

    print(f"\nOptimal PLS Components: {best_n_comp} (Mean OOF R^2: {best_overall_r2:.4f})")

    # 3. Final Evaluation & Lockbox Ensembling
    print("\n--- Running Final Lockbox Evaluation ---")
    oof_preds = np.clip(oof_by_ncomp[best_n_comp], a_min=0.0, a_max=None)
    lockbox_ensemble_preds = np.zeros((NUM_FOLDS, len(test_idx), Y_test_conc.shape[1]))

    # Parallelize the 7 fold models for the lockbox prediction
    def _train_and_predict_lockbox(k):
        train_mask = (cv_fold_id != k)
        X_tr, Y_tr = X_cv_spec[train_mask], Y_cv_conc[train_mask]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_test_scaled = scaler.transform(X_test_spec)

        pls = PLSRegression(n_components=best_n_comp, scale=False)
        pls.fit(X_tr_scaled, Y_tr)
        return pls.predict(X_test_scaled)

    lockbox_preds_list = Parallel(n_jobs=N_JOBS)(
        delayed(_train_and_predict_lockbox)(k) for k in range(NUM_FOLDS)
    )

    for k, pred in enumerate(lockbox_preds_list):
        lockbox_ensemble_preds[k] = pred

    lockbox_pred_avg = np.clip(lockbox_ensemble_preds.mean(axis=0), a_min=0.0, a_max=None)

    # 4. Convert back to physical wt% and report metrics
    true_cv_wt = Y_cv_conc * MAX_WT_PERCENT
    pred_cv_wt = oof_preds * MAX_WT_PERCENT

    true_test_wt = Y_test_conc * MAX_WT_PERCENT
    pred_test_wt = lockbox_pred_avg * MAX_WT_PERCENT

    active_elem_idx = [j for j in range(true_test_wt.shape[1]) if true_test_wt[:, j].max() > 1e-5]

    print("\n" + "=" * 80)
    print(f" PLS Benchmark - Out-Of-Fold (OOF) Performance (n_components={best_n_comp})")
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
    print(f" PLS Benchmark - Lockbox Test Set Performance (Ensemble of {NUM_FOLDS} Folds)")
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

    # 5. Generate Parity Plots
    oof_parity_path = OUTPUT_DIR / f"pls_oof_parity_{TIMESTAMP}.png"
    lockbox_parity_path = OUTPUT_DIR / f"pls_lockbox_parity_{TIMESTAMP}.png"

    plot_parity_grid(true_cv_wt, pred_cv_wt, target_names, 
                     title_tag=f"PLS OOF (n_comp={best_n_comp})", save_path=oof_parity_path)
    plot_parity_grid(true_test_wt, pred_test_wt, target_names, 
                     title_tag=f"PLS Lockbox Ensemble (n_comp={best_n_comp})", save_path=lockbox_parity_path)

    print(f"Pipeline completed in {(time.perf_counter() - t0_start) / 60:.2f} minutes.")

if __name__ == "__main__":
    main()