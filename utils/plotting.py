"""
jvlee_LIBS_ML > utils > plotting.py

Function definitions for plotting prediction evaluation plots.
"""

print('plotting.py loading...')

# region Imports
import gc
import sys
import torch

# region as
import matplotlib.pyplot as plt
import numpy as np
# endregion

# region from
from typing import Optional, Tuple, Dict, Any
from matplotlib.animation import FuncAnimation
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from setup import add_project_root_to_path
add_project_root_to_path(parent_generation=1)

from utils import (
    get_worker_logger,
    log,
    logging,
    get_h5_ds,
    check_h5_compositions
)
# endregion
# endregion


def animate_training(history, max_epochs, plot_animation=True):
    if not plot_animation:
        return None, None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Progress')
    ax.set_xlim(0, max_epochs + 1)
    ax.set_ylim(0, max(history['train_losses'] or [1]) * 1.1)

    train_line, = ax.plot([], [], 'b-', label='Train Loss')
    val_line = None
    if history.get('val_losses'):
        val_line, = ax.plot([], [], 'r-', label='Val Loss')
    ax.legend()
    ax.grid(True)

    def update(frame):
        train_line.set_data(range(1, frame + 2), history['train_losses'][:frame + 1])
        if val_line:
            val_line.set_data(range(1, frame + 2), history['val_losses'][:frame + 1])
        ax.relim()
        ax.autoscale_view()
        return (train_line, val_line) if val_line else (train_line,)

    ani = FuncAnimation(
        fig=fig,
        func=update,
        frames=max_epochs,
        interval=500,
        repeat=False,
        blit=True,
        cache_frame_data=False
    )

    plt.show(block=False)  # non-blocking

    return ani, update


def evaluate_and_plot_kfold(
    h5_path: str | Path,
    master_table_path: str | Path,
    model_prefix: str | Path,
    model_cls: Any,
    scaler_cls: Any,
    dataset_cls: Any,
    load_data_fn: Any,
    num_folds: int = 5,
    batch_size: int = 128,
    output_dir: str | Path = "plots",
    log_path: str | Path | None = None,
    device: str = "cuda",
    model_tag: str = "20261015_133024",
    max_wt_percent: float = 12.0,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if log_path is None:
        log_path = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/logs/p3vae_logs/default.txt").resolve()
    logger = get_worker_logger(Path(log_path).stem)

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    log(logger=logger, msg=f"Running K-Fold Ensemble Evaluation on: {device_obj}")

    data = load_data_fn(str(h5_path))
    test_idx_path = Path(f"{model_prefix}_test_idx.npy")
    if not test_idx_path.exists():
        raise FileNotFoundError(f"Could not locate test holdout indices file at: {test_idx_path}")

    test_idx = np.load(test_idx_path)
    log(logger=logger, msg=f"Loaded lockbox test set: {len(test_idx)} samples")

    wl_grid = torch.from_numpy(data["wavelengths"]).to(device_obj)
    target_names = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in data["names"]]

    input_dim = data["y"].shape[-1]
    n_elements = data["X"].shape[-1]
    N_test = len(test_idx)

    avg_pred_conc = np.zeros((N_test, n_elements), dtype=np.float32)
    avg_recon_spec = np.zeros((N_test, input_dim), dtype=np.float32)
    avg_phys_spec = np.zeros((N_test, input_dim), dtype=np.float32)
    avg_res_spec = np.zeros((N_test, input_dim), dtype=np.float32)

    loaded_folds = 0

    for k in range(num_folds):
        ckpt_path = Path(f"{model_prefix}_fold{k}.pt")
        if not ckpt_path.exists():
            log(logger=logger, msg=f"⚠️ Warning: Checkpoint for fold {k} not found ({ckpt_path.name}). Skipping fold.")
            continue

        log(logger=logger, msg=f"Running inference for Fold {k} model...")
        checkpoint = torch.load(ckpt_path, map_location=device_obj)

        model = model_cls(
            input_dim=input_dim,
            n_elements=n_elements,
            wl_grid=wl_grid,
            master_table_path=master_table_path,
            n_broadening=1,
        ).to(device_obj)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.eval()

        scaler = scaler_cls()
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

        test_spectra_scaled = scaler.transform(data["y"][test_idx])
        test_ds = dataset_cls(
            test_spectra_scaled, 
            data["X"][test_idx], 
            data["is_synth"][test_idx], 
            np.arange(len(test_idx))
        )
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        fold_c, fold_r, fold_p, fold_res = [], [], [], []
        with torch.no_grad():
            for batch in test_loader:
                spectra = batch["spectrum"].to(device_obj)
                out = model(spectra)
                
                fold_c.append(out["concentrations"].cpu().numpy())
                fold_r.append(scaler.inverse_transform(out["reconstructed_spectrum"].cpu().numpy()))
                fold_p.append(scaler.inverse_transform(out["ideal_physics_spectrum"].cpu().numpy()))
                fold_res.append(scaler.inverse_transform(out["matrix_residual"].cpu().numpy()))

        avg_pred_conc += np.concatenate(fold_c, axis=0)
        avg_recon_spec += np.concatenate(fold_r, axis=0)
        avg_phys_spec += np.concatenate(fold_p, axis=0)
        avg_res_spec += np.concatenate(fold_res, axis=0)
        loaded_folds += 1

        del model, checkpoint, fold_c, fold_r, fold_p, fold_res, test_spectra_scaled, test_loader
        gc.collect()
        if device_obj.type == 'cuda':
            torch.cuda.empty_cache()

    if loaded_folds == 0:
        raise RuntimeError("No valid model checkpoints were found for evaluation!")

    avg_pred_conc /= loaded_folds
    avg_recon_spec /= loaded_folds
    avg_phys_spec /= loaded_folds
    avg_res_spec /= loaded_folds

    true_conc_wt = np.asarray(data["X"][test_idx]) * max_wt_percent
    pred_conc_wt = avg_pred_conc * max_wt_percent

    true_spectra = np.asarray(data["y"][test_idx])
    wl = data["wavelengths"]

    log(logger=logger, msg=f"Successfully ensembled predictions across {loaded_folds} fold models.")

    # =========================================================================
    # PLOT 1: Concentration Parity Grid (Pred vs True wt%)
    # =========================================================================
    active_elem_idx = [j for j in range(n_elements) if true_conc_wt[:, j].max() > 1e-5]
    n_active = len(active_elem_idx)
    cols = 4
    rows = int(np.ceil(n_active / cols)) if n_active > 0 else 1

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.8, rows * 3.4))
    axes_flat = np.atleast_1d(axes).flatten()

    for idx_plot, j in enumerate(active_elem_idx):
        ax = axes_flat[idx_plot]
        name = target_names[j]
        y_true = true_conc_wt[:, j]
        y_pred = pred_conc_wt[:, j]

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-8
        r2 = 1.0 - (ss_res / ss_tot)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        ax.scatter(y_true, y_pred, alpha=0.35, s=10, color="#1f77b4", edgecolors="none")
        max_v = max(np.max(y_true), np.max(y_pred))
        min_v = min(np.min(y_true), np.min(y_pred))
        ax.plot([min_v, max_v], [min_v, max_v], "k--", alpha=0.7, lw=1)
        ax.set_title(f"{name}\n$R^2$: {r2:.3f} | RMSE: {rmse:.3f} wt% | MAE: {mae:.3f} wt%", fontsize=9, fontweight="bold")
        ax.set_xlabel("True wt%", fontsize=8)
        ax.set_ylabel("Pred wt%", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)

    for idx_plot in range(n_active, len(axes_flat)):
        fig.delaxes(axes_flat[idx_plot])

    plt.tight_layout()
    parity_path = output_dir / f"{model_tag}_kfold_concentration_parity_grid.png"
    plt.savefig(parity_path, dpi=300)
    plt.close()
    log(logger=logger, msg=f"Saved Concentration Parity Plot to: {parity_path}")

    # =========================================================================
    # PLOT 2: Spectral Reconstruction & Physics Breakdown Overlay
    # =========================================================================
    sample_idx = np.random.randint(0, len(true_spectra))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(wl, true_spectra[sample_idx], label="Measured Spectrum", color="black", alpha=0.7, lw=1)
    ax1.plot(wl, avg_recon_spec[sample_idx], label="Ensemble Model Reconstruction", color="crimson", alpha=0.8, lw=1)
    ax1.set_ylabel("Intensity (a.u.)", fontsize=10)
    ax1.set_title(f"Lockbox Test Sample #{sample_idx} - Full Spectrum Reconstruction", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(wl, avg_phys_spec[sample_idx], label="Ensemble Physics Component (NIST Lines)", color="forestgreen", lw=1)
    ax2.plot(wl, avg_res_spec[sample_idx], label="Ensemble Matrix Residual Component", color="orange", lw=1)
    ax2.set_xlabel("Wavelength (nm)", fontsize=10)
    ax2.set_ylabel("Intensity (a.u.)", fontsize=10)
    ax2.set_title("P3VAE Decoded Physics vs. Residual Breakdown", fontsize=11)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    spec_path = output_dir / f"{model_tag}_kfold_sample_{sample_idx}_spectral_breakdown.png"
    plt.savefig(spec_path, dpi=300)
    plt.close()
    log(logger=logger, msg=f"Saved Spectral Breakdown Plot to: {spec_path}")

    # =========================================================================
    # STRATIFIED DIAGNOSTICS: Non-Zero vs. Zero Background Performance
    # =========================================================================
    print("\n====================================================================================================")
    print(f" Ensemble Test Set Performance Across {loaded_folds} Folds ({len(test_idx)} Shots)")
    print("====================================================================================================")
    eps = 0.001

    for j in active_elem_idx:
        name = target_names[j]
        y_true = true_conc_wt[:, j]
        y_pred = pred_conc_wt[:, j]

        mae_all = np.mean(np.abs(y_true - y_pred))
        rmse_all = np.sqrt(np.mean((y_true - y_pred) ** 2))
        r2_all = r2_score(y_true, y_pred)

        mask_nz = y_true > eps
        mask_z = y_true <= eps

        if np.sum(mask_nz) > 0:
            y_true_nz = y_true[mask_nz]
            y_pred_nz = y_pred[mask_nz]
            r2_nz = r2_score(y_true_nz, y_pred_nz)
            rmse_nz = np.sqrt(mean_squared_error(y_true_nz, y_pred_nz))
            mae_nz = np.mean(np.abs(y_true_nz - y_pred_nz))
            n_nz = np.sum(mask_nz)
        else:
            r2_nz, rmse_nz, mae_nz, n_nz = 0.0, 0.0, 0.0, 0

        if np.sum(mask_z) > 0:
            y_pred_z = y_pred[mask_z]
            tnr = np.mean(y_pred_z <= eps)
            fpr = np.mean(y_pred_z > eps)
        else:
            tnr, fpr = 1.0, 0.0

        print(
            f"Element {name:<5} | Overall R^2: {r2_all:7.3f} | "
            f"Non-Zero R^2: {r2_nz:7.3f} (RMSE: {rmse_nz:.4f}, n={n_nz}) | "
            f"Zero Specificity (TNR): {tnr * 100:5.1f}% | FPR: {fpr * 100:4.1f}%"
        )
    print("====================================================================================================\n")


def plot_residual_history(
        history: Dict[str, Any],
        save_path: Optional[str] = None,
        show: bool = True
) -> None:
    means = history.get('val_residuals_mean')
    stds = history.get('val_residuals_std')
    if not means:
        print("No residual history to plot.")
        return
    
    epochs = range(1, len(means) + 1)
    means = np.array(means)
    stds = np.array(stds)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, means, 'k-o', linewidth=1.5, label='Mean residual (pred − true)')
    ax.fill_between(epochs, means - stds, means + stds, alpha=0.25, color='steelblue', label='±1 std')
    ax.axhline(0, color='red', linewidth=1, linestyle='--', label='Zero bias')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Residual (scaled units)')
    ax.set_title('Per-Epoch Validation Residual — Mean ± 1 Std')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
 
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Residual history plot saved to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_predicted_vs_actual(
        wavelengths:    np.ndarray,
        y_actual_raw:   np.ndarray,
        y_predicted:    np.ndarray,
        sample_indices: list,
        sample_labels:  Optional[list] = None,
        save_path:      Optional[str]  = None,
        show:           bool = True,
        ncols:          int  = 2,
) -> None:
    n      = len(sample_indices)
    nrows  = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle('Predicted vs Actual Spectra (physical units)', fontsize=13, y=1.01)
 
    for plot_idx, sample_idx in enumerate(sample_indices):
        row, col = divmod(plot_idx, ncols)
        ax       = axes[row][col]
 
        actual    = y_actual_raw[sample_idx]
        predicted = y_predicted[plot_idx]
 
        label = sample_labels[plot_idx] if sample_labels else f'Sample {sample_idx}'
 
        ax.plot(wavelengths, actual,    color='steelblue', linewidth=1.0,
                alpha=0.85, label='Actual')
        ax.plot(wavelengths, predicted, color='tomato',    linewidth=1.0,
                alpha=0.85, linestyle='--', label='Predicted')
 
        ax_res = ax.twinx()
        ax_res.fill_between(wavelengths, predicted - actual, 0,
                            alpha=0.15, color='purple', label='Residual')
        ax_res.set_ylabel('Residual', color='purple', fontsize=8)
        ax_res.tick_params(axis='y', labelcolor='purple', labelsize=7)
        ax_res.axhline(0, color='purple', linewidth=0.5, linestyle=':')
 
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('Wavelength')
        ax.set_ylabel('Intensity')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
 
    for empty_idx in range(n, nrows * ncols):
        row, col = divmod(empty_idx, ncols)
        axes[row][col].set_visible(False)
 
    plt.tight_layout()
 
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Spectral overlay plot saved to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_peak_error_heatmap(
        wavelengths:  np.ndarray,
        y_actual_raw: np.ndarray,
        y_predicted:  np.ndarray,
        sample_labels: Optional[list] = None,
        save_path:    Optional[str]  = None,
        show:         bool = True,
) -> None:
    n          = len(y_predicted)
    rel_errors = np.zeros((n, len(wavelengths)), dtype=np.float32)
    for i in range(n):
        denom            = np.maximum(y_actual_raw[i].max(), 1e-6)
        rel_errors[i]    = (y_actual_raw[i] - y_predicted[i]) / denom
 
    fig, ax = plt.subplots(figsize=(14, max(3, 0.5 * n)))
    im = ax.imshow(rel_errors, aspect='auto', cmap='RdBu_r',
                   vmin=-1, vmax=1,
                   extent=(float(wavelengths[0]), float(wavelengths[-1]), float(n - 0.5), float(-0.5)))
 
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('(actual - pred) / max(actual)')
 
    ax.set_xlabel('Wavelength')
    ax.set_ylabel('Sample')
    ax.set_title('Relative Prediction Error Heatmap')
 
    if sample_labels:
        ax.set_yticks(range(n))
        ax.set_yticklabels(sample_labels, fontsize=8)
 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
    if show:
        plt.show()
    plt.close(fig)