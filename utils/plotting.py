"""
jvlee_LIBS_ML>utils>plotting.py

Function definitions for plotting prediction evaluation plots.

"""

print('plotting.py loading...')

# region Imports
# region as
import matplotlib.pyplot as plt
import numpy as np
# endregion

# region from
from typing import Optional, Tuple, Dict, Any
from matplotlib.animation import FuncAnimation
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

    return ani, update  # ← return both!

def plot_residual_history(
        history: Dict[str, Any],
        save_path: Optional[str] = None,
        show: bool = True
) -> None:
    """
    Plot per-epoch mean residual (ppred - true, scaled space) +/- 1 std.
    A well trained model should hover near 0 with shrinking std.
    """
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
        wavelengths:    np.ndarray,        # (n_wavelengths,)
        y_actual_raw:   np.ndarray,        # (N, n_wavelengths) — already in physical units (post-clip, pre-scale)
        y_predicted:    np.ndarray,        # (N, n_wavelengths) — already inverse-transformed to physical units
        sample_indices: list,              # which rows to plot
        sample_labels:  Optional[list] = None,   # e.g. ["5% Ce", "10% Ce", ...]
        save_path:      Optional[str]  = None,
        show:           bool = True,
        ncols:          int  = 2,
) -> None:
    """
    For each sample index, overlay predicted vs actual spectrum.
    Panels are arranged in a grid (ncols wide).
 
    Args:
        wavelengths:    1-D array of wavelength values (nm or whatever unit)
        y_actual_raw:   Physical-unit spectra — use y_trn_raw / y_val_raw
                        (the clipped-but-not-scaled arrays), aligned with
                        whichever X array you ran inference on.
        y_predicted:    inverse_transform output from run_spectral_inference()
        sample_indices: row indices to plot
        sample_labels:  human-readable label per index (composition, etc.)
        ncols:          number of columns in the subplot grid
    """
    n      = len(sample_indices)
    nrows  = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle('Predicted vs Actual Spectra (physical units)', fontsize=13, y=1.01)
 
    for plot_idx, sample_idx in enumerate(sample_indices):
        row, col = divmod(plot_idx, ncols)
        ax       = axes[row][col]
 
        actual    = y_actual_raw[sample_idx]
        predicted = y_predicted[plot_idx]          # run_spectral_inference returns in sample_indices order
 
        label = sample_labels[plot_idx] if sample_labels else f'Sample {sample_idx}'
 
        ax.plot(wavelengths, actual,    color='steelblue', linewidth=1.0,
                alpha=0.85, label='Actual')
        ax.plot(wavelengths, predicted, color='tomato',    linewidth=1.0,
                alpha=0.85, linestyle='--', label='Predicted')
 
        # residual on a twin axis so it doesn't distort the spectrum scale
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
 
    # hide any unused subplots
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
        wavelengths:  np.ndarray,         # (W,)
        y_actual_raw: np.ndarray,         # (N, W) physical units
        y_predicted:  np.ndarray,         # (N, W) physical units
        sample_labels: Optional[list] = None,
        save_path:    Optional[str]  = None,
        show:         bool = True,
) -> None:
    """
    Heatmap of (actual - predicted) / max(actual) across all plotted samples.
    Rows = samples, columns = wavelengths.  Red = over-prediction, blue = under.
    Useful for spotting if the model is systematically wrong at certain wavelengths.
    """
    n          = len(y_predicted)
    rel_errors = np.zeros((n, len(wavelengths)), dtype=np.float32)
    for i in range(n):
        denom            = np.maximum(y_actual_raw[i].max(), 1e-6)
        rel_errors[i]    = (y_actual_raw[i] - y_predicted[i]) / denom
 
    fig, ax = plt.subplots(figsize=(14, max(3, 0.5 * n)))
    im = ax.imshow(rel_errors, aspect='auto', cmap='RdBu_r',
                   vmin=-1, vmax=1,
                   extent=[wavelengths[0], wavelengths[-1], n - 0.5, -0.5])
 
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
