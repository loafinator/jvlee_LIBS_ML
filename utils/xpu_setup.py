"""
jvlee_LIBS_ML > utils > xpu_setup.py

Function definitions setting up the trainer for models on the GPU. Note that this
is for an intel GPU, not NVIDIA so it is called an xpu instead of gpu in pytorch.

"""

print('xpu_setup.py loading...')

# region Imports
# region plain
import torch
import joblib
import time 
# endregion 

# region as
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# endregion

# region from
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional, Tuple, Dict, Any 
from datetime import datetime
from pathlib import Path
# endregion

# region custom
try:
    from utils import log, get_worker_logger
except ImportError:
    try:
        from .debug import log, get_worker_logger
    except ImportError:
        from debug import log, get_worker_logger
# endregion
# endregion

xpu = torch.device('xpu')

def init_cnn(module):
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)

def init_xpu_trainer(
        # region Arguments
        model: nn.Module,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.DataFrame,
        X_val: Optional[np.ndarray | pd.DataFrame] = None,
        y_val: Optional[np.ndarray | pd.DataFrame] = None,
        X_scaler: Optional[Any] = None,
        y_scaler: Optional[Any] = None,
        max_epochs: int = 10,
        device: torch.device = torch.device('xpu'),
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = False,
        learning_rate: float = 0.001,
        criterion: nn.Module = nn.MSELoss(),
        clip_grads: bool = True,
        optimizer_cls = optim.Adam,
        weight_decay: float = 1e-4,
        verbose: bool = True,
        plot_animation: bool = True,
        save_path: Optional[str] = None,
        log_path: Path | None = None,
        # endregion
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Trains a 1D CNN (or any nn.Module) on XPU with a simple loop.

    Args:
        model: Your nn.Module (e.g. Simple1DCNN)
        X_train, y_train: Training features and targets
        X_val, y_val: Optional validation data
        max_epochs: Number of epochs
        device: 'xpu' by default
        batch_size, shuffle, learning_rate: Training hyperparameters
        criterion: Loss function (MSELoss for regression by default)
        optimizer_cls: Optimizer class (Adam by default)
        verbose: Print progress or not

    Returns:
        (trained_model, history_dict)
        history_dict contains: train_losses, val_losses (if val provided)
    """
    
    if log_path is None:
        log_path = Path(r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\default_log.txt").resolve()
    logger = get_worker_logger(Path(log_path).stem)

    # region Setup
    # # region Convert inputs to numpy if pandas
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.to_numpy(dtype=np.float32)
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_numpy(dtype=np.float32)
    if X_val is not None and isinstance(X_val, pd.DataFrame):
        X_val = X_val.to_numpy(dtype=np.float32)
    if y_val is not None and isinstance(y_val, pd.Series):
        y_val = y_val.to_numpy(dtype=np.float32)

    # Guarantee numpy arrays for Pylance
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    if X_val is not None:
        X_val = np.asarray(X_val, dtype=np.float32)
    if y_val is not None:
        y_val = np.asarray(y_val, dtype=np.float32)
    # endregion

    # region Make sure y is 2D for regression (n_samples, 1)
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    if y_val is not None and y_val.ndim == 1:
        y_val = y_val.reshape(-1, 1)
    # endregion

    # region Create datasets
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = None
    if X_val is not None and y_val is not None:
        val_dataset = TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    # endregion

    # region Move model and optimizer to device
    model = model.to(device)
    optimizer = optimizer_cls(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # endregion

    # region Save directories resolution
    time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if save_path is None:
        save_dir = Path(r"G:\My Drive\RLSL\Python\LIBS")
    else:
        save_dir = Path(save_path).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / f'best_model_{time_stamp}.pt'
    fig_save_path = save_dir / f"training_history_{time_stamp}.png"
    # endregion

    # region Scheduler / Early Stopping
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 30
    # endregion

    # region Track losses
    history = {
        'train_losses': [],
        'val_losses': [] if val_loader else None,
        'val_mae': [] if val_loader else None,
        'val_residuals_mean' : [] if val_loader else None,
        'val_residuals_std' : [] if val_loader else None,
    }
    # endregion

    # region Print training size details
    if verbose:
        log(logger=logger, msg= f"Training on {device} | Epochs: {max_epochs} | Batch size: {batch_size}")
        log(logger=logger, msg= f"Train samples: {len(train_dataset)}")
    # endregion
    # endregion

    # region Train and Validate
    # region Epoch Loop
    epoch_counter = 0
    for epoch in range(max_epochs):
        epoch_counter += 1
        start_time = time.perf_counter()
        model.train()
        train_loss = 0.0
        n_batches = 0

        # region Training Loop
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            if clip_grads:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        # endregion

        # region Training Losses
        avg_train_loss = train_loss / n_batches
        history['train_losses'].append(avg_train_loss)
        # endregion

        # region Validation Loader
        val_loss = None
        val_mae = None
        if val_loader:
            model.eval()
            val_loss_sum = 0.0
            val_mae_sum = 0.0
            n_val = 0
            all_residuals = []

            # region Validation
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    outputs = model(batch_x)
                    val_loss_sum += criterion(outputs, batch_y).item()
                    val_mae_sum += mae(outputs, batch_y)
                    residual = (outputs - batch_y).cpu().numpy()
                    all_residuals.append(residual)
                    n_val += 1
            # endregion

            # region Validation Losses
            val_loss = val_loss_sum / n_val
            val_mae = val_mae_sum / n_val * 100
            history['val_losses'].append(val_loss)
            history['val_mae'].append(val_mae)
            # endregion
            
            # region Residuals calcs
            all_residuals = np.concatenate(all_residuals, axis=0)
            per_wl_mean = all_residuals.mean(axis=0)
            per_wl_std = all_residuals.std(axis=0)
            history['val_residuals_mean'].append(float(per_wl_mean.mean()))
            history['val_residuals_std'].append(float(per_wl_std.mean()))
            # endregion

            # region Schedule / Save / Early stop
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                joblib.dump(X_scaler, save_dir / f'X_scaler_{time_stamp}.pk1')
                joblib.dump(y_scaler, save_dir / f'y_scaler_{time_stamp}.pk1')
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    log(logger=logger, msg= f'Early stopping at epoch {epoch + 1}')
                    break
            # endregion
        # endregion

        # region Epoch Report
        end_time = time.perf_counter()
        msg = f'Epoch {epoch + 1}/{max_epochs} | Train Loss: {avg_train_loss:.6f}'
        if val_loss is not None:
            msg += (f' | Val Loss: {val_loss:.6f} | Val MAE: {val_mae:.4f}'
                    f' | Res μ: {history["val_residuals_mean"][-1]:+.4f}'
                    f' σ: {history["val_residuals_std"][-1]:.4f}')
        msg += f' | {(end_time - start_time)/60:.3f} min'
        log(logger=logger, msg=msg)
        # endregion
    # endregion

    # region Plotting loss
    if plot_animation:
        fig, ax = plt.subplots(figsize=(8, 5))
        epochs_range = range(1, len(history['train_losses']) + 1)
        ax.plot(epochs_range, history['train_losses'], 'b-o', label='Train Loss')
        if history.get('val_losses'):
            ax.plot(epochs_range, history['val_losses'], 'r-o', label='Val Loss')
        
        # region Plot Details
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training History')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        # endregion

        # region Saving the Plot Figure
        plt.savefig(fig_save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Training plot saved to {fig_save_path}")
        # endregion
        
        # region Force open the saved image with Windows default viewer
        import os
        os.startfile(os.path.abspath(fig_save_path))
        # endregion
    # endregion
    # endregion

    return model, history

def mae(outputs, labels):
    return torch.mean(torch.abs(outputs - labels)).item()

def mse(outputs, labels):
    return torch.mean((outputs - labels)**2).item()
