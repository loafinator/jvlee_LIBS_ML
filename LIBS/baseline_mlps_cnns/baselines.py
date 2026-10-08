from __future__ import annotations

"""
jvlee_LIBS_ML > LIBS > baseline_mlps_cnns > baselines.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from setup import add_project_root_to_path

add_project_root_to_path(parent_generation=1)

from LIBS.conc_to_spec.CNN_1D.LIBS_model_003 import LIBS_1D_CNN_003
from LIBS.conc_to_spec.MLP.cts_MLP_005 import LIBS_MLP_003
from LIBS.p3VAE.p3vae_005 import (
    BATCH_SIZE,
    LIBSSpectraDataset,
    create_test_and_cv_folds,
    load_crossval_data,
)
from utils import(
    get_worker_logger,
    log,
)


def train_baseline_kfold(
    h5_path: str,
    log_path: str,
    model_type: str = "mlp",
    num_folds: int = 5,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cuda",
):
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    
    log_path = Path(log_path or f"/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/logs/p3vae_logs/kfold_{timestamp}.txt").resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = get_worker_logger(log_path.stem)

    # Load dataset once
    data = load_crossval_data(h5_path)

    # Use exact same K-fold + lockbox split as P3VAE
    test_idx, cv_idx, cv_fold_id = create_test_and_cv_folds(
        data=data, num_folds=num_folds, min_groups_per_dopant=1, seed=42
    )

    # Save test indices for lockbox evaluation
    np.save(f"baseline_{model_type}_test_idx.npy", test_idx)

    # Slice cross-validation subset safely
    n_samples = len(data["X"])
    cv_data = {
        k: (v[cv_idx] if isinstance(v, np.ndarray) and v.shape[0] == n_samples else v)
        for k, v in data.items()
    }

    # Extract dimensions correctly
    input_dim = cv_data["y"].shape[-1]   # 10,000 spectral channels
    output_dim = cv_data["X"].shape[-1]  # 19 element target features

    fold_results = []

    for fold in range(num_folds):
        print(f"\n--- Starting Fold {fold + 1}/{num_folds} ({model_type.upper()}) ---")
        train_mask = cv_fold_id != fold
        val_mask = cv_fold_id == fold

        train_ds = LIBSSpectraDataset(
            cv_data["y"], cv_data["X"], cv_data["is_synth"], np.where(train_mask)[0]
        )
        val_ds = LIBSSpectraDataset(
            cv_data["y"], cv_data["X"], cv_data["is_synth"], np.where(val_mask)[0]
        )

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        # Instantiate fresh model for each fold
        if model_type == "mlp":
            model = LIBS_MLP_003(
                n_features=input_dim,
                hidden_dims=(512, 1024, 512),
                n_wavelengths=output_dim,
            ).to(device_obj)
        else:
            model = LIBS_1D_CNN_003(
                input_channels=1, n_features=input_dim, n_wavelengths=output_dim
            ).to(device_obj)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            # Training phase
            model.train()
            for batch in train_loader:
                spectra = batch["spectrum"].to(device_obj)
                targets = batch["targets"].to(device_obj)

                if model_type == "cnn" and spectra.ndim == 2:
                    spectra = spectra.unsqueeze(1)  # [B, 1, 10000] for Conv1d

                optimizer.zero_grad()
                preds = model(spectra)
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()

            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    spectra = batch["spectrum"].to(device_obj)
                    targets = batch["targets"].to(device_obj)

                    if model_type == "cnn" and spectra.ndim == 2:
                        spectra = spectra.unsqueeze(1)

                    preds = model(spectra)
                    val_loss += criterion(preds, targets).item() * spectra.size(0)

            val_loss /= len(val_ds)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"best_{model_type}_fold{fold}.pt")

            if epoch % 10 == 0 or epoch == epochs:
                print(f"Epoch [{epoch}/{epochs}] | Val MSE: {val_loss:.6f} | Best Val MSE: {best_val_loss:.6f}")

        print(f"[{model_type.upper()}] Fold {fold} Best Val MSE: {best_val_loss:.6f}")
        fold_results.append(best_val_loss)

    print(
        f"\n{model_type.upper()} Mean CV MSE: {np.mean(fold_results):.6f} +/- {np.std(fold_results):.6f}"
    )

h5_path = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/cts_noleak_crossval.h5"
NUM_FOLDS = 5
Num_EPOCHS = 100
LEARNING_RATE = 1e-03
DEVICE = 'cuda'

train_baseline_kfold(
    h5_path=h5_path,
    model_type='mlp',
    num_folds=NUM_FOLDS,
    epochs=Num_EPOCHS,
    lr=LEARNING_RATE,
    device=DEVICE
)