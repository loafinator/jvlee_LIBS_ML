"""
Python > LIBS > LIBS_MLP_002.py

"""
# region Imports
from setup import add_project_root_to_path
add_project_root_to_path(parent_generation=1)
try:
    from utils import (
        Logger, 
        gen_speak,
        get_worker_logger,
        log,
        init_xpu_mlp_trainer
    )
except ImportError:
    try:
        from ...utils import Logger, get_worker_logger, log, init_xpu_mlp_trainer
    except ImportError:
        try:
            # from ..utils.data_prep import enrich_with_progress, combine_and_save_as_HDF5, trn_val_splitter_HDF5, load_h5_dataset
            from ...utils.debug import Logger, get_worker_logger, log
            from ...utils.speak import gen_speak
            from ...utils.xpu_setup import init_xpu_mlp_trainer
        except ImportError as e:
            print(f'Utils import error: {e}')
import pandas as pd
import numpy as np
import time 
from torch import optim
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import torch
import torch.nn as nn
import h5py
import logging
from logging.handlers import QueueListener
import sys
import multiprocessing
from pathlib import Path
# endregion


class LIBS_MLP_002(nn.Module):
    def __init__(
            self, 
            n_features: int = 21,
            hidden_dims: tuple = (512, 1024, 2048, 1024, 512),
            dropout: float = 0.3,
            n_wavelengths: int = 451
    ):
        super().__init__()

        layers = []
        in_dim = n_features

        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            in_dim = h_dim

        self.hidden = nn.Sequential(*layers)

        self.output = nn.Linear(in_dim, n_wavelengths)

        self.skip = nn.Linear(n_features, n_wavelengths)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.output(self.hidden(x)) + self.skip(x)
    
ALLOWED_FEATURE_COLS = {
    'frac_LiCl', 'frac_KCl',
    'conc_Ce_wt%', 'conc_CeCl3_wt%', 'conc_CeN_wt%',
    'conc_Ca_wt%', 'conc_CaCl3_wt%',
    'conc_U_wt%', 'conc_UCl3_wt%',
    'conc_Sm_wt%', 'conc_SmCl3_wt%',
    'conc_Gd_wt%', 'conc_GdCl3_wt%',
    'conc_La_wt%', 'conc_LaCl3_wt%',
    'conc_Mg_wt%', 'conc_MgCl2_wt%',
    'conc_H2o_wt%', 'conc_Nd_wt%',
    'temperature_C', 'state_aerosol', 'state_molten', 'state_solid',
    'delay_study', 'delay', 
    'width_study', 'width', 
    'energy_study', 'energy',
    'qdelay_study', 'qdelay',
    'shot_study', 'shots',
    'flow_study', 'flow',
    'pressure_study', 'pressure', 
    'repetition', 'blank', 'kinetic', 'static_',
}

if __name__ == "__main__":
    # region Timing Setup
    start_time = time.perf_counter()
    start_time_a = time.perf_counter()
    # endregion

    # region Logging Setup
    logger_path = r"G:\My Drive\RLSL\Python\LIBS\LIBS_MLP_002_log.txt"

    my_logger = Logger(logger_path)
    my_logger.setFormatter(logging.Formatter('%(asctime)s  |  %(message)s'))

    sys.stdout = my_logger

    log_q = multiprocessing.Queue()
    listener = QueueListener(log_q, my_logger, respect_handler_level=True)
    listener.start()
    # endregion

    try:  
        logger = get_worker_logger(Path(logger_path).stem)

        # region Load H5 Data file
        with h5py.File(r"G:\My Drive\RLSL\Data\combined_CSVs\trn_val_split_LIBS.h5", 'r') as hf:
            y_trn_raw = hf['train/spectra'][:]
            y_val_raw   = hf['val/spectra'][:]
            wavelengths = hf['wavelengths'][:]

            all_cols = list(hf['train/metadata'].keys())
            feature_cols = [c for c in all_cols
                            if c in ALLOWED_FEATURE_COLS]
            
            X_trn_raw = np.stack([hf[f'train/metadata/{c}'][:] for c in feature_cols], axis=1).astype(np.float32)
            X_val_raw = np.stack([hf[f'val/metadata/{c}'][:] for c in feature_cols], axis=1).astype(np.float32)
            
            log(logger=logger, msg= f'Feature columns ({len(feature_cols)}): {feature_cols}')

        assert X_trn_raw.shape[0] == y_trn_raw.shape[0], \
            f'X/y row mismatch: {X_trn_raw.shape[0]} vs {y_trn_raw.shape[0]}'
        # endregion

        # region Drop Columns
        # drop_columns = ['Unnamed: 0', 'technique', 'file_id', 'file_path', 'scan_rate_mVs']
        selector = VarianceThreshold(threshold=1e-10)
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()

        # Find the fraction of NaN values as wavelength per row
        nan_frac_trn = np.isnan(y_trn_raw).mean(axis=1)
        nan_frac_val = np.isnan(y_val_raw).mean(axis=1)

        # Designate any row with more than __% as bad rows to be dropped
        bad_rows_trn = nan_frac_trn > 0.1
        bad_rows_val = nan_frac_val > 0.1
        log(logger=logger, msg= f'Dropping {bad_rows_trn.sum()} training rows and {bad_rows_val.sum()} validation rows with >10% NaN values in Spectrum')

        # Drop designated rows
        y_trn_raw = y_trn_raw[~bad_rows_trn]
        y_val_raw = y_val_raw[~bad_rows_val]

        row_maxes = y_trn_raw.max(axis=1)
        outlier_mask = row_maxes > np.percentile(row_maxes, 99)

        # Clip the extra bright shots so they don't skew the mean and standard deviation
        clip_threshold = np.nanpercentile(y_trn_raw, 99.5)
        log(logger=logger, msg= f'Clipping spectra intensities above: {clip_threshold:.1f}')
        y_trn_raw = np.clip(y_trn_raw, 0, clip_threshold)
        y_val_raw = np.clip(y_val_raw, 0, clip_threshold)   # val uses the same threshold as trn

        # Filter the X data
        selector.fit(X_trn_raw)
        surviving_cols = [c for c, keep in zip(feature_cols, selector.get_support()) if keep]
        X_trn_filtered = pd.DataFrame(
            selector.transform(X_trn_raw),
            columns=surviving_cols)
        X_val_filtered = pd.DataFrame(
            selector.transform(X_val_raw),
            columns=surviving_cols)
        
        # Scale the X data
        X_trn_scaled = X_scaler.fit_transform(X_trn_filtered)
        X_trn_scaled = np.nan_to_num(X_trn_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_scaled = X_scaler.transform(X_val_filtered)
        X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Match the X with y by applying the same bad rows mask
        X_trn_scaled_and_cleaned = X_trn_scaled[~bad_rows_trn]
        X_val_scaled_and_cleaned = X_val_scaled[~bad_rows_val]

        # Expand X data
        X_trn = X_trn_scaled_and_cleaned.astype(np.float32)
        X_val = X_val_scaled_and_cleaned.astype(np.float32)

        # Scale y data
        y_trn = np.nan_to_num(y_scaler.fit_transform(y_trn_raw), nan=0.0).astype(np.float32)
        y_val = np.nan_to_num(y_scaler.transform(y_val_raw), nan=0.0).astype(np.float32)
        # endregion
        
        # region 
        log(logger=logger, msg= f"Columns surviving VarianceThreshold: {X_trn_filtered.columns.tolist()}")

        log(logger=logger, msg= f"NaNs in trn_X: {np.isnan(X_trn).sum()}")
        log(logger=logger, msg= f"NaNs in trn_y: {np.isnan(y_trn).sum()}")
        log(logger=logger, msg= f"NaNs in val_X: {np.isnan(X_val).sum()}")
        log(logger=logger, msg= f"NaNs in val_y: {np.isnan(y_val).sum()}")
        log(logger=logger, msg= f"trn_y min/max: {y_trn.min():.4f} / {y_trn.max():.4f}")
        log(logger=logger, msg= f"trn_X min/max: {X_trn.min():.4f} / {X_trn.max():.4f}")

        n_features = X_trn.shape[1]
        log(logger=logger, msg= f'number of features is:          {n_features}')
        log(logger=logger, msg= f'Training features shape:        {X_trn.shape}')
        log(logger=logger, msg= f'Validation features shape:      {X_val.shape}')
        log(logger=logger, msg= f'Training targets shape:         {y_trn.shape}')
        log(logger=logger, msg= f'Validation targets shape:       {y_val.shape}')

        row_maxes_after_clip = y_trn_raw.max(axis=1)
        log(logger=logger, msg= f'Top 10 max intensities per spectrum before clip: {np.sort(row_maxes)[-10:]}')
        log(logger=logger, msg= f'Top 10 max intensities per spectrum after clip: {np.sort(row_maxes_after_clip)[-10:]}')
        log(logger=logger, msg= f'Spectra with extreme max values: {outlier_mask.sum()} / {len(y_trn_raw)}')

        model = LIBS_MLP_002(
            n_features=n_features,
            hidden_dims= (512, 1024, 1024, 512),
            dropout = 0.5,
            n_wavelengths=len(wavelengths)
        )
        end_time_a = time.perf_counter()
        log(logger=logger, msg= f'Training setup time: {(end_time_a - start_time_a)} seconds')


        start_time_b = time.perf_counter()
        training_model, history = init_xpu_mlp_trainer(
            model=model,
            X_train=X_trn,
            y_train=y_trn,
            X_val=X_val,
            y_val=y_val,
            max_epochs=400,
            batch_size=256,
            device= torch.device('xpu'),
            shuffle= True,
            num_workers= 0,
            pin_memory= False,
            learning_rate= 1e-3,
            criterion= nn.MSELoss(),
            clip_grads= True,
            optimizer_cls= optim.Adam,
            weight_decay=1e-4,
            verbose= True,
            plot_animation= True,
            save_path= r"G:\My Drive\RLSL\Python\LIBS\MLP_002_trn_Hist"
        )

        end_time_b = time.perf_counter()
        log(logger=logger, msg= f'Training time: {(end_time_b - start_time_b)/60} minutes')

        end_time = time.perf_counter()
        log(logger=logger, msg= f'Total training time: {(end_time - start_time) / 60} minutes')
    finally:
        gen_speak('Training complete!')
        listener.stop()
        sys.stdout = sys.__stdout__
        my_logger.close()


# region Imporvement ideas
    # - normalize each spectrum to itself, rather than globally, this could help by eliminating
    #   any variation in spectrum intensity from the varience in energy level of the laser.
    # - Potentially try Savitzky-Golay smoothing instead of normal noise removal.
    # - If I am able to figure out the baseline LIBS signal of the surrounding environment for
    #   the samples then I would be able to subtract that from the signal.
    # - Continuum normalization makes composition driven features more prominent by getting rid
    #   of plasma temperature variation.
    # - ResNet-style CNN?
    #       - lets gradients pass thru to deeper levels without vanishing, can go deeper without
    #         loosing training stability.
    # - Gaussion Process Regression
    #       - includes uncertainty estimations alongside predictions.
    # - Random Forest / Gradient Boosting (XGBoost,LightGBM)
    #       - Good on small datasets, consider as a benchmark?
    # **Partial Least Squares Regression**
    #       - chemometrics workhorse for LIBS. Extremely well-validated in spectroscopy literature.
    #       - strongly recommended.
    # - Decrease dataset size
    # - replace Dropout with L2 weight decay. 
    #       - penalizes large weights in optimizer instead of just cutting them out.
    #       - often more effective for regression tasks.
    # - Systematic learning rate search.
    #       - log scale grid search for best learning rate
    # - Bayesian hyperparameter optimization (Optuna, RayTune)
    #       - sample efficient tuning learning rate dropout, kernel sizes, filter counts
    # - **Simulated spectra augmentation** 
    #       - LIBS spectra can be simulated using NIST atomic line databases and Boltzmann/Saha
    #         equations. Augmenting training data with physics-based synthetic spectra for 
    #         compositions that we haven't measured. 
# endregion