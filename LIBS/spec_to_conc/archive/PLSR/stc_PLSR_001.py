"""
jvlee_LIBS_ML > LIBS > spec_to_conc > PLSR > stc_PLSR_001.py

NOTE: update all paths before running from new location(machine).

Full definition and implimentation of stc_CNN_1D_001().

"""


# region Imports
# region imports
import time
import torch
import h5py
import logging
import sys
import multiprocessing
# endregion

# region as
import torch.nn as nn
import pandas as pd
import numpy as np
# endregion

# region froms
from datetime import datetime
from torch import optim
from pathlib import Path
from h5py import Group
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error
from logging.handlers import QueueListener
# endregion

# region custom
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from setup import add_project_root_to_path
add_project_root_to_path(parent_generation=1)
from utils import (
    Logger, 
    gen_speak,
    get_worker_logger,
    log,
    init_xpu_trainer,
    plot_residual_history,
    plot_peak_error_heatmap,
    plot_predicted_vs_actual,
    run_spectral_inference,
    hf_get,
    load_h5_split_dataset
)

# endregion
# endregion

# region Paths
logger_path = r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\LIBS\spec_to_conc\PLSR\stc_PLSR_001_log.txt"
h5_path = r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\LIBS\trn_val_split_LIBS.h5"
eval_dir = Path(r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\LIBS\spec_to_conc\PLSR\eval_PLSR_001")
# endregion

if __name__ == "__main__":
    # region Timing Setup
    start_time = time.perf_counter()
    start_time_a = time.perf_counter()
    # endregion

    # region Logging Setup
    eval_dir.mkdir(parents=True, exist_ok=True)

    my_logger = Logger(logger_path)
    my_logger.setFormatter(logging.Formatter('%(asctime)s  |  %(message)s'))

    sys.stdout = my_logger

    log_q = multiprocessing.Queue()
    listener = QueueListener(log_q, my_logger, respect_handler_level=True)
    listener.start()
    # endregion

    ALLOWED_TARGET_COLS = {
        'frac_LiCl', 'frac_KCl',
        'conc_Ce_wt%', 'conc_CeCl3_wt%', 'conc_CeN_wt%',
        'conc_Ca_wt%', 'conc_CaCl3_wt%',
        'conc_U_wt%', 'conc_UCl3_wt%',
        'conc_Sm_wt%', 'conc_SmCl3_wt%',
        'conc_Gd_wt%', 'conc_GdCl3_wt%',
        'conc_La_wt%', 'conc_LaCl3_wt%',
        'conc_Mg_wt%', 'conc_MgCl2_wt%',
        'conc_H2o_wt%', 'conc_Nd_wt%',
        # 'temperature_C',
    }

    try:  
        logger = get_worker_logger(Path(logger_path).stem)

        # region Load H5 Data file
        # X was concentration and y was spectra. Need to switch it
        # return X_trn_raw, X_val_raw, y_trn_raw, y_val_raw, feature_cols, wavelengths, meta_val_df
        y_trn_raw, y_val_raw, X_trn_raw, X_val_raw, target_cols, wavelengths, meta_val_df = load_h5_split_dataset(
            h5_path=h5_path,
            allowed_cols=ALLOWED_TARGET_COLS,
        )

        assert X_trn_raw.shape[0] == y_trn_raw.shape[0], \
            f'X/y row mismatch: {X_trn_raw.shape[0]} vs {y_trn_raw.shape[0]}'

        log(logger=logger, msg=f'Target columns ({len(target_cols)}): {target_cols}')
        log(logger=logger, msg=f'wavelengths: {wavelengths}')
        log(logger=logger, msg=f'first shot intensities: {X_trn_raw[0]}')
        # endregion

        #   X = SPECTRUM = FEATURES = WAVELENGTH INTENSITIES
        #   y = CONCENTRATION = TARGETS

        # region Pre-processing
        # region Drop Columns
        # drop_columns = ['Unnamed: 0', 'technique', 'file_id', 'file_path', 'scan_rate_mVs']
        selector = VarianceThreshold(threshold=1e-10)
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        # endregion

        # region Filter out rows
        # Find the fraction of NaN values as wavelength per row
        nan_frac_trn = np.isnan(X_trn_raw).mean(axis=1)
        nan_frac_val = np.isnan(X_val_raw).mean(axis=1)

        # Designate any row with more than __% as bad rows to be dropped
        bad_rows_trn = nan_frac_trn > 0.1
        bad_rows_val = nan_frac_val > 0.1
        log(logger=logger, msg= f'Dropping {bad_rows_trn.sum()} training rows and {bad_rows_val.sum()} validation rows with >10% NaN values in Spectrum')
    
        # Drop designated y rows
        y_trn_raw = y_trn_raw[~bad_rows_trn]
        y_val_raw = y_val_raw[~bad_rows_val]
        # endregion

        # region Clip bright data
        # Clip the extra bright shots so they don't skew the mean and standard deviation
        clip_threshold = np.nanpercentile(X_trn_raw, 99.5)
        log(logger=logger, msg= f'Clipping spectra intensities above: {clip_threshold:.1f}')
        X_trn_raw = np.clip(X_trn_raw, 0, clip_threshold)
        X_val_raw = np.clip(X_val_raw, 0, clip_threshold)   # val uses the same threshold as trn
        # endregion 

        # region Keep physical copy
            # keep a copy of the clipped but not scaled for the predicted/actual plot
        y_val_physical = y_val_raw.copy()
        # endregion

        # region Filter the y data
        selector.fit(y_trn_raw)
        surviving_cols = [c for c, keep in zip(target_cols, selector.get_support()) if keep]
        y_trn_filtered = pd.DataFrame(
            np.array(selector.transform(y_trn_raw)),
            columns=surviving_cols)
        y_val_filtered = pd.DataFrame(
            np.array(selector.transform(y_val_raw)),
            columns=surviving_cols)
        # Filter the metadata df so that index alignment is preserved
        meta_val_df = meta_val_df[[c for c in surviving_cols if c in meta_val_df.columns]]
        # endregion
        
        # region Scale the X data
        X_trn_raw = X_trn_raw[~bad_rows_trn]
        X_val_raw = X_val_raw[~bad_rows_val]
        X_trn_scaled = X_scaler.fit_transform(X_trn_raw)
        X_val_scaled = X_scaler.transform(X_val_raw)
        X_trn_scaled = np.nan_to_num(X_scaler.fit_transform(X_trn_raw), nan=0.0, posinf=0.0, neginf=0.0)
        X_val_scaled = np.nan_to_num(X_scaler.transform(X_val_raw), nan=0.0, posinf=0.0, neginf=0.0)
        # endregion

        # region Expand X data
        X_trn = np.expand_dims(X_trn_scaled, axis=1).astype(np.float32)
        X_val = np.expand_dims(X_val_scaled, axis=1).astype(np.float32)
        # endregion

        # region Scale y data
        y_trn = np.nan_to_num(y_scaler.fit_transform(y_trn_filtered), nan=0.0).astype(np.float32)
        y_val = np.nan_to_num(y_scaler.transform(y_val_filtered), nan=0.0).astype(np.float32)
        # endregion

        # region Align metadata df to surviving bad-row mask
        meta_val_df = meta_val_df.iloc[np.where(~bad_rows_val)[0]].reset_index(drop=True)
        # endregion

        # region Report
        log(logger=logger, msg= f"Columns surviving VarianceThreshold: {y_trn_filtered.columns.tolist()}")

        log(logger=logger, msg= f"NaNs in X_trn: {np.isnan(X_trn).sum()}")
        log(logger=logger, msg= f"NaNs in y_trn: {np.isnan(y_trn).sum()}")
        log(logger=logger, msg= f"NaNs in X_val: {np.isnan(X_val).sum()}")
        log(logger=logger, msg= f"NaNs in y_val: {np.isnan(y_val).sum()}")
        log(logger=logger, msg= f"y_trn min/max: {y_trn.min():.4f} / {y_trn.max():.4f}")
        log(logger=logger, msg= f"X_trn min/max: {X_trn.min():.4f} / {X_trn.max():.4f}")

        n_features = X_trn.shape[2]
        n_targets = y_trn.shape[1]
        log(logger=logger, msg= f'Number of features is:          {n_features}')
        log(logger=logger, msg= f'Training features shape:        {X_trn.shape}')
        log(logger=logger, msg= f'Validation features shape:      {X_val.shape}')
        log(logger=logger, msg= f'Training targets shape:         {y_trn.shape}')
        log(logger=logger, msg= f'Validation targets shape:       {y_val.shape}')

        row_maxes = X_trn_raw.max(axis=1)
        outlier_mask = row_maxes > np.percentile(row_maxes, 99)
        log(logger=logger, msg= f'Top 10 max intensities per spectrum: {np.sort(row_maxes)[-10:]}')
        log(logger=logger, msg= f'Spectra with extreme max values: {outlier_mask.sum()} / {len(y_trn_raw)}')
        # endregion
        # endregion

        # region PLSR
        pls = PLSRegression(n_components=100)
        pls.fit(X_trn_scaled, y_trn)

        y_pred_val = pls.predict(X_val_scaled)

        val_mse = np.mean((y_pred_val - y_val) ** 2)
        y_pred_physical = y_scaler.inverse_transform(y_pred_val)
        y_val_physical_filtered = y_scaler.inverse_transform(y_val)
        val_mae = mean_absolute_error(y_val_physical_filtered, y_pred_physical)
        log(logger=logger, msg=f'PLS Val MSE: {val_mse:.6f} | Val MAE: {val_mae:.6f}')
        per_target_mae = np.abs(y_pred_physical - y_val_physical_filtered).mean(axis=0)
        for col, mae_val in zip(surviving_cols, per_target_mae):
            log(logger=logger, msg=f'  {col:30s} | MAE: {mae_val:.4f}')
        # endregion


        end_time = time.perf_counter()
        log(logger=logger, msg= f'Total training time: {((end_time - start_time) / 60):.3f} minutes')
    finally:
        gen_speak('Training complete!')
        listener.stop()
        sys.stdout = sys.__stdout__
        my_logger.close()
