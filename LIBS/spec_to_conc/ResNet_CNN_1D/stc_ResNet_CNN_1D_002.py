"""
jvlee_LIBS_ML > LIBS > spec_to_conc > ResNet_CNN_1D > stc_ResNet_CNN_1D_002.py

NOTE: update all paths before running from new location(machine).

Full definition and implimentation of stc_CNN_1D_001().

TODO: Try out AdamW instead of Adam for the optimizer. The weight decay implementation
        on regular Adam is not right it sounds like and the AdamW fixes it.

"""


# region Imports
# region imports
import time
import torch
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
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from logging.handlers import QueueListener
# endregion

# endregion

# region Paths
WRK_DIR = Path(__file__).parent.parent.parent.parent.resolve()
print(f"Working directory is {WRK_DIR}")
logger_path = WRK_DIR.joinpath(r"LIBS\spec_to_conc\ResNet_CNN_1D\stc_ResNet_CNN_1D_001_log.txt")
h5_path = WRK_DIR.joinpath(r"LIBS\trn_val_split_LIBS.h5")
eval_dir = WRK_DIR.joinpath(r"LIBS\spec_to_conc\ResNet_CNN_1D\eval_ResNet_CNN_1D_001")
# endregion

class ResBlock1D(nn.Module):
    """ 
    Singe residual block for 1D spectral data. 
    """
    def __init__(
            self,
            channels: int,
            kernel_size: int = 3,
            dropout: float = 0.2,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))         # This is the skip connection

class STC_ResNet_1D_CNN_001(nn.Module):
    def __init__(
            self, 
            input_channels: int = 1, 
            base_channels: int = 64,
            n_res_blocks: int = 4,
            adaptive_pool_out: int = 64,
            fc_hidden: int = 256,
            kernel_size: int = 7,
            dropout: float = 0.3,
            n_targets: int = 7):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, kernel_size=kernel_size, padding=(kernel_size//2)),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.res_blocks = nn.Sequential(
            *[ResBlock1D(base_channels, kernel_size=kernel_size)#, dropout=dropout)
              for _ in range(n_res_blocks)]
        )

        self.downsample = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            nn.AdaptiveAvgPool1d(adaptive_pool_out),
            nn.Flatten(),
        )

        fc_in = base_channels * adaptive_pool_out
        self.fc = nn.Sequential(
            nn.Linear(fc_in, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, (fc_hidden // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear((fc_hidden // 2), n_targets),
        )

    def forward(self, x):
        x1 = self.stem(x)
        x2 = self.res_blocks(x1)
        x3 = self.downsample(x2)
        return self.fc(x3)

if __name__ == "__main__":
    # region custom imports
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from setup import add_project_root_to_path
    add_project_root_to_path(parent_generation=1)
    from utils import (
        Logger, 
        gen_speak,
        get_worker_logger,
        log,
        init_xpu_trainer,
        init_gpu_trainer,
        plot_residual_history,
        # plot_peak_error_heatmap,
        # plot_predicted_vs_actual,
        # run_spectral_inference,
        # hf_get,
        load_h5_split_dataset
    )

    # endregion

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
    
        # Drop designated rows
        X_trn_raw = X_trn_raw[~bad_rows_trn]
        X_val_raw = X_val_raw[~bad_rows_val]
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
        X_trn_scaled = np.nan_to_num(X_scaler.fit_transform(X_trn_raw), nan=0.0, posinf=0.0, neginf=0.0)
        X_val_scaled = np.nan_to_num(X_scaler.transform(X_val_raw), nan=0.0, posinf=0.0, neginf=0.0)
        # endregion

        # region Downsample
        downsample_factor = 4
        X_trn_scaled = X_trn_scaled[:, ::downsample_factor]
        X_val_scaled = X_val_scaled[:, ::downsample_factor]
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
        meta_val_df = meta_val_df.reset_index(drop=True)
        # endregion

        # region Report
        n_features = X_trn.shape[2]
        n_targets = y_trn.shape[1]
        row_maxes = X_trn_raw.max(axis=1)
        outlier_mask = row_maxes > np.percentile(row_maxes, 99)
        log(logger=logger, msg= f"Columns surviving VarianceThreshold: {y_trn_filtered.columns.tolist()}")
        log(logger=logger, msg= f"NaNs in X_trn: {np.isnan(X_trn).sum()}")
        log(logger=logger, msg= f"NaNs in y_trn: {np.isnan(y_trn).sum()}")
        log(logger=logger, msg= f"NaNs in X_val: {np.isnan(X_val).sum()}")
        log(logger=logger, msg= f"NaNs in y_val: {np.isnan(y_val).sum()}")
        log(logger=logger, msg= f"y_trn min/max: {y_trn.min():.4f} / {y_trn.max():.4f}")
        log(logger=logger, msg= f"X_trn min/max: {X_trn.min():.4f} / {X_trn.max():.4f}")
        log(logger=logger, msg= f'Number of features is:          {n_features}')
        log(logger=logger, msg= f'Training features shape:        {X_trn.shape}')
        log(logger=logger, msg= f'Validation features shape:      {X_val.shape}')
        log(logger=logger, msg= f'Training targets shape:         {y_trn.shape}')
        log(logger=logger, msg= f'Validation targets shape:       {y_val.shape}')
        log(logger=logger, msg= f'Top 10 max intensities per spectrum: {np.sort(row_maxes)[-10:]}')
        log(logger=logger, msg= f'Spectra with extreme max values: {outlier_mask.sum()} / {len(y_trn_raw)}')
        # endregion
        # endregion

        # region Training
        model = STC_ResNet_1D_CNN_001(
            input_channels=1,
            base_channels=64,
            n_res_blocks=4,
            adaptive_pool_out=64,
            fc_hidden=256,
            kernel_size=7,
            dropout = 0.2,
            n_targets=n_targets
        )
        end_time_a = time.perf_counter()
        log(logger=logger, msg= f'Training setup time: {(end_time_a - start_time_a):.3f} seconds')

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(logger=logger, msg=f'Model parameters: {n_params:,}')

        log(logger=logger, msg='Start transfer data to xpu')
        time_t1 = time.perf_counter()
        X_trn_t = torch.from_numpy(X_trn).to('xpu')
        y_trn_t = torch.from_numpy(y_trn).to('xpu')
        X_val_t = torch.from_numpy(X_val).to('xpu')
        y_val_t = torch.from_numpy(y_val).to('xpu')
        time_t2 = time.perf_counter()
        log(logger=logger, msg=f'Data transfer to xpu took {(time_t2 - time_t1)} sec')


        start_time_b = time.perf_counter()
        training_model, history = init_xpu_trainer(
            model=model,
            X_train=X_trn,
            y_train=y_trn,
            X_val=X_val,
            y_val=y_val,
            X_scaler= None,
            y_scaler= None,
            max_epochs=5,
            device= torch.device('xpu'),
            batch_size=256,
            shuffle= True,
            num_workers= 0,
            persistent_workers=False,
            pin_memory= False,
            learning_rate= 3e-5,
            criterion= nn.MSELoss(),
            clip_grads= True,
            optimizer_cls= optim.AdamW,  # TODO: try AdamW
            weight_decay= 1e-4,
            verbose= True,
            plot_animation= True,
            save_path= str(eval_dir),
            log_path= Path(logger_path),
            ReduceLROnPlateau=True,
        )

        end_time_b = time.perf_counter()
        log(logger=logger, msg= f'Training time: {((end_time_b - start_time_b)/60):.3f} minutes')
        # endregion

        # region Residual Plot
        time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log(logger=logger, msg='Starting post-training spectral evaluation...')
        plot_residual_history(
            history=history,
            save_path=str(eval_dir / f'residual_history_{time_stamp}.png'),
            show=False
        )
        # endregion 


        end_time = time.perf_counter()
        log(logger=logger, msg= f'Total training time: {((end_time - start_time) / 60):.3f} minutes')
    finally:
        gen_speak('Training complete!')
        listener.stop()
        sys.stdout = sys.__stdout__
        my_logger.close()


# region Imporvements
    # - Update xpu_setup.py to log instead of print
# endregion


        # region Sample Plots
        # region plot prep
        # composition_col = next(
        #     (c for c in [
        #         'conc_CaCl3_wt%', 
        #         'conc_CeCl3_wt%', 
        #         'conc_UCl3_wt%', 
        #         'conc_SmCl3_wt%',
        #         'conc_GdCl3_wt%',
        #         'conc_LaCl3_wt%',
        #         'conc_MgCl2_wt%',
        #         ] if c in meta_val_df.columns),
        #     None
        # )
        # if composition_col is None:
        #     raise ValueError(f'Could not find any concentration columns in : {list(meta_val_df.columns)}')
        # log(logger=logger, msg=f'Composition-range sampling column: {composition_col}')

        # target_percentiles = [5, 15, 30, 50, 70, 85, 95]
        # conc_vals = meta_val_df[composition_col].to_numpy(dtype=np.float32)
        # eval_indices = []
        # for pct in target_percentiles:
        #     target_val = np.percentile(conc_vals, pct)
        #     closest = int(np.argmin(np.abs(conc_vals - target_val)))
        #     if closest not in eval_indices:
        #         eval_indices.append(closest)

        #     eval_labels = [
        #         f'{composition_col} = {conc_vals[i]:.4f} (p{p})'
        #         for i, p in zip(eval_indices, target_percentiles[:len(eval_indices)])
        #     ]

        #     log(logger=logger, msg = f'Evaluation sample indices: {eval_indices}')
        #     log(logger=logger, msg = f'Evaluation sample labels: {eval_labels}')
        #     # endregion

        #     # region Prediction
        #     X_eval = X_val[eval_indices]
        #     y_eval_true = y_val_physical[eval_indices]
        #     y_eval_pred = run_spectral_inference(
        #         model = training_model,
        #         X_samples= X_eval,
        #         y_scaler=y_scaler,
        #         device=torch.device('xpu')
        #     )
        #     # endregion

        #     # region Plot prediction
        #     plot_predicted_vs_actual(
        #         wavelengths=np.array(surviving_cols),
        #         y_actual_raw=y_eval_true,
        #         y_predicted=y_eval_pred,
        #         sample_indices=list(range(len(eval_indices))),
        #         sample_labels=eval_labels,
        #         save_path=str(eval_dir / f'spectral_overlay_{time_stamp}.png'),
        #         show=False,
        #         ncols=2,
        #     )
        #     # endregion

        #     # region Plot error heatmap
        #     plot_peak_error_heatmap(
        #         wavelengths=np.array(surviving_cols),
        #         y_actual_raw=y_eval_true,
        #         y_predicted=y_eval_pred,
        #         sample_labels=eval_labels,
        #         save_path=str(eval_dir / f'error_heatmap_{time_stamp}.png'),
        #         show=False,
        #     )
        #     # endregion

        #     log(logger=logger, msg=f'Post-training evaluation plots saved to: {eval_dir}')

        #     # region Report
        #     for i, label in enumerate(eval_labels):
        #         actual    = y_eval_true[i]
        #         predicted = y_eval_pred[i]
        #         mae_val   = np.mean(np.abs(predicted - actual))
        #         peak_err  = np.abs(predicted - actual).max()
        #         peak_col   = surviving_cols[np.abs(predicted - actual).argmax()]
        #         log(logger=logger, msg=(
        #             f'  {label:50s} | MAE={mae_val:8.2f} '
        #             f'| MaxErr={peak_err:8.2f} @ {peak_col}'
        #         ))
            # endregion
        # endregion