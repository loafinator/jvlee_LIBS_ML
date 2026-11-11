"""
jvlee_LIBS_ML > LIBS > spec_to_conc > ResNet_CNN_1D > stc_ResNet_CNN_1D_002.py

NOTE: update all paths before running from new location(machine).

Full definition and implimentation of stc_CNN_1D_001().

TODO: Try out AdamW instead of Adam for the optimizer. The weight decay implementation
        on regular Adam is not right it sounds like and the AdamW fixes it.

"""

# print('hello?')
# region Imports
# region imports
import time
import torch
import logging
import sys
import multiprocessing
# endregion
# print('check 1')
# region as
import torch.nn as nn
import pandas as pd
import numpy as np
# endregion
# print('check 2')

# region froms
from datetime import datetime
from torch import optim
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from logging.handlers import QueueListener
# endregion
# print('check 3')

# endregion

# region Paths
# print('check 4')
WRK_DIR = Path(__file__).parent.parent.parent.parent.resolve()
print(f"Working directory is {WRK_DIR}")
logger_path = WRK_DIR / "LIBS" / "spec_to_conc" / "ResNet_CNN_1D" / "stc_ResNet_CNN_1D_002_log.txt"
h5_path = WRK_DIR / "LIBS" / "training_ready_LIBS.h5"
eval_dir = WRK_DIR / "LIBS" / "spec_to_conc" / "ResNet_CNN_1D" / "eval_ResNet_CNN_1D_002"
X_scaler_path= WRK_DIR / "LIBS" / "X_scaler.pkl"
y_scaler_path= WRK_DIR / "LIBS" / "y_scaler.pkl"
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
        init_gpu_trainer,
        plot_residual_history,
        # plot_peak_error_heatmap,
        # plot_predicted_vs_actual,
        # run_spectral_inference,
        # hf_get,
        # load_h5_split_dataset,
        load_prepped_training_dataset,
        load_scalers
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

        # region Read prepped .h5
        X_trn, X_val, y_trn, y_val, y_val_physical, surviving_cols, meta_val_df = load_prepped_training_dataset(
            prepped_h5_path=h5_path
        )
        n_features = X_trn.shape[2]
        n_targets = y_trn.shape[1]
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

        log(logger=logger, msg='Start transfer data to GPU')
        time_t1 = time.perf_counter()
        X_trn_t = torch.from_numpy(X_trn).to('cuda')
        y_trn_t = torch.from_numpy(y_trn).to('cuda')
        X_val_t = torch.from_numpy(X_val).to('cuda')
        y_val_t = torch.from_numpy(y_val).to('cuda')
        time_t2 = time.perf_counter()
        log(logger=logger, msg=f'Data transfer to GPU took {(time_t2 - time_t1)} sec')

        # Import the scalers used in data_prep
        X_scaler, y_scaler = load_scalers(
            X_scaler_path= X_scaler_path,
            y_scaler_path= y_scaler_path
            )

        start_time_b = time.perf_counter()
        training_model, history = init_gpu_trainer(
            model=model,
            X_train=X_trn,
            y_train=y_trn,
            X_val=X_val,
            y_val=y_val,
            X_scaler= X_scaler,
            y_scaler= y_scaler,
            max_epochs=50,
            device= torch.device('cuda'),
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