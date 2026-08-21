"""
jvlee_LIBS_ML > LIBS > spec_to_conc > CNN_1D > stc_CNN_1D_002.py

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

# region Global Variables
max_epochs = 500
batch_size= 256
learning_rate= 3e-5
weight_decay= 1e-4
# endregion

# region Paths
# print('check 4')
WRK_DIR = Path(__file__).parent.parent.parent.parent.resolve()
print(f"Working directory is {WRK_DIR}")
logger_path = WRK_DIR / "LIBS" / "spec_to_conc" / "CNN_1D" / "stc_CNN_1D_002_log.txt"
h5_path = WRK_DIR / "LIBS" / "training_ready_LIBS.h5"
eval_dir = WRK_DIR / "LIBS" / "spec_to_conc" / "CNN_1D" / "eval_CNN_1D_002"
X_scaler_path= WRK_DIR / "LIBS" / "X_scaler.pkl"
y_scaler_path= WRK_DIR / "LIBS" / "y_scaler.pkl"
# endregion

class STC_1D_CNN_002(nn.Module):
    def __init__(
            self, 
            input_channels: int = 1, 
            layer_1_out: int = 128, 
            layer_2_out: int = 64,
            layer_3_out: int = 32,
            adaptive_pool_out: int = 64,
            fc_hidden: int = 256,
            kernel_size: int = 3,
            padding: int = 1,
            dropout: float = 0.3,
            n_targets: int = 21):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, layer_1_out, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(layer_1_out),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(layer_1_out, layer_2_out, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(layer_2_out),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(layer_2_out, layer_3_out, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(layer_3_out),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(adaptive_pool_out),
            nn.Flatten()
        )

        conv_out_size = layer_3_out * adaptive_pool_out
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size,fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, n_targets)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

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
        model = STC_1D_CNN_002(
            input_channels=1,
            layer_1_out= 128, 
            layer_2_out = 64,
            layer_3_out = 32,
            adaptive_pool_out=64,
            fc_hidden=256,
            kernel_size=3,
            padding=1,
            dropout = 0.2,
            n_targets=n_targets
        )
        end_time_a = time.perf_counter()
        log(logger=logger, msg= f'Training setup time: {(end_time_a - start_time_a):.3f} seconds')

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(logger=logger, msg=f'Model parameters: {n_params:,}')

        log(logger=logger, msg='Start transfer data to GPU')
        time_t1 = time.perf_counter()
        # X_trn_t = torch.from_numpy(X_trn).to('cuda')
        # y_trn_t = torch.from_numpy(y_trn).to('cuda')
        # X_val_t = torch.from_numpy(X_val).to('cuda')
        # y_val_t = torch.from_numpy(y_val).to('cuda')
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
            max_epochs=max_epochs,
            device= torch.device('cuda'),
            batch_size=batch_size,
            shuffle= True,
            num_workers= 3,
            persistent_workers=True,
            pin_memory= True,
            learning_rate= learning_rate,
            criterion= nn.MSELoss(),
            clip_grads= True,
            optimizer_cls= optim.AdamW,  # TODO: try AdamW
            weight_decay= weight_decay,
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

