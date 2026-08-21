from __future__ import annotations

"""
jvlee_LIBS_ML > LIBS > conc_to_spec > MLP > Predictions_MLP_003 > LIBS_MLP_002.py

NOTE: update all paths before running from new location(machine).

Currently this defines the LIBS_MLP_003 and sets up a 'full_run' function. The script
was nested into 'full_run' to be able to iterate through different hidden layer
structures. This looping takes place in the 'if __name__ == "__main__"'

"""


# region Imports
# region plain
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
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from logging.handlers import QueueListener
from typing import Optional
# endregion

# region custom
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from setup import add_project_root_to_path
add_project_root_to_path(parent_generation=1)
from utils import (
    gen_speak,
    plot_peak_error_heatmap,
    plot_predicted_vs_actual,
    run_spectral_inference,
)
# endregion
# endregion

# region Paths
# print('check 4')
WRK_DIR = Path(__file__).parent.parent.parent.parent.resolve()
print(f"Working directory is {WRK_DIR}")
logger_root = WRK_DIR / "LIBS" / "conc_to_spec" / "MLP" / "cts_MLP_004"
h5_path = WRK_DIR / "LIBS" / "training_ready_LIBS.h5"
eval_dir = WRK_DIR / "LIBS" / "conc_to_spec" / "MLP" / "eval_cts_MLP_004"
X_scaler_path= WRK_DIR / "LIBS" / "X_scaler.pkl"
y_scaler_path= WRK_DIR / "LIBS" / "y_scaler.pkl"
hidden_study_root = logger_root / "cts_MLP_004_trn_Hist"
# endregion

max_epochs= 500
batch_size= 256
learning_rate= 1e-03
weight_decay= 1e-04

class LIBS_MLP_003(nn.Module):
    def __init__(
            self, 
            n_features: int = 21,
            hidden_dims: tuple[int, ...] | list[int] = (512, 1024, 2048, 1024, 512),
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
}

def full_run(
        hiddens: Optional[list[int] | tuple[int, ...]] = None,
):
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
        load_prepped_training_dataset,
        load_scalers
    )

    # endregion
    
    if hiddens is None:
        hiddens = (512, 1024, 512)

    if isinstance(hiddens, (list, tuple)) and any(isinstance(el, int) for el in hiddens):
        hidden_tuple = tuple(hiddens)
    else:
        hidden_tuple = tuple(hiddens) if hiddens is not None else (512, 1024, 512)

    hidden_str = '_'.join(map(str, hidden_tuple))

    # region Timing Setup
    start_time = time.perf_counter()
    start_time_a = time.perf_counter()
    time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # endregion

    # region Logging Setup
    logger_path = Path(logger_root / f"hidden_run_{time_stamp}_{hidden_str}.txt")
    eval_dir = Path(logger_root / f"hidden_run_{time_stamp}_{hidden_str}")
    logger_path.parent.parent.mkdir(parents=True, exist_ok=True)
    logger_path.parent.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    my_logger = None

    try:
        my_logger = Logger(logger_path)
        my_logger.setFormatter(logging.Formatter('%(asctime)s  |  %(message)s'))

        sys.stdout = my_logger

        log_q = multiprocessing.Queue()
        listener = QueueListener(log_q, my_logger, respect_handler_level=True)
        listener.start()
    except Exception as e:
        print(f'Logger initialization failed: {e}', file=sys.__stderr__)
        if my_logger is not None:
            my_logger.close()
        raise
    # endregion

    try:  
        logger = get_worker_logger(Path(logger_path).stem)

        # region Read prepped .h5 
        # 1. Keep the original 7-variable unpacking that your function expects
        X_trn, X_val, y_trn, y_val, y_val_physical, surviving_cols, meta_val_df = load_prepped_training_dataset(
            prepped_h5_path=h5_path
        )

        # 2. Extract the actual wavelength array directly from the HDF5 keys
        with h5py.File(h5_path, 'r') as hf:
            wavelengths = np.array(hf['wavelengths'], dtype=np.float32)
        true_wavelength_count = X_trn.shape[-1]

        # 3. Print shapes to verify it worked perfectly
        print(f"wavelengths array shape = {wavelengths.shape}")
        n_features = y_trn.shape[1]
        print(f"n_features = {n_features}")
        # endregion
        
        # Import the scalers used in data_prep
        X_scaler, y_scaler = load_scalers(
            X_scaler_path= X_scaler_path,
            y_scaler_path= y_scaler_path
            )


        model = LIBS_MLP_003(
            n_features=n_features,
            hidden_dims= hiddens,
            dropout = 0.5,
            n_wavelengths=true_wavelength_count
        )
        end_time_a = time.perf_counter()
        log(logger=logger, msg= f'Training setup time: {(end_time_a - start_time_a)} seconds')

        X_trn_flat = X_trn.squeeze()
        X_val_flat = X_val.squeeze()

        start_time_b = time.perf_counter()
        training_model, history = init_gpu_trainer(
            # The X's and y's are switched because it was built out for stc and is
            # now being used on cts 
            model=model,
            X_train=y_trn,
            y_train=X_trn_flat,
            X_val=y_val,
            y_val=X_val_flat,
            X_scaler=y_scaler,
            y_scaler=X_scaler,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device= torch.device('cuda'),
            shuffle= True,
            num_workers= 3,
            pin_memory= False,
            learning_rate= learning_rate,
            criterion= nn.MSELoss(),
            clip_grads= True,
            optimizer_cls= optim.Adam,
            weight_decay= weight_decay,
            verbose= True,
            plot_animation= True,
            save_path= str(hidden_study_root) + f"hidden_run_{time_stamp}_{hidden_str}",
            log_path=logger_path
        )

        end_time_b = time.perf_counter()
        log(logger=logger, msg= f'Training time: {(end_time_b - start_time_b)/60} minutes')
        
        # region Residual Plot
        log(logger=logger, msg='Starting post-training spectral evaluation...')
        plot_residual_history(
            history=history,
            save_path=str(eval_dir / f'residual_history_{time_stamp}.png'),
            show=False
        )
        # endregion 

        # region Sample Plots
        # region plot prep
        composition_col = next(
            (c for c in [
                'conc_CaCl3_wt%', 
                'conc_CeCl3_wt%', 
                'conc_UCl3_wt%', 
                'conc_SmCl3_wt%',
                'conc_GdCl3_wt%',
                'conc_LaCl3_wt%',
                'conc_MgCl2_wt%',
                ] if c in meta_val_df.columns),
            None
        )
        if composition_col is None:
            raise ValueError(f'Could not find any concentration columns in : {list(meta_val_df.columns)}')
        log(logger=logger, msg=f'Composition-range sampling column: {composition_col}')

        target_percentiles = [5, 15, 30, 50, 70, 85, 95]
        conc_vals = meta_val_df[composition_col].to_numpy(dtype=np.float32)
        eval_indices = []
        for pct in target_percentiles:
            target_val = np.percentile(conc_vals, pct)
            closest = int(np.argmin(np.abs(conc_vals - target_val)))
            if closest not in eval_indices:
                eval_indices.append(closest)

            eval_labels = [
                f'{composition_col} = {conc_vals[i]:.4f} (p{p})'
                for i, p in zip(eval_indices, target_percentiles[:len(eval_indices)])
            ]

            log(logger=logger, msg = f'Evaluation sample indices: {eval_indices}')
            log(logger=logger, msg = f'Evaluation sample labels: {eval_labels}')
            # endregion

            # region Prediction
            # The X's and y's are switched because it was built out for stc and is
            # now being used on cts 
            X_eval = y_val[eval_indices]
            y_eval_true = X_val[eval_indices].squeeze()
            y_eval_pred = run_spectral_inference(
                model = training_model,
                X_samples= X_eval,
                y_scaler=y_scaler,
                device=torch.device('cuda')
            )
            # endregion

            # region Plot prediction
            plot_predicted_vs_actual(
                wavelengths=wavelengths,
                y_actual_raw=y_eval_true,
                y_predicted=y_eval_pred,
                sample_indices=list(range(len(eval_indices))),
                sample_labels=eval_labels,
                save_path=str(eval_dir / f'spectral_overlay_{time_stamp}.png'),
                show=False,
                ncols=2,
            )
            # endregion

            # region Plot error heatmap
            plot_peak_error_heatmap(
                wavelengths=wavelengths,
                y_actual_raw=y_eval_true,
                y_predicted=y_eval_pred,
                sample_labels=eval_labels,
                save_path=str(eval_dir / f'error_heatmap_{time_stamp}.png'),
                show=False,
            )
            # endregion

            log(logger=logger, msg=f'Post-training evaluation plots saved to: {eval_dir}')

            # region Report
            for i, label in enumerate(eval_labels):
                actual    = y_eval_true[i]
                predicted = y_eval_pred[i]
                mae_val   = np.mean(np.abs(predicted - actual))
                peak_err  = np.abs(predicted - actual).max()
                peak_wl   = wavelengths[np.abs(predicted - actual).argmax()]
                log(logger=logger, msg=(
                    f'  {label:50s} | MAE={mae_val:8.2f} '
                    f'| MaxErr={peak_err:8.2f} @ {peak_wl:.1f} nm'
                ))
            # endregion
        # endregion

        end_time = time.perf_counter()
        log(logger=logger, msg= f'Total training time: {((end_time - start_time) / 60):.3f} minutes')
    finally:
        gen_speak('Training complete!')
        if listener is not None:
            listener.stop()
        sys.stdout = sys.__stdout__
        if my_logger is not None:
            my_logger.close()

if __name__ == "__main__":
    loop_time_start = time.perf_counter()
    hiddens = [
        (256, 256),
        # (256, 512, 256),
        # (256, 512, 512, 256),
        # (256, 512, 1024, 512, 256),
        # (512, 512),
        # (512, 512, 512),
        # (512, 512, 512, 512),
        # (512, 1024, 512),
        # (1024, 1024),
        # (1024, 1024, 1024),
        # (1024, 2048, 1024,),
        # (1024, 1024, 1024, 1024)
    ]
    for hc, h in enumerate(hiddens):
        full_run(hiddens=h)
        # time.sleep(5)

    loop_time_end = time.perf_counter()
    print(f'The hidden layers sweep took: {(loop_time_end - loop_time_start)/60} minutes.')
    gen_speak('Complete run set complete!')
    