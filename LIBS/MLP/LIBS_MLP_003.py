"""
jvlee_LIBS_ML>LIBS>LIBS_MLP_002.py

NOTE: update all paths before running from new location(machine).

Currently this defines the LIBS_MLP_003 and sets up a 'full_run' function. The script
was nested into 'full_run' to be able to iterate through different hidden layer
structures. This looping takes place in the 'if __name__ == "__main__"'

"""

logger_root = Path(r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\LIBS\LIBS_MLP_003_log")
eval_root = Path(r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\LIBS\eval_MLP_003")
h5_path = r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\LIBS\trn_val_split_LIBS.h5"
hidden_study_root = r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\LIBS\MLP_003_trn_Hist"

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
from setup import add_project_root_to_path
add_project_root_to_path(parent_generation=1)
try:
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

    )
except ImportError:
    try:
        from ...utils import (
            Logger, 
            get_worker_logger, 
            log, 
            init_xpu_trainer,
            plot_residual_history,
            plot_peak_error_heatmap,
            plot_predicted_vs_actual,
            run_spectral_inference,
        )
    except ImportError:
        try:
            # from ..utils.data_prep import enrich_with_progress, combine_and_save_as_HDF5, trn_val_splitter_HDF5, load_h5_dataset
            from ...utils.debug import (
                Logger, 
                get_worker_logger, 
                log
            )
            from ...utils.speak import gen_speak
            from ...utils.xpu_setup import (
                init_xpu_trainer,
            ) 
            from ...utils.plotting import (
                plot_residual_history,
                plot_peak_error_heatmap,
                plot_predicted_vs_actual,
            )
            from ...utils.plotting import (
                run_spectral_inference,
            )
        except ImportError as e:
            print(f'Utils import error: {e}')
# endregion
# endregion


class LIBS_MLP_003(nn.Module):
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

def full_run(
        hiddens: Optional[list] = None,
):
    
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
    eval_dir = Path(eval_root / f"hidden_run_{time_stamp}_{hidden_str}")
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

        # region Load H5 Data file
        with h5py.File(h5_path, 'r') as hf:
            y_trn_raw = hf['train/spectra'][:]
            y_val_raw   = hf['val/spectra'][:]
            wavelengths = hf['wavelengths'][:]

            all_cols = list(hf['train/metadata'].keys())
            feature_cols = [c for c in all_cols
                            if c in ALLOWED_FEATURE_COLS]
            
            X_trn_raw = np.stack([hf[f'train/metadata/{c}'][:] for c in feature_cols], axis=1).astype(np.float32)
            X_val_raw = np.stack([hf[f'val/metadata/{c}'][:] for c in feature_cols], axis=1).astype(np.float32)

            meta_val_dict = {c: hf[f'val/metadata/{c}'][:] for c in feature_cols}
            meta_val_df = pd.DataFrame(meta_val_dict)
            
            log(logger=logger, msg= f'Feature columns ({len(feature_cols)}): {feature_cols}')

        assert X_trn_raw.shape[0] == y_trn_raw.shape[0], \
            f'X/y row mismatch: {X_trn_raw.shape[0]} vs {y_trn_raw.shape[0]}'
        # endregion

        # region Drop Columns
        # drop_columns = ['Unnamed: 0', 'technique', 'file_id', 'file_path', 'scan_rate_mVs']
        selector = VarianceThreshold(threshold=1e-10)
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        # endregion

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

        # region Keep physical copy 
            # kee a copy of the clipped but not scaled for the predicted/actual overlay
        y_val_physical = y_val_raw.copy()
        # endregion

        # region Filter the X data
        selector.fit(X_trn_raw)
        surviving_cols = [c for c, keep in zip(feature_cols, selector.get_support()) if keep]
        X_trn_filtered = pd.DataFrame(
            selector.transform(X_trn_raw),
            columns=surviving_cols)
        X_val_filtered = pd.DataFrame(
            selector.transform(X_val_raw),
            columns=surviving_cols)
        # Filter the metadata df so that index alignment is preserved
        meta_val_df = meta_val_df[[c for c in surviving_cols if c in meta_val_df.columns]]
        # endregion
        
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

        model = LIBS_MLP_003(
            n_features=n_features,
            hidden_dims= hiddens,
            dropout = 0.5,
            n_wavelengths=len(wavelengths)
        )
        end_time_a = time.perf_counter()
        log(logger=logger, msg= f'Training setup time: {(end_time_a - start_time_a)} seconds')


        start_time_b = time.perf_counter()
        training_model, history = init_xpu_trainer(
            model=model,
            X_train=X_trn,
            y_train=y_trn,
            X_val=X_val,
            y_val=y_val,
            X_scaler=X_scaler,
            y_scaler=y_scaler,
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
            save_path= hidden_study_root / f"hidden_run_{time_stamp}_{hidden_str}"
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
        conc_vals = meta_val_df[composition_col].values
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
            X_eval = X_val[eval_indices]
            y_eval_true = y_val_physical[eval_indices]
            y_eval_pred = run_spectral_inference(
                model = training_model,
                X_samples= X_eval,
                y_scaler=y_scaler,
                device=torch.device('xpu')
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
        full_run(hiddens=h,hidden_count=hc)
        # time.sleep(5)

    loop_time_end = time.perf_counter()
    print(f'The hidden layers sweep took: {(loop_time_end - loop_time_start)/60} minutes.')
    gen_speak('Complete run set complete!')
    

# region Imporvements
    # - Update xpu_setup.py to log instead of print
# endregion

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