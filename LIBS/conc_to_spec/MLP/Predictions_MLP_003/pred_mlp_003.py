"""
jvlee_LIBS_ML > LIBS > MLP > Predictions_MLP_003 > pred_mlp_003.py
"""

# region Imports
# region library sourced
import torch
import h5py
import joblib
import time
import sys
import multiprocessing

import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from logging.handlers import QueueListener
# endregion

# region custom
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from setup import add_project_root_to_path
add_project_root_to_path(parent_generation=1)
from LIBS.conc_to_spec.MLP.LIBS_MLP_003 import(
    LIBS_MLP_003, 
) 
from utils import(
    log,
    logging,
    Logger,
    get_worker_logger,
    gen_speak,
    run_spectral_inference,
    plot_peak_error_heatmap,
    plot_predicted_vs_actual,
    plot_residual_history,
    load_h5_split_dataset
)
# endregion
# endregion

# region Globals
# region Paths
h5_path = r"G:\My Drive\RLSL\Data\combined_CSVs\trn_val_split_LIBS.h5"
model_path = r"G:\My Drive\RLSL\Python\LIBS\MLP\MLP_003_trn_Hist\best_model_20260406_232918.pt"
pred_dir = r"G:\My Drive\RLSL\Python\LIBS\MLP\Predictions_MLP_003"
log_path = r"G:\My Drive\RLSL\Python\LIBS\MLP\Predictions_MLP_003\pred_mlp_003_logfile.txt"
# endregion

# region Variables
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
# endregion
# endregion

if __name__ == '__main__':
    # region Timing Setup
    start_time = time.perf_counter()
    start_time_a = time.perf_counter()
    # endregion

    # region Prediction Directory
    pred_dir = Path(pred_dir).resolve()
    pred_dir.mkdir(parents=True, exist_ok=True)
    # endregion

    # region Logging Setup
    my_logger = Logger(log_path)
    my_logger.setFormatter(logging.Formatter('%(asctime)s  |  %(message)s'))

    sys.stdout = my_logger

    log_q = multiprocessing.Queue()
    listener = QueueListener(log_q, my_logger, respect_handler_level=True)
    listener.start()
    # endregion

    try:
        logger = get_worker_logger(Path(log_path).stem)
        # region recreate scalers
        # NOTE This will not be needed once I re-train with the joblib.dump's in place in xpu_setup.py
            # NOTE The scaler saving has been implimented but I haven't re-trained with it yet.
            # To load the data:
            #       - X_scaler = joblib.load(path to X_scaler)
            #       - y_scaler = joblib.load(path to y_scaler) 
        # region Load H5 Data file
        X_trn_raw, X_val_raw, y_trn_raw, y_val_raw, feature_cols, wavelengths, meta_val_df = load_h5_split_dataset(
            h5_path=h5_path,
            allowed_feature_cols=ALLOWED_FEATURE_COLS,
        )

        assert X_trn_raw.shape[0] == y_trn_raw.shape[0], \
            f'X/y row mismatch: {X_trn_raw.shape[0]} vs {y_trn_raw.shape[0]}'

        log(logger=logger, msg=f'Feature columns ({len(feature_cols)}): {feature_cols}')
        # endregion

        # region Drop Columns
        # drop_columns = ['Unnamed: 0', 'technique', 'file_id', 'file_path', 'scan_rate_mVs']
        selector = VarianceThreshold(threshold=1e-15)
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        # endregion

        # region Remove NAN rows
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
        # endregion

        # region Clip Outliers
        # clip the extra bright shots so they don't skew the mean and standard deviation
        clip_threshold = np.nanpercentile(y_trn_raw, 99.5)
        log(logger=logger, msg= f'Clipping spectra intensities above: {clip_threshold:.1f}')
        y_trn_raw = np.clip(y_trn_raw, 0, clip_threshold)
        y_val_raw = np.clip(y_val_raw, 0, clip_threshold)   # val uses the same threshold as trn
        # endregion

        # region Keep physical copy 
            # kee a copy of the clipped but not scaled for the predicted/actual overlay
        y_val_physical = y_val_raw.copy()
        # endregion

        # region Filter the X data
        log(logger=logger, msg= f'Filtering X data')
        selector.fit(X_trn_raw)
        surviving_cols = [c for c, keep in zip(feature_cols, selector.get_support()) if keep]
        X_trn_filtered = pd.DataFrame(
            np.asarray(selector.transform(X_trn_raw), dtype=np.float32),
            columns=surviving_cols)
        X_val_filtered = pd.DataFrame(
            np.asarray(selector.transform(X_val_raw), dtype=np.float32),
            columns=surviving_cols)
        # Filter the metadata df so that index alignment is preserved
        meta_val_df = meta_val_df[[c for c in surviving_cols if c in meta_val_df.columns]]
        # endregion
        
        # region Scaling
        log(logger=logger, msg= f'Scaling data')
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

        # for col in X_val_filtered.columns:
        #     unique_vals = X_val_filtered[col].nunique()
        #     col_min = X_val_filtered[col].min()
        #     col_max = X_val_filtered[col].max()
        #     log(logger=logger, msg=f'  {col:30s} | unique={unique_vals:5d} | min={col_min:.4f} | max={col_max:.4f}')

        n_features = X_trn.shape[1]
        log(logger=logger, msg= f'Number of features is {n_features}')
        # log(logger=logger, msg= surviving_cols)
        
        # endregion

        # region Recreate and load in Model
        log(logger=logger, msg= f'Recreating Model')
        model = LIBS_MLP_003(
            n_features=n_features,
            hidden_dims=(512, 1024, 512),
            dropout=0.5,
            n_wavelengths=len(wavelengths)
        )

        log(logger=logger, msg= f'Loading in weights')
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

        # region Move to XPU
        log(logger=logger, msg= f'Moving to XPU')
        device = torch.device('xpu')
        model = model.to(device=device)
        # endregion
        # endregion

    
        # region Sample Plots
        # region plot prep
        composition_cols = [
                'conc_CeCl3_wt%',
                'conc_GdCl3_wt%',
                'conc_Mg_wt%',
                'conc_Nd_wt%',
                'conc_SmCl3_wt%',
                'conc_Sm_wt%', 
                'conc_UCl3_wt%', 
                ]
        log(logger=logger, msg=f'Composition-range sampling column: {composition_cols}')

        target_percentiles = [5, 35, 65, 95]
        for c in composition_cols:
            conc_vals = meta_val_df[c].to_numpy(dtype=np.float32)

            # Split into non-zero and zero indices
            nonzero_mask = conc_vals > 0
            nonzero_indices = np.where(nonzero_mask)[0]
            zero_indices = np.where(~nonzero_mask)[0]

            log(logger=logger, msg=f'{c} | non-zero samples: {len(nonzero_indices)} | zero samples: {len(zero_indices)}')

            # region Sample non-zero indices across concentration range
            pred_indices_nonzero = []
            if len(nonzero_indices) > 0:
                nonzero_vals = conc_vals[nonzero_indices]
                for pct in target_percentiles:
                    target_val = np.percentile(nonzero_vals, pct)
                    closest_local = int(np.argmin(np.abs(nonzero_vals - target_val)))
                    closest_global = int(nonzero_indices[closest_local])
                    if closest_global not in pred_indices_nonzero:
                        pred_indices_nonzero.append(closest_global)
            # endregion

            # region Sample blank (zero) indices — up to 4
            pred_indices_blank = []
            if len(zero_indices) > 0:
                blank_percentiles = [5, 35, 65, 95]
                for pct in blank_percentiles:
                    closest = int(zero_indices[int(np.percentile(range(len(zero_indices)), pct))])
                    if closest not in pred_indices_blank:
                        pred_indices_blank.append(closest)
                    if len(pred_indices_blank) >= 4:
                        break
            # endregion

            # region Plot non-zero samples
            if pred_indices_nonzero:
                pred_labels_nonzero = [
                    f'{c} = {conc_vals[i]:.4f} (p{p})'
                    for i, p in zip(pred_indices_nonzero, target_percentiles[:len(pred_indices_nonzero)])
                ]
                log(logger=logger, msg=f'Non-zero indices: {pred_indices_nonzero}')
                log(logger=logger, msg=f'Non-zero labels: {pred_labels_nonzero}')

                X_eval = X_val[pred_indices_nonzero]
                y_eval_true = y_val_physical[pred_indices_nonzero]
                y_eval_pred = run_spectral_inference(model=model, X_samples=X_eval, y_scaler=y_scaler, device=torch.device('xpu'))

                plot_predicted_vs_actual(
                    wavelengths=wavelengths, y_actual_raw=y_eval_true, y_predicted=y_eval_pred,
                    sample_indices=list(range(len(pred_indices_nonzero))), sample_labels=pred_labels_nonzero,
                    save_path=str(pred_dir / f'spectral_overlay_{c}_nonzero_MLP_003.png'), show=False, ncols=2,
                )
                plot_peak_error_heatmap(
                    wavelengths=wavelengths, y_actual_raw=y_eval_true, y_predicted=y_eval_pred,
                    sample_labels=pred_labels_nonzero,
                    save_path=str(pred_dir / f'error_heatmap_{c}_nonzero_MLP_003.png'), show=False,
                )

                for i, label in enumerate(pred_labels_nonzero):
                    actual, predicted = y_eval_true[i], y_eval_pred[i]
                    mae_val = np.mean(np.abs(predicted - actual))
                    peak_err = np.abs(predicted - actual).max()
                    peak_wl = wavelengths[np.abs(predicted - actual).argmax()]
                    log(logger=logger, msg=f'  {label:50s} | MAE={mae_val:8.2f} | MaxErr={peak_err:8.2f} @ {peak_wl:.1f} nm')
            else:
                log(logger=logger, msg=f'WARNING: No non-zero samples found for {c}, skipping non-zero plots.')
            # endregion

            # region Plot blank samples
            if pred_indices_blank:
                pred_labels_blank = [f'{c} = 0.0000 (blank {i+1})' for i in range(len(pred_indices_blank))]
                log(logger=logger, msg=f'Blank indices: {pred_indices_blank}')

                X_eval = X_val[pred_indices_blank]
                y_eval_true = y_val_physical[pred_indices_blank]
                y_eval_pred = run_spectral_inference(model=model, X_samples=X_eval, y_scaler=y_scaler, device=torch.device('xpu'))

                plot_predicted_vs_actual(
                    wavelengths=wavelengths, y_actual_raw=y_eval_true, y_predicted=y_eval_pred,
                    sample_indices=list(range(len(pred_indices_blank))), sample_labels=pred_labels_blank,
                    save_path=str(pred_dir / f'spectral_overlay_{c}_blanks_MLP_003.png'), show=False, ncols=2,
                )
                plot_peak_error_heatmap(
                    wavelengths=wavelengths, y_actual_raw=y_eval_true, y_predicted=y_eval_pred,
                    sample_labels=pred_labels_blank,
                    save_path=str(pred_dir / f'error_heatmap_{c}_blanks_MLP_003.png'), show=False,
                )
            # endregion

            log(logger=logger, msg=f'Predictions for {c} are complete.')
        # endregion

    finally:
        # region Close
        gen_speak('Predictions complete!')
        listener.stop()
        sys.stdout = sys.__stdout__
        my_logger.close()
        # endregion

# region Notes
#   blank                          | unique=    2 | min=0.0000 | max=1.0000
#   conc_CeCl3_wt%                 | unique=   48 | min=0.0000 | max=5.0000
#   conc_GdCl3_wt%                 | unique=   63 | min=0.0000 | max=4.9240
#   conc_Mg_wt%                    | unique=    2 | min=0.0000 | max=0.1000
#   conc_Nd_wt%                    | unique=    2 | min=0.0000 | max=2.0000
#   conc_SmCl3_wt%                 | unique=   33 | min=0.0000 | max=10.0000
#   conc_Sm_wt%                    | unique=    2 | min=0.0000 | max=1.0000
#   conc_UCl3_wt%                  | unique=   18 | min=0.0000 | max=8.7390
#   delay                          | unique=    8 | min=0.0000 | max=14.0000
#   delay_study                    | unique=    2 | min=0.0000 | max=1.0000
#   energy                         | unique=    4 | min=2.0000 | max=200.0000
#   energy_study                   | unique=    2 | min=0.0000 | max=1.0000
#   flow                           | unique=   11 | min=0.0000 | max=15.0000
#   flow_study                     | unique=    2 | min=0.0000 | max=1.0000
#   kinetic                        | unique=    2 | min=0.0000 | max=1.0000
#   pressure                       | unique=   10 | min=0.0000 | max=60.0000
#   pressure_study                 | unique=    2 | min=0.0000 | max=1.0000
#   qdelay                         | unique=    8 | min=70.0000 | max=150.0000
#   qdelay_study                   | unique=    2 | min=0.0000 | max=1.0000
#   repetition                     | unique=   25 | min=1.0000 | max=25.0000
#   shot_study                     | unique=    2 | min=0.0000 | max=1.0000
#   shots                          | unique=    9 | min=10.0000 | max=500.0000
#   state_aerosol                  | unique=    2 | min=0.0000 | max=1.0000
#   state_solid                    | unique=    2 | min=0.0000 | max=1.0000
#   static_                        | unique=    2 | min=0.0000 | max=1.0000
#   temperature_C                  | unique=    2 | min=20.0000 | max=500.0000
#   width                          | unique=    2 | min=3.0000 | max=8.0000
#   width_study                    | unique=    2 | min=0.0000 | max=1.0000
# endregion