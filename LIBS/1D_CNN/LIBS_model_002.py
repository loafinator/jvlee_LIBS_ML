"""
Python > LIBS > LIBS_model_002.py

"""

from setup import add_project_root_to_path
add_project_root_to_path(parent_generation=1)

from utils import init_xpu_1d_cnn_trainer
import pandas as pd
import numpy as np
import time 
from torch import optim
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import torch
import torch.nn as nn


class LIBS_1D_CNN_002(nn.Module):
    def __init__(
            self, 
            input_channels: int = 1, 
            layer_1_out: int = 32, 
            layer_2_out: int = 128,
            layer_3_out: int = 512,
            kernel_size: int = 3,
            padding: int = 1,
            n_features: int = 21, 
            dropout: float = 0.3,
            n_wavelengths: int = 451):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, layer_1_out, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(layer_1_out),
            nn.ReLU(),
            nn.Conv1d(layer_1_out, layer_2_out, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(layer_2_out),
            nn.ReLU(),
            nn.Flatten()
        )

        conv_out_size = layer_2_out * n_features
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size,layer_3_out),
            nn.BatchNorm1d(layer_3_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_3_out, n_wavelengths)
        )

    def forward(self, x):
        return self.fc(self.conv(x))
    

if __name__ == "__main__":
    start_time = time.perf_counter()
    start_time_a = time.perf_counter()

    # region number of wavelength columns
#    df = pd.read_csv(r"G:\My Drive\RLSL\Data\MATs_to_CSVs\KBrice\NEUP Keith Data\Data transfer\Sample Spectra\0.5 wt CeCl3 Samples\Data_Import_lamba_1.csv")
    df = pd.read_csv(r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations\SmCl3 Studies\SmCl3 CV and LIBS 1\Sm LIBS data\SL_Values_472_lambda_SLA.csv")
    n_wavelengths = len([c for c in df.columns if c.replace('.','').isnumeric()])
    print(f'Number of wavelengths is {n_wavelengths}')
    # endregion

    # region Preparing training and validation data
    trn_data = pd.read_csv(r"G:\My Drive\RLSL\Data\combined_CSVs\training_data.csv")
    val_data = pd.read_csv(r"G:\My Drive\RLSL\Data\combined_CSVs\validation_data.csv")
    #print(trn_data.head())

    numeric_column_mask = pd.to_numeric(trn_data.columns, errors='coerce').notna()
    targets = trn_data.columns[numeric_column_mask].tolist()

    drop_columns = ['Unnamed: 0', 'technique', 'file_id', 'file_path', 'scan_rate_mVs']
    selector = VarianceThreshold(threshold=1e-10)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Split raw data into features and targets
    trn_X_raw = trn_data.drop(columns=targets + drop_columns, errors='ignore')
    val_X_raw = val_data.drop(columns=targets + drop_columns, errors='ignore')
    trn_y_raw = trn_data[targets].values.astype(np.float32)
    val_y_raw = val_data[targets].values.astype(np.float32)

    # Find the fraction of NaN values as wavelength per row
    nan_frac_trn = np.isnan(trn_y_raw).mean(axis=1)
    nan_frac_val = np.isnan(val_y_raw).mean(axis=1)

    # Designate any row with more than __% as bad rows to be dropped
    bad_rows_trn = nan_frac_trn > 0.1
    bad_rows_val = nan_frac_val > 0.1
    print(f'Dropping {bad_rows_trn.sum()} training rows and {bad_rows_val.sum()} validation rows with >10% NaN values in Spectrum')

    # Drop designated rows
    trn_y_raw = trn_y_raw[~bad_rows_trn]
    val_y_raw = val_y_raw[~bad_rows_val]

    # Clip the extra bright shots so they don't skew the mean and standard deviation
    clip_threshold = np.nanpercentile(trn_y_raw, 99.5)
    print(f'Clipping spectra intensities above: {clip_threshold:.1f}')
    trn_y_raw = np.clip(trn_y_raw, 0, clip_threshold)
    val_y_raw = np.clip(val_y_raw, 0, clip_threshold)   # val uses the same threshold as trn

    # Ensure X data is all numeric
    trn_X_numeric = trn_X_raw.select_dtypes(include=[np.number])
    val_X_numeric = val_X_raw.select_dtypes(include=[np.number])

    # Filter the X data
    trn_X_filtered = pd.DataFrame(
        selector.fit_transform(trn_X_numeric),
        columns=trn_X_numeric.columns[selector.get_support()])
    val_X_filtered = pd.DataFrame(
        selector.transform(val_X_numeric),
        columns=trn_X_numeric.columns[selector.get_support()])
    
    # Scale the X data
    trn_X_scaled = scaler_X.fit_transform(trn_X_filtered)
    trn_X_scaled = np.nan_to_num(trn_X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    val_X_scaled = scaler_X.transform(val_X_filtered)
    val_X_scaled = np.nan_to_num(val_X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Match the X with y by applying the same bad rows mask
    trn_X_scaled_and_cleaned = trn_X_scaled[~bad_rows_trn]
    val_X_scaled_and_cleaned = val_X_scaled[~bad_rows_val]

    # Expand X data
    trn_X = np.expand_dims(trn_X_scaled_and_cleaned, axis=1).astype(np.float32)
    val_X = np.expand_dims(val_X_scaled_and_cleaned, axis=1).astype(np.float32)

    # Scale y data
    trn_y = np.nan_to_num(scaler_y.fit_transform(trn_y_raw), nan=0.0).astype(np.float32)
    val_y = np.nan_to_num(scaler_y.transform(val_y_raw), nan=0.0).astype(np.float32)
    # endregion
    
    # region 
    print(f"Columns surviving VarianceThreshold: {trn_X_filtered.columns.tolist()}")

    print(f"NaNs in trn_X: {np.isnan(trn_X).sum()}")
    print(f"NaNs in trn_y: {np.isnan(trn_y).sum()}")
    print(f"NaNs in val_X: {np.isnan(val_X).sum()}")
    print(f"NaNs in val_y: {np.isnan(val_y).sum()}")
    print(f"trn_y min/max: {trn_y.min():.4f} / {trn_y.max():.4f}")
    print(f"trn_X min/max: {trn_X.min():.4f} / {trn_X.max():.4f}")

    n_features = trn_X.shape[2]
    print(f'number of features is:          {n_features}')
    print(f'Training features shape:        {trn_X.shape}')
    print(f'Validation features shape:      {val_X.shape}')
    print(f'Training targets shape:         {trn_y.shape}')
    print(f'Validation targets shape:       {val_y.shape}')

    row_maxes = trn_y_raw.max(axis=1)
    print(f'Top 10 max intensities per spectrum: {np.sort(row_maxes)[-10:]}')
    outlier_mask = row_maxes > np.percentile(row_maxes, 99)
    print(f'Spectra with extreme max values: {outlier_mask.sum()} / {len(trn_y_raw)}')

    model = LIBS_1D_CNN_002(
        input_channels=1,
        layer_1_out= 32, 
        layer_2_out = 128,
        layer_3_out = 512,
        kernel_size=3,
        padding=1,
        n_features=n_features,
        dropout = 0.5,
        n_wavelengths=len(targets)
    )
    end_time_a = time.perf_counter()
    print(f'Training setup time: {(end_time_a - start_time_a)} seconds')


    start_time_b = time.perf_counter()
    training_model, history = init_xpu_1d_cnn_trainer(
        model=model,
        X_train=trn_X,
        y_train=trn_y,
        X_val=val_X,
        y_val=val_y,
        max_epochs=350,
        batch_size=256,
        device= torch.device('xpu'),
        shuffle= True,
        num_workers= 0,
        pin_memory= False,
        learning_rate= 0.00001,
        criterion= nn.MSELoss(),
        clip_grads= True,
        optimizer_cls= optim.Adam,
        scheduler_cls= torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_kwargs= None,
        verbose= True,
        plot_animation= True
    )

    end_time_b = time.perf_counter()
    print(f'Training time: {(end_time_b - start_time_b)/60} minutes')

    end_time = time.perf_counter()
    print(f'Total training time: {(end_time - start_time) / 60} minutes')