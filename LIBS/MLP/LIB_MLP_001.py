"""
Python > LIBS > LIBS_MLP_001.py

"""

from setup import add_project_root_to_path
add_project_root_to_path(parent_generation=1)

from utils import init_xpu_mlp_trainer, Logger
import pandas as pd
import numpy as np
import time 
from torch import optim
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import torch
import torch.nn as nn
import sys


class LIBS_MLP_001(nn.Module):
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
    

if __name__ == "__main__":
    # region Initiate log file
    log_path = r"G:\My Drive\RLSL\Python\LIBS\LIBS_MLP_001_log.txt"
    sys.stdout = Logger(log_path)
    # endrigion
    
    # region Start timers
    start_time = time.perf_counter()
    start_time_a = time.perf_counter()
    # endrigion

    # region number of wavelength columns
    #df = pd.read_csv(r"G:\My Drive\RLSL\Data\MATs_to_CSVs\KBrice\NEUP Keith Data\Data transfer\Sample Spectra\0.5 wt CeCl3 Samples\Data_Import_lamba_1.csv")
    df = pd.read_csv(r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations\SmCl3 Studies\SmCl3 CV and LIBS 1\Sm LIBS data\SL_Values_472_lambda_SLA.csv")
    n_wavelengths = len([c for c in df.columns if c.replace('.','').isnumeric()])
    print(f'Number of wavelengths is {n_wavelengths}')
    # endregion

    # region Preparing training and validation data
    trn_data = pd.read_csv(r"G:\My Drive\RLSL\Data\combined_CSVs\training_data.csv")
    val_data = pd.read_csv(r"G:\My Drive\RLSL\Data\combined_CSVs\validation_data.csv")

    print(f"Training file_ids: {trn_data['file_id'].nunique()}")
    print(f"Validation file_ids: {val_data['file_id'].nunique()}")
    print(f"Rows per file_id (train): {len(trn_data) / trn_data['file_id'].nunique():.1f} avg")

    comp_cols = [c for c in trn_data.columns if 'conc_' in c or 'frac_' in c]
    print(trn_data[comp_cols].nunique())  # check variance within each comp column
    #trn_data = trn_data.groupby('file_id').mean(numeric_only=True).reset_index()
    #val_data = val_data.groupby('file_id').mean(numeric_only=True).reset_index()

    # Column assignment
    numeric_column_mask = pd.to_numeric(trn_data.columns, errors='coerce').notna()
    targets = trn_data.columns[numeric_column_mask].tolist()
    drop_columns = ['Unnamed: 0', 'technique', 'file_id', 'file_path', 'scan_rate_mVs']

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

    shared_cols = [c for c in trn_X_numeric.columns if c in val_X_numeric.columns]
    trn_X_numeric = trn_X_numeric[shared_cols]
    val_X_numeric = val_X_numeric[shared_cols]

    # Filter the X data
    selector = VarianceThreshold(threshold=1e-10)
    trn_X_filtered = pd.DataFrame(
        selector.fit_transform(trn_X_numeric),
        columns=trn_X_numeric.columns[selector.get_support()])
    val_X_filtered = pd.DataFrame(
        selector.transform(val_X_numeric),
        columns=trn_X_numeric.columns[selector.get_support()])
    print(f"Columns surviving VarianceThreshold: {trn_X_filtered.columns.tolist()}")
    
    # Scale the X data
    scaler_X = StandardScaler()
    trn_X_scaled = scaler_X.fit_transform(trn_X_filtered)
    trn_X_scaled = np.nan_to_num(trn_X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    val_X_scaled = scaler_X.transform(val_X_filtered)
    val_X_scaled = np.nan_to_num(val_X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Match the X with y by applying the same bad rows mask
    trn_X_scaled_and_cleaned = trn_X_scaled[~bad_rows_trn]
    val_X_scaled_and_cleaned = val_X_scaled[~bad_rows_val]

    # Set X
    trn_X = trn_X_scaled_and_cleaned.astype(np.float32)
    val_X = val_X_scaled_and_cleaned.astype(np.float32)

    # Scale y data
    scaler_y = StandardScaler()
    trn_y = np.nan_to_num(scaler_y.fit_transform(trn_y_raw), nan=0.0).astype(np.float32)
    val_y = np.nan_to_num(scaler_y.transform(val_y_raw), nan=0.0).astype(np.float32)
    # endregion
    
    row_maxes = trn_y_raw.max(axis=1)
    outlier_mask = row_maxes > np.percentile(row_maxes, 99)

    n_features = trn_X.shape[1]
    n_wavelengths = trn_y.shape[1]

    end_time_a = time.perf_counter()
    
    # Print out all the checks:
    print(f"NaNs in trn_X:--------------------------{np.isnan(trn_X).sum()}")
    print(f"NaNs in trn_y:**************************{np.isnan(trn_y).sum()}")
    print(f"NaNs in val_X:--------------------------{np.isnan(val_X).sum()}")
    print(f"NaNs in val_y:**************************{np.isnan(val_y).sum()}")
    print(f"trn_y min/max:--------------------------{trn_y.min():.4f} / {trn_y.max():.4f}")
    print(f"trn_X min/max:**************************{trn_X.min():.4f} / {trn_X.max():.4f}")
    print(f'Training features shape:----------------{trn_X.shape}')
    print(f'Validation features shape:**************{val_X.shape}')
    print(f'Training targets shape:-----------------{trn_y.shape}')
    print(f'Validation targets shape:***************{val_y.shape}')
    print(f'Top 10 max intensities per spectrum:----{np.sort(row_maxes)[-10:]}')
    print(f'Spectra with extreme max values:********{outlier_mask.sum()} / {len(trn_y_raw)}')
    print(f'n_features:-----------------------------{n_features}')
    print(f'n_wavelengths:**************************{n_wavelengths}')
    print(f'Training setup time:--------------------{(end_time_a - start_time_a)} seconds')


    model = LIBS_MLP_001(
        n_features = n_features,
        hidden_dims = (512, 1024, 1024, 512),
        dropout = 0.3,
        n_wavelengths = len(targets)
    )


    start_time_b = time.perf_counter()
    training_model, history = init_xpu_mlp_trainer(
        model=model,
        X_train=trn_X,
        y_train=trn_y,
        X_val=val_X,
        y_val=val_y,
        max_epochs=300,
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
        save_path= r"G:\My Drive\RLSL\Python\LIBS\MLP_trn_Hist"
    )

    end_time_b = time.perf_counter()
    print(f'Training time: {(end_time_b - start_time_b)/60} minutes')

    end_time = time.perf_counter()
    print(f'Total training time: {(end_time - start_time) / 60} minutes')

    comp_cols = [c for c in trn_data.columns if 'conc_' in c or 'frac_' in c]
    print(trn_data[comp_cols].drop_duplicates().shape[0], 'unique compositions')

    sys.stdout.close()
    sys.stdout = sys.__stdout__