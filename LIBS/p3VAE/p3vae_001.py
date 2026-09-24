from __future__ import annotations

"""
jvlee_LIBS_ML > LIBS > p3VAE > p3vae_001.py
"""

# region Imports
import torch
import h5py
import time
import sys

import torch.nn as nn
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from typing import cast, Any, List
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from setup import add_project_root_to_path
add_project_root_to_path(parent_generation=1)
from utils import (
    # Logger, 
    # gen_speak,
    # get_worker_logger,
    # log,
    # init_gpu_trainer,
    # plot_residual_history,
    # load_prepped_training_dataset,
    # load_scalers,
    load_h5_base_dataset,
    save_merged_h5_dataset
)
# endregion

# region Global Vars
# define spectral grid      TODO use utils
wl_grid = torch.linspace(200.0, 1000.0, 2000)


max_epochs= 500
batch_size= 256
learning_rate= 1e-03
weight_decay= 1e-04

# region Default Columns
ALL_COLUMNS = [
    'frac_LiCl',        'frac_KCl',         'conc_Ce_wt%',
    'conc_CeCl3_wt%',   'conc_Ca_wt%',      'conc_CaCl3_wt%',
    'conc_U_wt%',       'conc_UCl3_wt%',    'conc_Sm_wt%',
    'conc_SmCl3_wt%',   'conc_Gd_wt%',      'conc_GdCl3_wt%',
    'conc_La_wt%',      'conc_LaCl3_wt%',   'conc_Mg_wt%',
    'conc_MgCl2_wt%',   'conc_H2o_wt%',     'conc_Nd_wt%',
    'conc_NdCl3_wt%',   'conc_CsCl_wt%',    'conc_SrCl2_wt%',
    'conc_BaCl2_wt%',   'conc_YCl3_wt%',    'conc_FeCl2_wt%',
    'conc_CrCl2_wt%',   'conc_NiCl2_wt%',   'conc_MnCl2_wt%',
    'temperature_C',    'scan_rate_mVs',    'technique',
    'file_path',        'og_path',          'state_aerosol',
    'state_molten',     'state_solid',      'delay_study',
    'delay',            'width_study',      'width',
    'energy_study',     'energy',           'qdelay_study',
    'qdelay',           'shot_study',       'shots',
    'flow_study',       'flow',             'pressure_study',
    'pressure',         'test_snr_study',   'test_snr',
    'static_',          'blank',            'kinetic',
    'repetition'
]
# endregion
# extract core line wavelengths and transition strengtions form nist
# ie. just use the data that I have already pulled down? TODO figure out if that data will work
#  
# endregion

# region Classes
class PhysicsSpectralDecoder(nn.Module):
    def __init__(self, wl_grid, nist_wls, nist_intens):
        super().__init__()
        # register fixed grid and NSIT constants (not trainable params)
        self.register_buffer('grid', wl_grid.unsqueeze(0))  # shape: (1, N_channels)
        self.register_buffer('line_centers', nist_wls.unsqueeze(0)) # shape: (1, N_channels)
        self.register_buffer('line_base_amps', nist_intens.unsqueeze(0))

    def forward(self, concentrations, broadening):
        """
        concentrations: Tensor of shape (batch_size, N_lines)
        broadening: Tensor of shape (batch_size, N_lines) -> Stark/instrumental HWHM
        """
        # shape expansion for vectorization: (batch_size, N_lines, N_channels)
        grid = self.grid.unsqueeze(1)
        centers = self.line_centers.unsqueeze(2)
        gammas = broadening.unsqueeze(2)

        # Effective peak amplitudes scaled by concentrations
        amps = (concentrations * self.line_base_amps).unsqueeze(2)

        # Differentiable Lorentzian line profile calculation (capturing Stark broadening)
        # Profile = Amp * (gamma / pi) / ((labda - center)^2 + gamma^2)
        profiles = amps * (gammas / np.pi) / ((grid - centers)**2 + gammas**2)

        # sum all line contributions across the wavelength grid
        ideal_spectrum = torch.sum(profiles, dim=1)
        return ideal_spectrum

class PhysicsInformedVAE(nn.Module):
    def __init__(
            self, 
            input_dim, 
            n_lines, 
            wl_grid, 
            nist_wls, 
            nist_intens):
        super().__init__()
        self.n_lines = n_lines

        # 1. Endcoder: Maps input spectrum to Latent params
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Latent heads: splits into physical params + matrix residual latents
        self.fc_mean_phys = nn.Linear(128, n_lines * 2) # [Concentrations, Broadening]
        self.fc_logvar_phys = nn.Linear(128, n_lines * 2)

        self.fc_mean_matrix = nn.Linear(128, 8) # Nuisance latent vector (8-dim)
        self.fc_logvar_matrix = nn.Linear(128, 8)

        # 2. Physics Decoder layer
        self.physics_decoder = PhysicsSpectralDecoder(
            wl_grid=wl_grid,
            nist_wls=nist_wls,
            nist_intens=nist_intens
        )

        # 3. Neural Matrix Residual Decoder (Learns baseline shift and liquid interaction)
        self.matrix_decoder = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid() # Keeps baseline scale constrained
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)

        # Sample Physical Latents
        mu_p, logvar_p = self.fc_mean_phys(h), self.fc_logvar_phys(h)
        z_phys = self.reparameterize(mu_p, logvar_p)

        # Split physical outputs (ensuring positive values via Softplus)
        concentrations = nn.functional.softplus(z_phys[:, :self.n_lines])
        broadening = nn.functional.softplus(z_phys[:, self.n_lines:])

        # Sample Matrix Latents
        mu_m, logvar_m = self.fc_mean_matrix(h), self.fc_logvar_matrix(h)
        z_matrix = self.reparameterize(mu_m, logvar_m)

        # Generate components
        ideal_physics_spectrum = self.physics_decoder(concentrations, broadening)
        matrix_residual = self.matrix_decoder(z_matrix)

        # Final Spectrum = Base Physics + Learned Matrix Residual
        reconstructed_spectrum = ideal_physics_spectrum + matrix_residual

        return reconstructed_spectrum, mu_p, logvar_p, mu_m, logvar_m

class MoltenSaltHDF5Dataset(Dataset):
    def __init__(self, h5_file_path, split='X_trn'):
        super().__init__()
        self.h5_path = str(h5_file_path)
        self.split = split

        # Read dataset lengths without keeping file open continuously
        with h5py.File(self.h5_path, 'r') as hf:
            obj = hf[self.split]
            if isinstance(obj, h5py.Dataset):
                self.length = obj.shape[0]
            else:
                raise TypeError(f"'{self.split}' is an HDF5 Group, not a Dataset")

    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        # Open per-worker to prevent hdf5 thread collisions
        with h5py.File(self.h5_path, 'r') as hf:
            dataset: Any = hf[self.split]
            # Load spectrum (reconstructed by VAE)
            spectrum = dataset[idx]

            # Remove redundant channel dimension if shape is (1, N_channels)
            if spectrum.ndim == 2 and spectrum.shape[0] == 1:
                spectrum = spectrum.squeeze(0)

            # Load corresponding physical labels (e.g., concentrations)
            # Assuming y_trn holds target concentrations
            label_key = 'y_trn' if 'trn' in self.split else 'y_val'
            keys: Any = hf[label_key]
            conc_labels = keys[idx]

        return torch.tensor(spectrum, dtype=torch.float32), torch.tensor(conc_labels, dtype=torch.float32)

# endregion

# region Functions
def combine_h5_files():
    # Load all three datasets
    expr_X_raw, expr_y_raw, expr_concentration_cols, expr_wavelengths, expr_meta = load_h5_base_dataset(expr_h5_path, allowed_cols=ALL_COLUMNS)
    nist2_X_raw, nist2_y_raw, nist2_concentration_cols, nist2_wavelengths, nist2_meta = load_h5_base_dataset(nist2_h5_path, allowed_cols=ALL_COLUMNS)
    nist3_X_raw, nist3_y_raw, nist3_concentration_cols, nist3_wavelengths, nist3_meta = load_h5_base_dataset(nist3_h5_path, allowed_cols=ALL_COLUMNS)

    expr_meta["data_source"] = "experimental"
    expr_meta["is_experimental"] = 1
    nist2_meta["data_source"] = "nist2"
    nist2_meta["is_experimental"] = 0
    nist3_meta["data_source"] = "nist3"
    nist3_meta["is_experimental"] = 0

    # 1. Validate Feature Space Alignment (surviving_cols)
    if not (list(expr_concentration_cols) == list(nist2_concentration_cols) == list(nist3_concentration_cols)):
        raise KeyError(
            "Feature space mismatch! `surviving_cols` differ between datasets.\n"
            f"Experimental cols ({len(expr_concentration_cols)}): {expr_concentration_cols}\n"
            f"NIST2 cols ({len(nist2_concentration_cols)}): {nist2_concentration_cols}\n"
            f"NIST3 cols ({len(nist3_concentration_cols)}): {nist3_concentration_cols}"
        )
    surviving_cols = expr_concentration_cols

    # Wavelengths check
    wls = [expr_wavelengths, nist2_wavelengths, nist3_wavelengths]

    if any(w is None for w in wls):
        missing = [
            name
            for name, w in zip(["expr", "nist2", "nist3"], wls)
            if w is None
        ]
        raise KeyError(f"Wavelengths missing from dataset(s): {missing}")

    # Assert for type checker (narrows np.ndarray | None -> np.ndarray)
    assert expr_wavelengths is not None
    assert nist2_wavelengths is not None
    assert nist3_wavelengths is not None

    if not (
        np.array_equal(expr_wavelengths, nist2_wavelengths)
        and np.array_equal(expr_wavelengths, nist3_wavelengths)
    ):
        raise KeyError(
            "Wavelength grids do not match across datasets.\n"
            f"Experimental shape ({expr_wavelengths.shape}): {expr_wavelengths}\n"
            f"NIST2 shape ({nist2_wavelengths.shape}): {nist2_wavelengths}\n"
            f"NIST3 shape ({nist3_wavelengths.shape}): {nist3_wavelengths}"
        )

    wavelengths = expr_wavelengths

    # 2. Combine Arrays (stacking along samples axis=0)
    X_raw = np.concatenate([expr_X_raw, nist2_X_raw, nist3_X_raw])
    y_raw = np.concatenate([expr_y_raw, nist2_y_raw, nist3_y_raw])

    # 3. Concatenate Validation Metadata into a single DataFrame
    meta_val_df = pd.concat([expr_meta, nist2_meta, nist3_meta], axis=0, ignore_index=True)

    print(f"Merge successful!")
    print(f"Dataset Shape -> Concentrations: {X_raw.shape}, Spectra: {y_raw.shape}")
    print(f"Combined Metadata Rows: {len(meta_val_df)}")

    save_merged_h5_dataset(
        output_h5_path=master_base_path,
        X_raw=X_raw,
        y_raw=y_raw,
        feature_cols=surviving_cols,
        wavelengths=wavelengths,
        meta_df=meta_val_df,
        compression_level=3
    )
# endregion

if __name__ == '__main__':
    print('hi')

    input_dim = 128
    n_lines = 10
    WRK_DIR = Path(__file__).parent.parent.parent.resolve()
    expr_h5_path = WRK_DIR / "LIBS" / "experimental.h5"
    nist2_h5_path = WRK_DIR / "LIBS" / "combined_pairs_LIBS.h5"
    nist3_h5_path = WRK_DIR / "LIBS" / "old_combined_LIBS.h5"           #NOTE: This is just a filler until trios finishes scrubbing 
    master_base_path = WRK_DIR / "LIBS" / "test_master_base.h5"

    # X = concentrations
    # y = spectra

    # combine_h5_files()

    # TODO: split dataset into training and validation 
    # remove file_id.1 and file_path columns, keep file_id and og_path
    base_X_raw, base_y_raw, base_concentration_cols, base_wavelengths, base_meta = load_h5_base_dataset(master_base_path, allowed_cols=ALL_COLUMNS)

    # Identify all "Date and Time:" columns
    date_cols = [c for c in base_meta.columns if c.startswith("Date and Time:")] # type: ignore
    junk_cols = date_cols + ['file_id.1', 'file_path', "?", "Unnamed: 0"] 

    # Define the element-to-chloride mapping
    elem_to_chloride = {
        'conc_Mg_wt%': 'conc_MgCl2_wt%',
        'conc_Nd_wt%': 'conc_NdCl3_wt%',
        'conc_Sm_wt%': 'conc_SmCl3_wt%'
    }

    if isinstance(base_meta, pd.DataFrame):
        print("=== INSPECTING ELEMENT VS CHLORIDE COLUMNS ===\n")
        
        for elem_col, chlor_col in elem_to_chloride.items():
            if elem_col not in base_meta.columns or chlor_col not in base_meta.columns:
                continue
                
            # Treat NaN, 0, or empty string as missing/empty
            elem_valid = base_meta[elem_col].notna() & (base_meta[elem_col] != 0) & (base_meta[elem_col] != "")
            chlor_valid = base_meta[chlor_col].notna() & (base_meta[chlor_col] != 0) & (base_meta[chlor_col] != "")
            
            n_elem_only = (elem_valid & ~chlor_valid).sum()
            n_chlor_only = (~elem_valid & chlor_valid).sum()
            n_both = (elem_valid & chlor_valid).sum()
            
            print(f"--- {elem_col} -> {chlor_col} ---")
            print(f"  Valid in {elem_col} ONLY: {n_elem_only}")
            print(f"  Valid in {chlor_col} ONLY: {n_chlor_only}")
            print(f"  Valid in BOTH: {n_both}")
            
            # Merge data: fill empty chloride values using non-empty element values
            base_meta[chlor_col] = base_meta[chlor_col].where(chlor_valid, base_meta[elem_col])
            print(f"  -> Merged non-zero {elem_col} values into {chlor_col}.\n")

            
        base_meta = base_meta.drop(columns=junk_cols, errors='ignore')
        blank_idx = base_meta.columns.get_loc('blank')
        additional_cols_to_drop = base_meta.columns[:blank_idx]
        base_meta = base_meta.drop(columns=additional_cols_to_drop, errors='ignore')
        concentration_cols_to_drop = ['conc_Ca_wt%', 'conc_CeN_wt%', 'conc_Ce_wt%', 'conc_Gd_wt%', 'conc_H2o_wt%', 
                              'conc_La_wt%', 'conc_Mg_wt%', 'conc_Nd_wt%', 'conc_Sm_wt%', 'conc_U_wt%']
        base_meta = base_meta.drop(columns=concentration_cols_to_drop, errors='ignore')

        # List all 219 dataset names inside metadata
        field_names = base_meta.columns.tolist()
        print("Available metadata fields (first 10):", field_names[:])
        print("Total fields:", len(field_names))






    # region Archive
        # if isinstance(base_meta, pd.DataFrame):
    #     base_meta = base_meta.drop(columns=junk_cols, errors='ignore')
    #     blank_idx = base_meta.columns.get_loc('blank')
    #     additional_cols_to_drop = base_meta.columns[:blank_idx]
    #     base_meta = base_meta.drop(columns=additional_cols_to_drop, errors='ignore')
    #     # concentration_cols_to_drop = ['conc_Ca_wt%', 'conc_CeN_wt%', 'conc_Ce_wt%', 'conc_Gd_wt%', 'conc_H2o_wt%', 
    #     #                       'conc_La_wt%', 'conc_Mg_wt%', 'conc_Nd_wt%', 'conc_Sm_wt%', 'conc_U_wt%']

    # # List all 219 dataset names inside metadata
    # field_names = base_meta.columns.tolist()
    # print("Available metadata fields (first 10):", field_names[:])
    # print("Total fields:", len(field_names))

    # junk_cols = ['conc_Ca_wt%', 'conc_CeN_wt%', 'conc_Ce_wt%', 'conc_Gd_wt%', 'conc_H2o_wt%', 
    #                           'conc_La_wt%', 'conc_Mg_wt%', 'conc_Nd_wt%', 'conc_Sm_wt%', 'conc_U_wt%']

    # print(f"Found {len(junk_cols)} junk columns.\n")

    # date_summary = []
    # for col in junk_cols:
    #     vals = base_meta[col] # type: ignore

    #     # Handle missing, null, 0, or NaN entries
    #     is_null_or_zero = vals.isna() | (vals == 0) | (vals == "0") | (vals == "")
    #     zero_count = is_null_or_zero.sum()
    #     total_count = len(vals)
    #     pct_empty = (zero_count / total_count) * 100

    #     date_summary.append(
    #         {
    #             "column": col,
    #             "empty_count": zero_count,
    #             "total": total_count,
    #             "pct_empty": pct_empty,
    #         }
    #     )

    # # Summary table
    # df_summary = pd.DataFrame(date_summary)
    # print(df_summary[["column", "empty_count", "total", "pct_empty"]])

    # # Quick decision check
    # completely_empty = df_summary[df_summary["pct_empty"] == 100]
    # print(
    #     f"\nTotal 'Date and Time' columns that are 100% empty/zero: {len(completely_empty)} / {len(junk_cols)}"
    # )

    # endregion 