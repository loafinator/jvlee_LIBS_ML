from __future__ import annotations

"""
jvlee_LIBS_ML > utils > data_prep.py

This is the labeling workhorse module. Contains specialized functions called in pipeline 
scripts (e.g., 'get_libs.py') to parse file metadata, clean spectral lines, align wavelength
grids, map stoichiometry to element weight percentages, and standardize datasets into 
HDF5 and CSV persistence formats.

Extraction and labeling are driven by regular expression pattern matching linked to hardcoded
concentration maps, stoichiometric conversion tables, and standard experimental defaults.
"""

print('data_prep.py loading ...')

# =============================================================================
# region: Imports
# =============================================================================
# Standard Library Imports
import re 
import logging
import os
import warnings
import h5py
import pickle

# Third-Party Scientific & Data Processing Imports
import pandas as pd 
import numpy as np
import multiprocessing as mp

# Typing, Concurrency, and Functional Utilities
from pathlib import Path 
from typing import List, Optional, cast, Any
from tqdm import tqdm
from h5py import Group, Dataset
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from logging import handlers
from collections.abc import Container
from numpy.typing import NDArray
# endregion

# =============================================================================
# region: Warnings & Global Runtime Configurations
# =============================================================================
# Suppress performance warnings caused by step-by-step DataFrame modifications (fragmentation)
warnings.filterwarnings('ignore', message='DataFrame is highly fragmented', category=pd.errors.PerformanceWarning)

# Global logging queue handle used across worker processes during multiprocessing execution
_worker_log_q = None

# Standard spectral wavelength sampling grid parameters across target detectors
TARGET_MIN_WL: float = 250.0   # Minimum target wavelength in nanometers (nm)
TARGET_MAX_WL: float = 1000.0  # Maximum target wavelength in nanometers (nm)
TARGET_N_PTS: int = 10000       # Target resolution (number of interpolated points)

# Pre-computed linear wavelength vector for uniform resampling across raw spectral files
target_grid: NDArray[np.float64] = np.linspace(TARGET_MIN_WL, TARGET_MAX_WL, TARGET_N_PTS)

# endregion

# =============================================================================
# region: Lookups & Mapping Dictionaries
# =============================================================================

# Map individual element symbols to their target output weight percentage column names
_SINGLES_CONC_MAP: dict[str, str] = {
    'Ba': 'conc_Ba_wt%',
    'Ca': 'conc_Ca_wt%',
    'Ce': 'conc_Ce_wt%',
    'Cr': 'conc_Cr_wt%',
    'Cs': 'conc_Cs_wt%',
    'Gd': 'conc_Gd_wt%',
    'Fe': 'conc_Fe_wt%',
    'La': 'conc_La_wt%',
    'Mg': 'conc_Mg_wt%',
    'Mn': 'conc_Mn_wt%',
    'Nd': 'conc_Nd_wt%',
    'Ni': 'conc_Ni_wt%',
    'Sm': 'conc_Sm_wt%',
    'Sr': 'conc_Sr_wt%',
    'U':  'conc_U_wt%',
    'Y':  'conc_Y_wt%',
    'Cl': 'conc_Cl_wt%',
    'K':  'conc_K_wt%',
    'Li': 'conc_Li_wt%',
}

# Lookup table mapping MS-series sample identifiers to CeCl3 concentrations (wt%)
_MS_CONC_MAP: dict[str, float] = {
    'ms4': 3.0,
    'ms5': 3.0,
    'ms6': 0.0,
    'ms8': 0.1,
    'ms10': 1.0,
    'ms11': 1.0,
    'ms12': 3.0,
    'ms14': 5.0,
    'ms16': 0.5,
}

# Lookup table mapping SampleU-series keys to concentrations: [UCl3 wt%, GdCl3 wt%]
_SAMPLE_U_MAP: dict[str, list[float]] = {
    'sampleu1' : [0.792, 0],
    'sampleu2' : [1.945, 0],
    'sampleu25' : [2.184, 0],       # Represents sample 2.5
    'sampleu3' : [2.868, 0],
    'sampleu4' : [3.767, 0],
    'sampleu5' : [4.733, 0],
    'sampleug1' : [0.831, 0.935],
    'sampleug2' : [1.748, 1.748],
    'sampleug3' : [2.765, 2.776],
    'sampleug4' : [3.978, 3.709],
    'sampleug5' : [5.017, 4.603],
    'sampleug6' : [2.879, 0.941],
    'sampleug7' : [3.852, 1.789],
    'sampleug8' : [0.836, 2.745],
    'sampleug9' : [2.782, 4.576],
    'sampleug10' : [1.878, 3.676],
    'sampleug11' : [8.739, 2.966],
    'sampleug12' : [4.879, 0.922],
    'sampleug13' : [0.814, 4.470],
}

# Lookup table mapping SL-series sample codes to SmCl3 concentration (wt%)
_SL___MAP: dict[str, float] = {
    'sla' : 1.04,
    'slb' : 2.88,
    'slc' : 6.40,
    'sld' : 7.69
}

# Lookup table mapping SampleMg-series keys to concentrations: [U wt%, Mg wt%]
_SAMPLE_MG_MAP: dict[str, list[float]] = {
    'samplem1_t1' : [0, 0.1],
}

# Lookup table mapping Sample1-T10 keys to concentrations: [SmCl3 wt%, GdCl3 wt%]
_SAMPLE_1_THRU_T10_MAP: dict[str, list[float]] = {
    'sample1_t1' : [0, 0.895],
    'sample2_t1' : [0, 1.845],
    'sample3_t1' : [0, 2.839],
    'sample4_t1' : [0, 3.702],
    'sample5_t1' : [0, 4.924],
    'sampleT1_t1' : [1.850, 0.913],
    'sampleT2_t1' : [6.164, 3.006],
    'sampleT3_t1' : [3.869, 3.707],
    'sampleT4_t1' : [1.958, 3.758],
    'sampleT5_t1' : [8.498, 4.045],
    'sampleT6_t1' : [8.535, 1.086],
    'sampleT7_t1' : [6.337, 2.071],
    'sampleT8_t1' : [4.101, 2.916],
    'sampleT9_t1' : [3.958, 1.919],
    'sampleT10_t1' : [4.064, 1.035],
}

# Lookup table mapping Keith-series experiment replicate strings to: [CeCl3 wt%, GdCl3 wt%]
_KEITH_CECL3_MAP: dict[str, list[float]] = {
    '1 rep ' : [0.288, 0.269],
    '2 rep ' : [0.294, 0.565],
    '3 rep ' : [0.327, 1.247],
    '4 rep ' : [0.306, 1.744],
    '5 rep ' : [0.287, 2.309],
    '6 rep ' : [0.318, 3.138],
    '7 rep ' : [0.674, 0.370],
    '8 rep ' : [0.674, 0.632],
    '9 rep ' : [0.556, 1.088],
    '10 rep ' : [0.580, 1.657],
    '11 rep ' : [0.592, 2.244],
    '12 rep ' : [0.586, 2.779],
    '13 rep ' : [1.138, 0.287],
    '14 rep ' : [1.147, 0.540],
    '15 rep ' : [1.153, 1.121],
    '16 rep ' : [1.165, 1.685],
    '17 rep ' : [1.135, 2.217],
    '18 rep ' : [1.146, 2.768],
    '19 rep ' : [1.741, 0.292],
    '20 rep ' : [1.693, 0.541],
    '21 rep ' : [1.671, 1.056],
    '22 rep ' : [1.687, 1.606],
    '23 rep ' : [1.746, 2.219],
    '24 rep ' : [1.738, 2.764],
    '25 rep ' : [2.212, 0.279],
    '26 rep ' : [2.212, 0.527],
    '27 rep ' : [2.273, 1.037],
    '28 rep ' : [2.192, 1.509],
    '29 rep ' : [2.257, 2.179],
    '30 rep ' : [2.284, 2.740],
    '31 rep ' : [2.823, 0.284],
    '32 rep ' : [2.792, 0.539],
    '33 rep ' : [2.819, 1.067],
    '34 rep ' : [2.796, 1.605],
    '35 rep ' : [2.896, 2.277],
    '36 rep ' : [2.817, 2.755],
}

# Lookup table mapping Trial 1 and 2 glovebox file strings to SmCl3 concentration (wt%)
_TRIAL_1_OR_2_MAP: dict[str, float] = {
    '0.5-smcl3' : 0.424,
    '0.75-smcl3' : 0.645,
    '1.0-smcl3' : 0.991,
    '2.0-smcl3' : 1.967,
    '3.0-smcl3' : 2.837,
    '4.0-smcl3' : 3.725,
    '5.0-smcl3' : 4.847,
    '7.0-smcl3' : 6.721,
    '8.0-smcl3' : 7.814,
    '10.0-smcl3' : 8.994
}

# Lookup table mapping solid salt (SSU1) and dissolved salt (DU1) runs to UCl3 concentration (wt%)
_SSU_AND_DU_MAP: dict[str, float] = {
    'ssu1 rep' : 1.0,       
    'du1 rep' : 1.0         # Solid salt vs dissolved salt 1.0 wt% UCl3 baseline
}
# endregion

# =============================================================================
# region: Compiled Regex Patterns for File Parsing
# =============================================================================
_SCAN_PATTERN       = re.compile(r'(\d+(\.\d+)?)\s*mvs')
_TECH_PATTERN       = re.compile(r'(cv|ocv|ca|lp|eis|libs)')
_GTD_PATTERN        = re.compile(r'gtd(\d+\.?\d*)')
_DELAY_PATTERN      = re.compile(r'(?:(?<!Q)delay\D*(\d+\.?\d*)\s?(us|ms|s)?|(\d+\.?\d*)\s?(ns|us|ms|s)?\D*(?<!Q)delay)')
_REP_PATTERN        = re.compile(r'(?<![a-z])(?:rep(?:licate)?|r(?=\d)|run)\s?(\d+)(?:[^a-z]|$)')
_WIDTH_PATTERN      = re.compile(r'(?:width\D*(\d+\.?\d*)\s?(us|ms|s)?|(\d+\.?\d*)\s?(us|ms|s)?\D*width)')
_ENERGY_PATTERN     = re.compile(r'(?:energy|enrgy)\D*(\d+\.?\d*)|(\d+\.?\d*)\s*(?:mj|milli)?\D*(?:energy|enrgy)')
_QDELAY_PATTERN     = re.compile(r'qdelay\D*(\d+\.?\d*)|(\d+\.?\d*)\s*qdelay')
_SHOT_PATTERN       = re.compile(r'(?:(?<!p)shots?\b\D*(\d+)|(\d+)\s*(?<!p)shots?\b)')
_FLOW_PATTERN       = re.compile(r'(\d+[.p]?\d*)\s?(?:mm|units)\sflow')
_PRESSURE_PATTERN   = re.compile(r'(\d+\.?\d*)\s?psi')
_TEMP_PATTERN       = re.compile(r'\b(\d{3})\s*(?:°?c)\b')
_BLANK_PATTERN      = re.compile(r'(blank|pure)')
_STATIC_PATTERN     = re.compile(r'(static)')
_CONC_PATTERN       = re.compile(r'(?:^|[^\w])(\d+\.?\d*)\s*(wt%|ppm|%)')
_NEUP_PATTERN       = re.compile(r'neup')

# Standardized experimental defaults applied when metadata parameters are unlisted
_STANDARD_DELAY: int = 14     # Default gate delay (µs)
_STANDARD_WIDTH: int = 8      # Default gate width (µs)
_STANDARD_QDELAY: int = 110   # Default Q-switch delay (µs)
_STANDARD_SHOTS: int = 50     # Default laser shot count per specimen
# endregion

# =============================================================================
# region: Standard Target Columns & Chemical Species Mappings
# =============================================================================
_DEFAULT_CONC_COLS: List[str] = [
    'frac_LiCl',        'frac_KCl', 
    'conc_Ba_wt%',      'conc_Ca_wt%',      'conc_Ce_wt%',      'conc_Cl_wt%',
    'conc_Cr_wt%',      'conc_Cs_wt%',      'conc_Fe_wt%',      'conc_Gd_wt%',
    'conc_K_wt%',       'conc_La_wt%',      'conc_Li_wt%',      'conc_Mg_wt%',
    'conc_Mn_wt%',      'conc_Nd_wt%',      'conc_Ni_wt%',      'conc_Sm_wt%',
    'conc_Sr_wt%',      'conc_U_wt%',       'conc_Y_wt%',

    'conc_CeCl3_wt%',   'conc_CaCl3_wt%',   'conc_UCl3_wt%',    'conc_SmCl3_wt%',
    'conc_GdCl3_wt%',   'conc_LaCl3_wt%',   'conc_MgCl2_wt%',   'conc_H2o_wt%',
    'conc_NdCl3_wt%',   'conc_CsCl_wt%',    'conc_SrCl2_wt%',   'conc_BaCl2_wt%',
    'conc_YCl3_wt%',    'conc_FeCl2_wt%',   'conc_CrCl2_wt%',   'conc_NiCl2_wt%',
    'conc_MnCl2_wt%',
]

# Canonical species name mapping to enforce consistent casing and structural naming
_SPECIES_MAP: dict[str, str] = {
    'CECL3': 'CeCl3',       'CE': 'Ce',
    'CACL3': 'CaCl3',       'CA': 'Ca',
    'UCL3':  'UCl3',        'U':  'U',
    'SMCL3': 'SmCl3',       'SM': 'Sm',
    'GDCL3': 'GdCl3',       'GD': 'Gd',
    'LACL3': 'LaCl3',       'LACL': 'LaCl3',        'LA': 'La',
    'MGCL2': 'MgCl2',       'MG': 'Mg',
    'NDCL3': 'NdCl3',       'ND': 'Nd',
    'cerium':    'Ce',
    'gadolinium': 'Gd',
    'water':     'H2o',
    'CEEN':      'CeN'
}

_DEFAULT_REQ_COLS: List[str] = list(_DEFAULT_CONC_COLS) + [
    'temperature_C', 
    'scan_rate_mVs', 
    'technique', 
    'file_path',
    'og_path'
]

_DEFAULT_SALT_STATES: List[str] = [
    'state_aerosol', 
    'state_molten', 
    'state_solid'
]

_DEFAULT_EXP_VARS_COLS: List[str] = [
    'delay_study', 'delay',
    'width_study', 'width',
    'energy_study', 'energy', 
    'qdelay_study', 'qdelay',
    'shot_study', 'shots',
    'flow_study', 'flow', 
    'pressure_study', 'pressure', 
    'test_snr_study', 'test_snr', 
    'static_', 
    'blank',
    'kinetic',
    'repetition'
]

_ALL_COLUMNS: list[str] = list(
    dict.fromkeys(
        _DEFAULT_CONC_COLS
        + _DEFAULT_REQ_COLS
        + _DEFAULT_SALT_STATES
        + _DEFAULT_EXP_VARS_COLS
    )
)

# Dynamically compile regular expressions for matching specific chemical species in string filenames
_COMPILED_SPECIES_PATTERNS: dict[re.Pattern, str] = {
    re.compile(rf'(?i)(?:^|[^\w.])(3quarters|half|quarter|\d+\.?\d*)'
                rf'\s*(?:wt%?|%|wt_)?\s*[-_ ]?{re.escape(key.lower())}'
                rf'(?=[_\s.\-/\\]|$)'
                ): nice_name
    for key, nice_name in _SPECIES_MAP.items()
}
# endregion

# =============================================================================
# region: Core File Processing and Cleaning Functions
# =============================================================================

def clean_single_technique_file(
        path: Path | str,
        technique: str = 'cv',
        drop_columns: Optional[List[str]] = None,
        rename_columns: Optional[dict] = None,
        required_dict: Optional[dict] = None,
        required_columns: Optional[List[str]] = None,
        min_wavelength: Optional[float] = None,
        max_wavelength: Optional[float] = None,
        log_path: Path | None = None,
) -> Optional[pd.DataFrame]:
    """ Clean and validate a single CSV file representing a specific diagnostic technique.

    Loads raw CSV spectra or electrochemistry data, confirms the presence of mandatory headers, 
    trims spectral wavelength columns to a specific bounding range, drops unwanted variables, 
    casts data types to numeric, and attaches a temporary file identifier.

    Parameters:
        path (Path | str): Filepath of the target CSV file.
        technique (str): Diagnostic method identifier (e.g., 'cv', 'ocv', 'ca', 'libs').
        drop_columns (Optional[List[str]]): Column names to explicitly drop from the DataFrame.
        rename_columns (Optional[dict]): Mapping dictionary for standardizing header names.
        required_dict (Optional[dict]): Mapping of technique identifiers to expected columns.
        required_columns (Optional[List[str]]): Specific columns that must exist in the file.
        min_wavelength (Optional[float]): Minimum wavelength limit for trimming spectral columns.
        max_wavelength (Optional[float]): Maximum wavelength limit for trimming spectral columns.
        log_path (Path | None): Custom file path destination for logging process output.

    Returns:
        Optional[pd.DataFrame]: Standardized DataFrame, or None if validation fails.
    """
    path = Path(path)
    sanitized_path = sanitize_path(path=path, log_path=log_path)
    if sanitized_path is None:
        return None
    path = cast(Path, sanitized_path)

    # Configure isolated process logger
    if log_path is None:
        log_path = Path(r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\default_log.txt").resolve()
    logger = get_worker_logger(Path(log_path).stem)

    # Establish fallback required dictionary for electrochemical techniques
    if required_dict is None:
        required_dict = {
            'cv': ['<I>/mA', '(Q-Qo)/C'],
            'ocv': ['time/s', 'Ewe/V'],
            'ca': ['time/s', 'Ewe/V', 'I/mA'],
            'lp': ['<I>/mA', 'time/s', 'Ewe/V']
        }

    # Determine structural constraints based on diagnostic technique
    if required_columns is None:
        if technique is None or technique.lower() == 'libs':
            required_columns = []
        else:
            required_columns = required_dict.get(technique.lower(), ['Ewe/V'])

    required_columns = required_columns or []

    try:
        # Step 1: Validate file header presence without reading the full body
        headers = pd.read_csv(long_path(path), nrows=0).columns.tolist()

        missing = [col for col in required_columns if col not in headers]
        if missing:
            log(logger=logger, msg=f"         Skipped (missing required columns): {path.name}")
            log(logger=logger, msg=f'                 {missing}')
            return None

        # Step 2: Read complete file into memory
        df = pd.read_csv(long_path(path), delimiter=',', header=0, skiprows=0, low_memory=False)
        log(logger=logger, msg=f"     Loaded: {path.name}  |  shape={df.shape}")

        # Step 3: Spectral Wavelength Trimming (if bound limits are defined)
        if min_wavelength is not None and max_wavelength is not None:
            cols = df.columns.astype(str).str.strip().tolist()
            numeric_cols = pd.to_numeric(cols, errors='coerce')
            numeric_arr = pd.Series(numeric_cols).to_numpy(dtype=float, na_value=np.nan)
            
            mask = ((numeric_arr >= min_wavelength) & (numeric_arr <= max_wavelength))
            non_wl_mask = np.isnan(numeric_arr)  # Keep non-spectral metadata columns intact
            df = df.loc[:, mask | non_wl_mask]

        df.columns = [str(c) for c in df.columns]
        df = df.loc[:, ~df.columns.str.contains(r'^\s*$|^nan$', case=False, na=True)]

        # Step 4: Drop completely empty rows
        df = df.dropna(how='all')

        # Step 5: Convert non-ID data fields into float numeric formats
        id_cols = {'file_id', 'cycle number', 'loop number', 'Ns', 'half cycles'}
        cols_to_convert = [c for c in df.columns if c not in id_cols and not pd.api.types.is_numeric_dtype(df[c])]
        if cols_to_convert:
            df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

        # Step 6: Column drop and structural renaming
        if drop_columns:
            df = df.drop(columns=drop_columns, errors='ignore')
        if rename_columns:
            actual_rename = {k: v for k, v in rename_columns.items() if k in df.columns}
            df = df.rename(columns=actual_rename)

        df['file_id'] = f"file_{path.stem}"
        log(logger=logger, msg=f"  ✅ Cleaned {path.name}  |  shape={df.shape}")
        return df

    except Exception as e:
        log(logger=logger, msg=f"  ❌ Failed to clean {path.name}: {e}")
        return None

def combine_hdf5_flexible_metadata(
    file1_path: str | Path,
    file2_path: str | Path,
    output_path: str | Path,
    compression: str = 'gzip',
    compression_level: int = 3,
) -> Optional[Path]:
    """ Merge two separate HDF5 spectral datasets into a unified dataset.

    Aligns target spectra shape and wavelength axes, resolving column discrepancies 
    by calculating the union across metadata keys and padding unpopulated records with NaN 
    or empty strings.

    Parameters:
        file1_path (str | Path): Primary source HDF5 file location.
        file2_path (str | Path): Secondary source HDF5 file location.
        output_path (str | Path): Path designation for combined output dataset.
        compression (str): Compression setting for dataset storage ('gzip' or 'lzf').
        compression_level (int): Specific compression density setting for Gzip operations.

    Returns:
        Optional[Path]: Filepath to the merged dataset, or None if validation fails.
    """
    file1_path = Path(file1_path).resolve()
    file2_path = Path(file2_path).resolve()
    output_path = Path(output_path).resolve()

    if not file1_path.exists() or not file2_path.exists():
        print("❌ One or both input HDF5 files do not exist.")
        return None

    print(f"Comparing structures: {file1_path.name} ↔ {file2_path.name}")

    with h5py.File(file1_path, 'r') as hf1, h5py.File(file2_path, 'r') as hf2:
        # Step 1: Structural Integrity Verification
        required_keys = {'spectra', 'wavelengths', 'metadata'}
        if not required_keys.issubset(hf1.keys()) or not required_keys.issubset(hf2.keys()):
            print("❌ Structure Mismatch: One or both files lack required datasets.")
            return None

        # Verify grid compatibility across files
        wl_ds1: Dataset = hf1['wavelengths']  # type: ignore[assignment]
        wl_ds2: Dataset = hf2['wavelengths']  # type: ignore[assignment]

        wl1 = wl_ds1[:]
        wl2 = wl_ds2[:]
        if len(wl1) != len(wl2) or not np.allclose(wl1, wl2, atol=1e-3):
            print("❌ Grid Mismatch: Wavelength axes length or values do not match.")
            return None

        spec_ds1: Dataset = hf1['spectra']  # type: ignore[assignment]
        spec_ds2: Dataset = hf2['spectra']  # type: ignore[assignment]

        spectra1 = spec_ds1[:]
        spectra2 = spec_ds2[:]
        combined_spectra = np.vstack([spectra1, spectra2])

        n_rows1 = spectra1.shape[0]
        n_rows2 = spectra2.shape[0]

        # Step 2: Compute Metadata Key Union
        meta_grp1: Group = hf1['metadata']  # type: ignore[assignment]
        meta_grp2: Group = hf2['metadata']  # type: ignore[assignment]

        meta_cols1 = set(meta_grp1.keys())
        meta_cols2 = set(meta_grp2.keys())
        all_meta_cols = sorted(list(meta_cols1 | meta_cols2))

        print(f"  File 1 Rows: {n_rows1} | File 2 Rows: {n_rows2}")
        print(f"  Merging {len(all_meta_cols)} total metadata columns across union...")

        combined_metadata: dict[str, np.ndarray] = {}

        for col in all_meta_cols:
            arr1: Optional[np.ndarray] = meta_grp1[col][:] if col in meta_cols1 else None  # type: ignore
            arr2: Optional[np.ndarray] = meta_grp2[col][:] if col in meta_cols2 else None  # type: ignore

            ref_arr: np.ndarray = arr1 if arr1 is not None else arr2  # type: ignore
            is_numeric = np.issubdtype(ref_arr.dtype, np.number)

            val1: np.ndarray = arr1 if arr1 is not None else np.full(n_rows1, np.nan if is_numeric else '', dtype=ref_arr.dtype)
            val2: np.ndarray = arr2 if arr2 is not None else np.full(n_rows2, np.nan if is_numeric else '', dtype=ref_arr.dtype)

            combined_metadata[col] = np.concatenate((val1, val2))

        attrs = dict(hf1.attrs)

    # Step 3: Write Combined Dataset to Disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comp_kwargs: dict[str, Any] = {'compression': compression}
    if compression == 'gzip':
        comp_kwargs['compression_opts'] = compression_level

    print(f"Writing combined HDF5 → {output_path.name}")
    with h5py.File(output_path, 'w') as hf_out:
        hf_out.create_dataset(
            'spectra',
            data=combined_spectra,
            chunks=(min(1000, combined_spectra.shape[0]), combined_spectra.shape[1]),
            **comp_kwargs
        )
        hf_out.create_dataset('wavelengths', data=wl1)

        meta_grp_out = hf_out.create_group('metadata')
        for col, data_arr in combined_metadata.items():
            if np.issubdtype(data_arr.dtype, np.number):
                meta_grp_out.create_dataset(col, data=data_arr, **comp_kwargs)
            else:
                dt = h5py.string_dtype(encoding='utf-8')
                meta_grp_out.create_dataset(col, data=data_arr, dtype=dt)

        for k, v in attrs.items():
            hf_out.attrs[k] = v
        hf_out.attrs['n_rows'] = combined_spectra.shape[0]
        hf_out.attrs['metadata_cols'] = list(all_meta_cols)

    size_mb = output_path.stat().st_size / 1_048_576
    print(f"✅ Combine Complete: Saved {output_path.name} | Total Rows: {combined_spectra.shape[0]} | Size: {size_mb:.1f} MB")
    
    return output_path

def parent_concentration_data(
        source_data_paths: List[str | Path],
        composition_columns: Optional[List[str]] = None,
        required_columns: Optional[List[str]] = None,
        salt_states: Optional[List[str]] = None,
        experimental_variation_columns: Optional[List[str]] = None,
        include_experimental_variation_columns: bool = True,
) -> pd.DataFrame:
    """ Extract experimental metadata and chemical composition values from filepaths using regex patterns.

    Parameters:
        source_data_paths (List[str | Path]): File paths to evaluate and parse.
        composition_columns (Optional[List[str]]): Target chemical composition column keys.
        required_columns (Optional[List[str]]): Required dataframe structural columns.
        salt_states (Optional[List[str]]): Chemical phase state indicators.
        experimental_variation_columns (Optional[List[str]]): Experimental parameter headers.
        include_experimental_variation_columns (bool): Toggle for including parameter columns.

    Returns:
        pd.DataFrame: Tabular DataFrame containing extracted metadata for every input path.
    """
    composition_columns = composition_columns or _DEFAULT_CONC_COLS
    required_columns = required_columns or _DEFAULT_REQ_COLS
        
    if salt_states is None:
        salt_states = _DEFAULT_SALT_STATES
        required_columns = required_columns + salt_states
        
    experimental_variation_columns = experimental_variation_columns or _DEFAULT_EXP_VARS_COLS

    if include_experimental_variation_columns:
        required_columns = required_columns + experimental_variation_columns

    rows = []
    for path_str in source_data_paths:
        if isinstance(path_str, Path):
            path_str = str(path_str)
        
        path = Path(path_str)
        all_text = (' '.join(path.parts) + ' ' + path.name).lower()

        data: dict[str, float | str | int | None] = {
            col: 0.0 for col in composition_columns if 'conc_' in col
        }
        data['frac_LiCl'] = 0.59  # Default LiCl-KCl eutectic mole fraction
        data['frac_KCl'] = 0.41
        data['temperature_C'] = None
        data['scan_rate_mVs'] = None
        data['technique'] = None
        data['file_path'] = path_str

        # Step 1: Parse electrochemistry scan rate
        scan_match = _SCAN_PATTERN.search(all_text)
        if scan_match:
            data['scan_rate_mVs'] = float(scan_match.group(1))

        # Step 2: Parse diagnostic technique
        tech_match = _TECH_PATTERN.search(all_text)
        if tech_match:
            data['technique'] = tech_match.group(1).upper()
        else:
            if 'cv' in all_text: data['technique'] = 'CV'
            elif 'ocv' in all_text: data['technique'] = 'OCV'
            elif 'ca' in all_text: data['technique'] = 'CA'
            elif 'lp' in all_text: data['technique'] = 'LP'

        # Step 3: Parse spectrometer gate delay
        gtd_match = _GTD_PATTERN.search(all_text)
        delay_match = _DELAY_PATTERN.search(all_text)
        if gtd_match:
            data['delay_study'] = 1
            data['delay'] = float(gtd_match.group(1))
        elif delay_match:
            data['delay_study'] = 1
            val = float(delay_match.group(1) or delay_match.group(3))
            unit = delay_match.group(2) or delay_match.group(4) or 'us'
            if unit == 's': val *= 1e6
            elif unit == 'ms': val *= 1e3
            elif unit == 'ns': val /= 1e3
            data['delay'] = val
        else:
            data['delay_study'] = 0
            data['delay'] = _STANDARD_DELAY

        # Step 4: Parse experiment replicate index
        rep_match = _REP_PATTERN.search(all_text)
        data['repetition'] = int(rep_match.group(1)) if rep_match else 1

        # Step 5: Parse spectrometer gate width
        width_match = _WIDTH_PATTERN.search(all_text)
        if width_match:
            data['width_study'] = 1
            val = float(width_match.group(1) or width_match.group(3))
            unit = width_match.group(2) or width_match.group(4) or 'us'
            if unit == 's': val *= 1e6
            elif unit == 'ms': val *= 1e3
            data['width'] = val
        else:
            data['width_study'] = 1 if 'width' in all_text else 0
            data['width'] = _STANDARD_WIDTH

        # Step 6: Parse laser energy
        energy_match = _ENERGY_PATTERN.search(all_text)
        neup_match = _NEUP_PATTERN.search(all_text)
        standard_energy = 200 if neup_match else 100

        if energy_match:
            data['energy_study'] = 1
            data['energy'] = float(energy_match.group(1) or energy_match.group(2))
        else:
            data['energy_study'] = 0
            data['energy'] = standard_energy

        # Step 7: Parse laser Q-switch delay
        qdelay_match = _QDELAY_PATTERN.search(all_text)
        if qdelay_match:
            data['qdelay_study'] = 1
            data['qdelay'] = float(qdelay_match.group(1) or qdelay_match.group(2))
        else:
            data['qdelay_study'] = 0
            data['qdelay'] = _STANDARD_QDELAY

        # Step 8: Parse laser shot count
        shot_match = _SHOT_PATTERN.search(all_text)
        if shot_match:
            data['shot_study'] = 1
            data['shots'] = float(shot_match.group(1) or shot_match.group(2))
        else:
            data['shot_study'] = 0
            data['shots'] = _STANDARD_SHOTS

        # Step 9: Parse carrier gas flow rate
        flow_match = _FLOW_PATTERN.search(all_text)
        if flow_match:
            data['flow_study'] = 1
            data['flow'] = float(flow_match.group(1).replace('p', '.'))
        else:
            data['flow_study'] = 0
            data['flow'] = 0.0

        # Step 10: Parse cell pressure
        pressure_match = _PRESSURE_PATTERN.search(all_text)
        if pressure_match:
            data['pressure_study'] = 1
            data['pressure'] = float(pressure_match.group(1))
        else:
            data['pressure_study'] = 0
            data['pressure'] = 0.0

        # Step 11: Identify fixed-spot vs rastered measurements
        data['static_'] = 1 if _STATIC_PATTERN.search(all_text) else 0

        # Step 12: Identify reaction kinetic measurement series
        data['kinetic'] = 1 if 'kinetic' in all_text else 0

        # Step 13: Identify blank baseline measurements
        if _BLANK_PATTERN.search(all_text):
            data['blank'] = 1
            for comp in composition_columns:
                if comp not in ('frac_LiCl', 'frac_KCl'):
                    data[comp] = 0.0
        else:
            data['blank'] = 0

        # Step 14: Determine salt phase state
        data['state_aerosol'] = 1 if 'aerosol' in all_text else 0
        data['state_molten'] = 1 if ('molten' in all_text and 'aerosol' not in all_text) else 0
        data['state_solid'] = 1 if (data['state_aerosol'] == 0 and data['state_molten'] == 0) else 0

        # Step 15: Determine temperature conditions
        temp_match = _TEMP_PATTERN.search(all_text)
        if data['state_aerosol'] == 1 or data['state_molten'] == 1:
            if temp_match:
                data['temperature_C'] = float(temp_match.group(1))
            elif any(s in all_text for s in ['sla_', 'slb_', 'slc_', 'sld_']):
                data['temperature_C'] = 500.0
            elif 'andrewsh' in all_text:
                data['temperature_C'] = 501.0
            else:
                for part in path.parts:
                    if part.isdigit() and 300 < int(part) < 1000:
                        data['temperature_C'] = float(part)
                        break
                else:
                    data['temperature_C'] = 500.0
        else:
            if temp_match:
                temp = float(temp_match.group(1))
                if temp > 360:  # Threshold above LiCl-KCl melting point (352 °C)
                    data['temperature_C'] = temp
                    data['state_molten'] = 1
                    data['state_solid'] = 0
            else:
                data['temperature_C'] = 20.0

        # Step 16: Extract Chemical Concentrations via Casing-Insensitive Parsing
        filename_text = path.name.upper()
        sorted_keys = sorted(_SPECIES_MAP.keys(), key=len, reverse=True)

        for key in sorted_keys:
            nice_name = _SPECIES_MAP[key]
            pattern = re.compile(rf"([0-9.]+)(?:_str|_wt|_wt%|wt%|ppm)?_?{re.escape(key)}(?=[_\s.\-/\\(]|$)", re.IGNORECASE)
            
            match = pattern.search(filename_text)
            if match:
                val_str = match.group(1)
                col = f"conc_{nice_name}_wt%"
                if nice_name == "H2o":
                    col = "conc_H2o_wt%"
                    
                if col in _DEFAULT_CONC_COLS:
                    try:
                        val = float(val_str)
                        context_around = filename_text[max(0, match.start()-5):match.end()+5]
                        if 'PPM' in context_around:
                            val = val / 10000.0  # Convert PPM to wt%
                        data[col] = val
                    except ValueError:
                        continue

        # Step 17: Apply Hardcoded Concentration Mapping Libraries
        if 'wavelength (nm),sum,' in all_text:
            for key, target_col in sorted(_SINGLES_CONC_MAP.items(), key=lambda x: -len(x[0])):
                pattern = rf'wavelength \(nm\),sum,{key.lower()}(?:\s+[ivxlcdm]+)?'
                if re.search(pattern, all_text, re.IGNORECASE):
                    if target_col in composition_columns:
                        data[target_col] = 1.0
                    break

        if re.search(r'(?<![a-z])ms\d', all_text) and not re.search(r'msu\d', all_text):
            av_conc = sum(_MS_CONC_MAP.values()) / len(_MS_CONC_MAP)
            for key, conc in sorted(_MS_CONC_MAP.items(), key=lambda x: -len(x[0])):
                if key in all_text:
                    data['conc_CeCl3_wt%'] = conc
                    break
            else:
                data['conc_CeCl3_wt%'] = av_conc

        if re.search(r'msu\d', all_text):
            if data.get('conc_UCl3_wt%', 0.0) == 0.0:
                data['conc_UCl3_wt%'] = 1.0

        if re.search('sampleu', all_text):
            for key, conc in sorted(_SAMPLE_U_MAP.items(), key=lambda x: -len(x[0])):
                if key in all_text:
                    data['conc_UCl3_wt%'] = conc[0]
                    data['conc_GdCl3_wt%'] = conc[1]
                    break

        if re.search(r'sl[abcd]', all_text):
            for key, conc in sorted(_SL___MAP.items(), key=lambda x: -len(x[0])):
                if key in all_text:
                    data['conc_SmCl3_wt%'] = conc
                    break

        if re.search('samplem', all_text):
            for key, conc in sorted(_SAMPLE_MG_MAP.items(), key=lambda x: -len(x[0])):
                if key in all_text:
                    data['conc_U_wt%'] = conc[0]
                    data['conc_Mg_wt%'] = conc[1]
                    break

        if re.search(r'sample\d+_t\d+|samplet\d+_t\d+', all_text):
            for key, conc in sorted(_SAMPLE_1_THRU_T10_MAP.items(), key=lambda x: -len(x[0])):
                if key.lower() in all_text:
                    data['conc_SmCl3_wt%'] = conc[0]
                    data['conc_GdCl3_wt%'] = conc[1]
                    break

        if re.search('keith', all_text):
            for key, conc in sorted(_KEITH_CECL3_MAP.items(), key=lambda x: -len(x[0])):
                if key in all_text:
                    data['conc_CeCl3_wt%'] = conc[0]
                    data['conc_GdCl3_wt%'] = conc[1]
                    break

        if re.search(r'trial 1|trail 2', all_text):
            for key, conc in sorted(_TRIAL_1_OR_2_MAP.items(), key=lambda x: -len(x[0])):
                if key in all_text:
                    data['conc_SmCl3_wt%'] = conc
                    break

        if re.search(r'ssu1|du1', all_text):
            for key, conc in sorted(_SSU_AND_DU_MAP.items(), key=lambda x: -len(x[0])):
                if key in all_text:
                    data['conc_UCl3_wt%'] = conc
                    break

        rows.append(data)
    
    df = pd.DataFrame(rows, columns=required_columns)
    return df

def recombine_h5_splits(
        h5_path: str | Path
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str], np.ndarray]:
    """ Recombine train and validation dataset splits stored in an HDF5 container into unified arrays. """
    with h5py.File(h5_path, "r") as hf:
        y_train = hf_get(hf, "train/spectra")
        y_val = hf_get(hf, "val/spectra")
        y_combined = np.concatenate([y_train, y_val], axis=0)

        train_meta_grp = hf["train/metadata"]
        assert isinstance(train_meta_grp, h5py.Group)

        all_cols = list(train_meta_grp.keys())
        feature_cols = [c for c in all_cols if np.issubdtype(train_meta_grp[c].dtype, np.number)]  # type: ignore[index]

        X_train = np.stack([hf_get(hf, f"train/metadata/{c}") for c in feature_cols], axis=1).astype(np.float32)
        X_val = np.stack([hf_get(hf, f"val/metadata/{c}") for c in feature_cols], axis=1).astype(np.float32)
        X_combined = np.concatenate([X_train, X_val], axis=0)

        train_meta_dict = {}
        val_meta_dict = {}

        for c in all_cols:
            tr_data = hf_get(hf, f"train/metadata/{c}")
            va_data = hf_get(hf, f"val/metadata/{c}")

            if tr_data.dtype.kind in ("O", "S", "V"):
                tr_data = [x.decode("utf-8") if isinstance(x, bytes) else x for x in tr_data]
                va_data = [x.decode("utf-8") if isinstance(x, bytes) else x for x in va_data]

            train_meta_dict[c] = tr_data
            val_meta_dict[c] = va_data

        train_meta_df = pd.DataFrame(train_meta_dict)
        val_meta_df = pd.DataFrame(val_meta_dict)
        meta_combined_df = pd.concat([train_meta_df, val_meta_df], axis=0, ignore_index=True)

        wavelengths = hf_get(hf, "wavelengths")

    return X_combined, y_combined, meta_combined_df, feature_cols, wavelengths

def remove_nans_h5(
    input_path: Path | str,
    output_path: Path | str,
) -> None:
    """ Filter out samples containing NaN values from an HDF5 dataset. """
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    print(f"Reading from: {input_path}")
    print(f"Writing clean dataset to: {output_path}")

    with h5py.File(input_path, 'r') as hf_in, h5py.File(output_path, 'w') as hf_out:
        for key in hf_in.keys():
            if key not in ['train', 'val', 'test']:
                print(f"Copying root-level item: '{key}'")
                hf_in.copy(key, hf_out)

        for split in ['train', 'val', 'test']:
            if split not in hf_in:
                continue
            
            print(f"\n--- Processing split: '{split}' ---")
            
            spectra_np = hf_in[split]['spectra'][:]     # type: ignore
            targets_np = hf_in[split]['metadata']['elem_comp_wt%'][:]     # type: ignore
            is_synth_np = hf_in[split]['metadata']['is_synthetic'][:]     # type: ignore

            valid_mask = ~np.isnan(spectra_np).any(axis=1) & ~np.isnan(targets_np).any(axis=1)
            dropped = len(spectra_np) - np.sum(valid_mask)     # type: ignore

            if dropped > 0:
                print(f"⚠️ Filtered out {dropped} NaN rows out of {len(spectra_np)} from '{split}'")     # type: ignore
            else:
                print(f"✅ Split '{split}' is clean (0 NaNs found).")

            split_group = hf_out.create_group(split)
            split_group.create_dataset('spectra', data=spectra_np[valid_mask], compression='gzip')      # type: ignore
            
            meta_group = split_group.create_group('metadata')
            meta_group.create_dataset('elem_comp_wt%', data=targets_np[valid_mask], compression='gzip')     # type: ignore
            meta_group.create_dataset('is_synthetic', data=is_synth_np[valid_mask], compression='gzip')     # type: ignore
            
            if 'elem_names' in hf_in[split]['metadata']:     # type: ignore
                meta_group.create_dataset('elem_names', data=hf_in[split]['metadata']['elem_names'][:])     # type: ignore

    print("\n🎉 Preprocessing complete! Clean HDF5 dataset saved successfully.")

def sanitize_path(
        path: Path | str | None = None,
        log_path: Path | None = None,
) -> Optional[Path]:
    """ Clean invalid Windows characters from paths by replacing them with underscores (`_`). """
    if path is None:
        raise FileExistsError('  ❌ No path provided.')
    
    path = Path(path)
    root = path.anchor
    parts = path.relative_to(root).parts

    if log_path is None:
        log_path = Path(r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\default_log.txt").resolve()
    logger = get_worker_logger(Path(log_path).stem)

    sanitized_parts = []
    for part in parts:
        safe = re.sub(r'[<>:"/\\|?*%]', '_', part)
        sanitized_parts.append(safe)

    current = Path(root)
    for original, sanitized in zip(parts, sanitized_parts):
        original_path = current / original
        sanitized_path = current / sanitized
        if original != sanitized and original_path.exists():
            try:
                original_path.rename(sanitized_path)
                log(logger=logger, msg=f"  ✅ Renamed: {original} → {sanitized}")
            except Exception as e:
                log(logger=logger, msg=f"  ❌ Could not rename {original}: {e}")
                return None
        current = sanitized_path

    return current

def verify_and_combine_hdf5(
    file1_path: str | Path,
    file2_path: str | Path,
    output_path: str | Path,
    compression: str = 'gzip',
    compression_level: int = 3,
) -> Optional[Path]:
    """Verifies that two HDF5 files share identical internal structures (wavelength grid,
    dataset keys, and metadata columns) and merges them vertically into a single output HDF5 file.

    Parameters
    ----------
    file1_path : str | Path
        Path to the first source HDF5 file.
    file2_path : str | Path
        Path to the second source HDF5 file.
    output_path : str | Path
        Destination path for the merged output HDF5 file.
    compression : str, default='gzip'
        Compression algorithm for HDF5 dataset creation.
    compression_level : int, default=3
        Gzip compression level (0-9).

    Returns
    -------
    Optional[Path]
        Path to combined HDF5 file if successful, otherwise None.
    """
    file1_path = Path(file1_path).resolve()
    file2_path = Path(file2_path).resolve()
    output_path = Path(output_path).resolve()

    if not file1_path.exists() or not file2_path.exists():
        print("❌ One or both input HDF5 files do not exist.")
        return None

    print(f"Comparing structures: {file1_path.name} ↔ {file2_path.name}")

    with h5py.File(file1_path, 'r') as hf1, h5py.File(file2_path, 'r') as hf2:
        # --- 1. Structure & Schema Verification ---
        required_keys = {'spectra', 'wavelengths', 'metadata'}
        if not required_keys.issubset(hf1.keys()) or not required_keys.issubset(hf2.keys()):
            print("❌ Structure Mismatch: One or both files lack required datasets ('spectra', 'wavelengths', 'metadata').")
            return None

        meta_grp1: Group = hf1['metadata']  # type: ignore[assignment]
        meta_grp2: Group = hf2['metadata']  # type: ignore[assignment]

        meta_cols1 = sorted(list(meta_grp1.keys()))
        meta_cols2 = sorted(list(meta_grp2.keys()))
        if meta_cols1 != meta_cols2:
            print(f"❌ Column Mismatch: Metadata columns do not match.\n File 1: {meta_cols1}\n File 2: {meta_cols2}")
            return None

        wl_ds1: Dataset = hf1['wavelengths']  # type: ignore[assignment]
        wl_ds2: Dataset = hf2['wavelengths']  # type: ignore[assignment]

        wl1 = wl_ds1[:]
        wl2 = wl_ds2[:]
        if len(wl1) != len(wl2) or not np.allclose(wl1, wl2, atol=1e-3):
            print("❌ Grid Mismatch: Wavelength axes length or values do not match.")
            return None

        print("✅ File structures match! Reading and combining datasets...")

        # --- 2. Data Concatenation ---
        spec_ds1: Dataset = hf1['spectra']  # type: ignore[assignment]
        spec_ds2: Dataset = hf2['spectra']  # type: ignore[assignment]

        spectra1 = spec_ds1[:]
        spectra2 = spec_ds2[:]
        combined_spectra = np.vstack([spectra1, spectra2])

        # Concatenate metadata dataset values per column
        combined_metadata = {}
        for col in meta_cols1:
            col_ds1: Dataset = meta_grp1[col]  # type: ignore[assignment]
            col_ds2: Dataset = meta_grp2[col]  # type: ignore[assignment]

            arr1 = col_ds1[:]
            arr2 = col_ds2[:]
            combined_metadata[col] = np.concatenate([arr1, arr2])

        attrs = dict(hf1.attrs)

    # --- 3. Output Creation ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    comp_kwargs: dict[str, Any] = {'compression': compression}
    if compression == 'gzip':
        comp_kwargs['compression_opts'] = compression_level

    print(f"Writing combined HDF5 → {output_path.name}")
    
    with h5py.File(output_path, 'w') as hf_out:
        hf_out.create_dataset(
            'spectra',
            data=combined_spectra,
            chunks=(min(1000, combined_spectra.shape[0]), combined_spectra.shape[1]),
            **comp_kwargs
        )

        hf_out.create_dataset('wavelengths', data=wl1)

        meta_grp_out = hf_out.create_group('metadata')
        for col, data_arr in combined_metadata.items():
            if np.issubdtype(data_arr.dtype, np.number):
                meta_grp_out.create_dataset(col, data=data_arr, **comp_kwargs)
            else:
                dt = h5py.string_dtype(encoding='utf-8')
                meta_grp_out.create_dataset(col, data=data_arr, dtype=dt)

        for k, v in attrs.items():
            hf_out.attrs[k] = v
        hf_out.attrs['n_rows'] = combined_spectra.shape[0]

    size_mb = output_path.stat().st_size / 1_048_576
    print(f"✅ Combine Complete: Saved {output_path.name} | Total Rows: {combined_spectra.shape[0]} | Size: {size_mb:.1f} MB")
    
    return output_path
# endregion

# =============================================================================
# region: Stoichiometric Conversions
# =============================================================================

def convert_to_elemental_comp(
    input_path: str | Path,
    output_path: str | Path,
) -> None:
    """ Convert salt species concentrations (wt%) to elemental concentrations (wt%).

    Parses metal chloride concentrations (e.g., UCl3, GdCl3, LiCl, KCl), applies exact 
    molar mass stoichiometry to extract pure element mass fractions, and rescales the remaining
    mass balance against the carrier salt matrix (LiCl-KCl eutectic) to ensure a 100 wt% balance.

    Parameters:
        input_path (str | Path): Source HDF5 filepath containing chloride concentrations.
        output_path (str | Path): Destination filepath for storing elemental concentrations.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # 1. Chemical stoichiometry map (Chlorine atom count per target metal cation)
    num_cl = {
        'Ba': 2, 'Ca': 2, 'Ce': 3, 'Cr': 3, 'Cs': 1, 'Fe': 2, 
        'Gd': 3, 'K': 1, 'La': 3, 'Li': 1, 'Mg': 2, 'Mn': 2, 
        'Nd': 3, 'Ni': 2, 'Sm': 3, 'Sr': 2, 'U': 3, 'Y': 3
    }

    # 2. IUPAC Standard Atomic Weights (g/mol)
    mw_elements = {
        'Ba': 137.327, 'Ca': 40.078, 'Ce': 140.116, 'Cr': 51.996, 'Cs': 132.905,
        'Fe': 55.845,  'Gd': 157.25,  'La': 138.905, 'Li': 6.941,  'K': 39.098,
        'Mg': 24.305,  'Mn': 54.938,  'Nd': 144.242, 'Ni': 58.693, 'Sm': 150.36,
        'Sr': 87.62,   'U': 238.029,  'Y': 88.906,   'Cl': 35.453
    }

    # Pre-calculate stoichiometric conversion factors for elemental mass fractions
    metal_factors = {}
    cl_factors = {}
    for elem, n_cl in num_cl.items():
        mw_m = mw_elements[elem]
        mw_cl = mw_elements['Cl']
        mw_salt = mw_m + (n_cl * mw_cl)

        metal_factors[elem] = mw_m / mw_salt
        cl_factors[elem] = (n_cl * mw_cl) / mw_salt

    # 3. Read metadata array components from input HDF5 container
    metadata_dict = {}
    with h5py.File(input_path, 'r') as hf:
        metadata_grp = hf['metadata']
        for key in metadata_grp.keys(): # type: ignore
            metadata_dict[key] = metadata_grp[key][:] # type: ignore

    df_raw = pd.DataFrame(metadata_dict)
    
    all_metals = sorted([k for k in num_cl.keys()])
    elemental_df = pd.DataFrame(index=df_raw.index)
    total_cl_wt = np.zeros(len(df_raw), dtype=np.float32)

    # Step A: Convert Dopant and Analyte Chlorides
    dopant_elems = [e for e in all_metals if e not in ('Li', 'K')]
    dopant_salt_wt_total = np.zeros(len(df_raw), dtype=np.float32)

    for elem in dopant_elems:
        possible_keys = [f'frac_{elem}Cl', f'conc_{elem}Cl{num_cl[elem]}_wt%', f'conc_{elem}Cl_wt%']
        matched_key = next((k for k in possible_keys if k in df_raw.columns), None)

        if matched_key:
            chloride_wt = df_raw[matched_key].values.astype(np.float32)

            if chloride_wt.max() <= 1.0: # type: ignore
                chloride_wt = chloride_wt * 100.0 # type: ignore

            dopant_salt_wt_total += chloride_wt # type: ignore
            elemental_df[elem] = chloride_wt * metal_factors[elem]
            total_cl_wt += chloride_wt * cl_factors[elem]
        else:
            elemental_df[elem] = np.zeros(len(df_raw), dtype=np.float32)

    # Step B: Rescale Carrier Base Salt Matrix (e.g., LiCl-KCl Eutectic)
    base_salt_scale = np.maximum(0.0, 100.0 - dopant_salt_wt_total) / 100.0

    # Step C: Convert Carrier Base Salt Components (LiCl, KCl)
    for elem in ['Li', 'K']:
        possible_keys = [f'frac_{elem}Cl', f'conc_{elem}Cl_wt%']
        matched_key = next((k for k in possible_keys if k in df_raw.columns), None)

        if matched_key:
            raw_val = df_raw[matched_key].values.astype(np.float32)
            
            if raw_val.max() <= 1.0: # type: ignore
                raw_val = raw_val * 100.0 # type: ignore

            scaled_salt_wt = raw_val * base_salt_scale # type: ignore
            elemental_df[elem] = scaled_salt_wt * metal_factors[elem]
            total_cl_wt += scaled_salt_wt * cl_factors[elem]
        else:
            elemental_df[elem] = np.zeros(len(df_raw), dtype=np.float32)

    # Re-order columns alphabetically and append total chlorine
    sorted_cols = sorted([c for c in elemental_df.columns if c != 'Cl']) + ['Cl']
    elemental_df['Cl'] = total_cl_wt
    elemental_df = elemental_df[sorted_cols]

    # Step D: Save Converted Datasets into New HDF5 Container
    with h5py.File(input_path, 'r') as hf_in, h5py.File(output_path, 'w') as hf_out:
        for key in hf_in.keys():
            if key != 'metadata':
                hf_in.copy(key, hf_out)

        if 'metadata' in hf_in:
            hf_in.copy('metadata', hf_out)
        meta_grp_out = hf_out['metadata']

        # Remove superseded salt concentration headers
        keys_to_remove = [k for k in meta_grp_out.keys() if k.startswith(('conc_', 'frac_'))] # type: ignore
        for old_key in keys_to_remove:
            del meta_grp_out[old_key] # type: ignore

        element_names = list(elemental_df.columns)
        meta_grp_out.create_dataset( # type: ignore
            'elem_comp_wt%',
            data=elemental_df.values.astype(np.float32),
            compression='gzip'
        )
        meta_grp_out.create_dataset( # type: ignore
            'elem_names',
            data=np.array(element_names, dtype='S')
        )

    print(f"Successfully converted compositions and saved output to {output_path}")
# endregion

# =============================================================================
# region: File Enrichment & Multiprocessing Infrastructure
# =============================================================================

def enrich_file_with_metadata(
        path: str | Path,
        enriched_root: str | Path,
        drop_columns: Optional[List[str]] = None,
        rename_columns: Optional[dict] = None,
        composition_columns: Optional[List[str]] = None,
        technique: str = 'libs',
        allowed_extensions: Optional[List[str]] = None,
        required_columns: Optional[List[str]] = None,
        salt_states: Optional[List[str]] = None,
        experimental_variation_columns: Optional[List[str]] = None,
        include_experimental_variation_columns: bool = True,
        min_wavelength: Optional[float] = None,
        max_wavelength: Optional[float] = None,
        log_path: Path | None = None,
) -> Optional[Path]:
    """ Clean a single spectral CSV file, attach parsed experimental metadata, and output the enriched file.

    Parameters:
        path (str | Path): Raw input spectrum file path.
        enriched_root (str | Path): Destination folder for enriched output CSVs.
        drop_columns (Optional[List[str]]): Target headers to strip from dataset.
        rename_columns (Optional[dict]): Header name mapping dictionary.
        composition_columns (Optional[List[str]]): Specific elemental tracking keys.
        technique (str): Analytical technique label (e.g., 'libs').
        allowed_extensions (Optional[List[str]]): Valid extensions for processing.
        required_columns (Optional[List[str]]): Mandatory columns required for processing.
        salt_states (Optional[List[str]]): Target sample salt phase state options.
        experimental_variation_columns (Optional[List[str]]): Experimental condition fields.
        include_experimental_variation_columns (bool): Toggle for including parametric metadata.
        min_wavelength (Optional[float]): Minimum spectral trimming boundary.
        max_wavelength (Optional[float]): Maximum spectral trimming boundary.
        log_path (Path | None): Custom log file destination.

    Returns:
        Optional[Path]: Enriched file output location, or None if processing fails.
    """
    if log_path is None:
        log_path = Path(r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\default_log.txt").resolve()
    logger = get_worker_logger(Path(log_path).stem)

    og_path = Path(path)
    sanitized_path = sanitize_path(path=og_path, log_path=log_path)
    if sanitized_path is None:
        log(logger=logger, msg=f'          Skipping (could not sanitize path): {og_path.name}')
        return None
    path = sanitized_path

    enriched_root = Path(enriched_root)
    enriched_root.mkdir(parents=True, exist_ok=True)

    file_w_parents = f'{path.name}'    
    enriched_path = enriched_root / file_w_parents

    if enriched_path.exists():
        log(logger=logger, msg=f'          Skipping (already exists): {enriched_path.name}')
        return enriched_path

    if allowed_extensions is None:
        allowed_extensions = ['.csv', '.mat', '.mpr', '.asc']
    
    if path.suffix.lower() not in allowed_extensions:
        log(logger=logger, msg=f'          Skipping (unsupported file type): {path.name}')
        return None

    log(logger=logger, msg=f'Enriching: {enriched_path.name}')

    df = clean_single_technique_file(
        path=path,
        technique=technique or 'libs',
        drop_columns=drop_columns,
        rename_columns=rename_columns,
        required_columns=required_columns,
        min_wavelength=min_wavelength,
        max_wavelength=max_wavelength,
        log_path=log_path,
    )

    if df is None:
        technique_label = technique.upper() if technique else 'UNKNOWN'
        log(logger=logger, msg=f'          Skipping (techniques={technique_label}): {path.name}')
        return None

    # Parse metadata from filepath structure and standard lookups
    meta_df = parent_concentration_data(
        source_data_paths=[path], 
        composition_columns=composition_columns,
        required_columns=required_columns,
        salt_states=salt_states,
        experimental_variation_columns=experimental_variation_columns,
        include_experimental_variation_columns=include_experimental_variation_columns
    )
    meta = meta_df.iloc[0].to_dict()

    meta_to_add = {col: value for col, value in meta.items() if col not in ('file_path', 'og_path')}
    meta_block = pd.DataFrame([{
        'file_id': f'file_{path.stem}',
        'og_path': str(og_path),
        **meta_to_add,
    }] * len(df)).reset_index(drop=True)

    df = pd.concat([meta_block, df.reset_index(drop=True)], axis=1)

    df.to_csv(long_path(enriched_path), index=False)
    log(logger=logger, msg=f"Enriched → {enriched_path.name}  |  shape={df.shape}")
    return enriched_path

def enrich_with_progress(
        files: List[Path],
        enriched_root: Path,
        technique_name: str = 'libs',
        drop_columns: Optional[List[str]] = None,
        rename_columns: Optional[dict] = None,
        composition_columns: Optional[List[str]] = None,
        min_wavelength: Optional[float] = None,
        max_wavelength: Optional[float] = None,
        allowed_extensions: Optional[List[str]] = None,
        required_columns: Optional[List[str]] = None,
        salt_states: Optional[List[str]] = None,
        experimental_variation_columns: Optional[List[str]] = None,
        include_experimental_variation_columns: bool = True,
        log_q: Any = None,
        log_path: Path | None = None,
) -> List[Path]:
    """ Parallel batch process multiple spectral files with a tqdm progress bar.

    Parameters:
        files (List[Path]): Target list of filepaths to process.
        enriched_root (Path): Output directory for enriched CSVs.
        technique_name (str): Label denoting diagnostic method.
        drop_columns (Optional[List[str]]): Target headers to remove.
        rename_columns (Optional[dict]): Dictionary mapping header replacements.
        composition_columns (Optional[List[str]]): Concentration tracking targets.
        min_wavelength (Optional[float]): Wavelength lower bound limit.
        max_wavelength (Optional[float]): Wavelength upper bound limit.
        allowed_extensions (Optional[List[str]]): Acceptable file suffix formats.
        required_columns (Optional[List[str]]): Required file column targets.
        salt_states (Optional[List[str]]): Target physical salt phases.
        experimental_variation_columns (Optional[List[str]]): Metadata conditions tracking.
        include_experimental_variation_columns (bool): Parameter toggle status.
        log_q (Any): Inter-process log handle for process pools.
        log_path (Path | None): Custom target log filepath.

    Returns:
        List[Path]: Collection of paths targeting enriched output files.
    """
    enriched_root.mkdir(parents=True, exist_ok=True)
    enriched_paths = []

    worker = partial(
        enrich_file_with_metadata,
        enriched_root=enriched_root,
        drop_columns=drop_columns,
        rename_columns=rename_columns,
        composition_columns=composition_columns,
        technique=technique_name,
        allowed_extensions=allowed_extensions,
        required_columns=required_columns,
        salt_states=salt_states,
        experimental_variation_columns=experimental_variation_columns,
        include_experimental_variation_columns=include_experimental_variation_columns,
        min_wavelength=min_wavelength,
        max_wavelength=max_wavelength,
        log_path=log_path
    )

    n_workers = min(3, os.cpu_count() or 3)
    print(f"Starting ProcessPoolExecutor with {n_workers} workers...")
    
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=worker_init,
        initargs=(log_q,)
    ) as executor:
        futures = {executor.submit(worker, path=file): file for file in files}
        for future in tqdm(
            as_completed(futures), 
            total=len(files),
            desc=f"Enriching {technique_name}", 
            unit="file"
        ):
            result = future.result()
            if result is not None:
                enriched_paths.append(result)

    return enriched_paths

def _process_single_file(
        args: tuple[Path | str, float, float, int, Path | None, bool]
) -> Optional[tuple[np.ndarray, pd.DataFrame, np.ndarray]]:
    """Parallel worker helper function for reading and standardizing individual CSV spectral files.

    Parameters
    ----------
    args : tuple
        Unpacked argument tuple containing:
        (file_path, min_wl, max_wl, n_points, log_path, transpose_needed).

    Returns
    -------
    Optional[tuple[np.ndarray, pd.DataFrame, np.ndarray]]
        A tuple of (spectra_matrix, metadata_dataframe, target_wavelength_grid), or None on failure.
    """
    f, min_wl, max_wl, n_points, log_path, transpose_needed = args
    try:
        swlg = standardize_wavelength_grid(
            file_path=f,
            min_wl=min_wl,
            max_wl=max_wl,
            n_points=n_points,
            log_path=log_path,
            transpose=transpose_needed
        )
        if swlg is None:
            return None
        
        std_df, target_grid = swlg
        
        # Split output into numeric feature matrix and metadata DataFrame
        col_index = std_df.columns.astype(str).str.strip().tolist()
        numeric_vals = pd.to_numeric(col_index, errors='coerce')
        numeric_mask = pd.Series(numeric_vals).notna().to_numpy().astype(bool)

        spectra = std_df.loc[:, numeric_mask].values.astype(np.float32)
        meta_df = std_df.loc[:, ~numeric_mask]
        
        return spectra, meta_df, target_grid
    except Exception as e:
        return None

def species_sort_key(item: tuple[str, str]) -> tuple[int, int]:
    """Sorting helper key for ordering chemical species/compounds.

    Prioritizes molecular compounds containing specific elements (Cl, N, O) ahead of
    pure elemental species, followed by descending character length of the key.

    Parameters
    ----------
    item : tuple[str, str]
        A tuple of (key, nice_name), where `key` is the species symbol/formula 
        and `nice_name` is the display string.

    Returns
    -------
    tuple[int, int]
        A tuple of (priority_flag, negative_key_length) for sorting.
    """
    key, nice_name = item
    # Check if the name represents a multi-element compound (contains Cl, N, or O)
    is_compound = any(s in nice_name for s in ['Cl', 'N', 'O'])
    
    # Priority rank: 0 for compounds (first), 1 for elements; tie-break by longer key length
    return (0 if is_compound else 1, -len(key))

def standardize_wavelength_grid(
    file_path: str | Path,
    output_path: str | Path | None = None,
    min_wl: float = 200.0,
    max_wl: float = 1000.0,
    n_points: int = 10000,
    reference_grid: np.ndarray | None = None,
    log_path: Path | None = None,
    transpose: bool = False,  # Kept for signature compatibility
) -> Optional[tuple[pd.DataFrame, np.ndarray]]:
    """Reads a CSV file containing spectral measurements, extracts numeric wavelength columns,
    and linearly interpolates all spectral shots onto a standardized uniform wavelength grid.

    Parameters
    ----------
    file_path : str | Path
        Path to the input CSV file containing raw spectral intensities.
    output_path : str | Path | None, optional
        Path where the standardized DataFrame should be saved as a CSV.
    min_wl : float, default=200.0
        Minimum wavelength for the linear grid (if `reference_grid` is None).
    max_wl : float, default=1000.0
        Maximum wavelength for the linear grid (if `reference_grid` is None).
    n_points : int, default=10000
        Number of points in the linear grid (if `reference_grid` is None).
    reference_grid : np.ndarray | None, optional
        Pre-defined 1D array of target wavelengths to interpolate onto.
    log_path : Path | None, optional
        File path for logger output.
    transpose : bool, default=False
        Unused parameter maintained for downstream signature compatibility.

    Returns
    -------
    Optional[tuple[pd.DataFrame, np.ndarray]]
        A tuple containing (resampled DataFrame with metadata, target wavelength grid),
        or None if processing fails.
    """
    # Initialize logger
    if log_path is None:
        log_path = Path(r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\default_log.txt").resolve()
    logger = get_worker_logger(Path(log_path).stem)

    file_path = Path(file_path).resolve()
    if not file_path.exists():
        log(logger=logger, msg=f'File not found: {file_path.name}')
        return None
        
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        log(logger=logger, msg=f'  ❌ Could not read {file_path.name}: {e}')
        return None

    try:
        # --- 1. Separate Numeric Wavelength Headers from Metadata Columns ---
        numeric_cols = []
        meta_cols = []

        for col in df.columns:
            try:
                numeric_cols.append((float(col), col))
            except ValueError:
                meta_cols.append(col)

        if not numeric_cols:
            log(logger=logger, msg=f'  ⚠️ No valid numeric wavelength columns found in {file_path.name}')
            return None

        # Sort original wavelengths sequentially for accurate monotonic np.interp evaluation
        numeric_cols.sort(key=lambda x: x[0])
        orig_wavelengths = np.array([x[0] for x in numeric_cols], dtype=np.float64)
        orig_col_names = [x[1] for x in numeric_cols]

        # Extract 2D intensity matrix (Shape: n_shots x n_orig_wavelengths)
        intensity_matrix = df[orig_col_names].to_numpy(dtype=np.float64)

        # --- 2. Determine Target Wavelength Axis ---
        if reference_grid is not None:
            target_grid = reference_grid.astype(np.float64)
        else:
            target_grid = np.linspace(min_wl, max_wl, n_points, dtype=np.float64)

        # --- 3. 1D Resampling/Interpolation Across Target Grid ---
        resampled_spectra = np.apply_along_axis(
            lambda row: np.interp(target_grid, orig_wavelengths, row, left=0.0, right=0.0),
            axis=1,
            arr=intensity_matrix
        ).astype(np.float32)

        # Format standardized DataFrame with column names formatted to 4 decimal places
        resampled_df = pd.DataFrame(
            resampled_spectra, 
            columns=[f"{wl:.4f}" for wl in target_grid]
        )

        # --- 4. Re-attach Non-Spectral Metadata ---
        if meta_cols:
            meta_df = df[meta_cols].reset_index(drop=True)
            combined_df = pd.concat([meta_df, resampled_df], axis=1)
        else:
            combined_df = resampled_df

        # --- 5. Export Standardized CSV ---
        if output_path is not None:
            out_p = Path(output_path).resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            combined_df.to_csv(out_p, index=False)
            log(logger=logger, msg=f'  ✅ Saved standardized file to: {out_p.name}')

        return combined_df, target_grid

    except Exception as e:
        log(logger=logger, msg=f'  ❌ Error standardizing {file_path.name}: {e}')
        return None
# endregion

# =============================================================================
# region: Dataset Loaders & I/O Helpers
# =============================================================================

def hf_get(hf: h5py.File, key: str) -> np.ndarray:
    """ Extract a dataset array from an HDF5 group or container safely and reutrn it as a ndarray. """
    return hf[key][:] # type: ignore[index]

def get_h5_ds(hf: h5py.File, key: str) -> h5py.Dataset:
    """ Extract a dataset array from a .h5 files and return it as a dataset. """
    from typing import cast
    return cast(h5py.Dataset, hf[key])

def h5_column_update(
    file_path: Path | str, target_columns: list[str]
) -> list[str]:
    """ Append missing metadata target columns filled with zero vectors to an HDF5 file. """
    file_path = Path(file_path)

    with h5py.File(file_path, "a") as hf:
        if "spectra" in hf:
            spectra_ds = hf["spectra"]
        elif "train/spectra" in hf:
            spectra_ds = hf["train/spectra"]
        else:
            raise KeyError(f"Could not locate 'spectra' or 'train/spectra' in {file_path}")

        assert isinstance(spectra_ds, h5py.Dataset)
        n_rows = spectra_ds.shape[0]

        if "metadata" in hf:
            meta_grp = hf["metadata"]
        elif "train/metadata" in hf:
            meta_grp = hf["train/metadata"]
        else:
            meta_grp = hf.create_group("metadata")

        assert isinstance(meta_grp, h5py.Group)

        existing_cols = set(meta_grp.keys())
        comp_kwargs = {"compression": "gzip", "compression_opts": 3}

        added_cols = []
        for col in target_columns:
            if col not in existing_cols:
                zeros_arr = np.zeros(n_rows, dtype=np.float32)
                meta_grp.create_dataset(col, data=zeros_arr, **comp_kwargs)
                added_cols.append(col)

        current_attrs = hf.attrs.get("metadata_cols", [])
        if isinstance(current_attrs, np.ndarray):
            current_attrs = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in current_attrs]
        else:
            current_attrs = list(current_attrs)

        updated_attrs = list(dict.fromkeys(current_attrs + target_columns))
        hf.attrs["metadata_cols"] = np.array(updated_attrs, dtype=h5py.string_dtype())

        print(f"Successfully updated {file_path.name}: Added {len(added_cols)} missing column(s).")

    return added_cols

def h5_to_xandy(
    input_h5: Path | str,
    output_h5: Path | str,
    feature_selector: bool = False,
    down_sample_rows: int = 0,
    down_sample_cols: int = 0,
    allowed_cols: set[str] | None = None,
    log_path: Path | None = None,
    conc_to_spec: bool = True,
) -> None:
    """ Preprocess raw HDF5 dataset splits into scaled X and y feature matrices for machine learning.

    Applies percentile intensity clipping (at the 99.5th percentile), scales features using MinMaxScaler, 
    filters low-variance targets, downsamples spatial spectral resolution, and writes formatted ML partitions.
    """
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.feature_selection import VarianceThreshold
    
    input_h5 = Path(input_h5).resolve()
    output_h5 = Path(output_h5).resolve()

    (
        X_train_raw, X_val_raw, X_test_raw, 
        y_train_raw, y_val_raw, y_test_raw, 
        train_syn, val_syn, test_syn, 
        feature_cols, wavelengths 
    ) = load_h5_split_dataset(
        h5_path=input_h5,
        allowed_cols=allowed_cols,
        log_path=log_path,
        conc_to_spec=conc_to_spec
    )

    # 1. Clip spectral intensity spikes at 99.5th percentile threshold
    if conc_to_spec:
        clip_threshold = np.nanpercentile(y_train_raw, 99.5)
        y_train_raw = np.clip(y_train_raw, 0, clip_threshold)
        y_val_raw   = np.clip(y_val_raw, 0, clip_threshold)
        y_test_raw  = np.clip(y_test_raw, 0, clip_threshold)
    else:
        clip_threshold = np.nanpercentile(X_train_raw, 99.5)
        X_train_raw = np.clip(X_train_raw, 0, clip_threshold)
        X_val_raw   = np.clip(X_val_raw, 0, clip_threshold)
        X_test_raw  = np.clip(X_test_raw, 0, clip_threshold)

    X_train_backup = X_train_raw.copy()
    X_val_backup   = X_val_raw.copy()
    X_test_backup  = X_test_raw.copy()
    y_train_backup = y_train_raw.copy()
    y_val_backup   = y_val_raw.copy()
    y_test_backup  = y_test_raw.copy()

    # 2. Fit MinMaxScaler on train set, transform val and test sets
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    y_train_scaled = y_scaler.fit_transform(y_train_raw)
    y_val_scaled   = y_scaler.transform(y_val_raw)
    y_test_scaled  = y_scaler.transform(y_test_raw)

    X_scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = X_scaler.fit_transform(X_train_raw)
    X_val_scaled   = X_scaler.transform(X_val_raw)
    X_test_scaled  = X_scaler.transform(X_test_raw)

    # 3. Filter zero/low variance features
    if feature_selector:
        selector = VarianceThreshold(threshold=1e-10)
        if conc_to_spec:
            X_train_scaled = selector.fit_transform(X_train_scaled)
            X_val_scaled   = selector.transform(X_val_scaled)
            X_test_scaled  = selector.transform(X_test_scaled)
        else:
            y_train_scaled = selector.fit_transform(y_train_scaled)
            y_val_scaled   = selector.transform(y_val_scaled)
            y_test_scaled  = selector.transform(y_test_scaled)

        retained_indices = selector.get_support(indices=True)
        feature_cols = [feature_cols[i] for i in retained_indices]

    # 4. Downsample spectral channels (columns)
    if down_sample_cols > 1:
        if conc_to_spec:
            y_train_scaled = y_train_scaled[:, ::down_sample_cols]
            y_val_scaled   = y_val_scaled[:, ::down_sample_cols]
            y_test_scaled  = y_test_scaled[:, ::down_sample_cols]
        else:
            X_train_scaled = X_train_scaled[:, ::down_sample_cols]
            X_val_scaled   = X_val_scaled[:, ::down_sample_cols]
            X_test_scaled  = X_test_scaled[:, ::down_sample_cols]

        if wavelengths is not None:
            wavelengths = wavelengths[::down_sample_cols]

    # 5. Downsample specimen row instances (training set only)
    if down_sample_rows > 1:
        X_train_scaled = X_train_scaled[::down_sample_rows]
        y_train_scaled = y_train_scaled[::down_sample_rows]
        train_syn      = train_syn[::down_sample_rows]
        X_train_backup = X_train_backup[::down_sample_rows]
        y_train_backup = y_train_backup[::down_sample_rows]

    # 6. Save formatted partitions to output HDF5
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 3}
    utf8_type = h5py.string_dtype(encoding='utf-8')

    with h5py.File(output_h5, 'w') as hf:
        hf.create_dataset('X_train', data=X_train_scaled, **comp_kwargs)
        hf.create_dataset('X_val', data=X_val_scaled, **comp_kwargs)
        hf.create_dataset('X_test', data=X_test_scaled, **comp_kwargs)
        
        hf.create_dataset('y_train', data=y_train_scaled, **comp_kwargs)
        hf.create_dataset('y_val', data=y_val_scaled, **comp_kwargs)
        hf.create_dataset('y_test', data=y_test_scaled, **comp_kwargs)
        
        hf.create_dataset('X_train_backup', data=X_train_backup, **comp_kwargs)
        hf.create_dataset('X_val_backup', data=X_val_backup, **comp_kwargs)
        hf.create_dataset('X_test_backup', data=X_test_backup, **comp_kwargs)
        
        hf.create_dataset('y_train_backup', data=y_train_backup, **comp_kwargs)
        hf.create_dataset('y_val_backup', data=y_val_backup, **comp_kwargs)
        hf.create_dataset('y_test_backup', data=y_test_backup, **comp_kwargs)
        
        hf.create_dataset('train_syn', data=train_syn, **comp_kwargs)
        hf.create_dataset('val_syn', data=val_syn, **comp_kwargs)
        hf.create_dataset('test_syn', data=test_syn, **comp_kwargs)
        
        if feature_cols is not None:
            hf.create_dataset('feature_cols', data=np.array(feature_cols, dtype=object), dtype=utf8_type)
            
        if wavelengths is not None:
            hf.create_dataset('wavelengths', data=wavelengths, **comp_kwargs)

    print(f"Dataset successfully saved to {output_h5}")

def load_h5_split_dataset(
        h5_path: str | Path,
        allowed_cols: set[str] | None = None,
        log_path: Path | None = None,
        conc_to_spec: bool = True
) -> tuple[
    NDArray[np.float32], NDArray[np.float32], NDArray[np.float32],
    NDArray[np.float32], NDArray[np.float32], NDArray[np.float32],
    NDArray[np.int_], NDArray[np.int_], NDArray[np.int_],
    list[str],
    NDArray[np.float32] | None
]:
    """ Load train, val, and test dataset splits from a pre-partitioned HDF5 file. """
    with h5py.File(h5_path, 'r') as hf:
        y_train_raw     = hf_get(hf, 'train/spectra')
        y_val_raw       = hf_get(hf, 'val/spectra')
        y_test_raw      = hf_get(hf, 'test/spectra')

        wavelengths     = hf_get(hf, 'wavelengths') if 'wavelengths' in hf else None

        raw_elem_names  = hf_get(hf, 'train/metadata/elem_names')
        feature_cols    = [
            e.decode('utf-8') if hasattr(e, 'decode') else str(e)
            for e in raw_elem_names
        ]

        X_train_raw     = hf_get(hf, 'train/metadata/elem_comp_wt%').astype(np.float32)
        X_val_raw       = hf_get(hf, 'val/metadata/elem_comp_wt%').astype(np.float32)
        X_test_raw      = hf_get(hf, 'test/metadata/elem_comp_wt%').astype(np.float32)

        if allowed_cols:
            col_indices     = [i for i, col in enumerate(feature_cols) if col in allowed_cols]
            feature_cols    = [feature_cols[i] for i in col_indices]
            X_train_raw     = X_train_raw[:, col_indices]
            X_val_raw       = X_val_raw[:, col_indices]
            X_test_raw      = X_test_raw[:, col_indices]

        train_syn       = hf_get(hf, 'train/metadata/is_synthetic').astype(int)
        val_syn         = hf_get(hf, 'val/metadata/is_synthetic').astype(int)
        test_syn        = hf_get(hf, 'test/metadata/is_synthetic').astype(int)

        # Invert input (X) and target (y) array roles if inverse modeling is selected
        if not conc_to_spec:
            y_train_raw, X_train_raw    = X_train_raw, y_train_raw
            y_val_raw, X_val_raw        = X_val_raw, y_val_raw
            y_test_raw, X_test_raw      = X_test_raw, y_test_raw

    return (
        X_train_raw, X_val_raw, X_test_raw, 
        y_train_raw, y_val_raw, y_test_raw, 
        train_syn, val_syn, test_syn,
        feature_cols, wavelengths
    )

def load_h5_base_dataset(
    h5_path: str | Path,
    allowed_cols: Container[str],
    log_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray | None, pd.DataFrame]:
    """ Load baseline spectra, concentrations, and metadata DataFrame from an unpartitioned HDF5 file. """
    with h5py.File(h5_path, "r") as hf:
        if "spectra" in hf:
            spectra_obj = hf["spectra"]
        elif "train/spectra" in hf:
            spectra_obj = hf["train/spectra"]
        else:
            raise KeyError("Could not locate 'spectra' dataset in file.")

        assert isinstance(spectra_obj, h5py.Dataset)
        y_raw = spectra_obj[:].astype(np.float32)

        if "wavelengths" in hf:
            wl_obj = hf["wavelengths"]
        elif "train/wavelengths" in hf:
            wl_obj = hf["train/wavelengths"]
        else:
            wl_obj = None

        if wl_obj is not None:
            assert isinstance(wl_obj, h5py.Dataset)
            wavelengths: np.ndarray | None = wl_obj[:]
        else:
            wavelengths = None

        if "metadata" in hf:
            meta_path = "metadata"
        elif "train/metadata" in hf:
            meta_path = "train/metadata"
        else:
            raise KeyError("Could not locate 'metadata' group in file.")

        meta_grp = hf[meta_path]
        assert isinstance(meta_grp, h5py.Group)

        all_cols = list(meta_grp.keys())
        concentration_cols = []
        for col in all_cols:
            if col in allowed_cols:
                ds = meta_grp[col]
                if isinstance(ds, h5py.Dataset) and np.issubdtype(ds.dtype, np.number):
                    concentration_cols.append(col)

        if not concentration_cols:
            raise KeyError(f"No numeric concentration columns found in {h5_path} matching allowed_cols!")

        X_raw = np.stack([hf_get(hf, f"{meta_path}/{c}") for c in concentration_cols], axis=1).astype(np.float32)

        meta_dict = {}
        for col in all_cols:
            data = hf_get(hf, f"{meta_path}/{col}")
            if data.dtype.kind in ("O", "S", "V"):
                meta_dict[col] = [x.decode("utf-8") if isinstance(x, bytes) else x for x in data]
            else:
                meta_dict[col] = data

        meta_df = pd.DataFrame(meta_dict)

    return X_raw, y_raw, concentration_cols, wavelengths, meta_df

def load_prepped_training_dataset(
        prepped_h5_path: str | Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """ Load preprocessed training and validation partitions from disk. """
    prepped_h5_path = str(prepped_h5_path)
    
    with h5py.File(prepped_h5_path, 'r') as hf:
        X_trn = hf_get(hf, 'X_trn')
        X_val = hf_get(hf, 'X_val')
        y_trn = hf_get(hf, 'y_trn')
        y_val = hf_get(hf, 'y_val')
        y_val_physical = hf_get(hf, 'y_val_physical')
        
        target_cols = [c.decode('utf-8') for c in hf_get(hf, 'target_cols')]
        surviving_cols = [c.decode('utf-8') for c in hf_get(hf, 'surviving_cols')]
        
        meta_dict = {}
        meta_grp = hf.get('metadata_val')
        if isinstance(meta_grp, h5py.Group):
            for col_name in meta_grp.keys():
                data = hf_get(hf, f'metadata_val/{col_name}')
                if data.dtype.kind in ('O', 'S', 'V'):
                    meta_dict[col_name] = [x.decode('utf-8') if isinstance(x, bytes) else x for x in data]
                else:
                    meta_dict[col_name] = data
                
        meta_val_df = pd.DataFrame(meta_dict)
        
    return X_trn, X_val, y_trn, y_val, y_val_physical, surviving_cols, meta_val_df

def load_scalers(
        y_scaler_path: str | Path | None = None,
        X_scaler_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """ Unpickle and load fitted feature and target scalers from disk. """
    if y_scaler_path is None:
        y_scaler_path = r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\LIBS\y_scaler.pkl"
    if X_scaler_path is None:
        X_scaler_path = r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\LIBS\X_scaler.pkl"

    with open(y_scaler_path, "rb") as f:
        y_scaler = pickle.load(f)
    with open(X_scaler_path, "rb") as f:
        X_scaler = pickle.load(f)

    return (y_scaler, X_scaler)

def long_path(
        path: Path | str,
        log_path: Path | None = None,
) -> str:
    r""" Format file paths with the Windows long-path prefix (`\\?\`) to bypass character length limits. """
    if log_path is None:
        log_path = Path(r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\default_log.txt").resolve()

    logger = get_worker_logger(Path(log_path).stem)

    if path is None:
        log(logger=logger, msg="No path provided")
        return ""
    else:
        s = str(Path(path).resolve())
        if len(s) > 240 and not s.startswith('\\\\?\\'):
            return '\\\\?\\' + s
        return s

def worker_init(log_q: Any) -> None:
    """Multiprocessing initialization routine called once per worker process startup.

    Configures queue-based logging handler for concurrent execution.

    Parameters
    ----------
    log_q : multiprocessing.Queue
        Shared queue used to aggregate log messages across worker processes.
    """
    global _worker_log_q
    _worker_log_q = log_q
    logger = logging.getLogger('worker')
    if not logger.handlers:
        logger.addHandler(handlers.QueueHandler(log_q))
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
# endregion


# =============================================================================
# region: File Saving
# =============================================================================

def combine_and_save_as_CSV(
        individuals_root: str | Path,
        combined_root: str | Path,
        technique: Optional[str] = None,
        log_path: Path | None = None,
) -> Optional[Path]:
    """ Concatenate multiple enriched CSV data files into a single master CSV dataset.

    Parameters:
        individuals_root (str | Path): Directory containing enriched individual CSV files.
        combined_root (str | Path): Output directory for saving the concatenated CSV.
        technique (Optional[str]): Technique label used in naming the output file.
        log_path (Path | None): Custom log file location.

    Returns:
        Optional[Path]: Filepath to the saved master CSV, or None if concatenation fails.
    """
    individuals_root = Path(individuals_root).resolve()
    combined_root = Path(combined_root).resolve()
    combined_root.mkdir(parents=True, exist_ok=True)
    
    if log_path is None:
        log_path = Path(r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\default_log.txt").resolve()
    logger = get_worker_logger(Path(log_path).stem)

    technique = technique or 'libs'
    enriched_files = list(individuals_root.glob('*.csv'))
    if not enriched_files:
        log(logger=logger, msg=f'  ❌ No {technique} files found.')
        return None
    
    log(logger=logger, msg=f'Found files: {len(enriched_files)}')
    log(logger=logger, msg='Reading files...')

    try:
        combined_df = pd.concat(
            [pd.read_csv(f) for f in enriched_files],
            ignore_index=True,
            sort=False
        )
        label = technique.upper()
        combined_path = combined_root / f'cleaned_combined_{label}.csv'
        combined_df.to_csv(combined_path, index=False)
        log(logger=logger, msg=f'Saved: {combined_path.name}  |  shape={combined_df.shape}')
        return combined_path
    except Exception as e:
        log(logger=logger, msg=f'  ❌ {e}')
        return None

def combine_and_save_as_HDF5(
        individuals_root: str | Path,
        combined_root: str | Path,
        technique: Optional[str] = None,
        compression: str = 'gzip',
        compression_level: int = 3,
        log_path: Path | None = None,
        transpose_needed: bool = False,
        n_workers: int | None = None,
) -> Optional[Path]:
    """ Parallel process individual spectral CSVs into a structured HDF5 file.

    Generates a standardized binary dataset split into discrete groups:
        /spectra   -> 2D float32 intensity matrix (n_rows, n_wavelengths)
        /metadata  -> 1D datasets corresponding to sample experimental variables
        /attrs     -> Global dataset attributes (wavelength ranges, execution tags)

    Parameters:
        individuals_root (str | Path): Source path containing individual enriched CSV files.
        combined_root (str | Path): Destination folder for the finalized HDF5 file.
        technique (Optional[str]): Operational label (e.g., 'libs', 'raman').
        compression (str): Lossless compression algorithm ('gzip', 'lzf', or None).
        compression_level (int): Gzip compression scaling level (1 to 9).
        log_path (Path | None): Custom filepath for output log registration.
        transpose_needed (bool): Indicates if spectral axes require transposition.
        n_workers (int | None): CPU core allocation count for parallel execution.

    Returns:
        Optional[Path]: Filepath to the resulting HDF5 file, or None if step fails.
    """
    individuals_root = Path(individuals_root).resolve()
    combined_root = Path(combined_root).resolve()
    combined_root.mkdir(parents=True, exist_ok=True)
    
    if log_path is None:
        log_path = Path(r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\default_log.txt").resolve()
    logger = get_worker_logger(Path(log_path).stem)

    technique = technique or 'libs'
    enriched_files = list(individuals_root.glob('*.csv'))
    if not enriched_files:
        log(logger=logger, msg=f'  ❌ No {technique} files found.')
        return None
    
    # Auto-detect compute cores using Slurm environment variables or system core count
    if n_workers is None:
        n_workers = int(os.getenv("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
        
    log(logger=logger, msg=f'Found files: {len(enriched_files)}')
    log(logger=logger, msg=f'Reading files across {n_workers} CPU workers...')

    all_spectra = []
    all_metadata = []
    target_grid = None

    tasks = [
        (f, TARGET_MIN_WL, TARGET_MAX_WL, TARGET_N_PTS, log_path, transpose_needed)
        for f in enriched_files
    ]

    completed_count = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_process_single_file, task) for task in tasks]
        
        for future in as_completed(futures):
            res = future.result()
            completed_count += 1
            
            if res is not None:
                spectra, meta_df, target_grid = res
                all_spectra.append(spectra)
                all_metadata.append(meta_df)

            if completed_count % 1000 == 0 or completed_count == len(enriched_files):
                log(logger=logger, msg=f'  Processed {completed_count}/{len(enriched_files)} files...')

    if not all_spectra:
        log(logger=logger, msg='  ❌ No valid files loaded.')
        return None
    
    log(logger=logger, msg='Stacking arrays...')
    spectra_array = np.vstack(all_spectra)
    metadata_df = pd.concat(all_metadata, ignore_index=True, sort=False)

    log(logger=logger, msg=f'  Spectra shape : {spectra_array.shape}')
    log(logger=logger, msg=f'  Metadata shape: {metadata_df.shape}')
    log(logger=logger, msg=f'  Wavelength range: {target_grid[0]:.1f} – {target_grid[-1]:.1f} nm')

    label = technique.upper()
    h5_path = combined_root / f'combined_{label}.h5'
    comp_kwargs = {'compression': compression}
    if compression == 'gzip':
        comp_kwargs['compression_opts'] = compression_level

    log(logger=logger, msg=f'Writing HDF5 → {h5_path.name}')

    with h5py.File(h5_path, 'w') as hf:
        hf.create_dataset(
            'spectra',
            data=spectra_array,
            chunks=(min(1000, spectra_array.shape[0]), spectra_array.shape[1]),
            **comp_kwargs
        )

        hf.create_dataset('wavelengths', data=target_grid)

        meta_grp = hf.create_group('metadata')
        for col in metadata_df.columns:
            series = metadata_df[col]
            if pd.api.types.is_numeric_dtype(series):
                arr = series.to_numpy(dtype=np.float32, na_value=np.nan)
                meta_grp.create_dataset(col, data=arr, **comp_kwargs)
            else:
                arr = series.fillna('').astype(str).to_numpy()
                dt = h5py.string_dtype(encoding='utf-8')
                meta_grp.create_dataset(col, data=arr, dtype=dt)

        hf.attrs['n_rows'] = spectra_array.shape[0]
        hf.attrs['n_wavelengths'] = spectra_array.shape[1]
        hf.attrs['min_wavelength'] = float(target_grid[0])
        hf.attrs['max_wavelength'] = float(target_grid[-1])
        hf.attrs['metadata_cols'] = list(metadata_df.columns)
        hf.attrs['technique'] = label

    size_mb = h5_path.stat().st_size / 1_048_576
    log(logger=logger, msg=f'✅ Saved: {h5_path.name}  |  {size_mb:.1f} MB')
    return h5_path

def combine_and_save_as_HDF5_one_at_a_time(
        individuals_root: str | Path,
        combined_root: str | Path,
        technique: Optional[str] = None,
        compression: str = 'gzip',
        compression_level: int = 3,
        log_path: Path | None = None,
        transpose_needed: bool = False,
) -> Optional[Path]:
    """ Sequentially parse and accumulate CSV files into an HDF5 dataset.

    Operates on a single thread to reduce memory overhead during dataset compilation on low-RAM systems.

    Parameters:
        individuals_root (str | Path): Folder path containing target individual CSVs.
        combined_root (str | Path): Folder destination for storing output HDF5 data.
        technique (Optional[str]): System diagnostic technique identifier tag.
        compression (str): HDF5 compression type ('gzip', 'lzf', or None).
        compression_level (int): Numerical scaling parameter for gzip compression algorithms.
        log_path (Path | None): File path specifying output log location.
        transpose_needed (bool): Standard flag enforcing orientation transpositions.

    Returns:
        Optional[Path]: Filepath targeting generated output dataset, or None if processing fails.
    """
    individuals_root = Path(individuals_root).resolve()
    combined_root = Path(combined_root).resolve()
    combined_root.mkdir(parents=True, exist_ok=True)
    
    if log_path is None:
        log_path = Path(r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\default_log.txt").resolve()
    logger = get_worker_logger(Path(log_path).stem)

    technique = technique or 'libs'
    enriched_files = list(individuals_root.glob('*.csv'))
    if not enriched_files:
        log(logger=logger, msg=f'  ❌ No {technique} files found.')
        return None
    
    log(logger=logger, msg=f'Found files: {len(enriched_files)}')
    log(logger=logger, msg='Reading files...')

    all_spectra = []
    all_metadata = []
    target_grid = None

    for i, f in enumerate(enriched_files):
        try:
            swlg = standardize_wavelength_grid(
                file_path=f,
                min_wl=TARGET_MIN_WL,
                max_wl=TARGET_MAX_WL,
                n_points=TARGET_N_PTS,
                transpose=transpose_needed
            )
            if swlg is None:
                log(logger=logger, msg=f'  ⚠️  Skipping {f.name} (standardize returned None)')
                continue
            std_df, target_grid = swlg
        except Exception as e:
            log(logger=logger, msg=f'  ❌ Could not read {f.name}: {e}')
            continue

        col_index = std_df.columns.astype(str).str.strip().tolist()
        numeric_vals = pd.to_numeric(col_index, errors='coerce')
        numeric_mask = pd.Series(numeric_vals).notna().to_numpy().astype(bool)

        spectra = std_df.loc[:, numeric_mask].values.astype(np.float32)
        meta_df = std_df.loc[:, ~numeric_mask]

        all_spectra.append(spectra)
        all_metadata.append(meta_df)

        if (i + 1) % 100 == 0:
            log(logger=logger, msg=f'  Read {i+1}/{len(enriched_files)}')

    if not all_spectra:
        log(logger=logger, msg='  ❌ No valid files loaded.')
        return None

    log(logger=logger, msg='Stacking arrays...')
    spectra_array = np.vstack(all_spectra)
    metadata_df = pd.concat(all_metadata, ignore_index=True, sort=False)

    log(logger=logger, msg=f'  Spectra shape : {spectra_array.shape}')
    log(logger=logger, msg=f'  Metadata shape: {metadata_df.shape}')
    log(logger=logger, msg=f'  Wavelength range: {target_grid[0]:.1f} – {target_grid[-1]:.1f} nm')

    label = technique.upper()
    h5_path = combined_root / f'combined_{label}.h5'
    comp_kwargs = {'compression': compression}
    if compression == 'gzip':
        comp_kwargs['compression_opts'] = compression_level

    log(logger=logger, msg=f'Writing HDF5 → {h5_path.name}')

    with h5py.File(h5_path, 'w') as hf:
        hf.create_dataset(
            'spectra',
            data=spectra_array,
            chunks=(min(1000, spectra_array.shape[0]), spectra_array.shape[1]),
            **comp_kwargs
        )

        hf.create_dataset('wavelengths', data=target_grid)

        meta_grp = hf.create_group('metadata')
        for col in metadata_df.columns:
            series = metadata_df[col]
            if pd.api.types.is_numeric_dtype(series):
                arr = series.to_numpy(dtype=np.float32, na_value=np.nan)
                meta_grp.create_dataset(col, data=arr, **comp_kwargs)
            else:
                arr = series.fillna('').astype(str).to_numpy()
                dt = h5py.string_dtype(encoding='utf-8')
                meta_grp.create_dataset(col, data=arr, dtype=dt)

        hf.attrs['n_rows'] = spectra_array.shape[0]
        hf.attrs['n_wavelengths'] = spectra_array.shape[1]
        hf.attrs['min_wavelength'] = float(target_grid[0])
        hf.attrs['max_wavelength'] = float(target_grid[-1])
        hf.attrs['metadata_cols'] = list(metadata_df.columns)
        hf.attrs['technique'] = label

    size_mb = h5_path.stat().st_size / 1_048_576
    log(logger=logger, msg=f'✅ Saved: {h5_path.name}  |  {size_mb:.1f} MB')
    return h5_path

def save_merged_h5_dataset(
    output_h5_path: str | Path,
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    feature_cols: list[str],
    wavelengths: np.ndarray | None,
    meta_df: pd.DataFrame,
    compression_level: int = 4,
) -> None:
    """ Write merged spectral features, target compositions, and metadata directly to an HDF5 container. """
    output_h5_path = Path(output_h5_path)
    output_h5_path.parent.mkdir(parents=True, exist_ok=True)

    comp_kwargs = {
        "compression": "gzip",
        "compression_opts": compression_level,
        "shuffle": True,
    }

    print(f"Writing merged dataset to {output_h5_path}...")

    with h5py.File(output_h5_path, "w") as hf:
        hf.create_dataset("spectra", data=y_raw.astype(np.float32), **comp_kwargs)

        if wavelengths is not None:
            hf.create_dataset(
                "wavelengths",
                data=wavelengths.astype(np.float32),
                **comp_kwargs,
            )

        meta_grp = hf.create_group("metadata")
        string_dt = h5py.string_dtype(encoding="utf-8")

        for col in meta_df.columns:
            series = meta_df[col]
            if pd.api.types.is_numeric_dtype(series):
                arr = series.to_numpy(dtype=np.float32, na_value=np.nan)
                meta_grp.create_dataset(col, data=arr, **comp_kwargs)
            else:
                arr = series.fillna("").astype(str).to_numpy()
                meta_grp.create_dataset(col, data=arr, dtype=string_dt)

        hf.attrs["n_rows"] = y_raw.shape[0]
        hf.attrs["n_wavelengths"] = y_raw.shape[1]
        hf.attrs["metadata_cols"] = list(meta_df.columns)

    print(f"Successfully saved merged HDF5 dataset to {output_h5_path}")

def save_recombined_h5(
    output_path: str | Path,
    X: np.ndarray,
    y: np.ndarray,
    meta_df: pd.DataFrame,
    wavelengths: np.ndarray,
) -> None:
    """Saves combined spectral features, target variables, wavelengths, and metadata to HDF5 format.

    Parameters
    ----------
    output_path : str | Path
        Destination path where the output HDF5 file will be written.
    X : np.ndarray
        Feature matrix (e.g., input intensities or preprocessed features).
    y : np.ndarray
        Target labels or elemental concentration matrix (saved under dataset 'spectra').
    meta_df : pd.DataFrame
        DataFrame containing sample-level metadata (categorical string attributes and/or numeric properties).
    wavelengths : np.ndarray
        1D array containing the wavelength axis values.
    """
    output_path = Path(output_path).resolve()
    
    with h5py.File(output_path, "w") as hf:
        # --- 1. Root Datasets ---
        hf.create_dataset("spectra", data=y)
        hf.create_dataset("wavelengths", data=wavelengths)

        # --- 2. Metadata Group ---
        # Iterate over metadata columns and assign appropriate fixed or variable-length string types
        meta_grp = hf.create_group("metadata")
        for col_name in meta_df.columns:
            data = meta_df[col_name].values
            
            # Identify object/string dtypes for dynamic variable-length string encoding
            if data.dtype == object or isinstance(
                data[0] if len(data) > 0 else "", str
            ):
                string_dt = h5py.special_dtype(vlen=str)
                meta_grp.create_dataset(
                    col_name, data=data.astype(object), dtype=string_dt
                )
            else:
                meta_grp.create_dataset(col_name, data=data)

    print(f"Successfully saved recombined dataset to {output_path}")
# endregion


# =============================================================================
# region: Training Splitting
# =============================================================================

def train_val_test_splitter_HDF5(
    h5_path: str | Path,
    output_path: str | Path | None = None,        # If None, appends '_split.h5' suffix
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    random_state: int = 42,
    log_path: Path | None = None,
) -> Path:
    """Performs domain-aware compositional splitting on an HDF5 dataset to prevent data leakage.

    Group-splits experimental samples by unique chemical compositions into Train/Val/Test splits,
    while assigning ALL synthetic samples directly to the Training split.

    Parameters
    ----------
    h5_path : str | Path
        Path to the raw combined source HDF5 file.
    output_path : str | Path | None, optional
        Path where the split HDF5 file will be exported.
    val_frac : float, default=0.1
        Fraction of total experimental samples reserved for validation.
    test_frac : float, default=0.1
        Fraction of total experimental samples reserved for testing.
    random_state : int, default=42
        Seed for reproducible random splitting.
    log_path : Path | None, optional
        Path to log file output.

    Returns
    -------
    Path
        Path to the generated split HDF5 file.
    """
    from sklearn.model_selection import train_test_split
    import h5py

    h5_path = Path(h5_path).resolve()
    if output_path is None:
        output_path = h5_path.with_name(h5_path.stem + '_split.h5')
    else:
        output_path = Path(output_path).resolve()

    if output_path == h5_path:
        raise ValueError("'output_path' cannot be identical to 'h5_path' when creating a split file.")
    
    # --- Logger Setup ---
    if log_path is None:
        log_path = h5_path.parent / "logs" / "splitter.log"
    else:
        log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = get_worker_logger(log_path.stem, log_file=log_path)

    # --- 1. Read Source Datasets into Memory ---
    log(logger=logger, msg=f'Loading from {h5_path.name}...')

    with h5py.File(h5_path, 'r') as hf:
        # Locate spectra dataset
        if 'spectra' in hf:
            spectra_obj = hf['spectra']
        elif 'train/spectra' in hf:
            spectra_obj = hf['train/spectra']
        else:
            raise KeyError("Could not locate 'spectra' dataset in file.")

        assert isinstance(spectra_obj, h5py.Dataset), "Expected a Dataset for spectra"
        spectra = spectra_obj[:]

        # Locate wavelengths dataset
        if 'wavelengths' in hf:
            wl_obj = hf['wavelengths']
        elif 'train/wavelengths' in hf:
            wl_obj = hf['train/wavelengths']
        else:
            wl_obj = None

        if wl_obj is not None:
            assert isinstance(wl_obj, h5py.Dataset), "Expected a Dataset for wavelengths"
            wavelengths = wl_obj[:]
        else:
            wavelengths = None

        # Locate metadata group
        if 'metadata' in hf:
            metadata_grp = hf['metadata']
        elif 'train/metadata' in hf:
            metadata_grp = hf['train/metadata']
        else:
            raise KeyError("Could not locate 'metadata' group in file.")

        assert isinstance(metadata_grp, h5py.Group), "Expected an h5py.Group for metadata"

        metadata = {}
        for col in metadata_grp.keys():
            item = metadata_grp[col]
            if isinstance(item, h5py.Dataset):
                metadata[col] = item[:]

        attrs = dict(hf.attrs)

    n_rows, n_features = spectra.shape
    log(logger=logger, msg=f'  Total rows: {n_rows}  |  Wavelengths: {spectra.shape[1]}')

    # --- 2. Domain-Aware Two-Step Compositional Splitting ---
    if 'is_synthetic' not in metadata:
        raise KeyError("'is_synthetic' is missing from metadata!")

    is_synth_flag = metadata['is_synthetic']
    exp_indices = np.where(is_synth_flag == 0)[0]
    syn_indices = np.where(is_synth_flag == 1)[0]

    # Step 1: Group experimental shots by unique elemental composition
    comp = metadata['elem_comp_wt%']
    exp_comp = np.round(comp[exp_indices], 4)
    _, group_ids = np.unique(exp_comp, axis=0, return_inverse=True)
    group_ids = group_ids.ravel()
    unique_groups = np.unique(group_ids)
    log(logger=logger, msg=f"Found {len(exp_indices)} experimental and  {len(syn_indices)} synthetic samples.")
    log(logger=logger, msg=f"Found {len(unique_groups)} UNIQUE experimental compositions.")

    # Step 2: Split unique experimental composition groups into Train vs Holdout (Val + Test)
    holdout_frac = val_frac + test_frac
    train_groups, holdout_groups = train_test_split(
        unique_groups,
        test_size=holdout_frac,
        random_state=random_state,
        shuffle=True
    )

    # Step 3: Split Holdout groups into Validation and Test composition groups
    relative_test_ratio = test_frac / holdout_frac
    val_groups, test_groups = train_test_split(
        holdout_groups,
        test_size=relative_test_ratio,
        random_state=random_state,
        shuffle=True
    )

    # Step 4: Map group assignments back to sample indices and merge synthetic data into train
    exp_trn_idx = exp_indices[np.isin(group_ids, train_groups)]
    trn_idx = np.sort(np.concatenate([syn_indices, exp_trn_idx]))
    val_idx = exp_indices[np.isin(group_ids, val_groups)]
    test_idx = exp_indices[np.isin(group_ids, test_groups)]

    log(logger=logger, msg=f' Train: {len(trn_idx)} ({len(syn_indices)} synth + {len(exp_trn_idx)} exp)')
    log(logger=logger, msg=f' Val:   {len(val_idx)} (100% exp)')
    log(logger=logger, msg=f' Test:  {len(test_idx)} (100% exp)')

    # Step 5: Element coverage verification across splits
    elem_names = [n.decode() if isinstance(n, bytes) else n for n in metadata['elem_names']]
    for name, idx in [('train(exp)', exp_trn_idx), ('val', val_idx), ('test', test_idx)]:
        nz = (comp[idx] > 0).sum(axis=0)
        log(logger=logger, msg=f"{name}: groups={len(np.unique(group_ids[np.searchsorted(exp_indices, idx)]))} "
            + ", ".join(f"{e}={n}" for e, n in zip(elem_names, nz) if n > 0))

    # --- 3. Write Datasets and Attributes to Output HDF5 ---
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 3, 'shuffle': True}
    log(logger=logger, msg=f'Writing splits → {output_path.name}')
    
    with h5py.File(output_path, 'w') as hf:
        if wavelengths is not None:
            hf.create_dataset('wavelengths', data=wavelengths)

        splits = [('train', trn_idx), ('val', val_idx), ('test', test_idx)]

        for split_name, idx in splits:
            grp = hf.create_group(split_name)

            grp.create_dataset(
                'spectra',
                data=spectra[idx],
                chunks=(min(1000, len(idx)), n_features),
                **comp_kwargs
            )

            meta_grp = grp.create_group('metadata')
            for col, arr in metadata.items():
                if len(arr) == n_rows:
                    if np.issubdtype(arr.dtype, np.number):
                        meta_grp.create_dataset(col, data=arr[idx], **comp_kwargs)
                    else:
                        dt = h5py.string_dtype(encoding='utf-8')
                        meta_grp.create_dataset(col, data=arr[idx], dtype=dt)
                else:
                    # Maintain global dataset variables unindexed (e.g., element string lists)
                    meta_grp.create_dataset(col, data=arr)

            grp.attrs['n_rows'] = len(idx)

        # Copy top-level metadata attributes
        for k, v in attrs.items():
            hf.attrs[k] = v
        hf.attrs['train_size'] = len(trn_idx)
        hf.attrs['val_size']   = len(val_idx)
        hf.attrs['test_size']  = len(test_idx)
        hf.attrs['random_state'] = random_state

    size_mb = output_path.stat().st_size / 1_048_576
    log(logger=logger, msg=f'✅ Saved: {output_path.name}  |  {size_mb:.1f} MB')
    return output_path

def trn_val_splitter_CSV( 
    df_path: str | Path,
    trn_path: str | Path,
    val_path: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
    shuffle: bool = True,
    log_path: Path | None = None,
) -> None:
    """Splits a tabular CSV dataset into training and validation CSV files.

    Parameters
    ----------
    df_path : str | Path
        Path to the source CSV dataset.
    trn_path : str | Path
        Destination path for saving the training CSV split.
    val_path : str | Path
        Destination path for saving the validation CSV split.
    test_size : float, default=0.2
        Proportion of dataset samples to include in the validation split.
    random_state : int, default=42
        Random state seed.
    shuffle : bool, default=True
        Whether to shuffle data before splitting.
    log_path : Path | None, optional
        Path for logger output.
    """
    df_path = Path(df_path)
    trn_path = Path(trn_path)
    val_path = Path(val_path)
    from sklearn.model_selection import train_test_split

    # --- Logger Setup ---
    if log_path is None:
        log_path = Path(r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\default_log.txt").resolve()
    logger = get_worker_logger(Path(log_path).stem)

    # --- Validate Input File ---
    if not df_path.is_file():
        raise FileNotFoundError(f'  ❌ Input file not found: {df_path}')
    
    log(logger=logger, msg=f'  ✅ Loading DataFrame from: {df_path}')
    df = pd.read_csv(df_path)

    if df.empty:
        raise ValueError('  ❌ Loaded DataFrame is empty!')
    
    log(logger=logger, msg=f'Original DataFrame Shape: {df.shape}')
    
    # --- Perform Split ---
    trn_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle
    )

    # --- Save Split Datasets ---
    trn_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    log(logger=logger, msg=f'Training DataFrame Shape: {trn_df.shape}')
    log(logger=logger, msg=f'Validation DataFrame Shape: {val_df.shape}')

    trn_df.to_csv(trn_path, index=False)
    val_df.to_csv(val_path, index=False)
    log(logger=logger, msg="Training and Validation Data have been split and saved.")

def training_ready_h5(
    h5_path: str | Path,
    prepped_h5_path: str | Path,
    allowed_cols: set[str] | None = None,
    downsample: bool = True
) -> None:
    """Preprocesses and normalizes raw spectral HDF5 datasets into a model-ready format for machine learning.

    Pipeline operations include:
      1. Dropping shots with high (>10%) NaN counts.
      2. Intensity clipping at 99.5th percentile to remove bright saturation artifacts.
      3. Variance thresholding to remove zero/constant target target features.
      4. MinMax scaling on features and target variables.
      5. Spectral 1D grid downsampling (optional factor of 4).
      6. Expanding dimensions to (N, 1, Channels) for 1D CNN frameworks.
      7. Exporting prepared datasets, metadata, and fitted scalers.

    Parameters
    ----------
    h5_path : str | Path
        Path to input raw split HDF5 file.
    prepped_h5_path : str | Path
        Path to output preprocessed HDF5 file.
    allowed_cols : set[str] | None, optional
        Target concentration columns to keep. Uses default suite if None.
    downsample : bool, default=True
        Whether to downsample spectral resolution by a factor of 4.
    """
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    if h5_path is None:
        h5_path = r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\LIBS\trn_val_split_LIBS.h5"
    if prepped_h5_path is None:
        prepped_h5_path = r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\LIBS\training_ready_LIBS.h5"
    if allowed_cols is None:
        allowed_cols = {
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

    prepped_h5_path = Path(prepped_h5_path)
    output_dir = prepped_h5_path.parent

    # --- Load Data Split arrays ---
    y_trn_raw, y_val_raw, y_test_raw, X_trn_raw, X_val_raw, X_test_raw, train_syn, val_syn, test_syn, target_cols, wavelengths = load_h5_split_dataset(
        h5_path=h5_path,
        allowed_cols=allowed_cols,
        log_path=None
    )

    assert X_trn_raw.shape[0] == y_trn_raw.shape[0], \
        f'X/y row mismatch: {X_trn_raw.shape[0]} vs {y_trn_raw.shape[0]}'

    selector = VarianceThreshold(threshold=1e-10)
    X_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))

    # --- 1. Filter Invalid Sample Rows ---
    nan_frac_trn = np.isnan(X_trn_raw).mean(axis=1)
    nan_frac_val = np.isnan(X_val_raw).mean(axis=1)

    bad_rows_trn = nan_frac_trn > 0.1
    bad_rows_val = nan_frac_val > 0.1

    X_trn_raw = X_trn_raw[~bad_rows_trn]
    X_val_raw = X_val_raw[~bad_rows_val]
    y_trn_raw = y_trn_raw[~bad_rows_trn]
    y_val_raw = y_val_raw[~bad_rows_val]

    meta_val_df = meta_val_df[~bad_rows_val].reset_index(drop=True) # type: ignore # TODO: needs to be fixed before using
                                                                                    # This will definitely throw an error 
                                                                                    # since the meta_val_df is no longer
                                                                                    # part of load_h5_split_dataset. 

    # --- 2. Clip Extreme Peak Intensities ---
    clip_threshold = np.nanpercentile(X_trn_raw, 99.5)
    X_trn_raw = np.clip(X_trn_raw, 0, clip_threshold)
    X_val_raw = np.clip(X_val_raw, 0, clip_threshold)

    # --- 3. Retain Unscaled Target Values for Benchmark Evaluation ---
    y_val_physical = y_val_raw.copy()

    # --- 4. Target Feature Selection (Variance Thresholding) ---
    selector.fit(y_trn_raw)
    surviving_cols = [c for c, keep in zip(target_cols, selector.get_support()) if keep]
    y_trn_filtered = pd.DataFrame(
        np.array(selector.transform(y_trn_raw)),
        columns=surviving_cols)
    y_val_filtered = pd.DataFrame(
        np.array(selector.transform(y_val_raw)),
        columns=surviving_cols)
    meta_val_df = meta_val_df[[c for c in surviving_cols if c in meta_val_df.columns]]

    # --- 5. Feature MinMax Scaling ---
    X_trn_scaled = np.nan_to_num(X_scaler.fit_transform(X_trn_raw), nan=0.0, posinf=0.0, neginf=0.0)
    X_val_scaled = np.nan_to_num(X_scaler.transform(X_val_raw), nan=0.0, posinf=0.0, neginf=0.0)

    # --- 6. Wavelength Downsampling ---
    if downsample:
        downsample_factor = 4
        X_trn_scaled = X_trn_scaled[:, ::downsample_factor]
        X_val_scaled = X_val_scaled[:, ::downsample_factor]

    # --- 7. Reshape to Channel-First Tensor Format (N, C=1, L) ---
    X_trn = np.expand_dims(X_trn_scaled, axis=1).astype(np.float32)
    X_val = np.expand_dims(X_val_scaled, axis=1).astype(np.float32)

    # --- 8. Target Scaling ---
    y_trn = np.nan_to_num(y_scaler.fit_transform(y_trn_filtered), nan=0.0).astype(np.float32)
    y_val = np.nan_to_num(y_scaler.transform(y_val_filtered), nan=0.0).astype(np.float32)

    meta_val_df = meta_val_df.reset_index(drop=True)

    # --- 9. Export Preprocessed Dataset to HDF5 ---
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 3}

    print(f"Writing HDF5 → {prepped_h5_path.name}")
    with h5py.File(str(prepped_h5_path), 'w') as hf:
        # Save arrays
        hf.create_dataset('X_trn', data=X_trn, **comp_kwargs)
        hf.create_dataset('X_val', data=X_val, **comp_kwargs)
        hf.create_dataset('y_trn', data=y_trn, **comp_kwargs)
        hf.create_dataset('y_val', data=y_val, **comp_kwargs)
        hf.create_dataset('y_val_physical', data=y_val_physical, **comp_kwargs)
        hf.create_dataset('wavelengths', data=wavelengths)

        # Save string column vectors
        dt_str = h5py.string_dtype(encoding='utf-8')
        hf.create_dataset('target_cols', data=np.array(target_cols, dtype=object), dtype=dt_str)
        hf.create_dataset('surviving_cols', data=np.array(surviving_cols, dtype=object), dtype=dt_str)

        # Save Validation Metadata natively
        meta_grp = hf.create_group('metadata_val')
        for col in meta_val_df.columns:
            series = meta_val_df[col]
            if pd.api.types.is_numeric_dtype(series):
                arr = series.to_numpy(dtype=np.float32, na_value=np.nan)
                meta_grp.create_dataset(col, data=arr, **comp_kwargs)
            else:
                arr = series.fillna('').astype(str).to_numpy()
                meta_grp.create_dataset(col, data=arr, dtype=dt_str)

        # Attach attributes
        hf.attrs['X_trn_shape'] = list(X_trn.shape)
        hf.attrs['X_val_shape'] = list(X_val.shape)
        hf.attrs['metadata_cols'] = list(meta_val_df.columns)

    # --- 10. Save Fitted Scalers to Disk ---
    with open(output_dir / "X_scaler.pkl", "wb") as f:
        pickle.dump(X_scaler, f)
    with open(output_dir / "y_scaler.pkl", "wb") as f:
        pickle.dump(y_scaler, f)

    size_mb = prepped_h5_path.stat().st_size / 1_048_576
    print(f"✅ Successfully saved training dataset: {prepped_h5_path.name} | {size_mb:.1f} MB")

def xandy_to_crossval(
    input_file,
    output_file    
):
    input_file = Path(input_file).resolve()
    output_file = Path(output_file).resolve()

    with h5py.File(input_file, 'r') as hf:
        feature_cols = hf["feature_cols"][:]        # type: ignore
        wavelengths = hf["wavelengths"][:]        # type: ignore
        
        # Feature matrices (compositions)
        X = np.vstack([hf["X_train"][:], hf["X_val"][:], hf["X_test"][:]])        # type: ignore
        
        # Target matrices (spectra)
        y = np.vstack([hf["y_train"][:], hf["y_val"][:], hf["y_test"][:]])        # type: ignore
        
        # Synthetic flags (0 = exp, 1 = syn)
        is_synthetic = np.concatenate([hf["train_syn"][:], hf["val_syn"][:], hf["test_syn"][:]])        # type: ignore

    print(f"Unified X shape: {X.shape}")
    print(f"Unified y shape: {y.shape}")

    # 2. Generate composition labels & group IDs across the whole dataset
    shot_labels, df_group_lookup = generate_comp_labels(
        comp_data=X,
        elem_names=feature_cols,
        host_elems=('Li', 'K', 'Cl')
    )

    # 3. Write out to unified HDF5 file
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 3}
    str_dt = h5py.string_dtype(encoding="utf-8")

    with h5py.File(output_file, "w") as hf_out:
        # --- Core Data Arrays ---
        hf_out.create_dataset("X", data=X, **comp_kwargs)
        hf_out.create_dataset("y", data=y, **comp_kwargs)
        hf_out.create_dataset("is_synthetic", data=is_synthetic, **comp_kwargs)
        
        # --- Group & System Labels for K-Fold ---
        hf_out.create_dataset("group_id", data=shot_labels["group_id"].to_numpy(dtype=np.int32), **comp_kwargs)
        hf_out.create_dataset("system_code", data=shot_labels["system_code"].astype(str).to_numpy(), dtype=str_dt)
        hf_out.create_dataset("label_short", data=shot_labels["label_short"].astype(str).to_numpy(), dtype=str_dt)
        hf_out.create_dataset("label_full", data=shot_labels["label_full"].astype(str).to_numpy(), dtype=str_dt)

        # --- Axis Metadata ---
        hf_out.create_dataset("feature_cols", data=feature_cols, dtype=str_dt)
        hf_out.create_dataset("wavelengths", data=wavelengths)

        # --- Unique Group Lookup Summary ---
        lookup_grp = hf_out.create_group("group_lookup")
        for col in df_group_lookup.columns:
            series = df_group_lookup[col]
            if pd.api.types.is_numeric_dtype(series):
                lookup_grp.create_dataset(col, data=series.to_numpy())
            else:
                lookup_grp.create_dataset(col, data=series.astype(str).to_numpy(), dtype=str_dt)

    print(f"✅ Successfully saved Group K-Fold dataset to: {output_file}")

# endregion


# =============================================================================
# region: Debug
# =============================================================================
def generate_comp_labels(
    comp_data,
    elem_names=None,
    host_elems=('Li', 'K', 'Cl'),
    round_decimals=4,
    conc_threshold=0.001
):
    """
    Labels unique elemental compositions with system letters, short alphanumeric codes, 
    integer group IDs, and full descriptive strings. 
    """
    # 1. Standardize input data into a Pandas DataFrame
    if isinstance(comp_data, pd.DataFrame):
        df_comp = comp_data.copy()
    else:
        if elem_names is None:
            raise ValueError("elem_names must be provided if comp_data is a NumPy array.")
        # Clean byte strings if loaded directly from h5
        clean_names = [e.decode('utf-8') if isinstance(e, bytes) else str(e) for e in elem_names]
        df_comp = pd.DataFrame(comp_data, columns=clean_names)

    # 2. Round values to eliminate tiny floating-point variations
    df_rounded = df_comp.round(round_decimals)

    # 3. Find unique compositions across all shots
    unique_comps, inverse_indices = np.unique(df_rounded.values, axis=0, return_inverse=True)

    all_elems = list(df_comp.columns)
    dopant_cols = [col for col in all_elems if col not in host_elems]
    dopant_cols.sort()

    # 4. First Pass: Categorize chemical systems (Pure Host, Ce, Ce-Gd, etc.)
    raw_groups = []
    for g_id, comp_row in enumerate(unique_comps):
        row_s = pd.Series(comp_row, index=all_elems)
        active_dopants = []
        dopant_strings = []

        for dopant in dopant_cols:
            wt = row_s[dopant]
            if wt > conc_threshold:
                active_dopants.append(dopant)
                dopant_strings.append(f"{dopant}({wt:.2f}%)")

        if not active_dopants:
            system_key = "PureHost"
            num_dopants = 0
        else:
            system_key = "-".join(active_dopants)
            num_dopants = len(active_dopants)

        raw_groups.append({
            'orig_g_id': g_id, 
            'comp_row': comp_row, 
            'num_dopants': num_dopants, 
            'system_key': system_key, 
            'active_dopants': active_dopants, 
            'dopant_strings': dopant_strings
        })

    # 5. Sort chemical systems: Pure Host (A) -> Single Dopants (B, C) -> Binary (E, F) -> Ternary
    unique_systems = sorted(
        list(set(g['system_key'] for g in raw_groups)),
        key=lambda s: (0 if s == 'PureHost' else len(s.split('-')), s)
    )
    # Map each chemical system to a letter code
    system_letter_map = {sys_key: get_system_letter(idx) for idx, sys_key in enumerate(unique_systems)}

    # 6. Second Pass: Build label table
    system_counts = {}
    group_info = []
    
    for item in raw_groups:
        g_id = item['orig_g_id']
        system_key = item['system_key']
        sys_letter = system_letter_map[system_key]

        system_counts[system_key] = system_counts.get(system_key, 0) + 1
        sys_num = system_counts[system_key]
        short_label = f"{sys_letter}{sys_num}"

        if not item['dopant_strings']:
            full_label = f"{g_id:02d}_{sys_letter}_PureHost"
        else:
            full_label = f"{g_id:02d}_{sys_letter}_" + "_".join(item['dopant_strings'])

        group_info.append({
            'group_id': g_id, 
            'system_key': system_key, 
            'system_code': sys_letter, 
            'label_short': short_label, 
            'label_full': full_label
        })

    # Convert group info into DataFrame after completing loop over all raw_groups
    df_group_lookup = pd.DataFrame(group_info)

    # Map group labels back to all original dataset shots
    shot_labels = df_group_lookup.iloc[inverse_indices].reset_index(drop=True)

    return shot_labels, df_group_lookup

def get_system_letter(index):
    """
    Generates letter codes: 0->A, 1->B, ..., 25->Z, 26-AA, ...
    """
    result = ""
    while True:
        result = chr(65 + (index % 26)) + result
        index = index // 26 - 1
        if index < 0:
            break
        return result

def get_worker_logger(
        name: str, log_file: Path | None = None
) -> logging.Logger:
    """ Instantiate or retrieve a logger configured for safe multiprocess logging. """
    logger = logging.getLogger(f'worker.{name}')
    logger.setLevel(logging.INFO)

    if log_file and not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def log(logger: logging.Logger, msg: str) -> None:
    """ Output log messages to active file handlers or print to standard output. """
    if logger.handlers:
        logger.info(msg)
    else:
        print(msg)

def summarize_compositions_in_h5(
    file_path: str | Path,
    label: str
) -> None:
    """Prints detailed summary statistics regarding elemental compositions and sample shot counts
    contained within a specific dataset split of an HDF5 file.

    Parameters
    ----------
    file_path : str | Path
        Path to the source HDF5 dataset file.
    label : str
        Target split designation label (e.g., 'train (experimental only)', 'val', or 'test').
    """
    # --- 1. Load Data Splits from HDF5 ---
    with h5py.File(file_path, 'r') as hf:
        names = [n.decode() if isinstance(n, bytes) else n for n in hf['feature_cols'][:]]      # type: ignore
        X   = {s: hf[f'X_{s}_backup'][:] for s in ['train', 'val', 'test']}      # type: ignore
        syn = {s: hf[f'{s}_syn'][:]      for s in ['train', 'val', 'test']}      # type: ignore

    # --- 2. Select Relevant Experimental/Synthetic Samples ---
    if label == 'train (experimental only)':
        Xs = X['train'][syn['train'] == 0]      # type: ignore
    elif label == 'val':
        Xs = X['val']      # type: ignore
    else:
        Xs = X['test']      # type: ignore

    # --- 3. Identify Unique Compositional Clusters ---
    Xr = np.round(Xs, 4)      # type: ignore
    comps, inv = np.unique(Xr, axis=0, return_inverse=True)
    inv = inv.ravel()
    sizes = np.sort(np.bincount(inv))[::-1]

    # --- 4. Print Summary Report ---
    print(f"\n=== {label}: {len(Xr)} shots, {len(comps)} unique compositions ===")
    print(f"largest groups (shots): {sizes[:5].tolist()}")
    print(f"{'elem':<4} {'shots>0':>9} {'groups>0':>9} {'max wt%':>9}")
    
    for j, e in enumerate(names):
        nz = Xr[:, j] > 0
        ng = len(np.unique(inv[nz])) if nz.any() else 0
        print(f"{e:<4} {nz.sum():>9} {ng:>9} {Xr[:, j].max():>9.3f}")

def check_h5_compositions(h5_path: str | Path):
    """
    Inspects an HDF5 dataset to count overall unique experimental compositions
    and displays a per-element breakdown of shots and unique composition groups.
    """
    h5_path = Path(h5_path)
    
    with h5py.File(h5_path, 'r') as hf:
        # 1. Load Concentrations Matrix X
        if 'X' in hf:
            X = hf['X'][:]      # type: ignore
        elif 'metadata/elem_comp_wt%' in hf:
            X = hf['metadata/elem_comp_wt%'][:]      # type: ignore
        else:
            raise KeyError("Could not locate concentration matrix ('X' or 'metadata/elem_comp_wt%').")

        # 2. Load Element Names
        if 'feature_cols' in hf:
            names = [n.decode('utf-8') if isinstance(n, bytes) else str(n) for n in hf['feature_cols'][:]]      # type: ignore
        elif 'metadata/elem_names' in hf:
            names = [n.decode('utf-8') if isinstance(n, bytes) else str(n) for n in hf['metadata/elem_names'][:]]      # type: ignore
        else:
            names = [f"Elem_{i}" for i in range(X.shape[1])]      # type: ignore

        # 3. Load Domain Label (0 = Experimental, 1 = Synthetic)
        if 'is_synthetic' in hf:
            is_synth = hf['is_synthetic'][:]      # type: ignore
        elif 'metadata/is_synthetic' in hf:
            is_synth = hf['metadata/is_synthetic'][:]      # type: ignore
        else:
            is_synth = np.zeros(X.shape[0], dtype=int)      # type: ignore

        # 4. Load or Derive Composition Group IDs
        if 'group_id' in hf:
            group_id = hf['group_id'][:]      # type: ignore
        else:
            # If group_id isn't pre-computed, group identical concentration vectors (rounded to 4 decimals)
            _, group_id = np.unique(np.round(X, 4), axis=0, return_inverse=True)      # type: ignore

    # Isolate experimental data (synthetic spectra are excluded from physical composition counts)
    exp_mask = (is_synth == 0)
    X_exp = X[exp_mask]      # type: ignore
    groups_exp = group_id[exp_mask]      # type: ignore

    # Calculate global unique composition count
    total_unique_exp_groups = len(np.unique(groups_exp))      # type: ignore

    print(f"\n========================================================")
    print(f" Dataset Composition Report: {h5_path.name}")
    print(f"========================================================")
    print(f"Total Samples (Shots): {len(X)} ({exp_mask.sum()} Exp | {(~exp_mask).sum()} Synth)")      # type: ignore
    print(f"Total Unique Experimental Composition Groups: {total_unique_exp_groups}")
    print(f"--------------------------------------------------------")
    print(f"{'Element':<12} {'Shots > 0':>10} {'Unique Groups':>15} {'Max wt%':>10}")
    print(f"--------------------------------------------------------")

    host_matrix = {'Li', 'K', 'Cl'}
    summary_records = []

    for j, elem_name in enumerate(names):
        active_shots = X_exp[:, j] > 0      # type: ignore
        n_shots = int(active_shots.sum())
        
        # Count unique group_ids where this specific element is present
        n_groups = len(np.unique(groups_exp[active_shots])) if n_shots > 0 else 0      # type: ignore
        max_wt = float(X_exp[:, j].max()) if n_shots > 0 else 0.0      # type: ignore

        label = f"{elem_name} (Host)" if elem_name in host_matrix else elem_name
        print(f"{label:<12} {n_shots:>10} {n_groups:>15} {max_wt:>10.3f}")

        summary_records.append({
            'element': elem_name,
            'shots': n_shots,
            'unique_groups': n_groups,
            'max_wt': max_wt
        })

    print(f"--------------------------------------------------------\n")
    return pd.DataFrame(summary_records)
# endregion

# --- Entry Point Script Execution ---
if __name__ == "__main__":
    print('Hi')

    xandy_path = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/cts_noleak_xandy.h5"
    crossval_path = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/cts_noleak_crossval.h5"

    # xandy_to_crossval(input_file=xandy_path, output_file=crossval_path)

    # with h5py.File(crossval_path, 'r') as hf:
    #     # Load the datasets into memory using [:]
    #     is_synthetic = hf['is_synthetic'][:]       # type: ignore
    #     group_id = hf['group_id'][:]       # type: ignore
        
    #     # Now the numpy operations will work
    #     exp = is_synthetic == 0
    #     print(np.unique(group_id[exp]).size)       # type: ignore
    #     print(np.unique(group_id[~exp]))       # type: ignore
    #     print(np.intersect1d(group_id[exp], group_id[~exp]).size)       # type: ignore   # expect 0: no ID shared by exp and synthetic rows
    #     print(np.unique(group_id[~exp]).size)       # type: ignore                       # roughly 162,948 if the IDs are disjoint

    # Example composition summarization run
    path = '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/cts_noleak_crossval.h5'
    summarize_compositions_in_h5(file_path=path, label='train (experimental only)')
    summarize_compositions_in_h5(file_path=path, label='val')
    summarize_compositions_in_h5(file_path=path, label='test')