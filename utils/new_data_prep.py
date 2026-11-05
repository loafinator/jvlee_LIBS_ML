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

# Standard Library Imports
import re 
import logging
import os
import warnings
import h5py
import hashlib
import time
import csv

# Third-Party Scientific & Data Processing Imports
import pandas as pd 
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

# Typing, Concurrency, and Functional Utilities
from pathlib import Path 
from typing import List, Optional, cast, Any
from tqdm import tqdm
from h5py import Group, Dataset
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from functools import partial
from logging import handlers
from collections.abc import Container
from numpy.typing import NDArray
from sklearn.feature_selection import VarianceThreshold
from scipy.ndimage import minimum_filter1d, uniform_filter1d

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
    'ms7': 0.1,
    'ms8': 0.1,
    'ms9': 0.5,
    'ms10': 1.0,
    'ms11': 1.0,
    'ms12': 3.0,
    'ms13': 5.0,
    'ms14': 5.0,
    'ms16': 0.5,
}

_MSU_CONC_MAP = {
    'msu1': 1.0, 
    'msu2': 1.0
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
    '1 rep' : [0.288, 0.269],
    '2 rep' : [0.294, 0.565],
    '3 rep' : [0.327, 1.247],
    '4 rep' : [0.306, 1.744],
    '5 rep' : [0.287, 2.309],
    '6 rep' : [0.318, 3.138],
    '7 rep' : [0.674, 0.370],
    '8 rep' : [0.674, 0.632],
    '9 rep' : [0.556, 1.088],
    '10 rep' : [0.580, 1.657],
    '11 rep' : [0.592, 2.244],
    '12 rep' : [0.586, 2.779],
    '13 rep' : [1.138, 0.287],
    '14 rep' : [1.147, 0.540],
    '15 rep' : [1.153, 1.121],
    '16 rep' : [1.165, 1.685],
    '17 rep' : [1.135, 2.217],
    '18 rep' : [1.146, 2.768],
    '19 rep' : [1.741, 0.292],
    '20 rep' : [1.693, 0.541],
    '21 rep' : [1.671, 1.056],
    '22 rep' : [1.687, 1.606],
    '23 rep' : [1.746, 2.219],
    '24 rep' : [1.738, 2.764],
    '25 rep' : [2.212, 0.279],
    '26 rep' : [2.212, 0.527],
    '27 rep' : [2.273, 1.037],
    '28 rep' : [2.192, 1.509],
    '29 rep' : [2.257, 2.179],
    '30 rep' : [2.284, 2.740],
    '31 rep' : [2.823, 0.284],
    '32 rep' : [2.792, 0.539],
    '33 rep' : [2.819, 1.067],
    '34 rep' : [2.796, 1.605],
    '35 rep' : [2.896, 2.277],
    '36 rep' : [2.817, 2.755],
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

# region: patterns
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
_SUMMARY_PATTERNS_REGEX = re.compile(
    r'^(Data_Import_.*\.csv|Static_.*_Values_.*\.csv|Static_Import.*\.csv|'
    r'Solid_Normalized_Spectra.*\.csv|Pressure_Import_.*\.csv|Energy_Import_.*\.csv|'
    r'Flow_Import_.*\.csv|.*_SNR_Import_.*\.csv|.*_SNR_Values_.*\.csv|Test_SNR_.*\.csv)$',
    re.IGNORECASE,
)
_EXCLUDED_DIR_PATTERNS_REGEX = re.compile(
    r'(Backup 9-12-2016|Williams_Ammon_Ned|Matlab analysis)', re.IGNORECASE
)
# endregion

# Standardized experimental defaults applied when metadata parameters are unlisted
# region: standard vals
_STANDARD_DELAY: int = 14     # Default gate delay (µs)
_STANDARD_WIDTH: int = 8      # Default gate width (µs)
_STANDARD_QDELAY: int = 110   # Default Q-switch delay (µs)
_STANDARD_SHOTS: int = 50     # Default laser shot count per specimen
# endregion

_DEFAULT_CONC_COLS: List[str] = [
    'frac_LiCl',        'frac_KCl', 
    'conc_Ba_wt%',      'conc_Ca_wt%',      'conc_Ce_wt%',      'conc_Cl_wt%',
    'conc_Cr_wt%',      'conc_Cs_wt%',      'conc_Fe_wt%',      'conc_Gd_wt%',
    'conc_K_wt%',       'conc_La_wt%',      'conc_Li_wt%',      'conc_Mg_wt%',
    'conc_Mn_wt%',      'conc_Nd_wt%',      'conc_Ni_wt%',      'conc_Sm_wt%',
    'conc_Sr_wt%',      'conc_U_wt%',       'conc_Y_wt%',

    'conc_CeCl3_wt%',   'conc_CaCl2_wt%',   'conc_UCl3_wt%',    'conc_SmCl3_wt%',
    'conc_GdCl3_wt%',   'conc_LaCl3_wt%',   'conc_MgCl2_wt%',   'conc_H2o_wt%',
    'conc_NdCl3_wt%',   'conc_CsCl_wt%',    'conc_SrCl2_wt%',   'conc_BaCl2_wt%',
    'conc_YCl3_wt%',    'conc_FeCl2_wt%',   'conc_CrCl2_wt%',   'conc_NiCl2_wt%',
    'conc_MnCl2_wt%',
]

# Canonical species name mapping to enforce consistent casing and structural naming
_SPECIES_MAP: dict[str, str] = {
    'CECL3': 'CeCl3',       'CE': 'Ce',
    'CACL2': 'CaCl2',       'CA': 'Ca',
    'UCL3':  'UCl3',        'U':  'U',
    'SMCL3': 'SmCl3',       'SM': 'Sm',
    'GDCL3': 'GdCl3',       'GD': 'Gd',
    'LACL3': 'LaCl3',       'LACL': 'LaCl3',        'LA': 'La',
    'MGCL2': 'MgCl2',       'MG': 'Mg',
    'NDCL3': 'NdCl3',       'ND': 'Nd',
    'CRCL2': 'CrCl2',       'CR': 'Cr',
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

# region Gloabl Inputs
DROP_COLUMNS = None
RENAME_COLUMNS = None
COMPOSITION_COLUMNS = _DEFAULT_CONC_COLS
_META_KEEP = set(_ALL_COLUMNS) | {'file_id'}

# Spectral Wavelength Alignment
MIN_WAVELENGTH = 250.0   # Aligned with TARGET_MIN_WL (250.0 nm)
MAX_WAVELENGTH = 1000.0  # Aligned with TARGET_MAX_WL (1000.0 nm)

ALLOWED_EXTENSIONS = ['.csv']
REQUIRED_COLUMNS = None
SALT_STATES = None       # Defaults to _DEFAULT_SALT_STATES inside data_prep
EXPERIMENTAL_VARIATION_COLUMNS = None
INCLUDE_EXPERIMENTAL_VARIATION_COLUMNS = True

# Logging & Multiprocessing
LOG_Q = None
LOG_PATH = Path('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/logs/data_rerun.txt')
N_WORKERS = 30

# HDF5 Compression
COMPRESSION = 'gzip'
COMPRESSION_LEVEL = 3
TRANSPOSE_NEEDED = False

# Dataset Cleaning & Filtering Parameters
TARGET_ELEMENTS = ['Ce', 'Gd', 'Sm', 'U']
CLIP_PERCENTILE = None   # Float (e.g., 99.5) or None
TARGET_DELAY = 14.0      # Float (e.g., 14.0) or None
TARGET_ENERGY = None    # Float (e.g., 100.0) or None
TARGET_STATE_MOLTEN = None  # Int (1) or None
# RAW_SATURATION_THRESH = 60000.0
# endregion

MW_ELEMENTS = {
    'Ba': 137.327, 'Ca': 40.078, 'Ce': 140.116, 'Cr': 51.996, 'Cs': 132.905,
    'Fe': 55.845,  'Gd': 157.25,  'La': 138.905, 'Li': 6.941,  'K': 39.098,
    'Mg': 24.305,  'Mn': 54.938,  'Nd': 144.242, 'Ni': 58.693, 'Sm': 150.36,
    'Sr': 87.62,   'U': 238.029,  'Y': 88.906,   'Cl': 35.453
}

NUM_CL = {
    'Ba': 2, 'Ca': 2, 'Ce': 3, 'Cr': 2, 'Cs': 1, 'Fe': 2, 
    'Gd': 3, 'K': 1, 'La': 3, 'Li': 1, 'Mg': 2, 'Mn': 2, 
    'Nd': 3, 'Ni': 2, 'Sm': 3, 'Sr': 2, 'U': 3, 'Y': 3
}
# =============================================================================
# 1. Enrich data
# =============================================================================
def enrich_with_progress(
        files: List[Path],
        enriched_root: Path,
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
        n_workers: int | None = None,
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
        allowed_extensions=allowed_extensions,
        required_columns=required_columns,
        salt_states=salt_states,
        experimental_variation_columns=experimental_variation_columns,
        include_experimental_variation_columns=include_experimental_variation_columns,
        min_wavelength=min_wavelength,
        max_wavelength=max_wavelength,
        log_path=log_path
    )

    if n_workers is None:
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
            unit="file"
        ):
            result = future.result()
            if result is not None:
                enriched_paths.append(result)

    return enriched_paths

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

    og_path = Path(path)

    all_text = (' '.join(og_path.parts) + ' ' + og_path.name).lower()

    if _SUMMARY_PATTERNS_REGEX.search(og_path.name) or _EXCLUDED_DIR_PATTERNS_REGEX.search(all_text):
        return None

    if log_path is None:
        log_path = Path(r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\default_log.txt").resolve()
    logger = get_worker_logger(Path(log_path).stem)

    sanitized_path = sanitize_path(path=og_path, log_path=log_path)
    if sanitized_path is None:
        log(logger=logger, msg=f'          Skipping (could not sanitize path): {og_path.name}')
        return None
    path = sanitized_path

    enriched_root = Path(enriched_root)
    enriched_root.mkdir(parents=True, exist_ok=True)

    uid = hashlib.md5(str(path).encode()).hexdigest()[:8]

    # file_w_parents = f'{path.name}'
    file_w_parents = f'{uid}_{path.name}'    
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

    df = clean_single_file(
        path=path,
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

# 1.a. Sanitize the path
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

# 1.b. Clean the file
def clean_single_file(
        path: Path | str,
        drop_columns: Optional[List[str]] = None,
        rename_columns: Optional[dict] = None,
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

# 1.c. Extract composition data from path
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
        data['frac_LiCl'] = 0.45  # Default LiCl eutectic wt% fraction
        data['frac_KCl'] = 0.55   # Default KCl eutectic wt% fraction
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
            data['energy'] = np.nan

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
            pattern = re.compile(
                rf"(?<![\d.])(\d+\.?\d*)\s*(?:_str|_wt%?|wt%?|ppm)?\s*[-_ ]?\s*{re.escape(key)}(?=[_\s.\-/\\(]|$)",
                re.IGNORECASE)
            
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

        # Step 16b: concentrations written in parent foldernames (ex. "5 wt_ CeCl3")
        folder_text = ' '.join(path.parts[:-1]).upper()
        salt_keys = [k for k in sorted_keys if k.endswith(('CL', 'CL2', 'CL3'))]
        for key in salt_keys:
            col = f"conc_{_SPECIES_MAP[key]}_wt%"
            if col not in _DEFAULT_CONC_COLS or data.get(col, 0.0) != 0.0:
                continue   # the filename already supplied a value
            m = re.search(rf"(?<![\d.])(\d+\.?\d*)\s*(?:WT%?|%|WT_)\s*[-_ ]?{re.escape(key)}(?=[_\s.\-/\\(]|$)", folder_text)
            if m:
                data[col] = float(m.group(1))

        # Step 16c: folder rule for ppm
        if 'CECL3-LICL-KCL-H2O' in folder_text and data.get('conc_Ce_wt%', 0.0) == 0.0:
            PPM = r'(\d+\.?\d*)\s*PPM(?![A-Z])'
            name_ppm = re.search(PPM, path.name.upper())
            dir_ppm = re.search(PPM, path.parent.name.upper())
            if name_ppm and dir_ppm and float(name_ppm.group(1)) != float(dir_ppm.group(1)):
                data['conc_Ce_wt%'] = np.nan      # filename and folder disagree: exclude
            elif name_ppm or dir_ppm:
                data['conc_Ce_wt%'] = float((name_ppm or dir_ppm).group(1)) / 10000.0 # type: ignore

        # Step 17: Apply Hardcoded Concentration Mapping Libraries
        if 'wavelength (nm),sum,' in all_text:
            for key, target_col in sorted(_SINGLES_CONC_MAP.items(), key=lambda x: -len(x[0])):
                pattern = rf'wavelength \(nm\),sum,{key.lower()}(?:\s+[ivxlcdm]+)?'
                if re.search(pattern, all_text, re.IGNORECASE):
                    if target_col in composition_columns:
                        data[target_col] = 1.0
                    break

        if re.search(r'(?<![a-z])ms\d', all_text) and not re.search(r'msu\d', all_text):
            if data.get('blank') == 1:
                data['conc_CeCl3_wt%'] = 0.0
            else:
                # av_conc = sum(_MS_CONC_MAP.values()) / len(_MS_CONC_MAP)
                for key, conc in sorted(_MS_CONC_MAP.items(), key=lambda x: -len(x[0])):
                    if key in all_text:
                        data['conc_CeCl3_wt%'] = conc
                        break
                else:
                    data['conc_CeCl3_wt%'] = np.nan

        if re.search(r'msu\d', all_text):
            if 'ms16_300' in all_text:
                data['conc_UCl3_wt%'] = np.nan        # misfiled? excluded
            elif data.get('blank') == 1:
                data['conc_UCl3_wt%'] = 0.0
            elif data.get('conc_UCl3_wt%', 0.0) == 0.0:
                m = re.search(r'msu(\d)', all_text)
                assert m is not None
                data['conc_UCl3_wt%'] = _MSU_CONC_MAP.get(f'msu{m.group(1)}', np.nan)

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

        if re.search(r'trial 1|trial 2', all_text):
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


# =============================================================================
# 2. Convert it all to .h5 file prior to full dataset processing
# =============================================================================
def combine_and_save_as_HDF5(
        individuals_root: str | Path,
        combined_root: str | Path,
        compression: str = 'gzip',
        compression_level: int = 3,
        log_path: Path | None = None,
        transpose_needed: bool = False,
        n_workers: int | None = None,
) -> Optional[Path]:
    """ Parallel process individual spectral CSVs into a structured HDF5 file 
    using direct dataset streaming to minimize RAM usage.
    """
    individuals_root = Path(individuals_root).resolve()
    combined_root = Path(combined_root).resolve()
    combined_root.mkdir(parents=True, exist_ok=True)
    
    if log_path is None:
        log_path = Path(r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\default_log.txt").resolve()
    logger = get_worker_logger(Path(log_path).stem)

    enriched_files = list(individuals_root.glob('*.csv'))
    if not enriched_files:
        log(logger=logger, msg=f'  ❌ No enriched files found.')
        return None
    
    if n_workers is None:
        n_workers = int(os.getenv("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
        
    log(logger=logger, msg=f'Found files: {len(enriched_files)}')
    log(logger=logger, msg=f'Streaming data across {n_workers} CPU workers into HDF5...')

    target_grid = np.linspace(TARGET_MIN_WL, TARGET_MAX_WL, TARGET_N_PTS)
    h5_path = combined_root / 'combined_libs.h5'
    
    comp_kwargs = {'compression': compression}
    if compression == 'gzip':
        comp_kwargs['compression_opts'] = compression_level # type: ignore

    all_metadata = []
    completed_count = 0
    CHUNK_ROWS = 64

    # Open HDF5 file and initialize a resizable dataset
    with h5py.File(h5_path, 'w', rdcc_nbytes=256 * 1024**2, rdcc_nslots=100_003) as hf:
        spectra_ds = hf.create_dataset(
            'spectra',
            shape=(0, TARGET_N_PTS),
            maxshape=(None, TARGET_N_PTS),
            chunks=(CHUNK_ROWS, TARGET_N_PTS),
            dtype=np.float32,
            **comp_kwargs,
        )
        hf.create_dataset('wavelengths', data=target_grid)

        tasks = [
            (f, TARGET_MIN_WL, TARGET_MAX_WL, TARGET_N_PTS, log_path, transpose_needed)
            for f in enriched_files
        ]

        task_iter = iter(tasks)
        t0 = time.time()

        executor = ProcessPoolExecutor(max_workers=n_workers)
        try:
            pending = set()

            def submit_next() -> None:
                task = next(task_iter, None)
                if task is not None:
                    pending.add(executor.submit(_process_single_file, task))

            for _ in range(n_workers * 2):
                submit_next()

            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    pending.discard(fut)
                    submit_next()
                    res = fut.result()
                    completed_count += 1

                    if res is not None:
                        spectra, meta_df, _ = res
                        n0 = spectra_ds.shape[0]
                        spectra_ds.resize(n0 + spectra.shape[0], axis=0)
                        spectra_ds[n0:n0 + spectra.shape[0], :] = spectra
                        all_metadata.append(meta_df)

                    if completed_count % 100 == 0 or completed_count == len(enriched_files):
                        log(logger=logger,
                            msg=f'  {completed_count}/{len(enriched_files)} files, {time.time() - t0:.0f}s')
        finally:
            executor.shutdown(wait=False)

        if not all_metadata:
            log(logger=logger, msg='  ❌ No valid files loaded.')
            return None

        # Write metadata table directly
        # Write metadata table directly
        log(logger=logger, msg='Writing metadata tables...')
        metadata_df = pd.concat(all_metadata, ignore_index=True, sort=False)
        if metadata_df.shape[1] > 200:
            raise RuntimeError(f'Metadata has {metadata_df.shape[1]} columns; header parsing is leaking wavelengths')
        
        meta_grp = hf.create_group('metadata')
        for col in metadata_df.columns:
            series = metadata_df[col]
            if pd.api.types.is_numeric_dtype(series):
                arr = series.to_numpy(dtype=np.float32, na_value=np.nan)
                meta_grp.create_dataset(col, data=arr, **comp_kwargs)
            else:
                # Convert strings to fixed-length bytes for instant HDF5 writing
                arr = series.fillna('').astype(str).to_numpy()
                arr_bytes = arr.astype('S')
                meta_grp.create_dataset(col, data=arr_bytes, **comp_kwargs)

        hf.attrs['n_rows'] = spectra_ds.shape[0]
        hf.attrs['n_wavelengths'] = TARGET_N_PTS
        hf.attrs['min_wavelength'] = float(target_grid[0])
        hf.attrs['max_wavelength'] = float(target_grid[-1])
        hf.attrs['metadata_cols'] = list(metadata_df.columns)

    size_mb = h5_path.stat().st_size / 1_048_576
    log(logger=logger, msg=f'✅ Saved: {h5_path.name}  |  {size_mb:.1f} MB')
    return h5_path

# 2.a. process a single file and then append it into the full dataset
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
    log_path = Path(r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\default_log.txt").resolve()
    logger = get_worker_logger(Path(log_path).stem)
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
        meta_df = meta_df.loc[:, [c for c in meta_df.columns if c in _META_KEEP]]
        
        return spectra, meta_df, target_grid
    except Exception as e:
        log(logger=logger, msg=f'Process single file error: {Path(f).name}: {e}')
        return None

# 2.a.i Conform file to the standard wavelength grid
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
        with open(file_path, newline='') as fh:
            hdr = next(csv.reader(fh), [])
    except Exception as e:
        log(logger=logger, msg=f'  ❌ Could not read {file_path.name}: {e}')
        return None
    n_dup = sum(bool(re.fullmatch(r'\d+\.\d+\.\d+', c)) for c in hdr)
    if n_dup > 100:
        log(logger=logger, msg=f'  ⚠️ Skipping {file_path.name}: {n_dup} repeated wavelength headers')
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


# =============================================================================
# 3. Dataset processing
# =============================================================================
# so this is where most of the actual data cleaning happens. These include things
# such as:
#   - asdf 
def dataset_cleaning(
    input_h5: Path | str,
    output_h5: Path | str,
    target_elements: list[str] | None = None,
    clip_percentile: float | None = None,# Optional parameters to filter conditions dynamically
    target_delay: float | None = 14.0,
    target_energy: float | None = None,
    target_state_molten: int | None = 1,
) -> None:
    """Prepares raw spectral HDF5 dataset into clean X (Spectra) and y (Concentrations) for PLS."""
    input_h5, output_h5 = Path(input_h5), Path(output_h5)

    EXCLUDE = ['Gd-Mg Study', 'Iron with rust', 'Fiber optic test', 'Repetability Test',
            'Pure Mg Study', 'MS U Analysis', 'MS Ce Analysis',
            'Values_lamba', 'Values_Li_lamba', 'Static_Import_lamba']
    
    with h5py.File(input_h5, 'r') as hf_in:
        raw_spectra = get_h5_ds(hf_in,'spectra')[:]
        wavelengths = get_h5_ds(hf_in,'wavelengths')[:]
        metadata = cast(h5py.Group, get_h5_ds(hf_in, 'metadata'))
        
        # Load metadata columns into DataFrame
        meta_dict = {}
        for k in metadata.keys():
            data = cast(h5py.Dataset, metadata[k])[:]

            if data.dtype.kind in ('O', 'S', 'V'):
                meta_dict[k] = [x.decode('utf-8') if isinstance(x, bytes) else x for x in data]
            else:
                meta_dict[k] = data
        df_meta = pd.DataFrame(meta_dict)
        
    # Step 1a: Remove NaN and zero-intensity rows
    valid_mask = ~np.isnan(raw_spectra).any(axis=1) & (raw_spectra.sum(axis=1) > 0)
    conc_cols = [c for c in df_meta.columns if c.startswith('conc_')]
    valid_mask &= ~df_meta[conc_cols].isna().any(axis=1).to_numpy()

    og = df_meta['og_path'].astype(str)
    for pat in EXCLUDE:
        hit = og.str.contains(re.escape(pat), case=False).to_numpy()
        print(f'exclude "{pat}": {(hit & valid_mask).sum()} rows')
        valid_mask &= ~hit

    # energy is only trustworthy when parsed explicitly from the path
    if {'energy', 'energy_study'} <= set(df_meta.columns):
        df_meta.loc[df_meta['energy_study'] == 0, 'energy'] = np.nan

    # Step 1a-ii: drop exact duplicate spectra (copies in several backup folders)
    og_arr = df_meta['og_path'].astype(str).to_numpy()
    conc_vals = df_meta[conc_cols].to_numpy(np.float64)
    study_cols = [c for c in df_meta.columns if c.endswith('_study')]
    explicit = (df_meta[study_cols].to_numpy(np.float64).sum(axis=1)
                if study_cols else np.zeros(len(df_meta)))

    groups_by_hash = {}
    for i in np.flatnonzero(valid_mask):
        h = hashlib.blake2b(raw_spectra[i].tobytes(), digest_size=16).digest()
        groups_by_hash.setdefault(h, []).append(i)

    keep = np.zeros(len(valid_mask), dtype=bool)
    n_conflict = 0
    for idx in groups_by_hash.values():
        if len(idx) > 1 and not (conc_vals[idx] == conc_vals[idx[0]]).all():
            n_conflict += 1                  # same spectrum, different concentration labels
            continue
        keep[min(idx, key=lambda i: (-explicit[i], og_arr[i]))] = True   # most explicit copy; ties by path

    print(f'duplicate rows removed: {int(valid_mask.sum() - keep.sum()) - n_conflict}; '
          f'groups dropped for conflicting concentrations: {n_conflict}')
    valid_mask &= keep

    spectra = raw_spectra[valid_mask]
    del raw_spectra
    df_meta = df_meta[valid_mask].reset_index(drop=True)

    # diagnostic only (delete once you've settled on filters)
    df_diag = compute_elemental_wt(df_meta)
    diag_cols = [c for c in ['state_molten', 'delay', 'energy'] if c in df_meta.columns]
    tmp = pd.concat([df_meta[diag_cols], df_diag[['Ce', 'Gd', 'Sm', 'U']].round(3)], axis=1)
    for e in ['Ce', 'Gd', 'Sm', 'U']:
        print(e)
        print(tmp[tmp[e] > 0].groupby(diag_cols, dropna=False)[e].agg(['size', 'nunique']))


    # Step 1b: Apply experimental condition mask
    condition_mask = np.ones(len(df_meta), dtype=bool)

    if target_delay is not None and 'delay' in df_meta.columns:
        condition_mask &= (df_meta['delay'] == target_delay)
        
    if target_energy is not None and 'energy' in df_meta.columns:
        condition_mask &= (df_meta['energy'] == target_energy)

    if target_state_molten is not None and 'state_molten' in df_meta.columns:
        condition_mask &= (df_meta['state_molten'] == target_state_molten)

    # Apply mask to keep spectra and metadata synchronized
    spectra = spectra[condition_mask]
    df_meta = df_meta[condition_mask].reset_index(drop=True)

    print(f"Filtered dataset down to {len(spectra)} shots matching condition parameters.")

    # --- Filter Cosmic Rays & Saturated Detector Shots on RAW Counts ---
    # Compute max counts directly on the synchronized spectra array
    raw_max_counts = spectra.max(axis=1)
    unsaturated_mask = raw_max_counts #<= RAW_SATURATION_THRESH

    # print(f"Dropping {(~unsaturated_mask).sum()} shots exceeding raw count threshold {RAW_SATURATION_THRESH}")

    spectra = spectra[unsaturated_mask]
    df_meta = df_meta[unsaturated_mask].reset_index(drop=True)

    # Step 2: Intensity Clipping & LIBS Spectral Cleaning (Baseline + Area Norm)
    X_clean = clean_libs_spectrum(spectra)

    # Step 3: Elemental Stoichiometric Conversion for Targets (y)
    df_elemental_all = compute_elemental_wt(df_meta)
    if target_elements:
        df_elemental = df_elemental_all[[e for e in target_elements if e in df_elemental_all.columns]]
    else:
        df_elemental = df_elemental_all
    aux_cols = [c for c in df_elemental_all.columns if c not in df_elemental.columns]
    
    for c in df_elemental.columns:
        nz = df_elemental[c] > 0
        print(f"{c}: {nz.sum()} shots, {np.unique(np.round(df_elemental.loc[nz, c], 3)).size} distinct levels")

    # Step 4: Drop Inactive Target Columns (Zero Variance)
    selector = VarianceThreshold(threshold=1e-6)
    y_filtered = selector.fit_transform(df_elemental)
    surviving_cols = df_elemental.columns[selector.get_support()].tolist()

# --- 4a. Compute Group IDs (with Zero-Group Splitting) ---
    _, initial_groups = np.unique(np.round(y_filtered, 3), axis=0, return_inverse=True)
    
    # Split all-zero shots by file_id so blank samples don't land in one single group
    is_zero_shot = (y_filtered < 1e-5).all(axis=1)
    if is_zero_shot.any():
        file_ids = df_meta['file_id'].to_numpy()
        _, zero_file_inv = np.unique(file_ids[is_zero_shot], return_inverse=True)
        max_nonzero_grp = initial_groups[~is_zero_shot].max() if (~is_zero_shot).any() else 0
        
        groups = initial_groups.copy()
        groups[is_zero_shot] = max_nonzero_grp + 1 + zero_file_inv
    else:
        groups = initial_groups

    # --- 4b. Generate Group-Stratified Lockbox Split (~15% of unique groups) ---
    unique_groups = np.unique(groups)
    n_lockbox_groups = max(1, int(0.15 * len(unique_groups)))

    # Determine dopant presence per unique group for stratification
    group_present = np.array([
        (y_filtered[groups == g] > 1e-5).any(axis=0) for g in unique_groups
    ])

    sig = (group_present.astype(int) * (1 << np.arange(group_present.shape[1]))).sum(axis=1)
    rng = np.random.default_rng(42)

    lockbox_groups = []
    for s in np.unique(sig):
        members = rng.permutation(unique_groups[sig == s])
        n_select = max(1, int(0.15 * len(members))) if len(members) > 1 else (1 if len(lockbox_groups) < n_lockbox_groups else 0)
        lockbox_groups.extend(members[:n_select])

    is_lockbox = np.isin(groups, lockbox_groups)

    # Step 5: Save Clean PLS-Ready Dataset to HDF5
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 3} #[cite: 4]
    with h5py.File(output_h5, 'w') as hf_out: #[cite: 4]
        meta_out = hf_out.create_group('meta') #[cite: 4]
        for c in ['delay', 'energy', 'repetition']: #[cite: 4]
            if c in df_meta.columns: #[cite: 4]
                meta_out.create_dataset(c, data=df_meta[c].to_numpy(np.float32)) #[cite: 4]
        meta_out.create_dataset(
            'file_id',
            data=np.char.encode(df_meta['file_id'].astype(str).to_numpy(dtype='U'), 'utf-8'),
            compression='gzip') #[cite: 4]
        if aux_cols: #[cite: 4]
            hf_out.create_dataset('y_aux', data=df_elemental_all[aux_cols].to_numpy(np.float32), **comp_kwargs) #[cite: 4]
            hf_out.create_dataset('y_aux_names', data=np.char.encode(np.array(aux_cols, dtype='U'), 'utf-8')) #[cite: 4]
        for c in ['frac_LiCl', 'frac_KCl', 'temperature_C']: #[cite: 4]
            if c in df_meta.columns: #[cite: 4]
                meta_out.create_dataset(c, data=df_meta[c].to_numpy(np.float32)) #[cite: 4]
        
        # Save group IDs and explicit lockbox mask
        hf_out.create_dataset('groups', data=groups.ravel().astype(np.int32))
        hf_out.create_dataset('is_lockbox', data=is_lockbox.astype(bool), **comp_kwargs)

        hf_out.create_dataset('X', data=X_clean.astype(np.float32, copy=False), **comp_kwargs) #[cite: 4]
        hf_out.create_dataset('y', data=y_filtered.astype(np.float32, copy=False), **comp_kwargs) #[cite: 4]
        hf_out.create_dataset('wavelengths', data=wavelengths.astype(np.float32, copy=False)) #[cite: 4]
        
        str_dt = h5py.string_dtype(encoding='utf-8') #[cite: 4]
        hf_out.create_dataset('target_elements', data=np.array(surviving_cols, dtype=object), dtype=str_dt) #[cite: 4]
        
    print(f"✅ Cleaned dataset saved to {output_h5}")
    print(f"   Spectra Matrix (X): {X_clean.shape}")
    print(f"   Target Matrix (y) : {y_filtered.shape} -> Elements: {surviving_cols}")

# 3.a Load in the initial h5 file
def get_h5_ds(hf: h5py.File, key: str) -> h5py.Dataset:
    """ Extract a dataset array from a .h5 files and return it as a dataset. """
    from typing import cast
    return cast(h5py.Dataset, hf[key])

# 3.b. Clean the spectra
def clean_libs_spectrum(
    spectra_matrix, 
    window_size=10, 
    chunk=5000
):
      out = np.empty(spectra_matrix.shape, dtype=np.float32)
      for a in range(0, len(spectra_matrix), chunk):
          X = spectra_matrix[a:a+chunk].astype(np.float32)
          base = uniform_filter1d(
              minimum_filter1d(X, size=window_size, axis=1, mode='nearest'),
              size=window_size, axis=1, mode='nearest')
          X = np.maximum(X - base, 0.0)
          area = X.sum(axis=1, keepdims=True)
          out[a:a+chunk] = np.divide(X, area, out=X, where=area > 0)
      return out

# 3.c. Convert from chlorides to elemental wt%
def compute_elemental_wt(df_meta: pd.DataFrame) -> pd.DataFrame:
    """Converts salt species or direct elemental inputs to pure elemental wt%."""
    elemental_df = pd.DataFrame(index=df_meta.index)
    total_cl_wt = np.zeros(len(df_meta), dtype=np.float32)
    dopant_salt_wt_total = np.zeros(len(df_meta), dtype=np.float32)

    dopants = [e for e in NUM_CL.keys() if e not in ('Li', 'K')]
    
    for elem in dopants:
        n_cl = NUM_CL[elem]
        mw_m, mw_cl = MW_ELEMENTS[elem], MW_ELEMENTS['Cl']
        mw_salt = mw_m + (n_cl * mw_cl)
        
        # Candidate columns generated by filename parsing or lookups
        col_salt = f'conc_{elem}Cl{n_cl}_wt%'
        col_elem = f'conc_{elem}_wt%'
        
        salt_wt = df_meta[col_salt].to_numpy(np.float32) if col_salt in df_meta else 0.0
        elem_wt = df_meta[col_elem].to_numpy(np.float32) if col_elem in df_meta else 0.0
        salt_wt = np.where(salt_wt > 0, salt_wt, elem_wt * (mw_salt / mw_m))  # prefer salt label
        dopant_salt_wt_total += salt_wt
        elemental_df[elem] = salt_wt * (mw_m / mw_salt)
        total_cl_wt += salt_wt * ((n_cl * mw_cl) / mw_salt)

    # Calculate carrier matrix mass fraction (LiCl-KCl Eutectic: 45 wt% LiCl / 55 wt% KCl)
    base_scale = np.maximum(0.0, 100.0 - dopant_salt_wt_total) / 100.0
    for elem, default_wt, n_cl in [('Li', 45.0, 1), ('K', 55.0, 1)]:
        mw_m, mw_cl = MW_ELEMENTS[elem], MW_ELEMENTS['Cl']
        mw_salt = mw_m + (n_cl * mw_cl)
        scaled_salt_wt = default_wt * base_scale
        elemental_df[elem] = scaled_salt_wt * (mw_m / mw_salt)
        total_cl_wt += scaled_salt_wt * ((n_cl * mw_cl) / mw_salt)

    elemental_df['Cl'] = total_cl_wt
    return elemental_df


def full_run(
        input_csv_dir: Path | str = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/csvs",
        enriched_csv_dir: Path | str = '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/csvs/enriched',
        combined_root: Path | str = '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/csvs/enriched',
        cleaned_h5: Path | str = '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/updated_pls_data.h5',
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
        compression: str = 'gzip',
        compression_level: int = 3,
        transpose_needed: bool = False,
        n_workers: int | None = None,
        target_elements: list[str] | None = None,
        clip_percentile: float | None = None,# Optional parameters to filter conditions dynamically
        target_delay: float | None = 14.0,
        target_energy: float | None = None,
        target_state_molten: int | None = None,
):
    input_csv_dir = Path(input_csv_dir).resolve()
    enriched_csv_dir = Path(enriched_csv_dir).resolve()
    combined_root = Path(combined_root).resolve()
    cleaned_h5 = Path(cleaned_h5).resolve()

    all_csvs = list(input_csv_dir.rglob('*.csv'))

    input_files = [
        f for f in all_csvs if 'data_import' not in f.name.lower() and 'lambda' not in f.name.lower()
    ]

    print(f"Total CSVs found: {len(all_csvs)} | Clean single-run files kept: {len(input_files)}")

    enrich_with_progress(
        files=input_files,
        enriched_root=enriched_csv_dir,
        drop_columns=drop_columns,
        rename_columns=rename_columns,
        composition_columns=composition_columns,
        min_wavelength=min_wavelength,
        max_wavelength=max_wavelength,
        allowed_extensions=allowed_extensions,
        required_columns=required_columns,
        salt_states=salt_states,
        experimental_variation_columns=experimental_variation_columns,
        include_experimental_variation_columns=include_experimental_variation_columns,
        log_q=log_q,
        log_path=log_path,
        n_workers=n_workers
    )

    combine_and_save_as_HDF5(
        individuals_root=enriched_csv_dir,
        combined_root=combined_root,
        compression=compression,
        compression_level=compression_level,
        log_path=log_path,
        transpose_needed=transpose_needed,
        n_workers=n_workers,
    )

    input_h5 = combined_root / f"combined_libs.h5"

    dataset_cleaning(
        input_h5=input_h5,
        output_h5=cleaned_h5,
        target_elements=target_elements,
        clip_percentile=clip_percentile,# Optional parameters to filter conditions dynamically
        target_delay=target_delay,
        target_energy=target_energy,
        target_state_molten=target_state_molten,
    )

# =============================================================================
# 4. Data inspection
# =============================================================================
def find_cosmic(
        h5_file: Path | str = '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/recleaned/combined_libs.h5',
):
    h5_file = Path(h5_file).resolve()

    with h5py.File(h5_file, 'r') as hf:
        raw_spectra = hf['spectra'][:] # type: ignore

    max_counts = raw_spectra.max(axis=1) # type: ignore

    # 1. Print high percentile distribution
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80,90, 95, 98, 99, 99.5, 99.9, 100]
    print("--- Raw Peak Counts Percentile Distribution ---")
    for p, val in zip(percentiles, np.percentile(max_counts, percentiles)):
        print(f"{p:>5.1f}% : {val:10.1f} counts")

    # 2. Check for Hardware Saturation flatlining (e.g., hits near 65,535 or 16,383)
    near_sat = (max_counts > 60000).sum()
    print(f"\nShots near 16-bit saturation (> 60,000 counts): {near_sat} / {len(max_counts)}")

    # 3. Inspect top 20 highest-intensity shots for cosmic ray spikes
    top_indices = np.argsort(max_counts)[-20:][::-1]
    print("\n--- Top 20 Peak Intensity Shots ---")
    for rank, idx in enumerate(top_indices, 1):
        spec = raw_spectra[idx] # type: ignore
        peak_val = spec.max() # type: ignore
        peak_wl_idx = spec.argmax() # type: ignore
        
        # Check spike width: cosmic rays usually affect only 1-2 adjacent channels
        left_val = spec[max(0, peak_wl_idx - 1)] # type: ignore
        right_val = spec[min(len(spec) - 1, peak_wl_idx + 1)] # type: ignore
        ratio = peak_val / (max(left_val, right_val) + 1e-5) # type: ignore
        
        spike_type = "Cosmic Ray Spike" if ratio > 5.0 else "Broad Emission Line"
        print(f"Rank {rank:2d} | Shot {idx:6d} | Max Counts: {peak_val:10.1f} | Adjacent Ratio: {ratio:6.2f} | {spike_type}")


# =============================================================================
# Appendix: utils
# =============================================================================
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
        if os.name == 'nt' and len(s) > 240 and not s.startswith('\\\\?\\'):
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

if __name__ == "__main__":
    print('Hi')

    # --------------------------------------------------------------------------------
    # region Enriched Directory Check
    # --------------------------------------------------------------------------------
    # enriched = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/csvs")

    # print(f"Directory exists? {enriched.exists()}")

    # if enriched.exists():
    #     all_files = list(enriched.iterdir())
    #     print(f"Total items directly in folder: {len(all_files)}")
        
    #     # Check recursively in subfolders just in case
    #     recursive_csvs = list(enriched.rglob("*.[cC][sS][vV]"))
    #     print(f"Total CSVs found recursively: {len(recursive_csvs)}")
        
    #     for p in recursive_csvs[:10]:  # print first 10
    #         print(" Found CSV:", p)
    # endregion

    # --------------------------------------------------------------------------------
    # region MS Spectra Comparison & Inspection
    # --------------------------------------------------------------------------------
    # import re
    # from pathlib import Path
    # known = ['ms4','ms5','ms6','ms8','ms10','ms11','ms12','ms14','ms16']
    # root = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/csvs")
    # # for p in root.rglob("*.csv"):
    # #     t = (' '.join(p.parts) + ' ' + p.name).lower()
    # #     if re.search(r'(?<![a-z])ms\d', t) and not re.search(r'msu\d', t) and not any(k in t for k in known):
    # #         print(p)
    # # for p in root.rglob("*.csv"):
    # #     t = (' '.join(p.parts) + ' ' + p.name).lower()
    # #     if re.search(r'msu\d', t):
    # #         print(p)

    # known = ['ms4','ms5','ms6','ms8','ms10','ms11','ms12','ms14','ms16']
    # root = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/csvs")
    # enriched = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/csvs")

    # def find(*substrs):
    #     for p in enriched.rglob("*.csv"):
    #         og = pd.read_csv(p, usecols=['og_path'], nrows=1)['og_path'].iloc[0].lower()
    #         if all(s.lower() in og for s in substrs):
    #             return p  # first match is fine; backup trees are copies
    #     raise FileNotFoundError(substrs)

    # def mean_spec(p):
    #     df = pd.read_csv(p)
    #     cols = []
    #     for c in df.columns:
    #         try: 
    #             float(c); cols.append(c)
    #         except ValueError: 
    #             pass
    #     wl = np.array([float(c) for c in cols])
    #     order = np.argsort(wl)
    #     return wl[order], df[cols].to_numpy(float).mean(axis=0)[order]

    # def strength(wl, y, w0):
    #     peak = y[np.abs(wl - w0) < 0.3].max()
    #     base = np.median(y[(np.abs(wl - w0) > 0.8) & (np.abs(wl - w0) < 1.5)])
    #     return peak - base

    # LINES = {'K I 766.49': 766.49, 'Li I 670.78': 670.78,
    #          'U II 385.96': 385.96, 'U II 409.01': 409.01}  # verify against NIST ASD

    # def compare(pure, ref, lines=LINES, out_path="plot.png"):
    #     (wp, yp), (wr, yr) = mean_spec(pure), mean_spec(ref)
    #     kp, kr = strength(wp, yp, 766.49), strength(wr, yr, 766.49)
        
    #     print(f"{'line':14s} {'pure/K':>8s} {'ref/K':>8s}")
    #     for name, w0 in lines.items():
    #         print(f"{name:14s} {strength(wp, yp, w0)/kp:8.3f} {strength(wr, yr, w0)/kr:8.3f}")
            
    #     fig, ax = plt.subplots(1, len(lines), figsize=(4 * len(lines), 3))
    #     for a, (name, w0) in zip(ax, lines.items()):
    #         for w, y, k, lab in [(wp, yp, kp, 'pure'), (wr, yr, kr, 'ref')]:
    #             m = np.abs(w - w0) < 1.5
    #             a.plot(w[m], y[m] / k, label=lab)
    #         a.set_title(name)
    #         a.legend()
            
    #     plt.tight_layout()
    #     plt.savefig(out_path, dpi=300, bbox_inches='tight')
    #     plt.close(fig)
    #     print(f"Saved plot to: {out_path}\n")

    # compare(find('6th EXP Pure', 'MSU1_Data_lamba_MSU1'),
    #         find('MSU1 1 wt_ UCl3', 'MSU1 Rep 1'),
    #         out_path='compare_msu1.png')

    # compare(find('6th EXP Pure', 'MSU2_Data_lamba_MSU2'),
    #         find('MSU2 1 wt_ UCl3', 'MSU2 Rep 1'),
    #         out_path='compare_msu2.png')
    # endregion

    # --------------------------------------------------------------------------------
    # region Full Run
    # --------------------------------------------------------------------------------
    full_run(
        input_csv_dir = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/csvs",
        enriched_csv_dir = '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/enriched_csvs',
        combined_root = '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/recleaned',
        cleaned_h5 = '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/recleaned/recleaned_data_no-leak_TD14_TEN.h5',
        drop_columns=DROP_COLUMNS,
        rename_columns=RENAME_COLUMNS,
        composition_columns=COMPOSITION_COLUMNS,
        min_wavelength=MIN_WAVELENGTH,
        max_wavelength=MAX_WAVELENGTH,
        allowed_extensions=ALLOWED_EXTENSIONS,
        required_columns=REQUIRED_COLUMNS,
        salt_states=SALT_STATES,
        experimental_variation_columns=EXPERIMENTAL_VARIATION_COLUMNS,
        include_experimental_variation_columns=INCLUDE_EXPERIMENTAL_VARIATION_COLUMNS,
        log_q=LOG_Q,
        log_path=LOG_PATH,
        compression=COMPRESSION,
        compression_level=COMPRESSION_LEVEL,
        transpose_needed=TRANSPOSE_NEEDED,
        n_workers=N_WORKERS,
        target_elements=TARGET_ELEMENTS,
        clip_percentile=CLIP_PERCENTILE, # Optional parameters to filter conditions dynamically
        target_delay=TARGET_DELAY,
        target_energy=TARGET_ENERGY,
        target_state_molten=TARGET_STATE_MOLTEN,
    )
    # endregion

    # --------------------------------------------------------------------------------
    # region HDF5 Duplicate Spectra Check
    # --------------------------------------------------------------------------------
    # print(df_meta.loc[df_meta['energy'].isin([2.0, 3.0]), 'og_path'].drop_duplicates().head(10).tolist())

    # import h5py, hashlib, numpy as np, pandas as pd
    # p = '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/recleaned/combined_libs.h5'
    # with h5py.File(p) as f:
    #     S, m = f['spectra'], f['metadata']
    #     hashes = []
    #     for i in range(0, S.shape[0], 20000): # type: ignore
    #         hashes += [hashlib.blake2b(r.tobytes(), digest_size=16).hexdigest() for r in S[i:i+20000]] # type: ignore
    #     cols = ['delay', 'energy'] + [c for c in m if c.startswith('conc_')] # type: ignore
    #     df = pd.DataFrame({c: m[c][:] for c in cols}) # type: ignore
    #     df['og_path'] = np.char.decode(m['og_path'][:], 'utf-8') # type: ignore
    # df['h'] = hashes
    # sizes = df.groupby('h').size()
    # print(len(df), 'rows;', (sizes > 1).sum(), 'duplicated spectra;', int((sizes[sizes > 1] - 1).sum()), 'extra copies')
    # print(sizes.value_counts().sort_index().head(8))
    # dup = df[df.h.isin(sizes[sizes > 1].index)]
    # nun = dup.groupby('h')[cols].nunique(dropna=False)
    # print('duplicate groups that disagree, by column:'); print((nun > 1).sum()[lambda s: s > 0])
    # ex = dup[dup.h.isin(nun.index[nun['energy'] > 1])].sort_values('h').head(12)
    # for _, r in ex.iterrows(): print(r['energy'], '|', r['og_path'].split('/csvs/')[1][:100])

    # endregion

    # --------------------------------------------------------------------------------
    # region Run cleaning/filtering directly
    # --------------------------------------------------------------------------------
    # # Target your existing 16 GB HDF5 file
    # combined_h5 = Path('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/recleaned/combined_libs.h5')
    # cleaned_h5  = Path('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/recleaned/recleaned_data_TDN_TEN.h5')

    # dataset_cleaning(
    #     input_h5=combined_h5,
    #     output_h5=cleaned_h5,
    #     target_elements=['Ce', 'Gd', 'Sm', 'U'],
    #     clip_percentile=None,
    #     target_delay=None,
    #     target_energy=TARGET_ENERGY,
    #     target_state_molten=TARGET_STATE_MOLTEN,
    # )
    # endregion

    # --------------------------------------------------------------------------------
    # region Enriched CSV Headers Inspection
    # --------------------------------------------------------------------------------
    # import glob, csv
    # def isnum(c):
    #     try: float(c); return True
    #     except ValueError: return False
    # cols = set()
    # for f in glob.glob('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/enriched_csvs/*.csv'):
    #     with open(f) as fh:
    #         cols.update(c for c in next(csv.reader(fh)) if not isnum(c))
    # print(len(cols))
    # print(sorted(cols)[:60])
    # endregion

    # --------------------------------------------------------------------------------
    # region Check Specific CSV Duplicate Columns
    # --------------------------------------------------------------------------------
    # import pandas as pd
    # df = pd.read_csv('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/enriched_csvs/0a83e8dc_Static_500ppm_Values_Li_lamba.csv', nrows=5)
    # dups = [c for c in df.columns if c.startswith('1000.039')]
    # print(len(dups)); print(df[dups].iloc[0].values[:10])
    # endregion

    # --------------------------------------------------------------------------------
    # region Check Leaked Wavelength Headers
    # --------------------------------------------------------------------------------
    # import glob, csv, re
    # from collections import Counter

    # def isnum(c):
    #     try: float(c); return True
    #     except ValueError: return False

    # bad = {}
    # for f in glob.glob('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/enriched_csvs/*.csv'):
    #     with open(f) as fh:
    #         hdr = next(csv.reader(fh))
    #     leaked = [c for c in hdr if not isnum(c) and re.fullmatch(r'\d+\.\d+\.\d+', c)]
    #     if leaked:
    #         bad[f] = len(leaked)

    # print(len(bad), 'files affected')
    # for f, n in Counter(bad).most_common(): pass
    # for f, n in sorted(bad.items(), key=lambda x: -x[1])[:10]:
    #     print(n, f)
    # endregion

    # --------------------------------------------------------------------------------
    # region Check Solid Normalized Spectra & Zero Target Shots
    # --------------------------------------------------------------------------------
    # import pandas as pd
    # f = '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/enriched_csvs/7c7b7577_Solid_Normalized_Spectra_lamba_S.csv'
    # df = pd.read_csv(f, nrows=5)
    # leaked = [c for c in df.columns if re.fullmatch(r'\d+\.\d+\.\d+', c)]
    # print(len(leaked))
    # print(df[leaked].iloc[:, :10])           # values
    # print(df[leaked].nunique().describe())   # do they vary?

    # import h5py, numpy as np, collections
    # p = '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/recleaned/recleaned_data_TD14_TEN.h5'
    # with h5py.File(p) as f:
    #     y = f['y'][:] # type: ignore
    #     names = [s.decode() if isinstance(s, bytes) else s for s in f['target_elements'][:]] # type: ignore
    #     fid = np.char.decode(f['meta/file_id'][:], 'utf-8') # type: ignore
    # zero = (y == 0).all(axis=1)
    # c = collections.Counter(fid[zero])
    # print(names, y.shape, f'{zero.sum()} all-zero shots in {len(c)} files') # type: ignore
    # for k, v in c.most_common(25): print(v, k)
    # endregion

    # --------------------------------------------------------------------------------
    # region Check Unlabeled Shots at Delay 14
    # --------------------------------------------------------------------------------
    # import sys, h5py, numpy as np, pandas as pd
    # sys.path.insert(0, '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/utils')
    # import new_data_prep as dp

    # p = '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/recleaned/combined_libs.h5'
    # with h5py.File(p) as f:
    #     m = f['metadata']
    #     df = pd.DataFrame({c: (np.char.decode(m[c][:], 'utf-8') if m[c].dtype.kind == 'S' else m[c][:]) for c in m}) # type: ignore

    # el = dp.compute_elemental_wt(df)
    # targets = ['Ce', 'Gd', 'Sm', 'U']
    # others = [c for c in el.columns if c not in targets + ['Li', 'K', 'Cl']]
    # df['tgt'] = (el[targets] > 0).any(axis=1)
    # df['oth'] = (el[others] > 0).any(axis=1)

    # d = df[df['delay'] == 14.0]
    # print(pd.Series(np.select([d.tgt, d.oth], ['target dopant', 'other dopant'], 'no dopant label')).value_counts())

    # z = d[~d.tgt & ~d.oth].copy()
    # z['folder'] = z['og_path'].str.rsplit('/', n=1).str[0]
    # print(z.groupby('folder').size().sort_values(ascending=False).head(25))
    # print(z[['frac_LiCl', 'frac_KCl', 'technique']].describe(include='all'))

    # print('file_ids spanning >1 og_path:', (df.groupby('file_id')['og_path'].nunique() > 1).sum())
    # endregion

    # --------------------------------------------------------------------------------
    # region Before
    # --------------------------------------------------------------------------------
    # import sys, h5py, numpy as np, pandas as pd
    # sys.path.insert(0, '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/utils')
    # import new_data_prep as dp

    # p = '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/recleaned/combined_libs.h5'
    # with h5py.File(p) as f:
    #     og = np.char.decode(f['metadata']['og_path'][:], 'utf-8') # type: ignore
    # paths = sorted(set(og))
    # df = dp.parent_concentration_data(paths)
    # df.index = paths
    # df.to_pickle('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/tmp/labels_before.pkl')
    # print(len(paths), 'unique paths')
    # endregion

    # --------------------------------------------------------------------------------
    # region After
    # --------------------------------------------------------------------------------
    # import sys, h5py, numpy as np, pandas as pd
    # sys.path.insert(0, '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/utils')
    # import new_data_prep as dp

    # p = '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/recleaned/combined_libs.h5'
    # with h5py.File(p) as f:
    #     og = np.char.decode(f['metadata']['og_path'][:], 'utf-8') # type: ignore
    # paths = sorted(set(og))
    # df = dp.parent_concentration_data(paths)
    # df.index = paths
    # df.to_pickle('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/tmp/labels_after.pkl')
    # print(len(paths), 'unique paths')
    # endregion
    
    # --------------------------------------------------------------------------------
    # region After2
    # --------------------------------------------------------------------------------
    # import sys, h5py, numpy as np, pandas as pd
    # sys.path.insert(0, '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/utils')
    # import new_data_prep as dp

    # p = '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/recleaned/combined_libs.h5'
    # with h5py.File(p) as f:
    #     og = np.char.decode(f['metadata']['og_path'][:], 'utf-8') # type: ignore
    # paths = sorted(set(og))
    # df = dp.parent_concentration_data(paths)
    # df.index = paths
    # df.to_pickle('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/tmp/labels_after2.pkl')
    # print(len(paths), 'unique paths')
    # endregion
    
    # --------------------------------------------------------------------------------
    # region After3
    # --------------------------------------------------------------------------------
    # import sys, h5py, numpy as np, pandas as pd
    # sys.path.insert(0, '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/utils')
    # import new_data_prep as dp

    # p = '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/recleaned/combined_libs.h5'
    # with h5py.File(p) as f:
    #     og = np.char.decode(f['metadata']['og_path'][:], 'utf-8') # type: ignore
    # paths = sorted(set(og))
    # df = dp.parent_concentration_data(paths)
    # df.index = paths
    # df.to_pickle('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/tmp/labels_after3.pkl')
    # print(len(paths), 'unique paths')
    # endregion

    # --------------------------------------------------------------------------------
    # region After4
    # --------------------------------------------------------------------------------
    # import sys, h5py, numpy as np, pandas as pd
    # sys.path.insert(0, '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/utils')
    # import new_data_prep as dp

    # p = '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/recleaned/combined_libs.h5'
    # with h5py.File(p) as f:
    #     og = np.char.decode(f['metadata']['og_path'][:], 'utf-8') # type: ignore
    # paths = sorted(set(og))
    # df = dp.parent_concentration_data(paths)
    # df.index = paths
    # df.to_pickle('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/tmp/labels_after4.pkl')
    # print(len(paths), 'unique paths')
    # endregion

    # --------------------------------------------------------------------------------
    # region Compare
    # --------------------------------------------------------------------------------
    # import h5py
    # import numpy as np
    # import pandas as pd

    # T = '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/tmp/'
    # a = pd.read_pickle(T + 'labels_before.pkl')
    # b = pd.read_pickle(T + 'labels_after4.pkl')
    # conc = [c for c in a.columns if c.startswith('conc_')]

    # # Align dataframes to common paths before performing element-wise comparisons
    # common_paths = a.index.intersection(b.index)
    # a_sub = a.loc[common_paths, conc]
    # b_sub = b.loc[common_paths, conc]

    # # 1) Harness check: does "before" reproduce what's stored in combined_libs.h5?
    # p = (
    #     '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/recleaned/combined_libs_old.h5'
    # )
    # with h5py.File(p) as f:
    #     m = f['metadata']
    #     stored = pd.DataFrame({c: m[c][:] for c in conc if c in m})  # type: ignore
    #     stored['og_path'] = np.char.decode(m['og_path'][:], 'utf-8')  # type: ignore

    # stored = (
    #     stored.drop_duplicates('og_path').set_index('og_path').reindex(a.index)
    # )
    # cols = [c for c in conc if c in stored]
    # same = np.isclose(
    #     stored[cols].to_numpy(float), a[cols].to_numpy(float), equal_nan=True
    # ).all(axis=1)
    # print(f'before matches stored labels for {same.mean():.1%} of paths')

    # # 2) What did the patch change across common raw paths?
    # diff = ~(a_sub.fillna(-1) == b_sub.fillna(-1)).all(axis=1)
    # was_unlabeled = a_sub.fillna(0).abs().sum(axis=1) == 0
    # print(
    #     diff.sum(),
    #     'of',
    #     len(common_paths),
    #     'common paths changed;',
    #     (diff & was_unlabeled).sum(),
    #     'were unlabeled before',
    # )
    # print(
    #     'changed but ALREADY labeled (want ~0):', (diff & ~was_unlabeled).sum()
    # )

    # for path in b_sub.index[diff][:40]:
    #     nz = b_sub.loc[path][b_sub.loc[path] != 0]
    #     print(dict(nz), '|', '/'.join(path.split('/')[-3:]))

    # # 3) Grouped breakdown of changes
    # short = lambda p: '/'.join(p.rsplit('/', 1)[0].split('/')[-3:])
    # c = b_sub[diff]
    # lab = c.apply(
    #     lambda r: str({k[5:-4]: v for k, v in r.items() if v != 0}), axis=1
    # )
    # print(
    #     pd.DataFrame({'folder': [short(p) for p in c.index], 'label': lab.values})
    #     .value_counts()
    #     .to_string()
    # )

    # # 4) Check remaining unlabeled paths in new run 'b'
    # bad = b.index[
    #     (b[conc].fillna(0).abs().sum(axis=1) == 0) & ~b[conc].isna().any(axis=1)
    # ]
    # with h5py.File(p) as f:
    #     og = np.char.decode(f['metadata']['og_path'][:], 'utf-8')  # type: ignore
    #     delay = f['metadata']['delay'][:]  # type: ignore

    # n = pd.Series(og[delay == 14.0]).value_counts()
    # s = n.reindex(bad).dropna()
    # print('still unlabeled shots:', int(s.sum()))
    # print(
    #     s.groupby([short(x) for x in s.index])
    #     .sum()
    #     .sort_values(ascending=False)
    #     .head(25)
    #     .to_string()
    # )

    # print((c != 0).sum()[lambda s: s > 0])
    # print('paths with >1 species:', ((c != 0).sum(axis=1) > 1).sum())
    # endregion

    # --------------------------------------------------------------------------------
    # region Print Files Paths
    # --------------------------------------------------------------------------------
    # import os
    # from pathlib import Path

    # TARGET_DIR = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/csvs")
    # # OUTPUT_FILE = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/csvs/Awilliams_files.txt")

    # PATTERNS = [
    #     "Awilliams"
    # ]

    # def search_filepaths(directory):
    #     if not directory.exists():
    #         print(f"Error: Directory '{directory}' does not exist.")
    #         return

    #     matching_count = 0
    #     print(f"Searching in: {directory.resolve()}\n" + "-" * 50)

    #     for path in directory.rglob("*"):
    #         if path.is_file():
    #             path_str = str(path)
    #             if any(pattern in path_str for pattern in PATTERNS):
    #                 print(path_str)
    #                 matching_count += 1

    #     print("-" * 50)
    #     print(f"Found {matching_count} matching files.")

    # search_filepaths(TARGET_DIR)
    # endregion

    # --------------------------------------------------------------------------------
    # region Unlabeled PPM Paths
    # --------------------------------------------------------------------------------
    # import pandas as pd
    # T = '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/tmp/'
    # b = pd.read_pickle(T + 'labels_after2.pkl')
    # conc = [c for c in b.columns if c.startswith('conc_')]
    # un = b.index[(b[conc].fillna(0).abs().sum(axis=1) == 0) & ~b[conc].isna().any(axis=1)]
    # for p in sorted(p for p in un if 'ppm' in p.lower() and 'Backup 9-12-2016' in p): print('/'.join(p.split('/')[-3:]))
    # endregion

    # --------------------------------------------------------------------------------
    # region Matlab Analysis 1000 ppm Row Comparison
    # --------------------------------------------------------------------------------
    # import h5py, numpy as np, pandas as pd
    # p = '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/recleaned/combined_libs.h5'
    # key = 'Backup 9-12-2016/Spectra/CeCl3-LiCl-KCl-H2O Test/Static/Calibration Curves/200 shots 5 repeats/Matlab analysis/1000 ppm'
    # with h5py.File(p) as f:
    #     og = np.char.decode(f['metadata']['og_path'][:], 'utf-8') # type: ignore
    #     idx = np.flatnonzero(np.char.find(og, key) >= 0)
    #     X = f['spectra'][idx.tolist()] # type: ignore
    # names = np.array([s.rsplit('/', 1)[1] for s in og[idx]])
    # print(pd.Series(names).value_counts())
    # rows = {n: X[names == n] for n in np.unique(names)} # type: ignore
    # others = np.vstack([r for n, r in rows.items() if 'Values' not in n]) # type: ignore
    # for n in [n for n in rows if 'Values' in n]:
    #     d = np.abs(rows[n][:, None, :] - others[None]).max(axis=2).min(axis=1) # type: ignore
    #     print(n, rows[n].shape, 'max diff to nearest non-Values row:', d.max()) # type: ignore

    # def norm(a): return a / a.sum(axis=1, keepdims=True)
    # O = norm(others)
    # scale = O.max()
    # for n in [n for n in rows if 'Values' in n]:
    #     V = norm(rows[n])
    #     d = np.abs(V[:, None, :] - O[None]).max(axis=2)          # 10 Values rows x 10 other rows
    #     print(n)
    #     print('  nearest-row distance / peak height:', np.round(d.min(axis=1) / scale, 4))
    #     print('  best-match row index:', d.argmin(axis=1))
    # a = rows['Static_1000ppm_Values_lamba.csv']; b = rows['Static_1000ppm_Values_Li_lamba.csv']
    # print('Values vs Values_Li identical:', np.array_equal(a, b), '| max abs diff:', np.abs(a - b).max()) # type: ignore
    # print('row sums  t-files:', np.round(others.sum(axis=1)[:5]), ' Values:', np.round(rows['Static_1000ppm_Values_lamba.csv'].sum(axis=1)[:5])) # type: ignore
    # endregion

    # --------------------------------------------------------------------------------
    # region Find Cosmic Rays
    # --------------------------------------------------------------------------------
    # find_cosmic(h5_file="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/recleaned/combined_libs.h5")
    # endregion