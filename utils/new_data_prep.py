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
from sklearn.feature_selection import VarianceThreshold
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


# =============================================================================
# 2. Convert it all to .h5 file prior to full dataset processing
# =============================================================================
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
    target_energy: float | None = 100.0,
    target_state_molten: int | None = 1,
) -> None:
    """Prepares raw spectral HDF5 dataset into clean X (Spectra) and y (Concentrations) for PLS."""
    input_h5, output_h5 = Path(input_h5), Path(output_h5)
    
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
    spectra = raw_spectra[valid_mask]
    df_meta = df_meta[valid_mask].reset_index(drop=True)

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
    

    # Step 2: Intensity Clipping & LIBS Spectral Cleaning (Baseline + Area Norm)
    if clip_percentile:
        clip_val = np.percentile(spectra, clip_percentile)
        spectra_clipped = np.clip(spectra, 0, clip_val)
        X_clean = clean_libs_spectrum(spectra_clipped)
    else:
        X_clean = clean_libs_spectrum(spectra)

    # Step 3: Elemental Stoichiometric Conversion for Targets (y)
    df_elemental = compute_elemental_wt(df_meta)
    if target_elements:
        df_elemental = df_elemental[[e for e in target_elements if e in df_elemental.columns]]

    # Step 4: Drop Inactive Target Columns (Zero Variance)
    selector = VarianceThreshold(threshold=1e-6)
    y_filtered = selector.fit_transform(df_elemental)
    surviving_cols = df_elemental.columns[selector.get_support()].tolist()

    # Step 5: Save Clean PLS-Ready Dataset to HDF5
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 3}
    with h5py.File(output_h5, 'w') as hf_out:
        hf_out.create_dataset('X', data=X_clean.astype(np.float32), **comp_kwargs)
        hf_out.create_dataset('y', data=y_filtered.astype(np.float32), **comp_kwargs)
        hf_out.create_dataset('wavelengths', data=wavelengths.astype(np.float32))
        
        str_dt = h5py.string_dtype(encoding='utf-8')
        hf_out.create_dataset('target_elements', data=np.array(surviving_cols, dtype=object), dtype=str_dt)
        
        # Preserve sample IDs for GroupKFold validation splitting
        if 'file_id' in df_meta.columns:
            hf_out.create_dataset('groups', data=df_meta['file_id'].astype(str).to_numpy(), dtype=str_dt)

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
        spectra_matrix: np.ndarray,
        window_size: int = 50
) -> np.ndarray:
    """Subtracts rolling baseline and applies area normalization across all shots."""
    cleaned_spectra = np.zeros_like(spectra_matrix, dtype=np.float32)
    
    for i in range(spectra_matrix.shape[0]):
        row = spectra_matrix[i]
        # 1. Rolling Minimum Baseline Estimation
        series = pd.Series(row)
        baseline = series.rolling(window=window_size, center=True, min_periods=1).min().to_numpy()
        corrected = np.maximum(0.0, row - baseline)
        
        # 2. Area Normalization (Sum to 1)
        area = np.sum(corrected)
        cleaned_spectra[i] = (corrected / area) if area > 0 else corrected
        
    return cleaned_spectra

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
        
        if col_salt in df_meta.columns and df_meta[col_salt].abs().sum() > 0:
            # Case 1: Input provided as chloride salt wt%
            salt_wt = df_meta[col_salt].to_numpy(dtype=np.float32)
            dopant_salt_wt_total += salt_wt
            elemental_df[elem] = salt_wt * (mw_m / mw_salt)
            total_cl_wt += salt_wt * ((n_cl * mw_cl) / mw_salt)
            
        elif col_elem in df_meta.columns and df_meta[col_elem].abs().sum() > 0:
            # Case 2: Input provided directly as pure metal wt%
            pure_elem_wt = df_meta[col_elem].to_numpy(dtype=np.float32)
            equivalent_salt_wt = pure_elem_wt * (mw_salt / mw_m)
            dopant_salt_wt_total += equivalent_salt_wt
            elemental_df[elem] = pure_elem_wt
            total_cl_wt += pure_elem_wt * ((n_cl * mw_cl) / mw_m)
            
        else:
            elemental_df[elem] = np.zeros(len(df_meta), dtype=np.float32)

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


def FullRun(
        input_csv_dir: Path | str = "",
):
    enrich_with_progress(
        files='',
        enriched_root='',
        drop_columns=,
        rename_columns=,
        composition_columns=,
        min_wavelength=,
        max_wavelength=,
        allowed_extensions=,
        required_columns=,
        salt_states=,
        experimental_variation_columns=,
        include_experimental_variation_columns=,
        log_q=,
        log_path=,
    )

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

if __name__ == "__main__":
    print('Hi')
