"""

jvlee_LIBS_ML > utils > data_prep.py

This is the labeling workhorse. Contains functions which are called in 'get_libs.py'
to clean and label data. This is done via many re.search() statements with associated
concentration maps and term dictionaries.

TODO:   - move the combine and save functions to file_type_processing.py
        - move sanitize_path to file_type_processing.py
        - rename _long_path to long_path and move to file_type_processing.py
            - or I could make a save_and_load.py and move them into that instead?
              I don't think that would really change much, but would it make more
              sense to have them there?
"""

print('data_prep.py loading ...')

# region Imports
# region plain
import re 
import logging
import os
import warnings
import h5py
import pickle
# endregion

# region as
import pandas as pd 
import numpy as np
# endregion

# region from
from pathlib import Path 
from typing import List, Optional, cast
from tqdm import tqdm
from h5py import Group
from concurrent.futures import ProcessPoolExecutor, as_completed #, ThreadPoolExecutor
from functools import partial
from logging import handlers
# endregion

# region custom
# from utils import (
#     recursive_file_extension_converter, 
#     file_segregator, 
#     gen_speak, 
#     log,
    # get_worker_logger
# )
# endregion
# endregion

# region Fragmented Warning
warnings.filterwarnings('ignore', message='DataFrame is highly fragmented', category=pd.errors.PerformanceWarning)
# endregion

# region Global Vars

_worker_log_q = None  # global in each worker process

TARGET_MIN_WL = 250.0
TARGET_MAX_WL = 1000.0
TARGET_N_PTS = 10000
target_grid = np.linspace(TARGET_MIN_WL, TARGET_MAX_WL, TARGET_N_PTS)

# region MS Map
_MS_CONC_MAP = {
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
# endregion

# region SampleU Map
_SAMPLE_U_MAP = {
    # "W:\Phongikaroon Group\AndrewsH\Backup\Experiments & Calculations - Copy\U-Gd Study\ICP-MS Concentration U1-UG9.xlsx"
    # "W:\Phongikaroon Group\AndrewsH\Backup\Experiments & Calculations - Copy\U-Gd Study\ICP-MS Concentration UG10-UG13.xlsx"
    # 'sample' : [UCl3, GdCl3]
    'sampleu1' : [0.792, 0],
    'sampleu2' : [1.945, 0],
    'sampleu25' : [2.184, 0],       # sample 2.5
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
# endregion

# region SL_ Map
_SL___MAP = {
    # "W:\Phongikaroon Group\AndrewsH\Backup\Experiments & Calculations - Copy\Pure Sm Studies\SmCl3 CV and LIBS 1\Sm LIBS data\Peak comparison.xlsx"
    # 'sample' : SmCl3
    'sla' : 1.04,
    'slb' : 2.88,
    'slc' : 6.40,
    'sld' : 7.69
}
# endregion

# region Mg Map
_SAMPLE_MG_MAP = {
    #"W:\Phongikaroon Group\AndrewsH\Backup\Experiments & Calculations - Copy\U-Mg Study\U-Mg Withdrawal.xlsx"
    # 'sample' : [U, Mg]
    'samplem1_t1' : [0, 0.1],
}
# endregion

# region 1 thru 10 Map
_SAMPLE_1_THRU_T10_MAP = {
    # "W:\Phongikaroon Group\AndrewsH\Backup\Experiments & Calculations - Copy\Sm-Gd Study\ICP-MS Concentration Sample1-T10.xlsx"
    # 'sample' : [SmCl3, GdCl3]
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
# endregion

# region Keith Map
_KEITH_CECL3_MAP = {
    # "W:\Phongikaroon Group\Awilliams\DATA BACKUP\Backup 2-17-2017\NEUP Keith Data\Data transfer\Sample Spectra\Analysis\Solid_Spectra_Data.xlsx"
    # 'sample' : [CeCl3, GdCl3]
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
# endregion

# region Trial 1 and 2 Map
_TRIAL_1_OR_2_MAP = {
    # "W:\Phongikaroon Group\AndrewsH\Backup\Experiments & Calculations - Copy\Pure Sm Studies\SmCl3 LIBS Glovebox\Cal investigation 610-811.xlsx"
    # 'sample' : SmCl3
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
# endregion

# region SSU1/DU1 Map
_SSU_AND_DU_MAP = {
    # "W:\Phongikaroon Group\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Solid Salt\SSU1\Data 10 us\SSU1_1wt_UCl3_LiCl_KCl.m"
    # 'sample' : UCl3
    'ssu1 rep' : 1.0,       
    'du1 rep' : 1.0         # NOTE since ssu1 stands for solid salt ucl3 1wt%, then it is highly likely that du1 stands for disolved ucl3 1 wt%     
}
# endregion

# region Extraction Patterns
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
# endregion

# region Standard Values
_STANDARD_DELAY = 14     # "W:\Phongikaroon Group\AndrewsH\Backup\Experiments & Calculations - Copy\Pure Sm Studies\SmCl3 LIBS Glovebox\Optimization\Gate delay\Gate delay optimization.xlsx"
_STANDARD_WIDTH = 8      # "W:\Phongikaroon Group\AndrewsH\Backup\Experiments & Calculations - Copy\Pure Sm Studies\SmCl3 LIBS Glovebox\Optimization\Gate width\Gate width optimization.xlsx"
_STANDARD_QDELAY = 110       # "W:\Phongikaroon Group\AndrewsH\Backup\Experiments & Calculations - Copy\Pure Sm Studies\SmCl3 LIBS Glovebox\Optimization\Energy\Energy optimization.xlsx"
_STANDARD_SHOTS = 50         # "W:\Phongikaroon Group\AndrewsH\Backup\Experiments & Calculations - Copy\Pure Sm Studies\SmCl3 LIBS Glovebox\Optimization\Shot count\Shot optimization.xlsx"
# endregion

# region Default Columns
_DEFAULT_CONC_COLS: List[str] = [
    'frac_LiCl',        'frac_KCl', 
    'conc_Ce_wt%',      'conc_CeCl3_wt%',
                        'conc_CeN_wt%',
    'conc_Ca_wt%',      'conc_CaCl3_wt%',
    'conc_U_wt%',       'conc_UCl3_wt%',
    'conc_Sm_wt%',      'conc_SmCl3_wt%',
    'conc_Gd_wt%',      'conc_GdCl3_wt%',
    'conc_La_wt%',      'conc_LaCl3_wt%',
    'conc_Mg_wt%',      'conc_MgCl2_wt%',
    'conc_H2o_wt%',
    'conc_Nd_wt%',
]

_SPECIES_MAP = {
    'CECL3': 'CeCl3',       'CE': 'Ce',         # CECL3 sorts before CE (longer)
    'CACL3': 'CaCl3',       'CA': 'Ca',
    'UCL3':  'UCl3',        'U':  'U',
    'SMCL3': 'SmCl3',       'SM': 'Sm',
    'GDCL3': 'GdCl3',       'GD': 'Gd',
    'LACL3': 'LaCl3',       'LACL': 'LaCl3',        'LA': 'La',
    'MGCL2': 'MgCl2',       'MG': 'Mg',
    'NDCL3': 'NdCl3',       'ND': 'Nd',         # add NdCl3 while you're at it
    'cerium':    'Ce',                          # word-form aliases — these are SHORTER
    'gadolinium': 'Gd',                         # than their chloride keys, so they sort after
    'water':     'H2o',
    'CEEN':      'CeN'
}

_DEFAULT_REQ_COLS: List[str] =  list(_DEFAULT_CONC_COLS) + [
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
# endregion

# region Compile Patterns
_COMPILED_SPECIES_PATTERNS = {
    re.compile(rf'(?i)(?:^|[^\w.])(3quarters|half|quarter|\d+\.?\d*)'
                rf'\s*(?:wt%?|%|wt_)?\s*[-_ ]?{re.escape(key.lower())}'
                rf'(?=[_\s.\-/\\]|$)'
                ): nice_name
    for key, nice_name in _SPECIES_MAP.items()
# endregion
}

# endregion

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
    """
    Clean one CSV using the shared cleaner, add metadata, and save enriched version
    in the same directory (file structure abolished).
    """

    # region Logger Setup
    if log_path is None:
        log_path = Path(r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\default_log.txt").resolve()
    logger = get_worker_logger(Path(log_path).stem)
    # endregion

    # region Path initiation
    og_path = Path(path)
    sanitized_path = sanitize_path(
        path=og_path,
        log_path=log_path
    )
    if sanitized_path is None:
        log(logger=logger, msg = f'          Skipping (could not sanitize path): {og_path.name}')
        return None
    path = sanitized_path

    enriched_root = Path(enriched_root)
    enriched_root.mkdir(parents=True, exist_ok=True)
    # endregion

    # region Duplicates
    # Originally the parent name was not included in the duplicate check. This made it
    # so that any files that shared names that were from different studies got skipped.
    # Adding in the parent folder name to the enriched path name makes it so that those
    # files do not flag as duplicates of each other. Now only true duplicates (same file
    # in the same study) get skipped.
    # 
    # Actually I am going to add "grandparent" name too 
    file_w_parents = f'{path.parent.parent.parent.name}_{path.parent.parent.name}_{path.parent.name}_{path.name}'    
    enriched_path = enriched_root / file_w_parents

    if enriched_path.exists():
        log(logger=logger, msg = f'          Skipping (already exists): {enriched_path.name}')
        return enriched_path
    # endregion
    
    # region File extension check
    if allowed_extensions is None:
        allowed_extensions = ['.csv', '.mat', '.mpr', '.asc']
    
    if path.suffix.lower() not in allowed_extensions:
        log(logger=logger, msg = f'          Skipping (unsupported file type): {path.name}')
        return None
    # endregion

    log(logger=logger, msg = f'Enriching: {enriched_path.name}')

    # region Cleaning
    df = clean_single_technique_file(
        path=path,
        technique=technique if technique else 'libs',
        drop_columns=drop_columns,
        rename_columns=rename_columns,
        required_columns=required_columns,
        min_wavelength=min_wavelength,
        max_wavelength=max_wavelength,
        log_path=log_path,
    )

    if df is None:
#        print(f'Skipped enrichment of {csv_path.name}: could not clean')
#        raise ValueError(f"Could not clean {csv_path.name}")
        technique_label = technique.upper() if technique else 'UNKNOWN'
        log(logger=logger, msg = f'          Skipping (techniques={technique_label}): {path.name}')
        return None
    # endregion

    # region Add Metadata
    meta_df = parent_concentration_data(
        source_data_paths = [path], 
        composition_columns = composition_columns,
        required_columns= required_columns,
        salt_states= salt_states,
        experimental_variation_columns= experimental_variation_columns,
        include_experimental_variation_columns= include_experimental_variation_columns
    )
    meta = meta_df.iloc[0].to_dict()


    meta_to_add = {col: value for col, value in meta.items() if col != 'file_path' and col != 'og_path'}
    meta_block = pd.DataFrame([{
        'file_id': f'file_{path.stem}',
        'og_path': str(og_path),
        **meta_to_add,
    }] * len(df)).reset_index(drop=True)

    df = pd.concat([meta_block, df.reset_index(drop=True)], axis=1)
    # endregion

    # region Save   
    df.to_csv(long_path(enriched_path), index=False)
    log(logger=logger, msg = f"Enriched → {enriched_path.name}  |  shape={df.shape}")
    return enriched_path
    # endregion

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
        log_q=None,
        # log_path: Path | None = None,
) -> List[Path]:
    """
    Reusable function with progress bar for enriching any technique.
    """
    enriched_root.mkdir(parents=True, exist_ok=True)
    enriched_paths = []

    # region File by file
    # # In enrich_with_progress, temporarily replace the executor block with:
    # for file in files:
    #     result = enrich_file_with_metadata(
    #         path=file,
    #         enriched_root=enriched_root,
    #         drop_columns=drop_columns,
    #         rename_columns=rename_columns,
    #         composition_columns=composition_columns,
    #         technique=technique_name,
    #         allowed_extensions=None,
    #         required_columns=None,
    #         min_wavelength=min_wavelength,
    #         max_wavelength=max_wavelength
    #     )
    #     if result is not None:
    #         enriched_paths.append(result)
    # endregion

    # region ThreadPool
    # with ThreadPoolExecutor() as executor:
    #     futures = {
    #         executor.submit(
    #             enrich_file_with_metadata,
    #             path=file, 
    #             enriched_root=enriched_root, 
    #             drop_columns=drop_columns,
    #             rename_columns=rename_columns, 
    #             composition_columns=composition_columns, 
    #             technique=technique_name,
    #             allowed_extensions=allowed_extensions,
    #             required_columns=required_columns,
    #             salt_states=salt_states,
    #             experimental_variation_columns=experimental_variation_columns,
    #             include_experimental_variation_columns=include_experimental_variation_columns,
    #             min_wavelength=min_wavelength,
    #             max_wavelength=max_wavelength
    #         ): file for file in files
    #     }
    #     for future in tqdm(as_completed(futures), total=len(files),
    #                        desc=f"Enriching {technique_name}", unit="file"):
    #         result = future.result()
    #         if result is not None:
    #             enriched_paths.append(result)
    # return enriched_paths
    # endregion

    # region ProcessPool
    # Wrap the function call so all args are picklable
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
        max_wavelength=max_wavelength
    )

    # max_workers=None lets Python pick (usually cpu_count())
    # For I/O-heavy work you can go higher, e.g. max_workers=os.cpu_count() * 2
    print(f"Starting ProcessPoolExecutor with {min(4, os.cpu_count() or 4)} workers...")
    with ProcessPoolExecutor(
        max_workers=min(4,os.cpu_count() or 4),
        initializer=worker_init,
        initargs=(log_q,)
    ) as executor:
        futures = {
            executor.submit(worker, path=file): file 
            for file in files
        }
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
    # endregion

def species_sort_key(item):
    key, nice_name = item
    is_compound = any(s in nice_name for s in ['Cl', 'N', 'O'])
    return (0 if is_compound else 1, -len(key))
        
def parent_concentration_data(
        source_data_paths: List[str | Path],
        composition_columns: Optional[List[str]] = None,
        required_columns: Optional[List[str]] = None,
        salt_states: Optional[List[str]] = None,
        experimental_variation_columns: Optional[List[str]] = None,
        include_experimental_variation_columns: bool = True,
        # log_path: Path | None = None,
) -> pd.DataFrame:
    # region composition columns, species map, required columns, and MS concentration map
    if composition_columns is None:
        composition_columns = _DEFAULT_CONC_COLS

    if required_columns is None:
        required_columns = _DEFAULT_REQ_COLS
        
    if salt_states is None:
        salt_states = _DEFAULT_SALT_STATES
        required_columns = required_columns + salt_states
        
    if experimental_variation_columns is None:
        experimental_variation_columns = _DEFAULT_EXP_VARS_COLS

    if include_experimental_variation_columns:
        required_columns = required_columns + experimental_variation_columns


    # endregion
    

    # region Explanation
                    # rf'' is a raw f-string, lets you put {key.lower()} without having to escape first
                    # (?i) is a case insensitive marker. Essentially it is telling the pattern to ignore
                    #       any complaints that come from case mismatches. Ex. SMCL3 = SmCl3 = smcl3
                    # (?:^|[^\w.]) is a non-capturing group that acts as a lef-side boundary check. It
                    #       ensures the match either starts at the very beginning of the string (^), OR is
                    #       preceded by a character that is NOT a word character (\w = letters, digits, 
                    #       underscore) and NOT a period (.). This prevents partial matches like matching 
                    #       "5" in "MS5" or in "15". 
                    #       (?:  )      -> non-capturing group
                    #                   using ?: instead of just ( makes it so that we don't actually save
                    #                   the stuff that this part finds as its own group. We don't need the
                    #                   info here longterm, it is just for grouping the OR options.
                    #       ^           -> start of the string 
                    #       |           -> means "OR"
                    #       [^\w.]      -> any single character that is NOT a word character or period 
                    # (3quarters|half|quarter|\d+\.?\d*) is the furst capturing group:
                    #       3quarters   -> looks for the literal word "3quarters"
                    #       |           -> means "OR"
                    #       half        -> looks for the literal word "half"
                    #       |           -> means "OR"
                    #       quarter     -> looks for the literal word "quarter"
                    #       |           -> means "OR"
                    #       \d+\.?\d*   -> looks for one or more digits (d+), with an optional decimal (.?),
                    #                   and an option for more digits after the decimal (d*)
                    # \s* is an indication for an optional space (" "). It accepts either zero or more
                    #       spaces
                    # (?:wt%|%)? is finding the units of the number above (unit = wt%,wt,% after the number)
                    #       (?:  )      -> non-capturing group
                    #       wt%?        -> looks for "wt" followed by an optional "%". The "?" is what makes
                    #                   the "%" optional
                    #       |           -> means "OR"
                    #       %           -> checks for any instances of "%" without "wt" in front.
                    #                   ex. 2.0%La
                    #       ()?         -> makes the entire group that we just looked thru optional. 
                    # \s* is an indication for an optional space (" "). It accepts either zero or more
                    #       spaces
                    # [-_ ]? is an optional single character filter.
                    #       - looks for "-"
                    #       _ looks for "_"
                    #       " " looks for " "
                    # {re.escape(key.lower())} inserts the species key (e.g. "smcl3", "gd", or "nd") into the
                    #       pattern at complie time. 
                    #       re.escape() -> ensures an special regex characters in the key are treated as 
                    #                   literals. 
                    #       key.lower() -> The key is lowercased to match all_text, which is also lowercased 
                    #                   before searching. 
                    # \b is a word boundary anchor. It asserts that the character immediately after the species
                    #       key is NOT a word character (letter, digit, or underscore). This prevents partial 
                    #       matches like "nd" matching inside "2nd" (ordinal number), or "sm" matching inside a 
                    #       longer word. The match must end cleanly at a word boundary - e.g. followed by a "_",
                    #       ".", " ", ")", or the end of the string. 
                    # 
                    # 
                    # #################### No longer in this script but still useful info ################### 
                    # {key.lower()} pulls the actual species name and makes it all lower case.
                    # (?=[\s_.-]|$) looking ahead at the rest of the file name for one of the following
                    #       characters or the end of the string but don't actually do anything with them.
                    #       (?=  )      -> This is a positive lookahead. It looks at the rest of the string
                    #                   without actually moving the location of the match/checker forward
                    #       [\s_.)-]    -> This is the character class that it is looking for, it needs to 
                    #                   match ONE of these
                    #                       \s          -> Indicates white space (aka space or tab)
                    #                       _           -> underscore
                    #                       .           -> period
                    #                       )           -> end parenthises 
                    #                       -           -> hyphen
                    #       |           -> means "OR"
                    #       $           -> This means the end of the string.
    # endregion 
    # region Loop setup
    rows = []
    for path_str in source_data_paths:
        if isinstance(path_str, Path):
            path_str = str(path_str)
        
        path = Path(path_str)
        all_text = ' '.join(path.parts) + ' ' + path.name
        all_text = all_text.lower()
        # endregion
        
        # region Default values
        data: dict[str, float | str | int | None] = {
            col: 0.0 for col in composition_columns if 'conc_' in col
        }
        data['frac_LiCl'] = 0.59  # Default eutectic mole frac
        data['frac_KCl'] = 0.41
        data['temperature_C'] = None
        data['scan_rate_mVs'] = None
        data['technique'] = None
        data['file_path'] = path_str
        # endregion
               
        # region Extract Non-Concentration Data
        # region Extract scan rate
        scan_match = _SCAN_PATTERN.search(all_text)
        if scan_match:
            data['scan_rate_mVs'] = float(scan_match.group(1))
        # endregion
        
        # region Extract technique
        tech_match = _TECH_PATTERN.search(all_text)
        if tech_match:
            data['technique'] = tech_match.group(1).upper()
        else:
            if 'cv' in all_text: data['technique'] = 'CV'
            elif 'ocv' in all_text: data['technique'] = 'OCV'
            elif 'ca' in all_text: data['technique'] = 'CA'
            elif 'lp' in all_text: data['technique'] = 'LP'
        # endregion

        # region Extract Gate Delay
        gtd_match = _GTD_PATTERN.search(all_text)
        delay_match = _DELAY_PATTERN.search(all_text)
        if gtd_match:
            data['delay_study'] = 1
            data['delay'] = float(gtd_match.group(1))
        elif delay_match:
            data['delay_study'] = 1
            val = float(delay_match.group(1) or delay_match.group(3))
            unit = delay_match.group(2) or delay_match.group(4) or 'us'
            if unit == 's':
                val *= 1e6
            elif unit == 'ms':
                val *= 1e3
            elif unit == 'ns':
                val /= 1e3
            data['delay'] = val
        else:
            data['delay_study'] = 0
            data['delay'] = _STANDARD_DELAY
        # endregion

        # region Extract Rep number
        rep_match = _REP_PATTERN.search(all_text)
        if rep_match:
            data['repetition'] = int(rep_match.group(1))
        else:
            data['repetition'] = 1
        # endregion

        # region Extract Gate Width
        width_match = _WIDTH_PATTERN.search(all_text)
        if width_match:
            data['width_study'] = 1
            val = float(width_match.group(1) or width_match.group(3))
            unit = width_match.group(2) or width_match.group(4) or 'us'
            if unit == 's':
                val *= 1e6
            elif unit == 'ms':
                val *= 1e3
            data['width'] = val
        else:
            data['width_study'] = 1 if 'width' in all_text else 0
            data['width'] = _STANDARD_WIDTH
        # endregion

        # region Extract Energy
        energy_match = _ENERGY_PATTERN.search(all_text)
        neup_match = _NEUP_PATTERN.search(all_text)
        if neup_match:
            standard_energy = 200   # "G:\My Drive\RLSL\Data\user\Awilliams\DATA BACKUP\NEUP LIBS Project\Preliminary Data and Information\Spectra\Laser Energy Test\energytest1.txt"
        else:
            standard_energy = 100

        if energy_match:
            data['energy_study'] = 1
            val = float(energy_match.group(1) or energy_match.group(2))
            data['energy'] = val    # in mJ
        else:
            data['energy_study'] = 0
            data['energy'] = standard_energy    # TODO this actually needs to be the energy 
                                                # of the laser for the other studies too, 
                                                # but that is going to need another set of
                                                # extraction logic that I don't want to do
                                                # right now lol. 
        # endregion

        # region Extract Qdelay
        qdelay_match = _QDELAY_PATTERN.search(all_text)
        if qdelay_match:
            data['qdelay_study'] = 1
            val = float(qdelay_match.group(1) or qdelay_match.group(2))
            data['qdelay'] = val
        else:
            data['qdelay_study'] = 0
            data['qdelay'] = _STANDARD_QDELAY
        # endregion

        # region Extract shots
        shot_match = _SHOT_PATTERN.search(all_text)
        if shot_match:
            data['shot_study'] = 1
            val = float(shot_match.group(1) or shot_match.group(2))
            data['shots'] = val
        else:
            data['shot_study'] = 0
            data['shots'] = _STANDARD_SHOTS
        # endregion

        # region Extract Flow
        flow_match = _FLOW_PATTERN.search(all_text)
        if flow_match:
            data['flow_study'] = 1
            val = float(flow_match.group(1).replace('p', '.'))
            data['flow'] = val
        else:
            data['flow_study'] = 0
            data['flow'] = 0 # TODO same as for energy.
        # endregion

        # region Extract Pressure
        pressure_match = _PRESSURE_PATTERN.search(all_text)
        if pressure_match:
            data['pressure_study'] = 1
            val = float(pressure_match.group(1))
            data['pressure'] = val
        else:
            data['pressure_study'] = 0
            data['pressure'] = 0.0
        # endregion

        # region Extract SNR
        # snr_pattern = r''
        # snr_match = re.search(snr_pattern, all_text)
        # if snr_match:
        #     data['test_snr_study'] = 1
        #     data['test_snr'] = 1
        # else:
        #     data['test_snr_study'] = 0
        #     data['test_snr'] = 0
        #         # TODO check that setting this to zero for any non snr study 
        #         # tests is actually viable.
        # endregion

        # region Extract static
        # I am pretty sure that this should just be a yes/no. Essentially if it
        # is static then the laser is impacting the same point every time, if it
        # isn't static then the laser impacts a new surface each time. It is
        # likely that static tests would have changes as you go further into
        # them because of the char from previous shots as well as the drilling
        # effect the laser has, drilling out a hole, so the laser focus and
        # intensity actually change shot to shot in a static study. 
        static_match = _STATIC_PATTERN.search(all_text)
        if static_match:
            data['static_'] = 1
        else:
            data['static_'] = 0
        # endregion

        # region Extract kinetic
        data['kinetic'] = 1 if 'kinetic' in all_text else 0
        # endregion

        # region Extract blank
        blank_match = _BLANK_PATTERN.search(all_text)
        if blank_match:
            data['blank'] = 1
            for comp in composition_columns:
                if comp not in ('frac_LiCl', 'frac_KCl'):
                    data[comp] = 0
        else:
            data['blank'] = 0
        # endregion

        # region Extract salt state
        data['state_aerosol'] = 1 if 'aerosol' in all_text else 0
        data['state_molten'] = 1 if ( 'molten' in all_text and 'aerosol' not in all_text) else 0
        data['state_solid'] = 0 if data['state_aerosol'] == 1 else(0 if data['state_molten'] == 1 else 1)
        # endregion

        # region Extract temperature
        temp_match = _TEMP_PATTERN.search(all_text)
        if data['state_aerosol'] == 1 or data['state_molten'] == 1:
            if temp_match:
                data['temperature_C'] = float(temp_match.group(1))
            elif any(s in all_text for s in ['sla_', 'slb_', 'slc_', 'sld_']):
                data['temperature_C'] = 500
            elif 'andrewsh' in all_text:
                data['temperature_C'] = 501
            else:
                for part in path.parts:
                    if part.isdigit() and 300 < int(part) < 1000:
                        data['temperature_C'] = float(part)
                        break
                else: data['temperature_C'] = 500
        else:
            if temp_match:
                temp = float(temp_match.group(1))
                if temp > 360:
                    # 352 C is melting point of LiCl-KCl, doing 360 for error. Reference:
                    # M. DEL ROCÍO RODRÍGUEZ-LAGUNA et al., “Effect of iodides on thermal behavior 
                    # and phase partitioning in LiCl-KCl,” J. Mol. Liq. 418, 126706 (2025); 
                    # https://doi.org/10.1016/j.molliq.2024.126706.
                    data['temperature_C'] = temp
                    data['state_molten'] = 1
                    data['state_solid'] = 0
            else:
                data['temperature_C'] = 20
        # endregion
        # endregion
        
        # region Extract Concentration Data
        # region Extract compositions
        path_species = []
        seen_cols = set()
        matched_nice_names = set()
        
        for key, nice_name in sorted(_SPECIES_MAP.items(), key=species_sort_key):
            match = re.match(r'[A-Za-z]+', nice_name)
            if match is None:
                continue
            base_element = match.group(0)
            if any(n.startswith(base_element) and n != nice_name for n in matched_nice_names):
                continue

            if re.search(rf'(?i)(?<![a-z]){re.escape(key.lower())}(?=[_\s.\-/\\(]|$)', all_text):
                col = f'conc_{nice_name}_wt%'
                if col in data and col not in seen_cols:
                    path_species.append((col, nice_name))
                    seen_cols.add(col)
                    matched_nice_names.add(nice_name)

        filename_text = path.name.lower()
        conc_match = _CONC_PATTERN.search(filename_text)

        if conc_match and len(path_species) == 1:
            val = float(conc_match.group(1))
            unit = conc_match.group(2).lower()
            if 'ppm' in unit:
                val = val / 10000.0
            col = path_species[0][0]
            data[col] = val
        
        word_vals = {'half': 0.5, '3quarters': 0.75, 'quarter': 0.25}
        for pattern, nice_name in _COMPILED_SPECIES_PATTERNS.items():
            for match in pattern.finditer(all_text):
                val_str = match.group(1)
                if val_str is None:
                    continue 
                val_str = val_str.lower()
                #print(f'   MATCH: val_str={repr(val_str)} | nice_name={nice_name} | full_match={repr(match.group(0))}')
                if val_str in word_vals:
                    val = word_vals[val_str]
                else:
                    try:
                        val = float(val_str)
                    except ValueError:
                        print(f'   Could not convert "{val_str}" to float, skipping match')
                        continue

                after_match = all_text[match.end():match.end()+5]
                if 'ppm' in after_match:
                    val = val / 10000.0
                # NOTE: this is just a common approximation for LiCl-KCl and really
                # should be more specific. What would be best is a conversion 
                # per added species. But for now the approximation is ok since
                # I am just trying to get this thing to work at all.
                # TODO: replace with species-specific ppm -> wt% conversion factors 
                col = f'conc_{nice_name}_wt%'
                if col in data:
                    data[col] = val
        # endregion

        # region Concentration Maps
        # region MS Map
        if re.search(r'(?<![a-z])ms\d', all_text) and not re.search(r'msu\d', all_text):
            av_conc = sum(_MS_CONC_MAP.values()) / len(_MS_CONC_MAP)
            for key, conc in sorted(_MS_CONC_MAP.items(), key=lambda x: -len(x[0])):
                if key in all_text:
                    data['conc_CeCl3_wt%'] = conc
                    break
            else:
                    data['conc_CeCl3_wt%'] = av_conc
        # endregion
        
        # region MSU Map
        if re.search(r'msu\d', all_text):
            if data.get('conc_UCl3_wt%', 0.0) == 0.0:
                data['conc_UCl3_wt%'] = 1.0
        # endregion
        
        # region SampleU Map
        if re.search('sampleu', all_text):
            for key, conc in sorted(_SAMPLE_U_MAP.items(), key=lambda x: -len(x[0])):
                if key in all_text:
                    data['conc_UCl3_wt%'] = conc[0]
                    data['conc_GdCl3_wt%'] = conc[1]
                    break
        # endregion

        # region SL_ Map
        if re.search(r'sl[abcd]', all_text):
            for key, conc in sorted(_SL___MAP.items(), key=lambda x: -len(x[0])):
                if key in all_text:
                    data['conc_SmCl3_wt%'] = conc
                    break
        # endregion

        # region SampleM Map
        if re.search('samplem', all_text):
            for key, conc in sorted(_SAMPLE_MG_MAP.items(), key=lambda x: -len(x[0])):
                if key in all_text:
                    data['conc_U_wt%'] = conc[0]
                    data['conc_Mg_wt%'] = conc[1]
                    break
        # endregion

        # region 1 thru T10 Map
        if re.search(r'sample\d+_t\d+|samplet\d+_t\d+', all_text):
            for key, conc in sorted(_SAMPLE_1_THRU_T10_MAP.items(), key=lambda x: -len(x[0])):
                if key.lower() in all_text:
                    data['conc_SmCl3_wt%'] = conc[0]
                    data['conc_GdCl3_wt%'] = conc[1]
                    break
        # endregion

        # region Keith Map
        if re.search('keith', all_text):
            for key, conc in sorted(_KEITH_CECL3_MAP.items(), key=lambda x: -len(x[0])):
                if key in all_text:
                    data['conc_CeCl3_wt%'] = conc[0]
                    data['conc_GdCl3_wt%'] = conc[1]
                    break
        # endregion

        # region Trials 1 and 2 Map
        # This is from AndrewsH SmCl3 Glovebox Sm LIBS raw data.
        if re.search(r'trial 1|trail 2', all_text):
            for key, conc in sorted(_TRIAL_1_OR_2_MAP.items(), key=lambda x: -len(x[0])):
                if key in all_text:
                    data['conc_SmCl3_wt%'] = conc
                    break
        # endregion

        # region SSU1/DU1 Map
        if re.search(r'ssu1|du1', all_text):
            for key, conc in sorted(_SSU_AND_DU_MAP.items(), key=lambda x: -len(x[0])):
                if key in all_text:
                    data['conc_UCl3_wt%'] = conc
                    break
        # endregion
        
        # endregion
        # endregion
        
        rows.append(data)
    
    df = pd.DataFrame(rows, columns=required_columns)
    return df

def clean_single_technique_file(
        path: Path,
        technique: str = 'cv',
        drop_columns: Optional[List[str]] = None,
        rename_columns: Optional[dict] = None,
        required_dict: Optional[dict] = None,
        required_columns: Optional[List[str]] = None,
        min_wavelength: Optional[float] = None,
        max_wavelength: Optional[float] = None,
        log_path: Path | None = None,
) -> Optional[pd.DataFrame]:
    """
    Clean and validate a SINGLE {technique} CSV file.
    Returns the cleaned DataFrame or None if it should be skipped.

    required_dict should be in form:
        required_dict = {
        'technique_1': ['req_col_1a','req_col_1b',...,'req_col_1z'],
        'technique_2': ['req_col_2a','req_col_2b',...,'req_col_2z'],
        ...,
        'technique_n': ['req_col_na','req_col_nb',...,'req_col_nz']
        }
    *NOTE: If required_dict is provided, any technique not included
    in it will silently fall back to requiring only ['Ewe/V']. Ensure
    that all expected techniques are covered to avoid unexpected
    behavior.*
    """
    
    path = Path(path)
    sanitized_path = sanitize_path(
        path=path,
        log_path=log_path
    )
    if sanitized_path is None:
        return None
    path = sanitized_path
    # assert isinstance(path, Path)
    path = cast(Path, sanitized_path)

    # region Logger Setup
    if log_path is None:
        log_path = Path(r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\default_log.txt").resolve()
    logger = get_worker_logger(Path(log_path).stem)
    # endregion

    # region Required Dictionary
    if required_dict is None:
        required_dict = {
            'cv': ['<I>/mA', '(Q-Qo)/C'],
            'ocv': ['time/s', 'Ewe/V'],
            'ca': ['time/s', 'Ewe/V', 'I/mA'],
            'lp': ['<I>/mA', 'time/s', 'Ewe/V']
        }
    # endregion

    # region Required COlumns
    if required_columns is None:
        if technique is None or technique.lower() == 'libs':
            required_columns = []
        else:
            required_columns = required_dict.get(technique.lower(), ['Ewe/V'])

    required_columns = required_columns or []   # fall back assertion for the 'missing = ...' in try
    # endregion

    try:
        # region Headers Check
        headers = pd.read_csv(long_path(path), nrows=0).columns.tolist()

        missing = [col for col in required_columns if col not in headers]
        if missing:
            log(logger=logger, msg = f"         Skipped (missing required columns): {path.name}")
            log(logger=logger, msg = f'                 {missing}')
            return None
        # endregion

        # region Loag Data
        df = pd.read_csv(long_path(path), delimiter=',', header=0, skiprows=0, low_memory=False)
        log(logger=logger, msg = f"     Loaded: {path.name}  |  shape={df.shape}")
        # endregion

        # region Wavelength Trim
        if min_wavelength is not None and max_wavelength is not None:
            cols = df.columns.astype(str).str.strip().tolist()
            numeric_cols = pd.to_numeric(cols, errors='coerce')
            numeric_arr = pd.Series(numeric_cols).to_numpy(dtype=float, na_value=np.nan)
            
            mask = ((numeric_arr >= min_wavelength) & (numeric_arr <= max_wavelength))
            non_wl_mask = np.isnan(numeric_arr)
            df = df.loc[:, mask | non_wl_mask]

        df.columns = [str(c) for c in df.columns]
        df = df.loc[:, ~df.columns.str.contains(r'^\s*$|^nan$', case=False, na=True)]
        # endregion

        # region Basic Cleaning
        df = df.dropna(how='all')
        # endregion

        # region Numeric Convertion
        id_cols = {'file_id', 'cycle number', 'loop number', 'Ns', 'half cycles'}
        cols_to_convert = [c for c in df.columns if c not in id_cols
                           and not pd.api.types.is_numeric_dtype(df[c])]
        if cols_to_convert:
            df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
        # endregion

        # region Drop / rename
        if drop_columns:
            df = df.drop(columns=drop_columns, errors='ignore')
        if rename_columns:
            actual_rename = {k: v for k, v in rename_columns.items() if k in df.columns}
            df = df.rename(columns=actual_rename)
        # endregion

        # Temporary file_id (will be overwritten later if needed)
        df['file_id'] = f"file_{path.stem}"

    # region Report Success
        log(logger=logger, msg = f"  ✅ Cleaned {path.name}  |  shape={df.shape}")
        return df
    # endregion

    # region Report Fail
    except Exception as e:
        log(logger=logger, msg = f"  ❌ Failed to clean {path.name}: {e}")
        return None
    # endregion

def trn_val_splitter_HDF5(
        h5_path: str | Path,
        output_path: str | Path | None = None,        # if None, splits in-place in same file
        test_size: float = 0.2,
        random_state: int = 42,
        log_path: Path | None = None,
) -> Path:
    """
    Reads a combined HDF5 file and writes train/val splits back into it
    (or into a new file) as:
        /train/spectra
        /train/metadata/<col>
        /val/spectra
        /val/metadata/<col>
    """
    from sklearn.model_selection import train_test_split
    import h5py

    h5_path = Path(h5_path)
    output_path = Path(output_path) if output_path else h5_path.with_name(
        h5_path.stem + '_split.h5'
    )
    
    # region Logger Setup
    if log_path is None:
        log_path = Path(r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\default_log.txt").resolve()
    logger = get_worker_logger(Path(log_path).stem)
    # endregion
    
    log(logger=logger, msg=f'Loading from {h5_path.name}...')

    with h5py.File(h5_path, 'r') as hf:
        spectra = hf_get(hf, 'spectra')
        wavelengths = hf_get(hf, 'wavelengths')
        metadata_grp = hf['train/metadata']
        assert isinstance(metadata_grp, Group), "Expected a Group at 'train/metadata'"
        meta_cols = list(metadata_grp.keys())
        metadata = {col: meta_cols[col][:] for col in meta_cols}
        attrs = dict(hf.attrs)

    n_rows = spectra.shape[0]
    log(logger=logger, msg=f'  Total rows: {n_rows}  |  Wavelengths: {spectra.shape[1]}')

    indices = np.arange(n_rows)
    trn_idx, val_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    log(logger=logger, msg=f'  Train: {len(trn_idx)}  |  Val: {len(val_idx)}')

    # Sort indices so HDF5 reads stay sequential (much faster)
    trn_idx = np.sort(trn_idx)
    val_idx  = np.sort(val_idx)

    comp_kwargs = {'compression': 'gzip', 'compression_opts': 3}

    log(logger=logger, msg=f'Writing splits → {output_path.name}')

    with h5py.File(output_path, 'w') as hf:
        hf.create_dataset('wavelengths', data=wavelengths)

        for split_name, idx in [('train', trn_idx), ('val', val_idx)]:
            grp = hf.create_group(split_name)

            grp.create_dataset(
                'spectra',
                data=spectra[idx],
                chunks=(min(1000, len(idx)), spectra.shape[1]),
                **comp_kwargs
            )

            meta_grp = grp.create_group('metadata')
            for col, arr in metadata.items():
                if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer):
                    meta_grp.create_dataset(col, data=arr[idx], **comp_kwargs)
                else:
                    # string columns
                    dt = h5py.string_dtype(encoding='utf-8')
                    meta_grp.create_dataset(col, data=arr[idx], dtype=dt)

            grp.attrs['n_rows'] = len(idx)

        # Copy top-level attrs
        for k, v in attrs.items():
            hf.attrs[k] = v
        hf.attrs['train_size'] = len(trn_idx)
        hf.attrs['val_size']   = len(val_idx)
        hf.attrs['test_size_frac'] = test_size
        hf.attrs['random_state']   = random_state

    size_mb = output_path.stat().st_size / 1_048_576
    log(logger=logger, msg=f'✅ Saved: {output_path.name}  |  {size_mb:.1f} MB')
    return output_path

def trn_val_splitter_CSV( 
    df_path: str | Path,
    trn_path: str | Path,
    val_path: str | Path,
    test_size: float=0.2,
    random_state: int=42,
    shuffle: bool=True,
    log_path: Path | None = None,
):
    df_path = Path(df_path)
    trn_path = Path(trn_path)
    val_path = Path(val_path)
    from sklearn.model_selection import train_test_split

    # region Logger Setup
    if log_path is None:
        log_path = Path(r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\default_log.txt").resolve()
    logger = get_worker_logger(Path(log_path).stem)
    # endregion

    # Ensure that the designated DataFrame is actually there
    if not df_path.is_file():
        raise FileNotFoundError(f'  ❌ Input file not found: {df_path}')
    
    # Load in the DataFrame from the provided path
    log(logger=logger, msg = f'  ✅ Loading DataFrame from: {df_path}')
    df = pd.read_csv(df_path)

    # Ensure that the DataFram actually has data
    if df.empty:
        raise ValueError('  ❌ Loaded DataFram is empty!')
    
    log(logger=logger, msg = f'Original DataFrame Shape: {df.shape}')
    
    # Split the DataFrame into training and validation data sets
    trn_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle
    )

    # Ensure that the output directories for the training and 
    # validation data set CSVs exist
    trn_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    log(logger=logger, msg = f'Training DataFrame Shape: {trn_df.shape}')
    log(logger=logger, msg = f'Validation DataFrame Shape: {val_df.shape}')

    # Save the training and validation data sets as CSVs
    trn_df.to_csv(trn_path, index=False)
    val_df.to_csv(val_path, index=False)
    log(logger=logger, msg = "Training and Validation Data have been split and saved.")

def standardize_wavelength_grid(
        # This is not currently used by any other functions in this file, but
        # I can't remember if it is used elsewhere, so I am leaving it in here
        # for now. If I don't find its application somewhere else, I will move
        # this to Python>utils>def_archive.py
        # 
        # NOTE if I do end up needing it, I need to update prints to logs 
        file_path: str | Path,
        min_wl: float=350.0,
        max_wl: float=800.0,
        n_points: int=451,
        log_path: Path | None = None,
) -> Optional[tuple[pd.DataFrame, np.ndarray]]:
    
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
    
    target_grid = np.linspace(min_wl, max_wl, n_points)
    cols = df.columns.astype(str).str.strip().tolist()
    numeric_cols = pd.to_numeric(cols, errors='coerce')
    wl_mask = pd.Series(numeric_cols).notna().to_numpy().astype(bool)

    wl_cols = df.columns[wl_mask].astype(float)
    spectra = df.loc[:, wl_mask].values.astype(np.float32)

    standardized = np.array([
        np.interp(target_grid, wl_cols, row) for row in spectra
    ])

    meta_cols = df.columns[~wl_mask]
    meta_df = df[meta_cols].reset_index(drop=True)
    spectra_df = pd.DataFrame(standardized, columns=target_grid.astype(str))

    return pd.concat([meta_df, spectra_df], axis=1), target_grid

def combine_and_save_as_HDF5(
        individuals_root: str | Path,
        combined_root: str | Path,
        technique: Optional[str] = None,
        compression: str = 'gzip',         # 'gzip', 'lzf', or None
        compression_level: int = 3,        # 1 (fast) to 9 (small), only for gzip
        log_path: Path | None = None,
) -> Optional[Path]:
    """
    Combines all enriched CSVs into a single HDF5 file with two groups:
        /spectra   → float32 array of shape (n_rows, n_wavelengths)
        /metadata  → one dataset per metadata column
        /attrs     → wavelength axis + column name lists

    Returns the path to the .h5 file.
    """
    import h5py
    
    individuals_root = Path(individuals_root).resolve()
    combined_root = Path(combined_root).resolve()
    combined_root.mkdir(parents=True, exist_ok=True)
    
    # region Logger Setup
    if log_path is None:
        log_path = Path(r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\default_log.txt").resolve()
    logger = get_worker_logger(Path(log_path).stem)
    # endregion

    if technique is None:
        technique = 'libs'

    enriched_files = list(individuals_root.glob('*.csv'))
    if not enriched_files:
        log(logger=logger, msg = f'  ❌ No {technique} files found.')
        return None
    
    log(logger=logger, msg = f'Found files: {len(enriched_files)}')
    log(logger=logger, msg = f'Reading files...')

    all_spectra = []            # list of 2D float arrays
    all_metadata = []           # list of dicts (one per row, non-spectral cols)

    for i, f in enumerate(enriched_files):
        try:
            swlg = standardize_wavelength_grid(
                file_path=f,
                min_wl=TARGET_MIN_WL,
                max_wl=TARGET_MAX_WL,
                n_points=TARGET_N_PTS
            )
            if swlg is None:
                log(logger=logger, msg=f'  ⚠️  Skipping {f.name} (standardize returned None)')
                continue
            std_df, target_grid = swlg
        except Exception as e:
            log(logger=logger, msg = f'  ❌ Could not read {f.name}: {e}')
            continue


        # Identify spectral columns (numeric column names = wavelengths)
        col_index = std_df.columns.astype(str).str.strip().tolist()
        numeric_vals = pd.to_numeric(col_index, errors='coerce')
        numeric_mask = pd.Series(numeric_vals).notna().to_numpy().astype(bool)

        spectra = std_df.loc[:, numeric_mask].values.astype(np.float32)
        meta_df = std_df.loc[:, ~numeric_mask]

        all_spectra.append(spectra)
        all_metadata.append(meta_df)

        if (i + 1) % 100 == 0:
            log(logger=logger, msg= f'  Read {i+1}/{len(enriched_files)}')

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
    log(logger=logger, msg= f'✅ Saved: {h5_path.name}  |  {size_mb:.1f} MB')
    return h5_path

def combine_and_save_as_CSV(
        individuals_root: str | Path,
        combined_root: str | Path,
        technique: Optional[str] = None,
        log_path: Path | None = None,
) -> Optional[Path]:
    individuals_root = Path(individuals_root).resolve()
    combined_root = Path(combined_root).resolve()
    combined_root.mkdir(parents=True, exist_ok=True)
    
    # region Logger Setup
    if log_path is None:
        log_path = Path(r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\default_log.txt").resolve()
    logger = get_worker_logger(Path(log_path).stem)
    # endregion

    if technique is None:
        technique = 'libs'

    enriched_files = list(individuals_root.glob('*.csv'))
    if not enriched_files:
        log(logger=logger, msg = f'  ❌ No {technique} files found.')
        return None
    
    log(logger=logger, msg = f'Found files: {len(enriched_files)}')
    log(logger=logger, msg = f'Reading files...')

    try:
        combined_df = pd.concat(
            [pd.read_csv(f) for f in enriched_files],
            ignore_index=True,
            sort=False
        )
        label = technique.upper() if technique else 'LIBS'
        combined_path = combined_root / f'cleaned_combined_{label}.csv'
        combined_df.to_csv(combined_path, index=False)
        log(logger=logger, msg = f'Saved: {combined_path.name}  |  shape={combined_df.shape}')
        return combined_path
    except Exception as e:
        log(logger=logger, msg = f'  ❌ {e}')
        return None

def training_ready_h5(
        h5_path: str | Path,
        prepped_h5_path: str | Path,
        allowed_cols: set[str],
):
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import StandardScaler

    if h5_path is None:
        h5_path = r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\LIBS\trn_val_split_LIBS.h5"
    if prepped_h5_path is None:
        prepped_h5_path = r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\LIBS\training_ready_LIBS.h5"
    if allowed_cols is None:
        allowed_cols= {
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

    y_trn_raw, y_val_raw, X_trn_raw, X_val_raw, target_cols, wavelengths, meta_val_df = load_h5_split_dataset(
                h5_path=h5_path,
                allowed_cols=allowed_cols,
                log_path=None
            )

    assert X_trn_raw.shape[0] == y_trn_raw.shape[0], \
                f'X/y row mismatch: {X_trn_raw.shape[0]} vs {y_trn_raw.shape[0]}'

    selector = VarianceThreshold(threshold=1e-10)
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    # region Filter out rows
    # Find the fraction of NaN values as wavelength per row
    nan_frac_trn = np.isnan(X_trn_raw).mean(axis=1)
    nan_frac_val = np.isnan(X_val_raw).mean(axis=1)

    # Designate any row with more than __% as bad rows to be dropped
    bad_rows_trn = nan_frac_trn > 0.1
    bad_rows_val = nan_frac_val > 0.1
    #log(logger=logger, msg= f'Dropping {bad_rows_trn.sum()} training rows and {bad_rows_val.sum()} validation rows with >10% NaN values in Spectrum')

    # Drop designated rows
    X_trn_raw = X_trn_raw[~bad_rows_trn]
    X_val_raw = X_val_raw[~bad_rows_val]
    y_trn_raw = y_trn_raw[~bad_rows_trn]
    y_val_raw = y_val_raw[~bad_rows_val]

    meta_val_df = meta_val_df[~bad_rows_val].reset_index(drop=True)
    # endregion

    # region Clip bright data
    # Clip the extra bright shots so they don't skew the mean and standard deviation
    clip_threshold = np.nanpercentile(X_trn_raw, 99.5)
    #log(logger=logger, msg= f'Clipping spectra intensities above: {clip_threshold:.1f}')
    X_trn_raw = np.clip(X_trn_raw, 0, clip_threshold)
    X_val_raw = np.clip(X_val_raw, 0, clip_threshold)   # val uses the same threshold as trn
    # endregion 

    # region Keep physical copy
        # keep a copy of the clipped but not scaled for the predicted/actual plot
    y_val_physical = y_val_raw.copy()
    # endregion

    # region Filter the y data
    selector.fit(y_trn_raw)
    surviving_cols = [c for c, keep in zip(target_cols, selector.get_support()) if keep]
    y_trn_filtered = pd.DataFrame(
        np.array(selector.transform(y_trn_raw)),
        columns=surviving_cols)
    y_val_filtered = pd.DataFrame(
        np.array(selector.transform(y_val_raw)),
        columns=surviving_cols)
    # Filter the metadata df so that index alignment is preserved
    meta_val_df = meta_val_df[[c for c in surviving_cols if c in meta_val_df.columns]]
    # endregion
    
    # region Scale the X data
    X_trn_scaled = np.nan_to_num(X_scaler.fit_transform(X_trn_raw), nan=0.0, posinf=0.0, neginf=0.0)
    X_val_scaled = np.nan_to_num(X_scaler.transform(X_val_raw), nan=0.0, posinf=0.0, neginf=0.0)
    # endregion

    # region Downsample
    downsample_factor = 4
    X_trn_scaled = X_trn_scaled[:, ::downsample_factor]
    X_val_scaled = X_val_scaled[:, ::downsample_factor]
    # endregion

    # region Expand X data
    X_trn = np.expand_dims(X_trn_scaled, axis=1).astype(np.float32)
    X_val = np.expand_dims(X_val_scaled, axis=1).astype(np.float32)
    # endregion

    # region Scale y data
    y_trn = np.nan_to_num(y_scaler.fit_transform(y_trn_filtered), nan=0.0).astype(np.float32)
    y_val = np.nan_to_num(y_scaler.transform(y_val_filtered), nan=0.0).astype(np.float32)
    # endregion

    # region Align metadata df to surviving bad-row mask
    meta_val_df = meta_val_df.reset_index(drop=True)
    # endregion

    # region Report
    n_features = X_trn.shape[2]
    n_targets = y_trn.shape[1]
    row_maxes = X_trn_raw.max(axis=1)
    outlier_mask = row_maxes > np.percentile(row_maxes, 99)

    # Set up compression just like your combiner function
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 3}

    print(f"Writing HDF5 → {prepped_h5_path.name}")
    with h5py.File(str(prepped_h5_path), 'w') as hf:
        # 1. Save the main arrays
        hf.create_dataset('X_trn', data=X_trn, **comp_kwargs)
        hf.create_dataset('X_val', data=X_val, **comp_kwargs)
        hf.create_dataset('y_trn', data=y_trn, **comp_kwargs)
        hf.create_dataset('y_val', data=y_val, **comp_kwargs)
        hf.create_dataset('y_val_physical', data=y_val_physical, **comp_kwargs)
        hf.create_dataset('wavelengths', data=wavelengths)

        # 2. Save lists of columns (encoded as native HDF5 strings)
        dt_str = h5py.string_dtype(encoding='utf-8')
        hf.create_dataset('target_cols', data=np.array(target_cols, dtype=object), dtype=dt_str)
        hf.create_dataset('surviving_cols', data=np.array(surviving_cols, dtype=object), dtype=dt_str)

        # 3. Save the Validation Metadata DataFrame natively column-by-column (Bypasses PyTables!)
        meta_grp = hf.create_group('metadata_val')
        for col in meta_val_df.columns:
            series = meta_val_df[col]
            if pd.api.types.is_numeric_dtype(series):
                arr = series.to_numpy(dtype=np.float32, na_value=np.nan)
                meta_grp.create_dataset(col, data=arr, **comp_kwargs)
            else:
                arr = series.fillna('').astype(str).to_numpy()
                meta_grp.create_dataset(col, data=arr, dtype=dt_str)

        # 4. Attach helpful global attributes
        hf.attrs['X_trn_shape'] = list(X_trn.shape)
        hf.attrs['X_val_shape'] = list(X_val.shape)
        hf.attrs['metadata_cols'] = list(meta_val_df.columns)

    # 5. Save the scalers alongside it
    with open(output_dir / "X_scaler.pkl", "wb") as f:
        pickle.dump(X_scaler, f)
    with open(output_dir / "y_scaler.pkl", "wb") as f:
        pickle.dump(y_scaler, f)

    size_mb = prepped_h5_path.stat().st_size / 1_048_576
    print(f"✅ Successfully saved training dataset: {prepped_h5_path.name} | {size_mb:.1f} MB")
    # endregion

def load_scalers(
        y_scaler_path: str | Path,
        X_scaler_path: str | Path,
):
    if y_scaler_path is None:
        y_scaler_path = r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\LIBS\y_scaler.pkl"
    if X_scaler_path is None:
        X_scaler_path = r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\LIBS\X_scaler.pkl"

    with open(y_scaler_path, "rb") as f:
        y_scaler = pickle.load(f)
    with open(X_scaler_path, "rb") as f:
        X_scaler = pickle.load(f)

    return(y_scaler,X_scaler)

def load_h5_split_dataset(
        h5_path: str | Path,
        allowed_cols: set[str],
        log_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, pd.DataFrame]:
    """
    Returns:
        X_trn_raw    : np.ndarray, shape (N_trn, n_features), float32
        X_val_raw    : np.ndarray, shape (N_val, n_features), float32
        y_trn_raw    : np.ndarray, shape (N_trn, W), float32
        y_val_raw    : np.ndarray, shape (N_val, W), float32
        feature_cols : list[str]
        wavelengths  : np.ndarray
        meta_val_df  : pd.DataFrame
    """
    with h5py.File(h5_path, 'r') as hf:
        y_trn_raw   = hf_get(hf, 'train/spectra')
        y_val_raw   = hf_get(hf, 'val/spectra')
        wavelengths = hf_get(hf, 'wavelengths')

        trn_meta_grp = hf['train/metadata']
        assert isinstance(trn_meta_grp, h5py.Group), "Expected a Group at 'train/metadata'"
        all_cols     = list(trn_meta_grp.keys())
        feature_cols = [c for c in all_cols if c in allowed_cols]

        X_trn_raw = np.stack(
            [hf_get(hf, f'train/metadata/{c}') for c in feature_cols], axis=1
        ).astype(np.float32)
        X_val_raw = np.stack(
            [hf_get(hf, f'val/metadata/{c}') for c in feature_cols], axis=1
        ).astype(np.float32)

        meta_val_df = pd.DataFrame(
            {c: hf_get(hf, f'val/metadata/{c}') for c in feature_cols}
        )

    return X_trn_raw, X_val_raw, y_trn_raw, y_val_raw, feature_cols, wavelengths, meta_val_df

def sanitize_path(
        path: Path | None = None,
        log_path: Path | None = None,
) -> Optional[Path]:
    """
    Renames any path components containing illegal Windows characters.
    Returns the new sanitized path.
    """
    if path is None:
        raise FileExistsError('  ❌ No path provided.')
    
    path = Path(path)
    root = path.anchor  # e.g. "G:\\"
    parts = path.relative_to(root).parts

    # region Logger Setup
    if log_path is None:
        log_path = Path(r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\default_log.txt").resolve()
    logger = get_worker_logger(Path(log_path).stem)
    # endregion

    sanitized_parts = []
    for part in parts:
        safe = re.sub(r'[<>:"/\\|?*%]', '_', part)
        sanitized_parts.append(safe)

    # Walk down and rename any folders that changed
    current = Path(root)
    for original, sanitized in zip(parts, sanitized_parts):
        original_path = current / original
        sanitized_path = current / sanitized
        if original != sanitized and original_path.exists():
            try:
                original_path.rename(sanitized_path)
                log(logger=logger, msg = f"  ✅ Renamed: {original} → {sanitized}")
            except Exception as e:
                log(logger=logger, msg = f"  ❌ Could not rename {original}: {e}")
                return None
        current = sanitized_path

    return current

def long_path(
        path: Path,
        log_path: Path | None = None,
) -> str:
    r"""
    Returns \\?\ prefixed string for windows long path support.
    Use this only at the point of file I/O, not for path manipulation.
    """
    if log_path is None:
        log_path = Path(r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\default_log.txt").resolve()

    logger = get_worker_logger(Path(log_path).stem)

    if path is None:
        log(logger=logger, msg="No path provided")

    else:
        s = str(Path(path).resolve())
        if len(s) > 240 and not s.startswith('\\\\?\\'):
            return '\\\\?\\' + s
        return s

def worker_init(log_q):
    """Called once per worker process at startup."""
    global _worker_log_q
    _worker_log_q = log_q
    # Set up a single logger for this worker process
    logger = logging.getLogger('worker')
    if not logger.handlers:
        logger.addHandler(handlers.QueueHandler(log_q))
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

def get_worker_logger(name):
    return logging.getLogger(f'worker.{name}')

def log(logger,msg):
    if logger.handlers:
        logger.info(msg)
    else:
        print(msg)
        
def hf_get(
        hf: h5py.File, 
        key: 'str'
) -> np.ndarray:
    return hf[key][:]   # type: ignore[index]

if __name__=="__main__":
    print("hi")
    allowed_cols= {
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
    training_ready_h5(
        h5_path= Path(r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\LIBS\trn_val_split_LIBS.h5"),
        prepped_h5_path= Path(r"C:\Users\leejv2\Documents\git_repos\jvlee_LIBS_ML\LIBS\training_ready_LIBS.h5"),
        allowed_cols=allowed_cols
        )

    #    recursive_mpr_to_csv(
    #        data_root= r"W:\Phongikaroon Group\AndrewsH",
    #        csv_root= r"G:\My Drive\Radiochemistry_&_Laser_Spectroscopy_Lab\Data\ML_practice_data_from_W_drive",
    #    )
    # combined_path = technique_csv_enrichment_and_combination(
    #     data_root= r"W:\Phongikaroon Group\AndrewsH",
    #     csv_root= r"G:\My Drive\Radiochemistry_&_Laser_Spectroscopy_Lab\Data\ML_practice_data_from_W_drive",
    #     technique= 'cv'
    # )

    # split_training_and_validation_dataframes(
    #     combined_path=combined_path
    # )

    #    trn_val_splitter(
    #        df_path=r"G:\My Drive\Radiochemistry_&_Laser_Spectroscopy_Lab\Data\LIBS\CSVs\CV_data\cleaned_combined_CV.csv",
    #        trn_path=r"G:\My Drive\Radiochemistry_&_Laser_Spectroscopy_Lab\Data\LIBS\CSVs\CV_data\CV_training_data.csv",
    #        val_path=r"G:\My Drive\Radiochemistry_&_Laser_Spectroscopy_Lab\Data\LIBS\CSVs\CV_data\CV_validation_data.csv",
    #        test_size=0.2,
    #        random_state=42,
    #        shuffle=True)
        
    #    paths = [r"W:\Phongikaroon Group\AndrewsH\Backup\Experiments & Calculations\LaCl-GdCl Study\Batch 2 - 0.5wt% Gd\Batch A\0.0 wt% LaCl3-0.5 wt% GdCl3 (A)\450\CA Files\(0.0%La-0.5%Gd) 450C 3mVs_02_CA_C01.mpr"]
    #    df_metadata = parent_concentration_data(paths)
    #    print(df_metadata)

    #    paths = [
    #    r"G:\My Drive\Radiochemistry_&_Laser_Spectroscopy_Lab\Data\ML_practice_data_from_W_drive\enriched_ca_data\d1_4wt%_SmCl3_150mVs(2)_05_CV_02_CA_C01.csv",
    #    r"G:\My Drive\Radiochemistry_&_Laser_Spectroscopy_Lab\Data\ML_practice_data_from_W_drive\enriched_ca_data\2wt_SmCl3_4wt_GdCl3_150mVs_d5(1)_01_CA_C01.csv",
    #    r"G:\My Drive\Radiochemistry_&_Laser_Spectroscopy_Lab\Data\ML_practice_data_from_W_drive\enriched_ca_data\2wt_UCl3_0.5wt_MgCl2_200mVs_d3(1)_01_CA_C01.csv",
    #    ]

    #    df = parent_concentration_data(paths)
    #    print(df[['file_path', 'conc_SmCl3_wt%', 'conc_GdCl3_wt%', 'conc_UCl3_wt%', 'conc_MgCl2_wt%', 'conc_LaCl3_wt%']]