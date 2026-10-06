from __future__ import annotations

"""
jvlee_LIBS_ML > utils > NIST_data.py

================================================================================
MODULE OVERVIEW: NIST LIBS SPECTRA GENERATOR & SYNTHETIC DATASET PIPELINE
================================================================================
This script provides an end-to-end data acquisition, modeling, and synthesis 
pipeline for Laser-Induced Breakdown Spectroscopy (LIBS) applied to molten salts 
(e.g., LiCl-KCl eutectic and FLiBe matrices) containing various metal chloride 
dopants (CeCl3, SmCl3, UCl3, etc.).

Key Functional Pipelines:
1. NIST ASD / LIBS Web Scraping:
   - Scrapes atomic line intensities from NIST Atomic Spectra Database (ASD)
     and NIST LIBS online spectrum generator via POST requests.
   - Handles multi-threaded parallel queries with rate-limiting and session retries.

2. Spectral Line Broadening:
   - Converts discrete atomic peak lines into continuous Gaussian profiles based
     on spectrometer resolving power ($R = \\lambda / \\Delta\\lambda$).

3. Salt Stoichiometry Calculations:
   - Converts mass weight percentages (wt%) of dopant salts into atomic concentration
     percentages (at%) required by NIST physics models, preserving charge balance.

4. Synthetic Dataset Generator & HDF5 Assembly:
   - Generates large-scale synthetic mixture spectra (e.g., 200k samples) using
     Dirichlet distribution sampling over sparse element combinations.
   - Combines synthetic and experimental spectral datasets into unified, 
     HDF5 (.h5) structures for machine learning model training.
"""

print('NIST_data.py loading ...')

# ==============================================================================
# region: IMPORTS & DEPENDENCIES
# ==============================================================================
import time
import requests
import re
import random
import itertools
import concurrent.futures
import threading
import zipfile
import os
import h5py
import io
import glob

import pandas as pd
import numpy as np

from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from datetime import datetime
from typing import Any, Optional, Dict
from scipy.signal import find_peaks
# endregion

# ==============================================================================
# region: GLOBAL CONFIGURATIONS & NETWORK CONSTANTS
# ==============================================================================
# Base URL for the NIST Atomic Spectra Database / LIBS endpoint
NIST_LIBS_URL = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl"

# Browser user agent header to prevent HTTP 403 request rejection by NIST
NIST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
}

# Thread synchronization locks
print_lock = threading.Lock()   # Prevents interleaved console log outputs
batch_lock = threading.Lock()   # Thread-safe access to completed CSV list

# Concurrency and Rate Limiting Controls
THREAD_DELAY = 1.2    # Delay (seconds) between NIST requests per thread to respect rate limits
MAX_WORKERS = 4       # Number of concurrent scraping threads

# Batch Packaging Settings
completed_files = []  # Tracks paths of generated CSV files awaiting ZIP archiving
batch_counter = 1
BATCH_SIZE = 1000     # Number of CSV files per compressed output archive

# Spectral Wavelength Grid Configuration (200 nm to 1000 nm at 0.1 nm resolution)
GRID_MIN = 200.0
GRID_MAX = 1000.0
GRID_STEP = 0.1
NUM_POINTS = int(round((GRID_MAX - GRID_MIN) / GRID_STEP)) + 1
WAVELENGTH_GRID = np.linspace(GRID_MIN, GRID_MAX, NUM_POINTS)

# ISO timestamp for batch labeling
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")

# Thread-local storage for thread-safe HTTP sessions
thread_local = threading.local()
# endregion

# ==============================================================================
# region: FILE SYSTEM & DIRECTORY SETUP
# ==============================================================================
# Output directory relative to script path (Lustre high-performance file system)
output_dir = Path(__file__).parent.parent / "LIBS" / "NIST_trio_data"
output_dir.mkdir(parents=True, exist_ok=True)

# Final archive destination for zipped NIST CSV output batch files
FINAL_OUTPUT_DIR = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/NIST_trio_data")
FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Path for single-element baseline spectra stored in HDF5 format
INDIVIDUAL_SPECTRA = Path('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/individual_spectra.h5')

# High-speed local compute node scratch directory for temporary CSV staging
SCRATCH_DIR = Path(os.environ.get("SLURM_TMPDIR", "/tmp")) / "nist_csv_scratch"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
# endregion

# ==============================================================================
# region: PLASMA PHYSICS & MATERIAL PARAMETERS
# ==============================================================================
# Physics configuration for NIST Saha-LTE equilibrium plasma spectral calculator
plasma_params = {
    "low_w": "200",         # Minimum wavelength limit (nm)
    "upp_w": "1000",        # Maximum wavelength limit (nm)
    "limits_type": "0",     # Limits condition (0 = Wavelength range bounds)
    "unit": "1",            # Wavelength unit indicator (1 = nm)
    "resolution": "1000",   # Spectrometer resolving power (R = lambda / delta_lambda)
    "temp": "1.0",          # Electron temperature Te in eV (~11,600 K)
    "eden": "1e17",         # Electron density Ne in cm^-3
    "maxcharge": "2",       # Maximum ion charge state included (e.g., 2 = doubly ionized)
    "min_rel_int": "0.1",   # Minimum relative line intensity cutoff
    "show_av": "2",         # Profile calculation method indicator
    "libs": "1"             # Enables online LIBS simulation engine mode
}

# Material Specifications: Molar mass of compound (mmc in g/mol) and associated halide atoms
DOPANT_SPECS = {
    'CeCl3': {'mmc': 246.475, 'num_bond_atoms': 3},
    'SmCl3': {'mmc': 256.72,  'num_bond_atoms': 3},
    'LaCl3': {'mmc': 245.264, 'num_bond_atoms': 3},
    'NdCl3': {'mmc': 250.601, 'num_bond_atoms': 3},
    'CsCl':  {'mmc': 168.358, 'num_bond_atoms': 1},
    'SrCl2': {'mmc': 158.53,  'num_bond_atoms': 2},
    'BaCl2': {'mmc': 208.233, 'num_bond_atoms': 2},
    'YCl3':  {'mmc': 195.265, 'num_bond_atoms': 3},
    'FeCl2': {'mmc': 126.751, 'num_bond_atoms': 2},
    'CrCl2': {'mmc': 122.902, 'num_bond_atoms': 2},
    'NiCl2': {'mmc': 129.599, 'num_bond_atoms': 2},
    'MnCl2': {'mmc': 125.844, 'num_bond_atoms': 2},
    'UCl3':  {'mmc': 344.388, 'num_bond_atoms': 3},
    'CaCl2': {'mmc': 110.984, 'num_bond_atoms': 2},
    'GdCl3': {'mmc': 263.610, 'num_bond_atoms': 3},
    'MgCl2': {'mmc': 95.211,  'num_bond_atoms': 2}
}

# Base Salt Component Molar Masses (g/mol)
mmc_lif = 25.939
mmc_bef = 47.009

mmc_licl = 42.39
mmc_kcl = 74.55

# Eutectic Salt Composition (LiCl-KCl: 59 mol% LiCl, 41 mol% KCl)
mfr_licl = 0.59
mfr_kcl = 0.41
mmc_eutectice = (mfr_licl * mmc_licl) + (mfr_kcl * mmc_kcl)  # Average eutectic molar mass

# Default grid of weight percentages for dopant combinations
wt_percents_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
# endregion

# ==============================================================================
# region: NETWORK SESSION & RETRY HANDLERS
# ==============================================================================
def get_nist_session() -> requests.Session:
    """
    Constructs a robust `requests.Session` configured with exponential backoff 
    retries for handling connection drops and server HTTP 5xx/429 throttling.

    Returns:
        requests.Session: Configured session with retry adapter.
    """
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })

    # Retry strategy: up to 10 retries with exponential backoff (3s, 6s, 12s...)
    retries = Retry(
        total=10,
        backoff_factor=3,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session

def get_thread_session() -> requests.Session:
    """
    Retrieves or creates a thread-local `requests.Session` instance to enforce 
    thread safety during multi-threaded web scraping.

    Returns:
        requests.Session: Thread-isolated HTTP session.
    """
    if not hasattr(thread_local, 'session'):
        thread_local.session = get_nist_session()
    return thread_local.session
# endregion

# ==============================================================================
# region: ARCHIVAL & FILE SYSTEM UTILITIES
# ==============================================================================
def archive_batch_if_ready(force: bool = False, run_timestamp: Optional[str] = None):
    """
    Thread-safe utility that bundles generated individual spectrum CSV files 
    from the scratch directory into compressed ZIP archives once `BATCH_SIZE` 
    is reached, deleting raw temporary CSV files afterwards.

    Args:
        force (bool): If True, packages remaining files regardless of batch size.
        run_timestamp (str, optional): Custom timestamp string for ZIP file naming.
    """
    global batch_counter, completed_files

    if run_timestamp is None:
        run_timestamp = RUN_TIMESTAMP

    files_to_zip = []
    current_batch_num = 0

    with batch_lock:
        if len(completed_files) >= BATCH_SIZE or (force and len(completed_files) > 0):
            take_count = BATCH_SIZE if not force else len(completed_files)
            files_to_zip = completed_files[:take_count]
            completed_files = completed_files[take_count:]
            current_batch_num = batch_counter
            batch_counter += 1

    if files_to_zip:
        archive_name = (
            FINAL_OUTPUT_DIR
            / f"{run_timestamp}_nist_libs_batch_{current_batch_num:04d}.zip"
        )
        with print_lock:
            print(
                f"📦 Packaging batch {current_batch_num} ({len(files_to_zip)} CSVs) -> {archive_name.name}..."
            )

        with zipfile.ZipFile(archive_name, "a", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in files_to_zip:
                if file_path.exists():
                    zf.write(file_path, arcname=file_path.name)
                    os.remove(file_path)  # Cleanup scratch file
# endregion

# ==============================================================================
# region: NIST ASD & LIBS SCRAPING ENGINE
# ==============================================================================
def fetch_asd_baseline_lines(element: str, plasma_params: dict) -> pd.DataFrame:
    """
    Queries the NIST Atomic Spectra Database (ASD) for raw spectral line data 
    of neutral (I) and singly-ionized (II) species of a given element using 
    direct tab-delimited output format (format=3).

    Args:
        element (str): Chemical symbol of element (e.g., 'Ce', 'Li', 'Cl').
        plasma_params (dict): Dictionary specifying wavelength limits and flags.

    Returns:
        pd.DataFrame: DataFrame containing 'Wavelength (nm)' and 'Intensity'.
    """
    all_lines = []
    
    for stage in ["I", "II"]:  # Query neutral (I) and singly ionized (II) states
        payload = {
            "spectra": f"{element} {stage}",
            "limits_type": "0",
            "low_w": str(plasma_params.get("low_w", GRID_MIN)),
            "upp_w": str(plasma_params.get("upp_w", GRID_MAX)),
            "unit": "1",          # Wavelength unit: 1 = nm
            "format": "3",        # Format 3 = Tab-delimited ASCII
            "ascii_out": "1",     # Enable ASCII mode
            "line_out": "0",
            "remove_j": "on",
            "page_size": "0",     # Unlimited results
            "show_obs_wl": "1",
            "show_calc_wl": "1",
            "intens_out": "on",
            "order_out": "0",
        }
        
        try:
            resp = requests.post(NIST_LIBS_URL, data=payload, headers=NIST_HEADERS, timeout=30)
            
            if resp.status_code != 200 or "Input Error" in resp.text or "No lines" in resp.text:
                continue
            
            # Parse response lines and strip header comments
            raw_lines = resp.text.splitlines()
            data_lines = [line for line in raw_lines if "\t" in line and not line.startswith("Spectrum")]
            
            if len(data_lines) < 2:
                continue
                
            tsv_payload = "\n".join(data_lines).replace('"', '')
            df = pd.read_csv(io.StringIO(tsv_payload), sep="\t", engine="python", on_bad_lines="skip")
            df.columns = [str(c).strip() for c in df.columns]
            
            # Identify columns containing wavelength and relative intensity values
            wl_col = next((c for c in df.columns if "Observed" in c or "Ritz" in c or "Wavelength" in c), None)
            int_col = next((c for c in df.columns if "Rel." in c or "Int" in c), None)
            
            if not wl_col or not int_col:
                continue
                
            sub_df = pd.DataFrame()
            
            # Extract regex matching floating point numbers from messy string columns
            sub_df["Wavelength (nm)"] = (
                df[wl_col].astype(str).str.extract(r'([0-9]+\.?[0-9]*)')[0].astype(float)
            )
            sub_df["Intensity"] = (
                df[int_col].astype(str).str.extract(r'([0-9]+\.?[0-9]*)')[0].astype(float)
            )
            
            # Fill unrated or missing relative intensity entries with a baseline 1.0
            sub_df["Intensity"] = sub_df["Intensity"].fillna(1.0)
            sub_df = sub_df.dropna(subset=["Wavelength (nm)"])
            
            if not sub_df.empty:
                all_lines.append(sub_df)
                
        except Exception as e:
            print(f" -> Failed fetching ASD lines for {element} {stage}: {e}")

    if all_lines:
        return pd.concat(all_lines, ignore_index=True)
    return pd.DataFrame(columns=["Wavelength (nm)", "Intensity"])

def fetch_nist_libs_data(comp_string: str, plasma_params: dict) -> Optional[pd.DataFrame]:
    """
    Queries the NIST LIBS simulation calculation engine for a multi-element 
    composition string and parses the embedded JavaScript `var lines = [...]` array 
    containing individual line intensities.

    Args:
        comp_string (str): Formatted element composition string 
                           (e.g., 'Li:30.0;K:20.0;Cl:45.0;Ce:5.0').
        plasma_params (dict): Physics simulation parameters (temperature, density, etc.).

    Returns:
        Optional[pd.DataFrame]: Structured discrete line list with wavelength, intensity,
                                energy levels, and elemental identifiers, or None on failure.
    """
    pairs = [p.strip() for p in comp_string.split(";") if p.strip()]
    elements = [p.split(":")[0].strip() for p in pairs]
    percentages = [p.split(":")[1].strip() for p in pairs]

    # Construct POST form payload structure required by NIST LIBS CGI handler
    payload = [
        ("form", "libs"),
        ("action", "libs"),
        ("low_w", str(plasma_params.get("low_w", GRID_MIN))),
        ("upp_w", str(plasma_params.get("upp_w", GRID_MAX))),
        ("unit", "1"),                                     # 1 = nm
        ("de_unit", "0"),                                  # eV
        ("line_out", "1"),
        ("remove_j", "on"),
        ("temp", str(plasma_params.get("temp", "1.0"))),   # Te in eV
        ("eden", str(plasma_params.get("eden", "1e17"))),  # Ne in cm^-3
        ("resolution", str(plasma_params.get("resolution", "1000"))),
        ("min_rel_int", str(plasma_params.get("min_rel_int", "0.1"))),
        ("maxcharge", str(plasma_params.get("maxcharge", "2"))),
        ("show_av", str(plasma_params.get("show_av", "2"))),
        ("libs", "1"),
        ("composition", comp_string),
        ("num_sl", str(len(elements))),
        ("spectra", ",".join(elements)),
    ]

    # Append elemental composition array keys (`mytext[]` and `myperc[]`)
    for el, pct in zip(elements, percentages):
        payload.append(("mytext[]", el))
        payload.append(("myperc[]", str(float(pct))))

    time.sleep(random.uniform(1.5, 3.0))  # Jittered delay to mitigate rate limiting
    session = get_thread_session() if 'thread_local' in globals() else get_nist_session()

    try:
        response = session.post(NIST_LIBS_URL, data=payload, timeout=120)
        
        if response.status_code != 200:
            print(f' -> Server error: Status code {response.status_code}')
            return None
            
        html_content = response.text
        if "Error:" in html_content or "Incorrect element" in html_content:
            print(f" -> NIST returned a form validation error for composition: {comp_string}")
            return pd.DataFrame()

        # Parse JS line data matrix: var lines = [[wl, int, E1, code1, code2, E2], ...];
        match = re.search(r"var\s+lines\s*=\s*\[(.*?)\];", html_content, re.DOTALL)
        if not match:
            print(f' -> Warning: Response received for {comp_string}, but "var lines" data block wasn\'t found.')
            return pd.DataFrame()
            
        raw_data_block = match.group(1).strip()
        if not raw_data_block:
            return pd.DataFrame()

        row_strings = re.findall(r"\[([^\]]+)\]", raw_data_block)
        headers = [
            "Wavelength (nm)", 
            "Intensity", 
            "Energy Level 1", 
            "Element Code 1", 
            "Element Code 2", 
            "Energy Level 2"
        ]
        
        parsed_rows = [[float(v.strip()) for v in r.split(",")] for r in row_strings]
        return pd.DataFrame(parsed_rows, columns=headers)
        
    except Exception as e:
        print(f' -> Network failure (timeout/disconnect): {str(e)}')
        return None
# endregion

# ==============================================================================
# region: GAUSSIAN PROFILE BROADENING MODEL
# ==============================================================================
def generate_simple_intensity_profile(
    df_discrete: pd.DataFrame, 
    low_w: float = 200.0, 
    upp_w: float = 1000.0, 
    step: float = 0.1, 
    resolution: float = 1000.0
) -> pd.DataFrame:
    """
    Converts discrete atomic spectral transition lines into a continuous 
    spectrometer intensity profile across a uniform grid using Gaussian broadening.

    Physics Formulation:
        FWHM = \\lambda_0 / R
        \\sigma = \frac{\text{FWHM}}{2\\sqrt{2\\ln(2)}}
        I(\\lambda) = \\sum_i I_i \\cdot \frac{1}{\\sigma_i \\sqrt{2\\pi}} \\exp\\left( -\frac{(\\lambda - \\lambda_{0,i})^2}{2\\sigma_i^2} \right)

    Args:
        df_discrete (pd.DataFrame): Peak table containing 'Wavelength (nm)' and 'Intensity'.
        low_w (float): Minimum grid wavelength limit (nm).
        upp_w (float): Maximum grid wavelength limit (nm).
        step (float): Wavelength spacing delta (nm).
        resolution (float): Spectrometer resolving power R = \\lambda / \\Delta\\lambda.

    Returns:
        pd.DataFrame: Continuous spectrum array on uniform wavelength axis.
    """
    if df_discrete is None or df_discrete.empty:
        return pd.DataFrame(columns=["Wavelength (nm)", "Intensity"])

    # Build uniform linear wavelength grid
    num_points = int(round((upp_w - low_w) / step)) + 1
    wavelength_grid = np.linspace(low_w, upp_w, num_points)
    intensity_array = np.zeros_like(wavelength_grid)
    
    # Calculate Gaussian peak envelope per transition line
    for _, row in df_discrete.iterrows():
        lambda_0 = row["Wavelength (nm)"]
        peak_intensity = row["Intensity"]
        
        # Calculate standard deviation sigma from resolving power R
        fwhm = lambda_0 / resolution
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        
        # Local window optimization (+/- 5 standard deviations) to save computation time
        window = 5 * sigma
        mask = (wavelength_grid >= lambda_0 - window) & (wavelength_grid <= lambda_0 + window)
        relevant_grid = wavelength_grid[mask]
        
        if len(relevant_grid) == 0:
            continue
            
        # Evaluate Gaussian distribution kernel
        gaussian_shape = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((relevant_grid - lambda_0) ** 2) / (2 * sigma ** 2))
        intensity_array[mask] += peak_intensity * gaussian_shape

    return pd.DataFrame({
        "Wavelength (nm)": wavelength_grid,
        "Intensity": intensity_array
    })
# endregion

# ==============================================================================
# region: STOICHIOMETRIC GRID SIMULATION ENGINES
# ==============================================================================
def gather_2_data(
    dopant_1: str, mmc_dopant_1: float, num_bond_atoms_1: int,
    dopant_2: str, mmc_dopant_2: float, num_bond_atoms_2: int,
    mfr_salt_a: float, mmc_salt_a: float,
    mfr_salt_b: float, mmc_salt_b: float,
    salt: str, wt_percents: list,
):
    """
    Executes a 2-dopant concentration grid search, converting dopant weight 
    fractions (wt%) into overall atomic fractions (at%) and querying NIST LIBS.
    """
    grid = [(w1, w2) for w1 in wt_percents for w2 in wt_percents]
    experiment_df = pd.DataFrame(grid, columns=[f"{dopant_1}_wt%", f"{dopant_2}_wt%"])
    
    print(f"Total simulated runs to process: {len(experiment_df)}")
    
    for idx, (_, row) in enumerate(experiment_df.iterrows(), start=1):
        wt_dopant_1 = row[f"{dopant_1}_wt%"]
        wt_dopant_2 = row[f"{dopant_2}_wt%"]

        filename = f"nist_libs_{wt_dopant_1}_wt_{dopant_1}_and_{wt_dopant_2}_wt_{dopant_2}.csv"
        file_path = output_dir / filename

        if file_path.exists():
            print(f"Skipping {filename}, already exists...")
            continue
        
        wt_dopants = wt_dopant_1 + wt_dopant_2
        wt_salt = 100.0 - wt_dopants
        
        if wt_salt < 0:
            print(f"Skipping Run {idx}: Combined dopant weights exceed 100%!")
            continue

        if salt == 'ClLiK':
            mmc_salt = mmc_eutectice
            salt_a, salt_b, bond_element = 'Li', 'K', 'Cl'
        elif salt == 'FLiBe':
            mmc_salt = 1
            salt_a, salt_b, bond_element = 'Li', 'Be', 'F'
        else:
            print(f'Skipping Run {idx}: No host salt!')
            continue
            
        # Calculate component mass fractions and moles
        wt_salt_a = wt_salt * ((mfr_salt_a * mmc_salt_a) / mmc_salt)
        wt_salt_b = wt_salt * ((mfr_salt_b * mmc_salt_b) / mmc_salt)
        
        moles_salt_a = wt_salt_a / mmc_salt_a
        moles_salt_b = wt_salt_b / mmc_salt_b
        moles_dopant_1 = wt_dopant_1 / mmc_dopant_1
        moles_dopant_2 = wt_dopant_2 / mmc_dopant_2
        
        # Calculate atomic proportions
        salt_a_atoms = moles_salt_a
        salt_b_atoms = moles_salt_b
        dopant_1_atoms = moles_dopant_1
        dopant_2_atoms = moles_dopant_2
        bond_atoms = moles_salt_a + moles_salt_b + (moles_dopant_1 * num_bond_atoms_1) + (moles_dopant_2 * num_bond_atoms_2)
        
        total_atoms = salt_a_atoms + salt_b_atoms + dopant_1_atoms + dopant_2_atoms + bond_atoms
        
        # Convert to atomic percentages required for NIST composition input
        salt_a_val = (salt_a_atoms / total_atoms) * 100
        salt_b_val = (salt_b_atoms / total_atoms) * 100
        bond_val = (bond_atoms / total_atoms) * 100
        dopant_1_val = (dopant_1_atoms / total_atoms) * 100
        dopant_2_val = (dopant_2_atoms / total_atoms) * 100

        match1 = re.match(r"([A-Z][a-z]?)", dopant_1)
        match2 = re.match(r"([A-Z][a-z]?)", dopant_2)

        element_1 = match1.group(1) if match1 else dopant_1
        element_2 = match2.group(1) if match2 else dopant_2
        
        comp_string = f"{salt_a}:{salt_a_val:.5f};{salt_b}:{salt_b_val:.5f};{bond_element}:{bond_val:.5f};{element_1}:{dopant_1_val:.5f};{element_2}:{dopant_2_val:.5f}"
        
        print(f"Processing Run {idx}/{len(experiment_df)}: {dopant_1}={wt_dopant_1}wt%, {dopant_2}={wt_dopant_2}wt%")
        
        df_discrete = fetch_nist_libs_data(comp_string, plasma_params)
        
        if df_discrete is None or df_discrete.empty:
            print(f' -> Warning: Fetch issue or no lines found for Run {idx}.')
            time.sleep(2.0)
            continue

        df_continuous = generate_simple_intensity_profile(
            df_discrete,
            low_w=float(plasma_params["low_w"]),
            upp_w=float(plasma_params["upp_w"]),
            step=0.1,
            resolution=float(plasma_params["resolution"])
        )
        
        if not df_continuous.empty:
            tmp_path = file_path.with_suffix(".csv.tmp")
            df_continuous.to_csv(tmp_path, index=False)
            tmp_path.replace(file_path)  # Atomic POSIX file replace
            print(f' -> Successfully parsed, broadened, and saved data to: {filename}')

        time.sleep(2.0)

    print('\n--- Matrix data collection complete ---')

def gather_3_data(
    dopant_1: str, mmc_dopant_1: float, num_bond_atoms_1: int,
    dopant_2: str, mmc_dopant_2: float, num_bond_atoms_2: int,
    dopant_3: str, mmc_dopant_3: float, num_bond_atoms_3: int,
    mfr_salt_a: float, mmc_salt_a: float,
    mfr_salt_b: float, mmc_salt_b: float,
    salt: str, wt_percents: list,
):
    """
    Executes a 3D grid scan over 3 dopant salts in host matrix, converting wt% 
    stoichiometry to atomic percent and outputting broadened continuous spectra.
    """
    grid = [(w1, w2, w3) for w1 in wt_percents for w2 in wt_percents for w3 in wt_percents]
    experiment_df = pd.DataFrame(grid, columns=[f"{dopant_1}_wt%", f"{dopant_2}_wt%", f"{dopant_3}_wt%"])

    print(f"Total simulated runs to process: {len(experiment_df)}")

    for idx, (_, row) in enumerate(experiment_df.iterrows(), start=1):
        wt_dopant_1 = row[f"{dopant_1}_wt%"]
        wt_dopant_2 = row[f"{dopant_2}_wt%"]
        wt_dopant_3 = row[f"{dopant_3}_wt%"]

        filename = (
            f"nist_libs_{wt_dopant_1}_wt_{dopant_1}_and_"
            f"{wt_dopant_2}_wt_{dopant_2}_and_"
            f"{wt_dopant_3}_wt_{dopant_3}.csv"
        )
        file_path = SCRATCH_DIR / filename

        if file_path.exists():
            continue

        wt_dopants = wt_dopant_1 + wt_dopant_2 + wt_dopant_3
        wt_salt = 100.0 - wt_dopants

        if wt_salt < 0:
            continue

        if salt == "ClLiK":
            mmc_salt = mmc_eutectice
            salt_a, salt_b, bond_element = "Li", "K", "Cl"
        elif salt == "FLiBe":
            mmc_salt = 1
            salt_a, salt_b, bond_element = "Li", "Be", "F"
        else:
            continue

        wt_salt_a = wt_salt * ((mfr_salt_a * mmc_salt_a) / mmc_salt)
        wt_salt_b = wt_salt * ((mfr_salt_b * mmc_salt_b) / mmc_salt)

        moles_salt_a = wt_salt_a / mmc_salt_a
        moles_salt_b = wt_salt_b / mmc_salt_b
        moles_dopant_1 = wt_dopant_1 / mmc_dopant_1
        moles_dopant_2 = wt_dopant_2 / mmc_dopant_2
        moles_dopant_3 = wt_dopant_3 / mmc_dopant_3

        salt_a_atoms = moles_salt_a
        salt_b_atoms = moles_salt_b
        dopant_1_atoms = moles_dopant_1
        dopant_2_atoms = moles_dopant_2
        dopant_3_atoms = moles_dopant_3

        bond_atoms = (
            moles_salt_a + moles_salt_b
            + (moles_dopant_1 * num_bond_atoms_1)
            + (moles_dopant_2 * num_bond_atoms_2)
            + (moles_dopant_3 * num_bond_atoms_3)
        )

        total_atoms = (
            salt_a_atoms + salt_b_atoms + dopant_1_atoms + dopant_2_atoms + dopant_3_atoms + bond_atoms
        )

        salt_a_val = (salt_a_atoms / total_atoms) * 100
        salt_b_val = (salt_b_atoms / total_atoms) * 100
        bond_val = (bond_atoms / total_atoms) * 100
        dopant_1_val = (dopant_1_atoms / total_atoms) * 100
        dopant_2_val = (dopant_2_atoms / total_atoms) * 100
        dopant_3_val = (dopant_3_atoms / total_atoms) * 100

        match1 = re.match(r"([A-Z][a-z]?)", dopant_1)
        match2 = re.match(r"([A-Z][a-z]?)", dopant_2)
        match3 = re.match(r"([A-Z][a-z]?)", dopant_3)

        element_1 = match1.group(1) if match1 else dopant_1
        element_2 = match2.group(1) if match2 else dopant_2
        element_3 = match3.group(1) if match3 else dopant_3

        comp_string = (
            f"{salt_a}:{salt_a_val:.5f};{salt_b}:{salt_b_val:.5f};{bond_element}:{bond_val:.5f};"
            f"{element_1}:{dopant_1_val:.5f};{element_2}:{dopant_2_val:.5f};{element_3}:{dopant_3_val:.5f}"
        )

        df_discrete = fetch_nist_libs_data(comp_string, plasma_params)

        if df_discrete is None or df_discrete.empty:
            time.sleep(2.0)
            continue

        df_continuous = generate_simple_intensity_profile(
            df_discrete,
            low_w=float(plasma_params["low_w"]),
            upp_w=float(plasma_params["upp_w"]),
            step=0.1,
            resolution=float(plasma_params["resolution"]),
        )

        if not df_continuous.empty:
            tmp_path = file_path.with_suffix(".csv.tmp")
            df_continuous.to_csv(tmp_path, index=False)
            tmp_path.replace(file_path)

        time.sleep(2.0)
# endregion

# ==============================================================================
# region: PEAK DETECTION & MASTER LINE TABLE COMPILER
# ==============================================================================
def master_line_table():
    """
    Parses single-element NIST CSV spectrum files, computes mean baseline intensity, 
    detects prominent spectral peaks via `scipy.signal.find_peaks`, and exports a 
    sorted master line reference table (`master_line_table.csv`).
    """
    ELEMENTS = [
        'Ba', 'Ca', 'Ce', 'Cr', 'Cs', 'Fe', 'Gd', 'K', 'La', 
        'Li', 'Mg', 'Mn', 'Nd', 'Ni', 'Sm', 'Sr', 'U', 'Y', 'Cl'
    ]

    NIST_DIR = '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/NIST_singles/Enriched'
    OUTPUT_CSV = 'master_line_table.csv'

    # Exclude non-spectral metadata column headers
    METADATA_COLS = {
        'conc_Ba_wt%', 'frac_LiCl', 'temperature_C', 'scan_rate_mVs',
        'delay', 'energy', 'static_', 'blank', 'kinetic', 'repetition'
    }

    lines_data = []

    for elem_idx, elem in enumerate(ELEMENTS):
        pattern = os.path.join(NIST_DIR, f'*{elem}*.csv')
        files = glob.glob(pattern)

        if not files:
            print(f'Warning: No NIST CSV file found for the element {elem}')
            continue

        df = pd.read_csv(files[0])

        # Filter numeric wavelength headers
        wavelength_cols = []
        for col in df.columns:
            if col in METADATA_COLS:
                continue
            try:
                wl_val = float(col)
                wavelength_cols.append((col, wl_val))
            except ValueError:
                continue

        if not wavelength_cols:
            continue

        # Sort columns chronologically by wavelength
        wavelength_cols.sort(key=lambda x: x[1])
        col_names = [c[0] for c in wavelength_cols]
        wls = np.array([c[1] for c in wavelength_cols])

        # Compute mean spectrum across all spectrum rows
        intens = df[col_names].mean(axis=0).values
        intens = np.nan_to_num(intens, nan=0.0)        # type: ignore
        intens = np.clip(intens, 0, None)

        # Detect discrete peak centroids using SciPy
        peaks, _ = find_peaks(
            intens, height=0.01, prominence=0.005, distance=3
        )

        for p in peaks:
            lines_data.append({
                'element': elem,
                'elem_idx': elem_idx,
                'wavelength_nm': wls[p],
                'base_intensity': intens[p],
            })

    master_df = pd.DataFrame(lines_data)
    # Sort by wavelength to allow fast binary searches and PyTorch embedding
    master_df = master_df.sort_values(by='wavelength_nm').reset_index(drop=True)
    master_df.to_csv(OUTPUT_CSV, index=False)

    print(
        f'Master Line Table created successfully with {len(master_df)} total'
        f' lines across {len(ELEMENTS)} elements.'
    )
# endregion

# ==============================================================================
# region: MULTI-THREADED SIMULATION RUNNER
# ==============================================================================
def process_single_simulation(task_args: tuple) -> str:
    """
    Worker function executed inside `ThreadPoolExecutor`. Processes one specific 
    3-dopant concentration sample run, converts concentration to atomic fraction, 
    scrapes NIST LIBS, broadens line profiles, and stages the output CSV to disk.

    Args:
        task_args (tuple): Unpacked task parameters (indices, chemical symbols, wt%).

    Returns:
        str: Status string indicating execution outcome.
    """
    (
        idx, total_runs, d1, d2, d3,
        wt_dopant_1, wt_dopant_2, wt_dopant_3,
        DOPANT_SPECS, mfr_salt_a, mmc_salt_a, mfr_salt_b, mmc_salt_b,
        salt, plasma_params, SCRATCH_DIR,
    ) = task_args

    filename = f"nist_libs_{wt_dopant_1}_wt_{d1}_and_{wt_dopant_2}_wt_{d2}_and_{wt_dopant_3}_wt_{d3}.csv"
    file_path = SCRATCH_DIR / filename

    if file_path.exists():
        return "skipped_exists"

    wt_dopants = wt_dopant_1 + wt_dopant_2 + wt_dopant_3
    wt_salt = 100.0 - wt_dopants

    if wt_salt < 0:
        return "skipped_over_100"

    if salt == "ClLiK":
        mmc_salt = mmc_eutectice
        salt_a, salt_b, bond_element = "Li", "K", "Cl"
    elif salt == "FLiBe":
        mmc_salt = 1
        salt_a, salt_b, bond_element = "Li", "Be", "F"
    else:
        return "error_salt"

    # Fetch chemical parameters from DOPANT_SPECS dictionary
    mmc_d1, num_bond_1 = DOPANT_SPECS[d1]["mmc"], DOPANT_SPECS[d1]["num_bond_atoms"]
    mmc_d2, num_bond_2 = DOPANT_SPECS[d2]["mmc"], DOPANT_SPECS[d2]["num_bond_atoms"]
    mmc_d3, num_bond_3 = DOPANT_SPECS[d3]["mmc"], DOPANT_SPECS[d3]["num_bond_atoms"]

    wt_salt_a = wt_salt * ((mfr_salt_a * mmc_salt_a) / mmc_salt)
    wt_salt_b = wt_salt * ((mfr_salt_b * mmc_salt_b) / mmc_salt)

    moles_salt_a = wt_salt_a / mmc_salt_a
    moles_salt_b = wt_salt_b / mmc_salt_b
    moles_d1 = wt_dopant_1 / mmc_d1
    moles_d2 = wt_dopant_2 / mmc_d2
    moles_d3 = wt_dopant_3 / mmc_d3

    salt_a_atoms = moles_salt_a
    salt_b_atoms = moles_salt_b
    d1_atoms, d2_atoms, d3_atoms = moles_d1, moles_d2, moles_d3
    bond_atoms = moles_salt_a + moles_salt_b + (moles_d1 * num_bond_1) + (moles_d2 * num_bond_2) + (moles_d3 * num_bond_3)

    total_atoms = salt_a_atoms + salt_b_atoms + d1_atoms + d2_atoms + d3_atoms + bond_atoms

    salt_a_val = (salt_a_atoms / total_atoms) * 100
    salt_b_val = (salt_b_atoms / total_atoms) * 100
    bond_val = (bond_atoms / total_atoms) * 100
    d1_val = (d1_atoms / total_atoms) * 100
    d2_val = (d2_atoms / total_atoms) * 100
    d3_val = (d3_atoms / total_atoms) * 100

    match1 = re.match(r"([A-Z][a-z]?)", d1)
    match2 = re.match(r"([A-Z][a-z]?)", d2)
    match3 = re.match(r"([A-Z][a-z]?)", d3)

    e1 = match1.group(1) if match1 else d1
    e2 = match2.group(1) if match2 else d2
    e3 = match3.group(1) if match3 else d3

    comp_string = (
        f"{salt_a}:{salt_a_val:.5f};{salt_b}:{salt_b_val:.5f};{bond_element}:{bond_val:.5f};"
        f"{e1}:{d1_val:.5f};{e2}:{d2_val:.5f};{e3}:{d3_val:.5f}"
    )

    df_discrete = fetch_nist_libs_data(comp_string, plasma_params)
    time.sleep(THREAD_DELAY)

    if df_discrete is None:
        with print_lock:
            print(f"[Run {idx}/{total_runs}] ❌ Network drop/timeout: {filename}")
        return "failed_fetch"

    df_continuous = generate_simple_intensity_profile(
        df_discrete,
        low_w=float(plasma_params["low_w"]),
        upp_w=float(plasma_params["upp_w"]),
        step=0.1,
        resolution=float(plasma_params["resolution"]),
    )

    if not df_continuous.empty:
        tmp_path = file_path.with_suffix(".csv.tmp")
        df_continuous.to_csv(tmp_path, index=False)
        tmp_path.replace(file_path)
        with print_lock:
            completed_files.append(file_path)
            print(f"[Run {idx}/{total_runs}] ✅ Saved: {filename}")

        archive_batch_if_ready(force=False, run_timestamp=None)
        return "success"

    return "failed_broadening"

def master_parallel_run():
    """
    Main orchestration loop for parallel scraping.
    Scans persistent archives for existing ZIP contents to eliminate redundant queries,
    pre-calculates missing 3-dopant permutations, and distributes workload across
    a multi-threaded execution pool (`ThreadPoolExecutor`).
    """
    print("🔍 Inspecting existing ZIP archives in persistent storage...")
    existing_archived_files = set()
    for zip_path in FINAL_OUTPUT_DIR.glob("*.zip"):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                existing_archived_files.update(zf.namelist())
        except Exception as e:
            print(f"⚠️ Warning reading {zip_path.name}: {e}")

    print(f"ℹ️ Found {len(existing_archived_files)} CSVs already packaged in ZIP files.")

    # All unique 3-dopant combinations (286 triplets from 16 dopants)
    dopant_triplets = list(itertools.combinations(DOPANT_SPECS.keys(), 3))

    all_tasks = []
    run_counter = 1

    print("🔍 Pre-scanning grid to find uncompleted runs...")

    for d1, d2, d3 in dopant_triplets:
        grid = [(w1, w2, w3) for w1 in wt_percents_list for w2 in wt_percents_list for w3 in wt_percents_list]

        for w1, w2, w3 in grid:
            fn = f"nist_libs_{w1}_wt_{d1}_and_{w2}_wt_{d2}_and_{w3}_wt_{d3}.csv"
            if fn in existing_archived_files or (SCRATCH_DIR / fn).exists():
                continue

            if (w1 + w2 + w3) > 100.0:
                continue

            task = (
                run_counter, 0, d1, d2, d3, w1, w2, w3,
                DOPANT_SPECS, mfr_licl, mmc_licl, mfr_kcl, mmc_kcl,
                "ClLiK", plasma_params, SCRATCH_DIR,
            )
            all_tasks.append(task)
            run_counter += 1

    total_tasks = len(all_tasks)
    print(f"🚀 Found {total_tasks} remaining simulation runs to execute.")
    print(f"⚡ Processing with {MAX_WORKERS} concurrent threads ({THREAD_DELAY}s worker delay)...")

    final_tasks = [(*t[:1], total_tasks, *t[2:]) for t in all_tasks]

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_simulation, task) for task in final_tasks]

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                with print_lock:
                    print(f"❌ Exception occurred in worker thread: {e}")

    archive_batch_if_ready(force=False, run_timestamp=None)
# endregion

# ==============================================================================
# region: SYNTHETIC DATASET GENERATOR & HDF5 EXPORTER
# ==============================================================================
def load_single_element_library(input_dir: Path):
    """
    Reads single-element CSV files, dynamically identifies active chemical element 
    columns, normalizes spectral intensity to 100% basis concentration, and returns 
    a dictionary of pure elemental basis spectra array curves.

    Args:
        input_dir (Path): Directory path containing single-element NIST CSVs.

    Returns:
        tuple[dict, np.ndarray]: (element_spectra dict mapping symbol to mean spectrum, 
                                  wavelength array)
    """
    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    element_spectra = {}
    wavelength_cols = None

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        
        if wavelength_cols is None:
            wavelength_cols = []
            for col in df.columns:
                try:
                    float(col)
                    wavelength_cols.append(col)
                except ValueError:
                    continue

        conc_cols = [c for c in df.columns if c.startswith("conc_") and c.endswith("_wt%")]
        active_element = None
        
        for c_col in conc_cols:
            if (df[c_col] > 0).any():
                formula = c_col.replace("conc_", "").replace("_wt%", "").replace("%", "").strip("_")
                element_count = sum(1 for char in formula if char.isupper())
                if element_count > 1:
                    continue  # Skip polyatomic formulas
                active_element = formula

        if active_element is None:
            continue

        spectral_data = df[wavelength_cols].values.astype(np.float32)
        mean_spectrum = np.mean(spectral_data, axis=0)

        conc_val = df[f"conc_{active_element}_wt%"].iloc[0]
        if conc_val > 0:
            mean_spectrum = mean_spectrum / (conc_val / 100.0)  # Normalize to 100% pure basis

        element_spectra[active_element] = mean_spectrum

    return element_spectra, np.array([float(w) for w in wavelength_cols], dtype=np.float32)        # type: ignore

def synthetic_data_generator(
    input_dir: str | Path,
    output_dir: str | Path,
    num_samples: int = 200000,
    dopant_names: list = ['Ba', 'Ca', 'Ce', 'Cr', 'Cs', 'Fe', 'Gd', 'La', 'Mg', 'Mn', 'Nd', 'Ni', 'Sm', 'Sr', 'U', 'Y'],
    host_sigma: float = 0.015,
    dopant_max_total_wt: float = 0.05,
    alpha_dopants: Optional[np.ndarray] = None,
    singles: float = 0.4,
    doubles: float = 0.2,
    triples: float = 0.2,
    pure_host: float = 0.2,
):
    """
    Generates large-scale synthetic mixture LIBS spectra (e.g., N=200,000) using 
    Dirichlet distribution composition sampling across single, binary, and ternary 
    dopant mixtures in a perturbed LiCl-KCl host matrix. Outputs directly to HDF5 (.h5).

    Mathematical Pipeline:
        1. Classify sample category (pure host, single dopant, binary, ternary).
        2. Sample total metal dopant mass fraction log-uniformly in [1e-5, dopant_max_total_wt].
        3. Sample active dopant ratios via Dirichlet sub-distribution Dirichlet(\alpha).
        4. Calculate stoichiometrically required Cl- mass from metal cations:
           m_{\text{Cl, extra}} = \\sum_i m_{\text{metal}, i} \\cdot \\left( \frac{v_i \\cdot M_{\text{Cl}}}{M_{\text{metal}, i}} \right)
        5. Apply Gaussian noise perturbation to LiCl-KCl host stoichiometry.
        6. Compute linear basis matrix dot-product synthesis:
           \\mathbf{S}_{\text{samples} \times \text{wavelengths}} = \\mathbf{C}_{\text{samples} \times \text{elements}} \\cdot \\mathbf{B}_{\text{elements} \times \text{wavelengths}}
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(input_dir).resolve()

    print("Extracting base spectra from CSV library...")
    elem_library, wavelengths = load_single_element_library(input_path)
    
    all_target_elements = ['Li', 'K', 'Cl'] + dopant_names
    missing_elems = [e for e in all_target_elements if e not in elem_library]
    if missing_elems:
        print(f"Warning: The following requested elements were not found in CSVs: {missing_elems}")
        all_target_elements = [e for e in all_target_elements if e in elem_library]
        dopant_names = [d for d in dopant_names if d in elem_library]

    num_dopants = len(dopant_names)
    if alpha_dopants is None:
        alpha_dopants = np.ones(num_dopants)

    # Chloride Stoichiometry Constants (Metal Cation Valency & Molar Mass in g/mol)
    dopant_cl_valency: Dict[str, float] = {
        'Ba': 2.0, 'Ca': 2.0, 'Ce': 3.0, 'Cr': 3.0, 'Cs': 1.0, 
        'Fe': 2.0, 'Gd': 3.0, 'La': 3.0, 'Mg': 2.0, 'Mn': 2.0, 
        'Nd': 3.0, 'Ni': 2.0, 'Sm': 3.0, 'Sr': 2.0, 'U': 3.0, 'Y': 3.0
    }
    
    dopant_molar_mass: Dict[str, float] = {
        'Ba': 137.33, 'Ca': 40.08, 'Ce': 140.12, 'Cr': 51.996, 'Cs': 132.91,
        'Fe': 55.845, 'Gd': 157.25, 'La': 138.91, 'Mg': 24.305, 'Mn': 54.938,
        'Nd': 144.24, 'Ni': 58.693, 'Sm': 150.36, 'Sr': 87.62, 'U': 238.03, 'Y': 88.906
    }
    
    M_CL = 35.453  # Molar mass of Chlorine (g/mol)

    # Grams of associated Chlorine required per gram of metal cation
    cl_per_metal_mass = np.array([
        (dopant_cl_valency[d] * M_CL) / dopant_molar_mass[d] for d in dopant_names
    ], dtype=np.float64)

    # Assign mixture category per sample (1=single, 2=binary, 3=ternary, 0=pure host)
    category_probs = np.array([singles, doubles, triples, pure_host])
    category_probs /= np.sum(category_probs)
    categories = np.random.choice([1, 2, 3, 0], size=num_samples, p=category_probs)

    # Sparse Sampling of Metal Cation Weight Fractions
    dopant_metal_wt = np.zeros((num_samples, num_dopants), dtype=np.float64)
    log_min, log_max = np.log10(1e-5), np.log10(dopant_max_total_wt)

    for i in range(num_samples):
        k = categories[i]
        if k == 0:
            continue
        
        tot_wt = 10**np.random.uniform(log_min, log_max)
        active_indices = np.random.choice(num_dopants, size=k, replace=False)
        
        sub_alpha = alpha_dopants[active_indices]
        ratios = np.random.dirichlet(sub_alpha)
        
        dopant_metal_wt[i, active_indices] = ratios * tot_wt

    # Stoichiometric Calculation of Chlorine Contribution
    dopant_cl_wt = np.sum(dopant_metal_wt * cl_per_metal_mass, axis=1, keepdims=True)
    total_dopant_salt_wt = np.sum(dopant_metal_wt, axis=1, keepdims=True) + dopant_cl_wt
    remaining_matrix_wt = 1.0 - total_dopant_salt_wt

    # Host Salt Composition Perturbation (Gaussian fluctuation around eutectic ratio)
    li_rel_target = 0.0966 / (0.0966 + 0.2150)
    li_rel_perturbed = np.random.normal(loc=li_rel_target, scale=host_sigma, size=(num_samples, 1))
    li_rel_perturbed = np.clip(li_rel_perturbed, 0.20, 0.45)
    k_rel_perturbed = 1.0 - li_rel_perturbed

    m_li = li_rel_perturbed * 6.94
    m_k = k_rel_perturbed * 39.10
    m_cl_host = 35.453
    m_host_total = m_li + m_k + m_cl_host

    host_li_wt = remaining_matrix_wt * (m_li / m_host_total)
    host_k_wt = remaining_matrix_wt * (m_k / m_host_total)
    host_cl_wt = remaining_matrix_wt * (m_cl_host / m_host_total)

    total_cl_wt = host_cl_wt + dopant_cl_wt

    # Build Composition Matrix DataFrame (wt%)
    compositions_dict = {
        'Li': host_li_wt.flatten() * 100.0,
        'K': host_k_wt.flatten() * 100.0,
        'Cl': total_cl_wt.flatten() * 100.0
    }

    for idx, name in enumerate(dopant_names):
        compositions_dict[name] = dopant_metal_wt[:, idx] * 100.0

    compositions_df = pd.DataFrame(compositions_dict)
    compositions_df.to_csv(output_dir / "synthetic_compositions_wt_pct.csv", index=False)
    print(f"Saved compositions manifest to: {output_dir / 'synthetic_compositions_wt_pct.csv'}")

    # Chunked Matrix Multiplication & Compressed HDF5 Storage
    print("Combining spectra and saving directly to HDF5 (.h5)...")
    
    basis_matrix = np.zeros((len(all_target_elements), len(wavelengths)), dtype=np.float32)
    for row_idx, elem in enumerate(all_target_elements):
        basis_matrix[row_idx, :] = elem_library[elem]

    num_wavelengths = len(wavelengths)
    h5_file_path = output_dir / "synthetic_spectra_200k.h5"

    with h5py.File(h5_file_path, "w") as hf:
        hf.create_dataset("wavelengths", data=wavelengths)

        spectra_ds = hf.create_dataset(
            "spectra", 
            shape=(num_samples, num_wavelengths), 
            dtype=np.float32,
            chunks=(2000, num_wavelengths),
            compression="gzip",
            compression_opts=3
        )

        chunk_size = 20000
        for start_idx in range(0, num_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, num_samples)
            
            comp_chunk = (compositions_df.iloc[start_idx:end_idx][all_target_elements].values / 100.0).astype(np.float32)
            # Dot product synthesis: [chunk_size, elements] x [elements, wavelengths] -> [chunk_size, wavelengths]
            spectra_ds[start_idx:end_idx, :] = comp_chunk @ basis_matrix
            
            print(f"  Processed and saved chunk {start_idx} to {end_idx} / {num_samples}")

    print(f"Successfully generated and saved {num_samples} spectra array to {h5_file_path}")
    return compositions_df
# endregion

# ==============================================================================
# region: EXPERIMENTAL & SYNTHETIC DATASET CONCATENATION
# ==============================================================================
def final_combo():
    """
    Loads experimental LIBS datasets and generated synthetic datasets, aligns 
    elemental column orderings, appends domain labels (`is_synthetic`: 0=Exp, 1=Syn), 
    and saves a unified consolidated HDF5 file for machine learning training.
    """
    exp_h5_path = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/elemental_experimental.h5")
    syn_h5_path = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/NIST_combined/synthetic_spectra_200k.h5")
    syn_csv_path = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/NIST_combined/synthetic_compositions_wt_pct.csv") 

    formatted_syn_h5 = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/formatted_synthetic_200k.h5")
    merged_h5_out = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/combined_exp_syn_dataset.h5")

    # 1. Inspect Experimental Element Order
    with h5py.File(exp_h5_path, 'r') as hf_exp:
        exp_elem_names = [
            name.decode('utf-8') if isinstance(name, bytes) else name 
            for name in hf_exp['metadata/elem_names'][:]        # type: ignore
        ]

    print(f"Target Experimental Element Order ({len(exp_elem_names)}): {exp_elem_names}")

    # 2. Align Synthetic Manifest Column Order to Match Experimental Exactly
    df_syn = pd.read_csv(syn_csv_path)
    missing_elems = set(exp_elem_names) - set(df_syn.columns)
    if missing_elems:
        raise ValueError(f"Synthetic CSV is missing elements present in experimental data: {missing_elems}")

    df_syn_reordered = df_syn[exp_elem_names]
    syn_comp_matrix = df_syn_reordered.values.astype(np.float32)

    # 3. Save Structured Metadata into Formatted Synthetic Dataset
    with h5py.File(syn_h5_path, 'r') as hf_in, h5py.File(formatted_syn_h5, 'w') as hf_out:
        hf_in.copy('spectra', hf_out)
        wl_data = hf_in['wavelengths'][:].astype(np.float32)        # type: ignore
        hf_out.create_dataset('wavelengths', data=wl_data)

        meta_grp = hf_out.create_group('metadata')
        meta_grp.create_dataset('elem_comp_wt%', data=syn_comp_matrix, compression='gzip')
        meta_grp.create_dataset('elem_names', data=np.array(exp_elem_names, dtype='S'))

    # 4. Merge Experimental and Synthetic Datasets into Unified HDF5
    with h5py.File(exp_h5_path, 'r') as hf_exp, \
         h5py.File(formatted_syn_h5, 'r') as hf_syn, \
         h5py.File(merged_h5_out, 'w') as hf_out:

        exp_spectra = hf_exp['spectra'][:]        # type: ignore
        syn_spectra = hf_syn['spectra'][:]        # type: ignore

        exp_comp = hf_exp['metadata/elem_comp_wt%'][:]        # type: ignore
        syn_comp = hf_syn['metadata/elem_comp_wt%'][:]        # type: ignore

        # Stack along sample axis (0)
        merged_spectra = np.vstack([exp_spectra, syn_spectra]).astype(np.float32)        # type: ignore
        merged_comp = np.vstack([exp_comp, syn_comp]).astype(np.float32)        # type: ignore

        # Domain Labels: 0 = Experimental, 1 = Synthetic
        domain_labels = np.zeros(len(merged_spectra), dtype=np.int32)
        domain_labels[len(exp_spectra):] = 1        # type: ignore

        hf_out.create_dataset('spectra', data=merged_spectra, compression='gzip')
        hf_out.create_dataset('wavelengths', data=hf_exp['wavelengths'][:].astype(np.float32))        # type: ignore

        meta_grp = hf_out.create_group('metadata')
        meta_grp.create_dataset('elem_comp_wt%', data=merged_comp, compression='gzip')
        meta_grp.create_dataset('elem_names', data=np.array(exp_elem_names, dtype='S'))
        meta_grp.create_dataset('is_synthetic', data=domain_labels)

    print(f"Successfully created unified HDF5 dataset at: {merged_h5_out}")
    print(f"Total merged samples: {len(merged_spectra)} ({len(exp_spectra)} experimental + {len(syn_spectra)} synthetic)")        # type: ignore
# endregion 

# ==============================================================================
# REGION: ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    print('Hi')
    # Generate line lookup master table for PyTorch model feature extraction
    master_line_table()