from __future__ import annotations
"""
jvlee_LIBS_ML > utils > NIST_data.py
"""
print('NIST_data.py loading ...')

# region Imports
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
# from utils import enrich_file_with_metadata
# endregion

# region NIST URL
NIST_LIBS_URL = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl"
NIST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
}
# endregion

print_lock = threading.Lock()
batch_lock = threading.Lock()

THREAD_DELAY = 1.2
MAX_WORKERS = 4

completed_files = []
batch_counter = 1
BATCH_SIZE = 1000
GRID_MIN = 200.0
GRID_MAX = 1000.0
GRID_STEP = 0.1
NUM_POINTS = int(round((GRID_MAX - GRID_MIN) / GRID_STEP)) + 1
WAVELENGTH_GRID = np.linspace(GRID_MIN, GRID_MAX, NUM_POINTS)

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")

# region File Paths
# file path: /lustre/home/leejv2/git_repos/jvlee_LIBS_ML/utils/NIST_data.py
output_dir = Path(__file__).parent.parent / "LIBS" / "NIST_trio_data"
output_dir.mkdir(parents=True, exist_ok=True)

# Output location for final ZIP files (Persistent Lustre storage)
FINAL_OUTPUT_DIR = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/NIST_trio_data")
FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Output location for individual spectra
INDIVIDUAL_SPECTRA = Path('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/individual_spectra.h5')
# INDIVIDUAL_SPECTRA.mkdir(parents=True, exist_ok=True)

# Scratch location for temporary CSVs (Fast local compute node disk)
SCRATCH_DIR = Path(os.environ.get("SLURM_TMPDIR", "/tmp")) / "nist_csv_scratch"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
# endregion

plasma_params = {
    "low_w": "200",         # Wavelength Minimum (nm)
    "upp_w": "1000",        # Wavelength Maximum (nm)
    "limits_type": "0",     # 0 for Wavelength range bounds
    "unit": "1",            # Wavelength unit: 1 for nm
    "resolution": "1000",   # Instead of resolving_power
    "temp": "1.0",          # Instead of te (Electron temperature in eV)
    "eden": "1e17",         # Instead of ne (Electron density in cm^-3)
    "maxcharge": "2",       # Max ion charge (e.g., 2+)
    "min_rel_int": "0.1",   # Minimum relative intensity threshold
    "show_av": "2",         # Profile calculation flag
    "libs": "1"             # Activates the LIBS mode calculation
}

# region Material constants (g/mol)

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
    # 'CeN': {'mmc': 154.123, 'num_bond_atoms':},
    'CaCl2': {'mmc': 110.984, 'num_bond_atoms': 2},
    'GdCl3': {'mmc': 263.610, 'num_bond_atoms': 3},
    'MgCl2': {'mmc': 95.211, 'num_bond_atoms': 2}
}
        # chromium
        #  

mmc_lif = 25.939
mmc_bef = 47.009

mmc_licl = 42.39
mmc_kcl = 74.55

mfr_licl = 0.59
mfr_kcl = 0.41
mmc_eutectice = (mfr_licl * mmc_licl) + (mfr_kcl * mmc_kcl)

# endregion

wt_percents_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

thread_local = threading.local()

def archive_batch_if_ready(force=False, run_timestamp=None):
    global batch_counter, completed_files

    if run_timestamp is None:
        run_timestamp = RUN_TIMESTAMP  # Fallback to module-level global

    files_to_zip = []
    current_batch_num = 0

    with batch_lock:
        if len(completed_files) >= BATCH_SIZE or (
            force and len(completed_files) > 0
        ):
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

        with zipfile.ZipFile(
            archive_name, "a", compression=zipfile.ZIP_DEFLATED
        ) as zf:
            for file_path in files_to_zip:
                if file_path.exists():
                    zf.write(file_path, arcname=file_path.name)
                    os.remove(file_path)

def fetch_asd_baseline_lines(element: str, plasma_params: dict) -> pd.DataFrame:
    """
    Fetches raw NIST ASD atomic spectral lines using direct Tab-Delimited output (format=3)
    via POST requests.
    """
    all_lines = []
    
    for stage in ["I", "II"]:
        # NIST requires POST data (not GET params) for format=3 ASCII output
        payload = {
            "spectra": f"{element} {stage}",
            "limits_type": "0",
            "low_w": str(plasma_params.get("low_w", GRID_MIN)),
            "upp_w": str(plasma_params.get("upp_w", GRID_MAX)),
            "unit": "1",          # 1 = nm
            "format": "3",        # 3 = Tab-delimited ASCII
            "ascii_out": "1",     # Required flag for ASCII output
            "line_out": "0",
            "remove_j": "on",
            "page_size": "0",     # Return all lines
            "show_obs_wl": "1",
            "show_calc_wl": "1",
            "intens_out": "on",
            "order_out": "0",
        }
        
        try:
            resp = requests.post(NIST_LIBS_URL, data=payload, headers=NIST_HEADERS, timeout=30)
            
            # Check for input error or missing data in HTML
            if resp.status_code != 200 or "Input Error" in resp.text or "No lines" in resp.text:
                continue
            
            # Split lines and filter out non-tabbed NIST metadata headers
            raw_lines = resp.text.splitlines()
            data_lines = [line for line in raw_lines if "\t" in line and not line.startswith("Spectrum")]
            
            if len(data_lines) < 2:
                continue
                
            tsv_payload = "\n".join(data_lines).replace('"', '')
            
            # Parse clean TSV
            df = pd.read_csv(io.StringIO(tsv_payload), sep="\t", engine="python", on_bad_lines="skip")
            df.columns = [str(c).strip() for c in df.columns]
            
            # Dynamically select Wavelength and Intensity columns
            wl_col = next((c for c in df.columns if "Observed" in c or "Ritz" in c or "Wavelength" in c), None)
            int_col = next((c for c in df.columns if "Rel." in c or "Int" in c), None)
            
            if not wl_col or not int_col:
                continue
                
            sub_df = pd.DataFrame()
            
            # Extract clean floating-point numbers from string values
            sub_df["Wavelength (nm)"] = (
                df[wl_col]
                .astype(str)
                .str.extract(r'([0-9]+\.?[0-9]*)')[0]
                .astype(float)
            )
            
            sub_df["Intensity"] = (
                df[int_col]
                .astype(str)
                .str.extract(r'([0-9]+\.?[0-9]*)')[0]
                .astype(float)
            )
            
            # Fill unrated line intensities with 1.0 baseline
            sub_df["Intensity"] = sub_df["Intensity"].fillna(1.0)
            
            # Drop invalid/blank wavelength rows
            sub_df = sub_df.dropna(subset=["Wavelength (nm)"])
            
            if not sub_df.empty:
                all_lines.append(sub_df)
                
        except Exception as e:
            print(f" -> Failed fetching ASD lines for {element} {stage}: {e}")

    if all_lines:
        return pd.concat(all_lines, ignore_index=True)
    return pd.DataFrame(columns=["Wavelength (nm)", "Intensity"])

    if all_lines:
        return pd.concat(all_lines, ignore_index=True)
    return pd.DataFrame(columns=["Wavelength (nm)", "Intensity"])

def fetch_nist_libs_data(
    comp_string: str, 
    plasma_params: dict
) -> pd.DataFrame | None:
    """
    Sends a query to the NIST LIBS database and parses the hidden JavaScript 
    array into a discrete peak list DataFrame.
    """
    # Parse elements and concentrations
    pairs = [p.strip() for p in comp_string.split(";") if p.strip()]
    elements = [p.split(":")[0].strip() for p in pairs]
    percentages = [p.split(":")[1].strip() for p in pairs]

    # Use a tuple list instead of a dict so repeated keys like mytext[] serialize cleanly
    payload = [
        ("form", "libs"),
        ("action", "libs"),
        ("low_w", str(plasma_params.get("low_w", GRID_MIN))),
        ("upp_w", str(plasma_params.get("upp_w", GRID_MAX))),
        ("unit", "1"),                                     # 1 = nm
        ("de_unit", "0"),                                  # eV
        ("line_out", "1"),                                 # Line output
        ("remove_j", "on"),
        ("temp", str(plasma_params.get("temp", "1.0"))),   # Electron temp (eV)
        ("eden", str(plasma_params.get("eden", "1e17"))),  # Electron density (cm^-3)
        ("resolution", str(plasma_params.get("resolution", "1000"))),
        ("min_rel_int", str(plasma_params.get("min_rel_int", "0.1"))),
        ("maxcharge", str(plasma_params.get("maxcharge", "2"))),
        ("show_av", str(plasma_params.get("show_av", "2"))),
        ("libs", "1"),
        ("composition", comp_string),
        ("num_sl", str(len(elements))),
        ("spectra", ",".join(elements)),
    ]

    # Append array parameters as individual repeated keys
    for el, pct in zip(elements, percentages):
        payload.append(("mytext[]", el))
        payload.append(("myperc[]", str(float(pct))))

    time.sleep(random.uniform(1.5, 3.0))
    session = get_thread_session() if 'thread_local' in globals() else get_nist_session()

    try:
        response = session.post(NIST_LIBS_URL, data=payload, timeout=120)
        
        if response.status_code != 200:
            print(f' -> Server error: Status code {response.status_code}')
            return None
            
        html_content = response.text
        
        # Check if NIST returned an explicit error block
        if "Error:" in html_content or "Incorrect element" in html_content:
            print(f" -> NIST returned a form validation error for composition: {comp_string}")
            return pd.DataFrame()

        # Search for javascript 'var lines = [...]' block
        match = re.search(r"var\s+lines\s*=\s*\[(.*?)\];", html_content, re.DOTALL)
        if not match:
            print(f' -> Warning: Response received for {comp_string}, but "var lines" data block wasn\'t found.')
            # Debug tip: Inspect what NIST sent back if needed
            # print("DEBUG - Response snippet:", html_content[:500])
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

def gather_2_data(
    dopant_1: str,
    mmc_dopant_1: float,
    num_bond_atoms_1: int,
    dopant_2: str,
    mmc_dopant_2: float,
    num_bond_atoms_2: int,
    mfr_salt_a: float,
    mmc_salt_a: float,
    mfr_salt_b: float,
    mmc_salt_b: float,
    salt: str,
    wt_percents: list,
):
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
            salt_a = 'Li'
            salt_b = 'K'
            bond_element = 'Cl'
        elif salt == 'FLiBe':
            mmc_salt = 1
            salt_a = 'Li'
            salt_b = 'Be'
            bond_element = 'F'
        else:
            print(f'Skipping Run {idx}: No host salt!')
            continue
            
        wt_salt_a = wt_salt * ((mfr_salt_a * mmc_salt_a) / mmc_salt)
        wt_salt_b = wt_salt * ((mfr_salt_b * mmc_salt_b) / mmc_salt)
        
        moles_salt_a = wt_salt_a / mmc_salt_a
        moles_salt_b = wt_salt_b / mmc_salt_b
        moles_dopant_1 = wt_dopant_1 / mmc_dopant_1
        moles_dopant_2 = wt_dopant_2 / mmc_dopant_2
        
        salt_a_atoms = moles_salt_a * 1
        salt_b_atoms = moles_salt_b * 1
        dopant_1_atoms = moles_dopant_1 * 1
        dopant_2_atoms = moles_dopant_2 * 1
        bond_atoms = (moles_salt_a * 1) + (moles_salt_b * 1) + (moles_dopant_1 * num_bond_atoms_1) + (moles_dopant_2 * num_bond_atoms_2)
        
        total_atoms = salt_a_atoms + salt_b_atoms + dopant_1_atoms + dopant_2_atoms + bond_atoms
        
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
        
        # Step 1: Fetch discrete peak lines
        df_discrete = fetch_nist_libs_data(comp_string, plasma_params)
        # df_discrete = fetch_nist_libs_data(comp_string, plasma_params,num_chunks=2)
        
        if df_discrete is None:
            print(f' -> Error: Server fetch failed for Run {idx}. Skipping save.')
            time.sleep(2.0)
            continue
            
        if df_discrete.empty:
            print(f' -> Warning: No spectral lines found for Run {idx}.')
            time.sleep(2.0)
            continue

        # Step 2: Broaden profiles
        df_continuous = generate_simple_intensity_profile(
            df_discrete,
            low_w=float(plasma_params["low_w"]),
            upp_w=float(plasma_params["upp_w"]),
            step=0.1,
            resolution=float(plasma_params["resolution"])
        )
        
        # Step 3: Single atomic write to disk
        if not df_continuous.empty:
            tmp_path = file_path.with_suffix(".csv.tmp")
            df_continuous.to_csv(tmp_path, index=False)
            tmp_path.replace(file_path)  # atomic overwrite across POSIX filesystems
            print(f' -> Successfully parsed, broadened, and saved data to: {filename}')
        else:
            print(' -> Warning: Broadened profile returned 0 data points.')
            
        # Server rate-limit delay
        time.sleep(2.0)

    print('\n--- Matrix data collection complete ---')

def gather_3_data(
    dopant_1: str,
    mmc_dopant_1: float,
    num_bond_atoms_1: int,
    dopant_2: str,
    mmc_dopant_2: float,
    num_bond_atoms_2: int,
    dopant_3: str,
    mmc_dopant_3: float,
    num_bond_atoms_3: int,
    mfr_salt_a: float,
    mmc_salt_a: float,
    mfr_salt_b: float,
    mmc_salt_b: float,
    salt: str,
    wt_percents: list,
):
    # 3D Grid Generation for the 3 dopants
    grid = [
        (w1, w2, w3)
        for w1 in wt_percents
        for w2 in wt_percents
        for w3 in wt_percents
    ]
    experiment_df = pd.DataFrame(
        grid,
        columns=[
            f"{dopant_1}_wt%",
            f"{dopant_2}_wt%",
            f"{dopant_3}_wt%",
        ],
    )

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
            print(f"Skipping {filename}, already exists...")
            continue

        wt_dopants = wt_dopant_1 + wt_dopant_2 + wt_dopant_3
        wt_salt = 100.0 - wt_dopants

        if wt_salt < 0:
            print(
                f"Skipping Run {idx}: Combined dopant weights exceed 100%! "
                f"({wt_dopants:.2f}%)"
            )
            continue

        if salt == "ClLiK":
            mmc_salt = mmc_eutectice
            salt_a = "Li"
            salt_b = "K"
            bond_element = "Cl"
        elif salt == "FLiBe":
            mmc_salt = 1
            salt_a = "Li"
            salt_b = "Be"
            bond_element = "F"
        else:
            print(f"Skipping Run {idx}: No host salt!")
            continue

        # Salt sub-component weight fractions
        wt_salt_a = wt_salt * ((mfr_salt_a * mmc_salt_a) / mmc_salt)
        wt_salt_b = wt_salt * ((mfr_salt_b * mmc_salt_b) / mmc_salt)

        # Molar Calculations
        moles_salt_a = wt_salt_a / mmc_salt_a
        moles_salt_b = wt_salt_b / mmc_salt_b
        moles_dopant_1 = wt_dopant_1 / mmc_dopant_1
        moles_dopant_2 = wt_dopant_2 / mmc_dopant_2
        moles_dopant_3 = wt_dopant_3 / mmc_dopant_3

        # Atomic Proportions
        salt_a_atoms = moles_salt_a * 1
        salt_b_atoms = moles_salt_b * 1
        dopant_1_atoms = moles_dopant_1 * 1
        dopant_2_atoms = moles_dopant_2 * 1
        dopant_3_atoms = moles_dopant_3 * 1

        bond_atoms = (
            (moles_salt_a * 1)
            + (moles_salt_b * 1)
            + (moles_dopant_1 * num_bond_atoms_1)
            + (moles_dopant_2 * num_bond_atoms_2)
            + (moles_dopant_3 * num_bond_atoms_3)
        )

        total_atoms = (
            salt_a_atoms
            + salt_b_atoms
            + dopant_1_atoms
            + dopant_2_atoms
            + dopant_3_atoms
            + bond_atoms
        )

        # Atomic Percentages
        salt_a_val = (salt_a_atoms / total_atoms) * 100
        salt_b_val = (salt_b_atoms / total_atoms) * 100
        bond_val = (bond_atoms / total_atoms) * 100
        dopant_1_val = (dopant_1_atoms / total_atoms) * 100
        dopant_2_val = (dopant_2_atoms / total_atoms) * 100
        dopant_3_val = (dopant_3_atoms / total_atoms) * 100

        # Element extraction (e.g., handles formulas like "CaCl2" -> "Ca")
        match1 = re.match(r"([A-Z][a-z]?)", dopant_1)
        match2 = re.match(r"([A-Z][a-z]?)", dopant_2)
        match3 = re.match(r"([A-Z][a-z]?)", dopant_3)

        element_1 = match1.group(1) if match1 else dopant_1
        element_2 = match2.group(1) if match2 else dopant_2
        element_3 = match3.group(1) if match3 else dopant_3

        comp_string = (
            f"{salt_a}:{salt_a_val:.5f};"
            f"{salt_b}:{salt_b_val:.5f};"
            f"{bond_element}:{bond_val:.5f};"
            f"{element_1}:{dopant_1_val:.5f};"
            f"{element_2}:{dopant_2_val:.5f};"
            f"{element_3}:{dopant_3_val:.5f}"
        )

        print(
            f"Processing Run {idx}/{len(experiment_df)}: "
            f"{dopant_1}={wt_dopant_1}wt%, "
            f"{dopant_2}={wt_dopant_2}wt%, "
            f"{dopant_3}={wt_dopant_3}wt%"
        )

        # Step 1: Fetch discrete peak lines
        df_discrete = fetch_nist_libs_data(comp_string, plasma_params)

        if df_discrete is None:
            print(f" -> Error: Server fetch failed for Run {idx}. Skipping save.")
            time.sleep(2.0)
            continue

        if df_discrete.empty:
            print(f" -> Warning: No spectral lines found for Run {idx}.")
            time.sleep(2.0)
            continue

        # Step 2: Broaden profiles
        df_continuous = generate_simple_intensity_profile(
            df_discrete,
            low_w=float(plasma_params["low_w"]),
            upp_w=float(plasma_params["upp_w"]),
            step=0.1,
            resolution=float(plasma_params["resolution"]),
        )

        # Step 3: Single atomic write to disk
        if not df_continuous.empty:
            tmp_path = file_path.with_suffix(".csv.tmp")
            df_continuous.to_csv(tmp_path, index=False)
            tmp_path.replace(file_path)  # atomic overwrite
            print(
                f" -> Successfully parsed, broadened, and saved data to: {filename}"
            )
        else:
            print(" -> Warning: Broadened profile returned 0 data points.")

        # Server rate-limit delay
        time.sleep(2.0)

    print("\n--- Matrix data collection complete ---")

def generate_simple_intensity_profile(
        df_discrete: pd.DataFrame, 
        low_w: float = 200.0, 
        upp_w: float = 1000.0, 
        step: float = 0.1, 
        resolution: float = 1000.0
) -> pd.DataFrame:
    """
    Converts discrete peak lines into a uniform wavelength intensity array.
    Outputs a clean DataFrame filled with zeros and broadened intensity shapes.
    """
    if df_discrete is None or df_discrete.empty:
        return pd.DataFrame(columns=["Wavelength (nm)", "Intensity"])

    # 1. Create a bulletproof uniform wavelength grid using np.linspace
    num_points = int(round((upp_w - low_w) / step)) + 1
    wavelength_grid = np.linspace(low_w, upp_w, num_points)
    intensity_array = np.zeros_like(wavelength_grid)
    
    # 2. Apply Gaussian Broadening around each peak
    for _, row in df_discrete.iterrows():
        lambda_0 = row["Wavelength (nm)"]
        peak_intensity = row["Intensity"]
        
        # FWHM = lambda / R
        fwhm = lambda_0 / resolution
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        
        # Target local grid segment within 5 standard deviations to speed up math
        window = 5 * sigma
        mask = (wavelength_grid >= lambda_0 - window) & (wavelength_grid <= lambda_0 + window)
        relevant_grid = wavelength_grid[mask]
        
        if len(relevant_grid) == 0:
            continue
            
        # Gaussian distribution function evaluation
        gaussian_shape = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((relevant_grid - lambda_0) ** 2) / (2 * sigma ** 2))
        intensity_array[mask] += peak_intensity * gaussian_shape

    return pd.DataFrame({
        "Wavelength (nm)": wavelength_grid,
        "Intensity": intensity_array
    })

def get_nist_session() -> requests.Session:
    """Creates a requests Session with automatic backoff retries."""
    session = requests.Session()

    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })

    retries = Retry(
        total=10,                  # Try up to 5 times
        backoff_factor=3,         # Wait 2s, 4s, 8s, 16s...
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session

def get_thread_session() -> requests.Session:
    if not hasattr(thread_local, 'session'):
        thread_local.session = get_nist_session()
    return thread_local.session

def master_line_table():
    ELEMENTS = [
        'Ba',
        'Ca',
        'Ce',
        'Cr',
        'Cs',
        'Fe',
        'Gd',
        'K',
        'La',
        'Li',
        'Mg',
        'Mn',
        'Nd',
        'Ni',
        'Sm',
        'Sr',
        'U',
        'Y',
        'Cl',
    ]

    NIST_DIR = (
        '/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/NIST_singles/Enriched'
    )
    OUTPUT_CSV = 'master_line_table.csv'

    # Known non-wavelength metadata columns
    METADATA_COLS = {
        'conc_Ba_wt%',
        'frac_LiCl',
        'temperature_C',
        'scan_rate_mVs',
        'delay',
        'energy',
        'static_',
        'blank',
        'kinetic',
        'repetition',
    }

    lines_data = []

    for elem_idx, elem in enumerate(ELEMENTS):
        # Find matching CSV file for this element
        pattern = os.path.join(NIST_DIR, f'*{elem}*.csv')
        files = glob.glob(pattern)

        if not files:
            print(f'Warning: No NIST CSV file found for the element {elem}')
            continue

        df = pd.read_csv(files[0])

        # 1. Identify wavelength columns vs metadata columns
        wavelength_cols = []
        for col in df.columns:
            if col in METADATA_COLS:
                continue
            try:
                # Convert header name to float wavelength value
                wl_val = float(col)
                wavelength_cols.append((col, wl_val))
            except ValueError:
                # Skip non-numeric header strings that aren't wavelengths
                continue

        if not wavelength_cols:
            print(
                f'Warning: No valid wavelength headers found in file for {elem}'
            )
            continue

        # Sort columns chronologically by wavelength
        wavelength_cols.sort(key=lambda x: x[1])
        col_names = [c[0] for c in wavelength_cols]
        wls = np.array([c[1] for c in wavelength_cols])

        # 2. Extract intensities across wavelengths (mean spectrum across all rows)
        intens = df[col_names].mean(axis=0).values

        # 3. Clean NaN / negative values
        intens = np.nan_to_num(intens, nan=0.0)     # type: ignore
        intens = np.clip(intens, 0, None)

        # 4. Peak detection
        peaks, properties = find_peaks(
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
    # Sort by wavelength for memory-efficient PyTorch operations
    master_df = master_df.sort_values(by='wavelength_nm').reset_index(
        drop=True
    )
    master_df.to_csv(OUTPUT_CSV, index=False)

    print(
        f'Master Line Table created successfully with {len(master_df)} total'
        f' lines across {len(ELEMENTS)} elements.'
    )

def master_parallel_run():
    # 1. Scan existing ZIP archives to avoid re-downloading completed runs
    
    print("🔍 Inspecting existing ZIP archives in persistent storage...")
    existing_archived_files = set()
    for zip_path in FINAL_OUTPUT_DIR.glob("*.zip"):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                existing_archived_files.update(zf.namelist())
        except Exception as e:
            print(f"⚠️ Warning reading {zip_path.name}: {e}")

    print(f"ℹ️ Found {len(existing_archived_files)} CSVs already packaged in ZIP files.")

    dopant_triplets = list(itertools.combinations(DOPANT_SPECS.keys(), 3))

    # Build the full list of missing tasks across ALL triplets
    all_tasks = []
    run_counter = 1

    print("🔍 Pre-scanning grid to find uncompleted runs...")

    for d1, d2, d3 in dopant_triplets:
        grid = [
            (w1, w2, w3)
            for w1 in wt_percents_list
            for w2 in wt_percents_list
            for w3 in wt_percents_list
        ]

        for w1, w2, w3 in grid:
            # Quick check to avoid building tasks for files that already exist
            fn = f"nist_libs_{w1}_wt_{d1}_and_{w2}_wt_{d2}_and_{w3}_wt_{d3}.csv"
            if fn in existing_archived_files or (SCRATCH_DIR / fn).exists():
                continue

            if (w1 + w2 + w3) > 100.0:
                continue

            # Task payload
            task = (
                run_counter,
                0,  # placeholder total
                d1,
                d2,
                d3,
                w1,
                w2,
                w3,
                DOPANT_SPECS,
                mfr_licl,
                mmc_licl,
                mfr_kcl,
                mmc_kcl,
                "ClLiK",
                plasma_params,
                SCRATCH_DIR,
            )
            all_tasks.append(task)
            run_counter += 1

    total_tasks = len(all_tasks)
    print(f"🚀 Found {total_tasks} remaining simulation runs to execute.")
    print(
        f"⚡ Processing with {MAX_WORKERS} concurrent threads ({THREAD_DELAY}s worker delay)..."
    )

    # Re-assign total task count for accurate logging
    final_tasks = [
        (*t[:1], total_tasks, *t[2:]) for t in all_tasks
    ]

    # Execute Thread Pool
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:
        futures = [
            executor.submit(process_single_simulation, task)
            for task in final_tasks
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
            except Exception as e:
                with print_lock:
                    print(f"❌ Exception occurred in worker thread: {e}")

    archive_batch_if_ready(force=False, run_timestamp=None)

def process_single_simulation(task_args):
    """Worker function that processes ONE specific 3-dopant concentration run."""
    (
        idx,
        total_runs,
        d1,
        d2,
        d3,
        wt_dopant_1,
        wt_dopant_2,
        wt_dopant_3,
        DOPANT_SPECS,
        mfr_salt_a,
        mmc_salt_a,
        mfr_salt_b,
        mmc_salt_b,
        salt,
        plasma_params,
        SCRATCH_DIR,
    ) = task_args

    filename = (
        # f"{RUN_TIMESTAMP}_nist_libs_{wt_dopant_1}_wt_{d1}_and_{wt_dopant_2}_wt_{d2}_and_{wt_dopant_3}_wt_{d3}.csv"
        f"nist_libs_{wt_dopant_1}_wt_{d1}_and_{wt_dopant_2}_wt_{d2}_and_{wt_dopant_3}_wt_{d3}.csv"
    )
    file_path = SCRATCH_DIR / filename

    # 1. Skip if already downloaded
    if file_path.exists():
        return "skipped_exists"

    # 2. Check total dopant weight
    wt_dopants = wt_dopant_1 + wt_dopant_2 + wt_dopant_3
    wt_salt = 100.0 - wt_dopants

    if wt_salt < 0:
        return "skipped_over_100"

    # 3. Setup salt parameters
    if salt == "ClLiK":
        mmc_salt = mmc_eutectice
        salt_a, salt_b, bond_element = "Li", "K", "Cl"
    elif salt == "FLiBe":
        mmc_salt = 1
        salt_a, salt_b, bond_element = "Li", "Be", "F"
    else:
        return "error_salt"

    # 4. Molar & Atomic Calculations
    mmc_d1, num_bond_1 = (
        DOPANT_SPECS[d1]["mmc"],
        DOPANT_SPECS[d1]["num_bond_atoms"],
    )
    mmc_d2, num_bond_2 = (
        DOPANT_SPECS[d2]["mmc"],
        DOPANT_SPECS[d2]["num_bond_atoms"],
    )
    mmc_d3, num_bond_3 = (
        DOPANT_SPECS[d3]["mmc"],
        DOPANT_SPECS[d3]["num_bond_atoms"],
    )

    wt_salt_a = wt_salt * ((mfr_salt_a * mmc_salt_a) / mmc_salt)
    wt_salt_b = wt_salt * ((mfr_salt_b * mmc_salt_b) / mmc_salt)

    moles_salt_a = wt_salt_a / mmc_salt_a
    moles_salt_b = wt_salt_b / mmc_salt_b
    moles_d1 = wt_dopant_1 / mmc_d1
    moles_d2 = wt_dopant_2 / mmc_d2
    moles_d3 = wt_dopant_3 / mmc_d3

    salt_a_atoms = moles_salt_a
    salt_b_atoms = moles_salt_b
    d1_atoms = moles_d1
    d2_atoms = moles_d2
    d3_atoms = moles_d3
    bond_atoms = (
        moles_salt_a
        + moles_salt_b
        + (moles_d1 * num_bond_1)
        + (moles_d2 * num_bond_2)
        + (moles_d3 * num_bond_3)
    )

    total_atoms = (
        salt_a_atoms
        + salt_b_atoms
        + d1_atoms
        + d2_atoms
        + d3_atoms
        + bond_atoms
    )

    # Convert to atomic percentages
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

    # 5. Fetch from NIST API
    df_discrete = fetch_nist_libs_data(comp_string, plasma_params)

    # Per-thread delay to protect NIST server
    time.sleep(THREAD_DELAY)

    if df_discrete is None:  # Only None indicates a network failure
        with print_lock:
            print(f"[Run {idx}/{total_runs}] ❌ Network drop/timeout: {filename}")
        return "failed_fetch"

    # 6. Broaden profiles
    df_continuous = generate_simple_intensity_profile(
        df_discrete,
        low_w=float(plasma_params["low_w"]),
        upp_w=float(plasma_params["upp_w"]),
        step=0.1,
        resolution=float(plasma_params["resolution"]),
    )

    # 7. Safe atomic write to disk
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

def load_single_element_library(input_dir: Path):
    """
    Reads all CSVs in input_dir, dynamically extracts wavelength columns,
    identifies which element is active in each CSV, and averages its spectra.
    """
    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    element_spectra = {}
    wavelength_cols = None

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        
        # Detect wavelength columns (columns that can be converted to float)
        if wavelength_cols is None:
            wavelength_cols = []
            for col in df.columns:
                try:
                    float(col)
                    wavelength_cols.append(col)
                except ValueError:
                    continue

        # Identify element from non-zero concentration columns
        conc_cols = [c for c in df.columns if c.startswith("conc_") and c.endswith("_wt%")]
        active_element = None
        
        for c_col in conc_cols:
            if (df[c_col] > 0).any():
                # Cleanly extract the chemical formula (e.g., 'conc_Ba_wt%' -> 'Ba')
                formula = c_col.replace("conc_", "").replace("_wt%", "").replace("%", "").strip("_")
                
                # Count uppercase letters to count distinct elements in the formula
                element_count = sum(1 for char in formula if char.isupper())
                
                # Skip if it contains 2 or more elements (e.g., NaCl, BaCl2, LiCl)
                if element_count > 1:
                    continue
                    
                active_element = formula

        if active_element is None:
            continue

        # Extract spectral intensities (average across rows in file to get mean basis spectrum)
        spectral_data = df[wavelength_cols].values.astype(np.float32)
        mean_spectrum = np.mean(spectral_data, axis=0)

        # Normalize basis spectrum by its pure concentration weight if needed
        conc_val = df[f"conc_{active_element}_wt%"].iloc[0]
        if conc_val > 0:
            mean_spectrum = mean_spectrum / (conc_val / 100.0)

        element_spectra[active_element] = mean_spectrum

    return element_spectra, np.array([float(w) for w in wavelength_cols], dtype=np.float32) # type: ignore

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
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(input_dir).resolve()

    # --- 0.a. LOAD BASE SPECTRA LIBRARY ---
    print("Extracting base spectra from CSV library...")
    elem_library, wavelengths = load_single_element_library(input_path)
    
    # Ensure all required elements exist in loaded files
    all_target_elements = ['Li', 'K', 'Cl'] + dopant_names
    missing_elems = [e for e in all_target_elements if e not in elem_library]
    if missing_elems:
        print(f"Warning: The following requested elements were not found in CSVs: {missing_elems}")
        all_target_elements = [e for e in all_target_elements if e in elem_library]
        dopant_names = [d for d in dopant_names if d in elem_library]

    num_dopants = len(dopant_names)
    if alpha_dopants is None:
        alpha_dopants = np.ones(num_dopants)

    # --- 0.b. DEFINE CHLORIDE STOICHIOMETRY (molar masses in g/mol)---
    # Defines number of Cl atoms associated with 1 atom of metal cation
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
    
    M_CL = 35.453  # Molar mass of Chlorine

    # Compute conversion factor: grams of associated Cl added per gram of metal cation
    cl_per_metal_mass = np.array([
        (dopant_cl_valency[d] * M_CL) / dopant_molar_mass[d] for d in dopant_names
    ], dtype=np.float64)

    # --- 1. DETERMINE CATEGORY PER SAMPLE ---
    category_probs = [singles, doubles, triples, pure_host]
    category_probs = np.array(category_probs) / np.sum(category_probs)
    categories = np.random.choice([1, 2, 3, 0], size=num_samples, p=category_probs)

    # --- 2. SAMPLE SPARSE METAL CATION WEIGHT FRACTIONS ---
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

    # --- 3. CALCULATE ASSOCIATED CHLORINE FROM DOPANT SALTS ---
    # Vectorized multiplication: [num_samples, num_dopants] * [num_dopants]
    dopant_cl_wt = np.sum(dopant_metal_wt * cl_per_metal_mass, axis=1, keepdims=True)
    
    # Total mass of added dopant salts (Metal Cations + Associated Chlorine)
    total_dopant_salt_wt = np.sum(dopant_metal_wt, axis=1, keepdims=True) + dopant_cl_wt

    # Mass remainder belonging strictly to the LiCl-KCl host salt
    remaining_matrix_wt = 1.0 - total_dopant_salt_wt

    # --- 4. HOST SALT PERTURBATION (wt%) ---
    li_rel_target = 0.0966 / (0.0966 + 0.2150)
    li_rel_perturbed = np.random.normal(loc=li_rel_target, scale=host_sigma, size=(num_samples, 1))
    li_rel_perturbed = np.clip(li_rel_perturbed, 0.20, 0.45)
    k_rel_perturbed = 1.0 - li_rel_perturbed

    # Stoichiometric breakdown of host LiCl-KCl
    m_li = li_rel_perturbed * 6.94
    m_k = k_rel_perturbed * 39.10
    m_cl_host = 35.453
    m_host_total = m_li + m_k + m_cl_host

    host_li_wt = remaining_matrix_wt * (m_li / m_host_total)
    host_k_wt = remaining_matrix_wt * (m_k / m_host_total)
    host_cl_wt = remaining_matrix_wt * (m_cl_host / m_host_total)

    # Total Chlorine = Host Chlorine + Extra Chlorine brought by metal chloride dopants
    total_cl_wt = host_cl_wt + dopant_cl_wt

    # --- 5. BUILD COMPOSITION MATRIX (wt%) ---
    compositions_dict = {
        'Li': host_li_wt.flatten() * 100.0,
        'K': host_k_wt.flatten() * 100.0,
        'Cl': total_cl_wt.flatten() * 100.0
    }

    for idx, name in enumerate(dopant_names):
        compositions_dict[name] = dopant_metal_wt[:, idx] * 100.0

    compositions_df = pd.DataFrame(compositions_dict)

    # Save concentration targets manifest
    compositions_df.to_csv(output_dir / "synthetic_compositions_wt_pct.csv", index=False)
    print(f"Saved compositions manifest to: {output_dir / 'synthetic_compositions_wt_pct.csv'}")

    # --- 6. CHUNKED SPECTRA GENERATION & DIRECT H5 SAVE ---
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
            spectra_ds[start_idx:end_idx, :] = comp_chunk @ basis_matrix
            
            print(f"  Processed and saved chunk {start_idx} to {end_idx} / {num_samples}")

    print(f"Successfully generated and saved {num_samples} spectra array to {h5_file_path}")

    return compositions_df

def final_combo():
        # Paths
    exp_h5_path = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/elemental_experimental.h5")
    syn_h5_path = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/NIST_combined/synthetic_spectra_200k.h5")
    syn_csv_path = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/NIST_combined/synthetic_compositions_wt_pct.csv") 

    formatted_syn_h5 = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/formatted_synthetic_200k.h5")
    merged_h5_out = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/combined_exp_syn_dataset.h5")

    # -------------------------------------------------------------
    # 1. Read Experimental Element Names to Establish Target Order
    # -------------------------------------------------------------
    with h5py.File(exp_h5_path, 'r') as hf_exp:
        exp_elem_names = [name.decode('utf-8') if isinstance(name, bytes) else name 
                        for name in hf_exp['metadata/elem_names'][:]] # type: ignore

    print(f"Target Experimental Element Order ({len(exp_elem_names)}):")
    print(exp_elem_names)

    # -------------------------------------------------------------
    # 2. Load Synthetic CSV and Re-order Columns
    # -------------------------------------------------------------
    df_syn = pd.read_csv(syn_csv_path)

    # Verify all required elements are present
    missing_elems = set(exp_elem_names) - set(df_syn.columns)
    if missing_elems:
        raise ValueError(f"Synthetic CSV is missing elements present in experimental data: {missing_elems}")

    # Re-index synthetic DataFrame to match experimental column ordering EXACTLY
    df_syn_reordered = df_syn[exp_elem_names]

    syn_comp_matrix = df_syn_reordered.values.astype(np.float32)
    print(f"Re-ordered Synthetic Composition Matrix Shape: {syn_comp_matrix.shape}")

    # -------------------------------------------------------------
    # 3. Create Formatted Synthetic HDF5 File (Adding metadata)
    # -------------------------------------------------------------
    print("\nWriting metadata into formatted synthetic HDF5 file...")

    with h5py.File(syn_h5_path, 'r') as hf_in, h5py.File(formatted_syn_h5, 'w') as hf_out:
        # Copy spectra and wavelengths
        hf_in.copy('spectra', hf_out)
        
        # Cast wavelengths to float32 to match dtype consistency
        wl_data = hf_in['wavelengths'][:].astype(np.float32) # type: ignore
        hf_out.create_dataset('wavelengths', data=wl_data)

        # Build the missing metadata group
        meta_grp = hf_out.create_group('metadata')
        meta_grp.create_dataset(
            'elem_comp_wt%', 
            data=syn_comp_matrix, 
            compression='gzip'
        )
        meta_grp.create_dataset(
            'elem_names', 
            data=np.array(exp_elem_names, dtype='S')
        )

    print(f"Formatted synthetic dataset saved to: {formatted_syn_h5}")

    # -------------------------------------------------------------
    # 4. Concatenate Experimental and Synthetic Datasets into One HDF5
    # -------------------------------------------------------------
    print("\nConcatenating experimental and synthetic datasets...")

    with h5py.File(exp_h5_path, 'r') as hf_exp, \
        h5py.File(formatted_syn_h5, 'r') as hf_syn, \
        h5py.File(merged_h5_out, 'w') as hf_out:

        exp_spectra = hf_exp['spectra'][:] # type: ignore
        syn_spectra = hf_syn['spectra'][:] # type: ignore

        exp_comp = hf_exp['metadata/elem_comp_wt%'][:] # type: ignore
        syn_comp = hf_syn['metadata/elem_comp_wt%'][:] # type: ignore

        # Stack along sample axis (0)
        merged_spectra = np.vstack([exp_spectra, syn_spectra]).astype(np.float32) # type: ignore
        merged_comp = np.vstack([exp_comp, syn_comp]).astype(np.float32) # type: ignore

        # Track domain origin: 0 = Experimental, 1 = Synthetic
        domain_labels = np.zeros(len(merged_spectra), dtype=np.int32)
        domain_labels[len(exp_spectra):] = 1 # type: ignore

        # Save merged arrays
        hf_out.create_dataset('spectra', data=merged_spectra, compression='gzip')
        hf_out.create_dataset('wavelengths', data=hf_exp['wavelengths'][:].astype(np.float32)) # type: ignore

        meta_grp = hf_out.create_group('metadata')
        meta_grp.create_dataset('elem_comp_wt%', data=merged_comp, compression='gzip')
        meta_grp.create_dataset('elem_names', data=np.array(exp_elem_names, dtype='S'))
        meta_grp.create_dataset('is_synthetic', data=domain_labels)

    print(f"Successfully created unified HDF5 dataset at: {merged_h5_out}")
    print(f"Total merged samples: {len(merged_spectra)} ({len(exp_spectra)} experimental + {len(syn_spectra)} synthetic)") # type: ignore



if __name__ == "__main__":

    master_line_table()

    # final_combo()

    # synthetic_data_generator(
    #     input_dir='/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/NIST_singles/Enriched',
    #     output_dir='/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/NIST_combined',
    #     num_samples=200000,
    #     dopant_names= ['Ba', 'Ca', 'Ce', 'Cr', 'Cs', 'Fe', 'Gd', 'La', 'Mg', 'Mn', 'Nd', 'Ni', 'Sm', 'Sr', 'U', 'Y'],
    #     host_sigma=0.015,
    #     dopant_max_total_wt=0.05,
    #     alpha_dopants=None
    # )

    # elements = [
    #     'Li', 'K', 'Cl', 'Ce',
    #     'Sm', 'La', 'Nd', 'Cs',
    #     'Sr', 'Ba', 'Y', 'Fe',
    #     'Cr', 'Ni', 'Mn', 'U',
    #     'Ca', 'Gd', 'Mg'
    # ]

    # # elements = """ 
    # #     'Li':100.0;'K':100.0;'Cl':100.0;'Ce':100.0;
    # #     'Sm':100.0;'La':100.0;'Nd':100.0;'Cs':100.0;
    # #     'Sr':100.0;'Ba':100.0;'Y':100.0;'Fe':100.0;
    # #     'Cr':100.0;'Ni':100.0;'Mn':100.0;'U':100.0;
    # #     'Ca':100.0;'Gd':100.0;'Mg':100.0
    # # """
    # element_spectra = {}

    # for el in elements:
    #     print(f"Fetching baseline spectrum for 100% {el}...")

    #     # Fetch lines via direct ASD tab-delimited GET request
    #     df_discrete = fetch_asd_baseline_lines(el, plasma_params)

    #     if df_discrete.empty:
    #         print(f" -> Warning: No lines generated for {el}. Storing zero array.")
    #         element_spectra[el] = np.zeros_like(WAVELENGTH_GRID, dtype=np.float32)
    #         continue

    #     df_continuous = generate_simple_intensity_profile(
    #         df_discrete=df_discrete,
    #         low_w=GRID_MIN,
    #         upp_w=GRID_MAX,
    #         step=GRID_STEP,
    #         resolution=float(plasma_params['resolution'])
    #     )

    #     element_spectra[el] = df_continuous["Intensity"].to_numpy(dtype=np.float32)

    # print(f"\nSaving baseline spectra to {INDIVIDUAL_SPECTRA}...")

    # with h5py.File(INDIVIDUAL_SPECTRA, "w") as hf:
    #     hf.create_dataset("wavelengths", data=WAVELENGTH_GRID, dtype=np.float32)
    #     spectra_group = hf.create_group('baseline_spectra')

    #     for el_symbol, spectrum_array in element_spectra.items():
    #         spectra_group.create_dataset(
    #             el_symbol,
    #             data=spectrum_array,
    #             dtype=np.float32,
    #             compression="gzip",
    #             compression_opts=3
    #         )

    # print("✅ Successfully saved baseline spectra library!")

    # print(SCRATCH_DIR)

    # master_parallel_run()


    # # Get all unique 3-dopant combinations from the 13 available (286 triplets total)
    # dopant_triplets = list(itertools.combinations(DOPANT_SPECS.keys(), 3))

    # # Calculate total simulations based on the size of your wt_percents_list grid (N^3)
    # grid_size_per_triplet = len(wt_percents_list) ** 3

    # print(f"==========================================================")
    # print(
    #     f"Starting Master 3-Dopant Grid Scan: {len(dopant_triplets)} total triplet configurations found."
    # )
    # print(
    #     f"Total projected file matrix: {len(dopant_triplets) * grid_size_per_triplet} simulations."
    # )
    # print(f"==========================================================\n")

    # for triplet_idx, (d1, d2, d3) in enumerate(dopant_triplets, start=1):
    #     print(
    #         f"\n--- [Triplet {triplet_idx}/{len(dopant_triplets)}] Running configuration for {d1} + {d2} + {d3} ---"
    #     )

    #     # Execute 3-dopant generator function dynamically using parameters from DOPANT_SPECS
    #     gather_3_data(
    #         dopant_1=d1,
    #         mmc_dopant_1=DOPANT_SPECS[d1]["mmc"],
    #         num_bond_atoms_1=DOPANT_SPECS[d1]["num_bond_atoms"],
    #         dopant_2=d2,
    #         mmc_dopant_2=DOPANT_SPECS[d2]["mmc"],
    #         num_bond_atoms_2=DOPANT_SPECS[d2]["num_bond_atoms"],
    #         dopant_3=d3,
    #         mmc_dopant_3=DOPANT_SPECS[d3]["mmc"],
    #         num_bond_atoms_3=DOPANT_SPECS[d3]["num_bond_atoms"],
    #         mfr_salt_a=mfr_licl,
    #         mmc_salt_a=mmc_licl,
    #         mfr_salt_b=mfr_kcl,
    #         mmc_salt_b=mmc_kcl,
    #         salt="ClLiK",
    #         wt_percents=wt_percents_list,
    #     )

    # print("\n==========================================================")
    # print(
    #     f"--- ALL {len(dopant_triplets)} DOPANT TRIPLETS SUCCESSFULLY PROCESSED ---"
    # )
    # print("==========================================================")


    # # Get all unique 2-dopant combinations from the 13 available (78 pairs total)
    # dopant_pairs = list(itertools.combinations(DOPANT_SPECS.keys(), 2))
    
    # print(f"==========================================================")
    # print(f"Starting Master Grid Scan: {len(dopant_pairs)} total dopant configurations found.")
    # print(f"Total projected file matrix: {len(dopant_pairs) * 100} simulations.")
    # print(f"==========================================================\n")
    
    # for pair_idx, (d1, d2) in enumerate(dopant_pairs, start=1):
    #     print(f"\n--- [Pair {pair_idx}/{len(dopant_pairs)}] Running configuration for {d1} + {d2} ---")
        
    #     # Execute your main generator function dynamically passing parameters from the dictionary
    #     gather_2_data(
    #         dopant_1 = d1,
    #         mmc_dopant_1 = DOPANT_SPECS[d1]['mmc'],
    #         num_bond_atoms_1 = DOPANT_SPECS[d1]['num_bond_atoms'],
            
    #         dopant_2 = d2,
    #         mmc_dopant_2 = DOPANT_SPECS[d2]['mmc'],
    #         num_bond_atoms_2 = DOPANT_SPECS[d2]['num_bond_atoms'],
            
    #         mfr_salt_a = mfr_licl,
    #         mmc_salt_a = mmc_licl,
    #         mfr_salt_b = mfr_kcl,
    #         mmc_salt_b = mmc_kcl,
    #         salt = 'ClLiK',
    #         wt_percents = wt_percents_list
    #     )
        
    # print("\n==========================================================")
    # print("--- ALL 78 DOPANT COMBINATIONS SUCCESSFULLY PROCESSED ---")
    # print("==========================================================")