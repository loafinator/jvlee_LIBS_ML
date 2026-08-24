from __future__ import annotations
"""
jvlee_LIBS_ML > utils > NIST_data.py
"""
print('NIST_data.py loading ...')

# region Imports
import time
import requests
import re
import itertools
import pandas as pd
import numpy as np
from pathlib import Path
# from utils import enrich_file_with_metadata
# endregion

# region NIST URL
# NIST_LIBS_URL = "https://physics.nist.gov/cgi-bin/ASD/libs-1.pl"
NIST_LIBS_URL = "https://physics.nist.gov/cgi-bin/ASD/libs-form.cgi"
# endregion

def fetch_nist_libs_data(comp_string: str, plasma_params: dict) -> pd.DataFrame | None:
    """
    Sends a query to the NIST LIBS database and parses the hidden JavaScript 
    array into a discrete peak list DataFrame.
    """
    elements = [pair.split(":")[0] for pair in comp_string.split(";")]

    payload = {
        **plasma_params,
        "composition": comp_string,
        "spectra": ",".join(f"{el}0-2" for el in elements),
        "mytext[]": elements,
        "myperc[]": [val.split(":")[1] for val in comp_string.split(";")]
    }
    
    try:
        response = requests.post(NIST_LIBS_URL, data=payload, timeout=30)
        if response.status_code != 200:
            print(f' -> Server error: Received status code {response.status_code}')
            return None
            
        html_content = response.text
        
        # Pull out the embedded raw javascript data array loop
        match = re.search(r"var lines = \[(.*?)\];", html_content, re.DOTALL)
        if not match:
            print(' -> Warning: Response received, but "var lines" data block wasn\'t found.')
            return None
            
        raw_data_block = match.group(1)
        row_strings = re.findall(r"\[([^\]]+)\]", raw_data_block)
        
        headers = [
            "Wavelength (nm)", 
            "Intensity", 
            "Energy Level 1", 
            "Element Code 1", 
            "Element Code 2", 
            "Energy Level 2"
        ]
        
        parsed_rows = []
        for row_str in row_strings:
            numeric_values = [float(val) for val in row_str.split(",")]
            parsed_rows.append(numeric_values)
            
        return pd.DataFrame(parsed_rows, columns=headers)
        
    except Exception as e:
        print(f' -> Network failure: {str(e)}')
        return None


def generate_simple_intensity_profile(df_discrete: pd.DataFrame, low_w: float = 200.0, upp_w: float = 1000.0, step: float = 0.1, resolution: float = 1000.0) -> pd.DataFrame:
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

def _fetch_single_nist_chunk(
    comp_string: str,
    plasma_params: dict,
    low_w: float,
    upp_w: float,
    max_retries: int = 3
) -> pd.DataFrame | None:
    """Queries NIST ASD LIBS endpoint using tuple parameter pairs and proper User-Agent headers."""
    elements = [pair.split(":")[0].strip() for pair in comp_string.split(";")]
    percents = [pair.split(":")[1].strip() for pair in comp_string.split(";")]
    
    spectra_str = ",".join(f"{el}0-2" for el in elements)

    # Standard browser headers to ensure connection approval
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://physics.nist.gov/PhysRefData/ASD/plots/libs.html"
    }

    # Complete NIST LIBS parameter payload formatted as tuples
    payload = [
        ("element", "All"),
        ("low_w", f"{low_w:.1f}"),
        ("upp_w", f"{upp_w:.1f}"),
        ("limits_type", "0"),
        ("unit", "1"),
        ("format", "0"),
        ("line_out", "0"),
        ("remove_d", "on"),
        ("A_unit", "0"),
        ("resolution", str(plasma_params.get("resolution", "1000"))),
        ("temp", str(plasma_params.get("temp", "1.0"))),
        ("eden", str(plasma_params.get("eden", "1e17"))),
        ("maxcharge", str(plasma_params.get("maxcharge", "2"))),
        ("min_rel_int", str(plasma_params.get("min_rel_int", "0.1"))),
        ("show_av", "2"),
        ("libs", "1"),
        ("spectra", spectra_str),
    ]

    # Append elemental composition array arguments
    for el, perc in zip(elements, percents):
        payload.append(("mytext[]", el))
        payload.append(("myperc[]", perc))

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(NIST_LIBS_URL, data=payload, headers=headers, timeout=60)
            
            if response.status_code in (503, 504):
                print(f'   [Window {low_w}-{upp_w}nm] Server busy ({response.status_code}). Retry {attempt}/{max_retries} in 5s...')
                time.sleep(5)
                continue
                
            if response.status_code != 200:
                print(f'   [Window {low_w}-{upp_w}nm] Server error: HTTP {response.status_code}')
                return None
                
            match = re.search(r"var lines = \[(.*?)\];", response.text, re.DOTALL)
            if not match:
                if "Input Error" in response.text:
                    err_match = re.search(r"<h3>(.*?)</h3>", response.text, re.DOTALL | re.IGNORECASE)
                    msg = err_match.group(1).strip() if err_match else "Form submission rejected."
                    print(f"   [Window {low_w}-{upp_w}nm] NIST Input Error: {msg}")
                else:
                    print(f"   [Window {low_w}-{upp_w}nm] No spectral lines block found.")
                return pd.DataFrame()
                
            raw_data_block = match.group(1).strip()
            if not raw_data_block:
                return pd.DataFrame()

            row_strings = re.findall(r"\[([^\]]+)\]", raw_data_block)
            if not row_strings:
                return pd.DataFrame()

            headers_list = [
                "Wavelength (nm)", "Intensity", "Energy Level 1", 
                "Element Code 1", "Element Code 2", "Energy Level 2"
            ]
            
            parsed_rows = [[float(val.strip()) for val in row.split(",")] for row in row_strings]
            return pd.DataFrame(parsed_rows, columns=headers_list)
            
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            print(f'   [Window {low_w}-{upp_w}nm] Connection dropped. Retry {attempt}/{max_retries} in 5s...')
            time.sleep(5)
        except Exception as e:
            print(f'   [Window {low_w}-{upp_w}nm] Exception: {str(e)}')
            return None

    return None

def fetch_nist_libs_data_new(comp_string: str, plasma_params: dict, num_chunks: int = 2) -> pd.DataFrame | None:
    """
    Splits the main wavelength range into smaller windows, queries NIST sequentially,
    and returns a consolidated discrete peak list.
    """
    start_w = float(plasma_params["low_w"])
    end_w = float(plasma_params["upp_w"])
    
    # Generate chunk boundaries (e.g., [200.0, 600.0, 1000.0])
    boundaries = np.linspace(start_w, end_w, num_chunks + 1)
    
    collected_dfs = []
    
    for i in range(num_chunks):
        low_w = boundaries[i]
        upp_w = boundaries[i + 1]
        
        df_chunk = _fetch_single_nist_chunk(comp_string, plasma_params, low_w, upp_w)
        
        if df_chunk is None:
            # If a chunk fails completely, abort the run so bad/incomplete data isn't saved
            return None
            
        if not df_chunk.empty:
            collected_dfs.append(df_chunk)
            
        # Short breather between internal chunk requests
        time.sleep(1.5)

    if not collected_dfs:
        return pd.DataFrame()

    # Combine all wavelength sub-tables and deduplicate any boundary boundary overlaps
    full_df = pd.concat(collected_dfs, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=["Wavelength (nm)", "Intensity"])
    return full_df

# file path: /lustre/home/leejv2/git_repos/jvlee_LIBS_ML/utils/NIST_data.py
output_dir = Path(__file__).parent.parent / "LIBS" / "NIST_data"
output_dir.mkdir(parents=True, exist_ok=True)

plasma_params = {
    "low_w": "200",         # Wavelength Minimum (nm)
    "upp_w": "600",        # Wavelength Maximum (nm)
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

# Material constants (g/mol)
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
    'UCl3':  {'mmc': 344.388, 'num_bond_atoms': 3}
}
mmc_cecl3 = 246.475
mmc_smcl3 = 256.72
mmc_lacl3 = 245.264
mmc_ndcl3 = 250.601
mmc_cscl = 168.358
mmc_srcl2 = 158.53
mmc_bacl2 = 208.233
mmc_ycl3 = 195.265
mmc_fecl2 = 126.751
mmc_crcl2 = 122.902
mmc_nicl2 = 129.599
mmc_mncl2 = 125.844
mmc_ucl3 = 344.388
mmc_ = 1
mmc_ = 1

mmc_lif = 25.939
mmc_bef = 47.009

mmc_licl = 42.39
mmc_kcl = 74.55

mfr_licl = 0.59
mfr_kcl = 0.41
mmc_eutectice = (mfr_licl * mmc_licl) + (mfr_kcl * mmc_kcl)

wt_percents_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]



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

if __name__ == "__main__":
    # Get all unique 2-dopant combinations from the 13 available (78 pairs total)
    dopant_pairs = list(itertools.combinations(DOPANT_SPECS.keys(), 2))
    
    print(f"==========================================================")
    print(f"Starting Master Grid Scan: {len(dopant_pairs)} total dopant configurations found.")
    print(f"Total projected file matrix: {len(dopant_pairs) * 100} simulations.")
    print(f"==========================================================\n")
    
    for pair_idx, (d1, d2) in enumerate(dopant_pairs, start=1):
        print(f"\n--- [Pair {pair_idx}/{len(dopant_pairs)}] Running configuration for {d1} + {d2} ---")
        
        # Execute your main generator function dynamically passing parameters from the dictionary
        gather_2_data(
            dopant_1 = d1,
            mmc_dopant_1 = DOPANT_SPECS[d1]['mmc'],
            num_bond_atoms_1 = DOPANT_SPECS[d1]['num_bond_atoms'],
            
            dopant_2 = d2,
            mmc_dopant_2 = DOPANT_SPECS[d2]['mmc'],
            num_bond_atoms_2 = DOPANT_SPECS[d2]['num_bond_atoms'],
            
            mfr_salt_a = mfr_licl,
            mmc_salt_a = mmc_licl,
            mfr_salt_b = mfr_kcl,
            mmc_salt_b = mmc_kcl,
            salt = 'ClLiK',
            wt_percents = wt_percents_list
        )
        
    print("\n==========================================================")
    print("--- ALL 78 DOPANT COMBINATIONS SUCCESSFULLY PROCESSED ---")
    print("==========================================================")