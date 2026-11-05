#!/usr/bin/env python3
"""
plasma_diagnostics.py
---------------------
Plasma Diagnostics Module for LIBS Spectra taken in Argon Atmosphere.

Calculates:
  1. Electron Temperature (T_e) via Ar I Boltzmann Plot Method (eV & Kelvin).
  2. Electron Density (N_e) via Ar I 763.51 nm Stark Broadening (Voigt Fit).

Author: LIBS ML Pipeline Component
"""

import h5py
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from scipy.stats import linregress
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, Dict, Any, Optional

# =============================================================================
# SPECTROSCOPIC CONSTANTS & ARGON I REFERENCE DATA (NIST ASD)
# =============================================================================
EV_TO_KELVIN = 11604.518  # 1 eV in Kelvin

# Ar I Line Data for Boltzmann Plot
# Wavelength (nm), Upper Level Energy E_k (eV), Degeneracy g_k, Transition Prob A_ki (s^-1)
AR_I_BOLTZMANN_LINES = [
    {"wl": 696.543, "Ek": 13.328, "gk": 3, "Aki": 6.39e6},
    {"wl": 706.722, "Ek": 13.303, "gk": 5, "Aki": 3.80e6},
    {"wl": 738.398, "Ek": 13.303, "gk": 5, "Aki": 8.47e6},
    {"wl": 750.387, "Ek": 13.480, "gk": 1, "Aki": 4.45e7},
    {"wl": 763.511, "Ek": 13.171, "gk": 5, "Aki": 2.45e7},
    {"wl": 801.479, "Ek": 13.095, "gk": 5, "Aki": 9.28e6},
    {"wl": 811.531, "Ek": 13.076, "gk": 7, "Aki": 3.31e7},
]

# Primary line for Stark Broadening: Ar I 763.51 nm
STARK_LINE_WL = 763.511  # Center wavelength (nm)
STARK_ELECTRON_IMPACT_W = 0.0145  # Electron impact parameter w (nm) at N_ref = 1e16 cm^-3
STARK_N_REF = 1e16  # Reference electron density (cm^-3)


# =============================================================================
# VOIGT PROFILE FITTING FUNCTION
# =============================================================================
def voigt_model(x: np.ndarray, amp: float, x0: float, sigma: float, gamma: float, bg: float) -> np.ndarray:
    """
    Voigt line shape profile: convolution of Gaussian (sigma) and Lorentzian (gamma).
      - sigma: Gaussian HWHM parameter (Instrumental + Doppler broadening)
      - gamma: Lorentzian HWHM parameter (Stark broadening)
    """
    return amp * voigt_profile(x - x0, sigma, gamma) + bg


# =============================================================================
# CORE DIAGNOSTIC FUNCTIONS
# =============================================================================
def estimate_plasma_temperature(
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
    search_window_nm: float = 0.4
) -> Tuple[float, float, float]:
    """
    Estimates Plasma Temperature (T_e) using the Boltzmann Plot Method.

    Formula: ln( (I * lambda) / (g_k * A_ki) ) = - E_k / (k_B * T_e) + C
    
    Returns:
        (T_e_eV, T_e_Kelvin, r_squared_fit)
    """
    x_points = []  # E_k (eV)
    y_points = []  # ln( I*lambda / g_k*A_ki )

    for line in AR_I_BOLTZMANN_LINES:
        center_wl = line["wl"]
        mask = (wavelengths >= center_wl - search_window_nm) & (wavelengths <= center_wl + search_window_nm)
        
        if not np.any(mask):
            continue

        wl_sub = wavelengths[mask]
        spec_sub = spectrum[mask]

        # Local background estimate (minimum of region)
        bg = np.min(spec_sub)
        net_intensity = spec_sub - bg
        
        # Integrate net peak area (Trapezoidal rule)
        peak_area = np.trapz(net_intensity, wl_sub)

        if peak_area > 0:
            # Boltzmann y-value calculation
            y_val = np.log((peak_area * center_wl) / (line["gk"] * line["Aki"]))
            x_points.append(line["Ek"])
            y_points.append(y_val)

    # Require at least 4 valid line fits for reliable linear regression
    if len(x_points) < 4:
        return np.nan, np.nan, 0.0

    x_arr = np.array(x_points)
    y_arr = np.array(y_points)

    # Linear regression: slope m = -1 / T_e (in eV)
    res = linregress(x_arr, y_arr)
    slope = res.slope
    r_squared = res.rvalue ** 2

    # Physical check: slope must be negative for valid LTE temperature
    if slope >= 0 or np.isnan(slope):
        return np.nan, np.nan, r_squared

    t_e_ev = -1.0 / slope
    t_e_k = t_e_ev * EV_TO_KELVIN

    return float(t_e_ev), float(t_e_k), float(r_squared)


def estimate_electron_density(
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
    target_wl: float = STARK_LINE_WL,
    window_nm: float = 1.2,
    w_impact: float = STARK_ELECTRON_IMPACT_W,
    n_ref: float = STARK_N_REF
) -> Tuple[float, float]:
    """
    Estimates Electron Density (N_e) from Stark Broadening via Voigt fitting.

    Formula: N_e = N_ref * (FWHM_Stark / (2 * w))
             where FWHM_Stark = 2 * gamma (Lorentzian HWHM from Voigt fit)

    Returns:
        (N_e_cm3, Lorentzian_FWHM_nm)
    """
    mask = (wavelengths >= target_wl - window_nm) & (wavelengths <= target_wl + window_nm)
    if not np.any(mask):
        return np.nan, np.nan

    wl_sub = wavelengths[mask]
    spec_sub = spectrum[mask]

    # Initial parameter guesses: [amp, x0, sigma, gamma, bg]
    bg_guess = np.min(spec_sub)
    amp_guess = np.max(spec_sub) - bg_guess
    x0_guess = wl_sub[np.argmax(spec_sub)]
    sigma_guess = 0.05  # Initial Gaussian HWHM estimate (instrumental)
    gamma_guess = 0.05  # Initial Lorentzian HWHM estimate (Stark)

    p0 = [amp_guess, x0_guess, sigma_guess, gamma_guess, bg_guess]
    bounds = (
        [0.0, target_wl - 0.3, 0.001, 0.001, 0.0],
        [np.inf, target_wl + 0.3, 0.5, 0.5, np.inf]
    )

    try:
        popt, _ = curve_fit(voigt_model, wl_sub, spec_sub, p0=p0, bounds=bounds, maxfev=1500)
        _, _, _, gamma_fit, _ = popt

        # Lorentzian FWHM = 2 * gamma (HWHM)
        fwhm_stark = 2.0 * gamma_fit

        # Calculate N_e (cm^-3)
        n_e_cm3 = n_ref * (fwhm_stark / (2.0 * w_impact))

        return float(n_e_cm3), float(fwhm_stark)

    except Exception:
        # Fit failed or non-convergent peak
        return np.nan, np.nan


# =============================================================================
# SINGLE SPECTRUM WORKER FUNCTION FOR MULTIPROCESSING
# =============================================================================
def _process_single_shot(args: Tuple[np.ndarray, np.ndarray]) -> Tuple[float, float, float, float]:
    wavelengths, spectrum = args
    t_ev, _, r2 = estimate_plasma_temperature(wavelengths, spectrum)
    n_e, _ = estimate_electron_density(wavelengths, spectrum)
    return t_ev, r2, n_e


# =============================================================================
# HDF5 DATASET BATCH PROCESSING PIPELINE
# =============================================================================
def append_plasma_diagnostics_to_hdf5(
    h5_path: str,
    output_h5_path: Optional[str] = None,
    num_workers: int = 8
) -> None:
    """
    In-place or output enrichment of HDF5 dataset with 'meta/plasma_temp_eV' 
    and 'meta/electron_density_cm3'.
    """
    in_path = Path(h5_path)
    out_path = Path(output_h5_path) if output_h5_path else in_path

    print(f"Loading spectral dataset from: {in_path}")
    with h5py.File(in_path, 'r') as hf:
        X = hf['X'][:]
        wavelengths = hf['wavelengths'][:]
        
        # Load existing datasets if writing to same file
        all_keys = list(hf.keys())

    num_spectra = X.shape[0]
    print(f"Extracted {num_spectra} spectra across {len(wavelengths)} wavelength channels.")
    print(f"Processing diagnostics on {num_workers} CPU workers...")

    # Build argument tuples for parallel pool
    arg_list = [(wavelengths, X[i]) for i in range(num_spectra)]

    t_ev_arr = np.zeros(num_spectra, dtype=np.float32)
    r2_arr = np.zeros(num_spectra, dtype=np.float32)
    n_e_arr = np.zeros(num_spectra, dtype=np.float32)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_process_single_shot, arg_list, chunksize=500))

    for i, (t_ev, r2, n_e) in enumerate(results):
        t_ev_arr[i] = t_ev
        r2_arr[i] = r2
        n_e_arr[i] = n_e

    # Report Stats
    valid_t = ~np.isnan(t_ev_arr)
    valid_n = ~np.isnan(n_e_arr)
    print("\n--- Diagnostic Calculation Results ---")
    print(f"Valid T_e Fits: {np.sum(valid_t)} / {num_spectra} ({100*np.mean(valid_t):.1f}%)")
    if np.any(valid_t):
        print(f"  Mean T_e: {np.nanmean(t_ev_arr):.3f} eV ({np.nanmean(t_ev_arr)*EV_TO_KELVIN:.0f} K)")
    print(f"Valid N_e Fits: {np.sum(valid_n)} / {num_spectra} ({100*np.mean(valid_n):.1f}%)")
    if np.any(valid_n):
        print(f"  Mean N_e: {np.nanmean(n_e_arr):.3e} cm^-3")

    # Write back to HDF5
    print(f"\nWriting calculated metadata vectors to: {out_path}")
    mode = 'a' if out_path == in_path else 'w'
    with h5py.File(out_path, mode) as hf:
        if 'meta' not in hf:
            meta_grp = hf.create_group('meta')
        else:
            meta_grp = hf['meta']

        # Overwrite or create datasets
        for key, arr in [('plasma_temp_eV', t_ev_arr), ('boltzmann_r2', r2_arr), ('electron_density_cm3', n_e_arr)]:
            if key in meta_grp:
                del meta_grp[key]
            meta_grp.create_dataset(key, data=arr, compression="gzip")

    print("Pipeline Diagnostics Successfully Enriched!")


# =============================================================================
# CLI / TESTING ENTRYPOINT
# =============================================================================
if __name__ == "__main__":
    # Example standalone run on dataset
    target_h5 = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/recleaned/exp-syn_TD14_TEN.h5"
    if Path(target_h5).exists():
        append_plasma_diagnostics_to_hdf5(target_h5, num_workers=8)
    else:
        print(f"Target HDF5 file not found at: {target_h5}")