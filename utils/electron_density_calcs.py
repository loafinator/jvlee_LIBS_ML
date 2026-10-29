from __future__ import annotations

"""
LIBS > utils > electron_density_calcs.py

Estimate the electron densities of spectral signatures from LIBS
experiments for machine learning.
"""

from pathlib import Path
import h5py
import numpy as np
from lmfit.models import LinearModel, VoigtModel


def calc_rho_e(
    h5_input: Path | str,
    h5_out: Path | str,
    target_center: float = 610.36,
    window_width: float = 5.0,  # Wavelength span in nm
    electron_impact_w: float = 0.02,  # Impact parameter w (nm) for target line
    sigma_inst: float = 0.025,  # Fixed instrumental Gaussian width (nm)
) -> np.ndarray:
    """Calculates electron density (N_e) from Stark line broadening in LIBS spectra.

    Saves results to an output HDF5 file.
    """
    h5_input_path = Path(h5_input).resolve()
    h5_out_path = Path(h5_out).resolve()

    # 1. Load data from HDF5
    with h5py.File(h5_input_path, "r") as f_in:
        wavelength = np.array(f_in["wavelengths"])
        intensity = np.array(f_in["intensity"])

    # Ensure intensity is 2D: (n_shots, n_channels)
    if intensity.ndim == 1:
        intensity = intensity[np.newaxis, :]

    n_shots = intensity.shape[0]
    electron_densities = np.full(n_shots, np.nan)

    # 2. Slice spectral region using boolean masking (nm space, not array indices)
    mask = (wavelength >= target_center - window_width / 2) & (
        wavelength <= target_center + window_width / 2
    )
    x_sub = wavelength[mask]

    # 3. Define local multi-peak composite model
    model = (
        LinearModel(prefix="bg_")
        + VoigtModel(prefix="target_")
        + VoigtModel(prefix="interf1_")
        + VoigtModel(prefix="interf2_")
    )

    # 4. Iterate over shots and fit spectra
    for i in range(n_shots):
        y_sub = intensity[i, mask]

        params = model.make_params()

        # Center constraints (+/- 0.05 nm)
        params["target_center"].set(
            value=target_center,
            min=target_center - 0.05,
            max=target_center + 0.05,
        )
        params["interf1_center"].set(value=609.82, min=609.77, max=609.87)
        params["interf2_center"].set(value=610.85, min=610.80, max=610.90)

        # Fix Gaussian width to instrument function
        params["target_sigma"].set(value=sigma_inst, vary=False)
        params["interf1_sigma"].set(value=sigma_inst, vary=False)
        params["interf2_sigma"].set(value=sigma_inst, vary=False)

        # Initial amplitude and Lorentzian gamma guesses for stability
        y_max = np.max(y_sub) if len(y_sub) > 0 else 1.0
        params["target_amplitude"].set(value=y_max * 0.5, min=0)
        params["interf1_amplitude"].set(value=y_max * 0.2, min=0)
        params["interf2_amplitude"].set(value=y_max * 0.2, min=0)
        params["target_gamma"].set(value=0.05, min=0.001, max=1.0)

        try:
            result = model.fit(y_sub, params, x=x_sub)

            # Quality check: ensure fit converged with reasonable R^2
            if result.success and result.rsquared > 0.85:
                # lmfit Voigt gamma parameter is HWHM of Lorentzian component
                stark_fwhm = 2.0 * result.params["target_gamma"].value

                # Griem Stark formula: N_e = 10^16 * (Delta_lambda_Stark / 2w)
                n_e = 1e16 * (stark_fwhm / (2.0 * electron_impact_w))
                electron_densities[i] = n_e

        except Exception:
            # Bad fit or numerical error; leave as NaN for dataset cleaning downstream
            continue

    # 5. Write electron density dataset to output HDF5
    with h5py.File(h5_out_path, "a") as f_out:
        if "electron_density" in f_out:
            del f_out["electron_density"]
        f_out.create_dataset("electron_density", data=electron_densities)

    return electron_densities