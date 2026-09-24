from __future__ import annotations

"""

jvlee_LIBS_ML > utils > saha_boltzmann.py

"""

# region Imports
import os
import h5py

import numpy as np
import pandas as pd

from pathlib import Path
from utils import hf_get
from scipy import sparse
from scipy.sparse.linalg import spsolve
# endregion

# region Gloabal Vars
kb = 8.617333262145e-5  # Boltzmann constant in eV/K
# endregion

def estb_baseline(
        file_path: Path,
        lam: float = 1e5,
        p: float = 0.001,
        niter: int = 10
):
    with h5py.File(file_path, 'r') as hf:
        if 'spectra' in hf:
            spectra = hf_get(hf, 'spectra')
        else:
            raise KeyError("'spectra' not found in dataset")

    spectra = np.asarray(spectra)

    if spectra.ndim == 1:
        return _als_baseline_1d(spectra, lam, p, niter)

    baselines = np.zeros_like(spectra)
    for i in range(spectra.shape[0]):
        baselines[i] = _als_baseline_1d(spectra[i], lam, p, niter)

    return baselines

def single_t_pert(
    file_path: Path,
    output_path: Path,
    ej_map: np.ndarray,
    T0: float = 10000.0,
    T1: float = 10500.0,
    noise_std: float = 0.005,
) -> None:
    """
    Applies Saha-Boltzmann temperature perturbation to spectra in an HDF5 file
    and saves the augmented dataset to output_path.
    
    Parameters:
    - file_path: Path to source .h5 file containing 'wavelengths' and 'spectra'
    - output_path: Path to save augmented .h5 file
    - ej_map: 1D array (same length as wavelengths) containing upper state energy Ej (eV).
              Use 0.0 for wavelengths without assigned lines.
    - T0: Baseline plasma temperature (K)
    - T1: Target perturbed plasma temperature (K)
    - noise_std: Standard deviation scale for synthetic readout noise
    """
    with h5py.File(file_path, 'r') as hf:
        wavelengths = hf_get(hf, 'wavelengths') if 'wavelengths' in hf else None
        spectra = hf_get(hf, 'spectra') if 'spectra' in hf else None

        if wavelengths is None or spectra is None:
            raise KeyError("Both 'wavelengths' and 'spectra' must exist in dataset!")

        wavelengths = np.asarray(wavelengths)
        spectra = np.asarray(spectra)

    # 1. Calculate Baseline Continuum
    baseline = estb_baseline(file_path)

    # 2. Isolate Peaks (Ensure positive values)
    peaks = np.maximum(spectra - baseline, 0)

    # 3. Compute Saha-Boltzmann Scaling Factor
    # Ratio = exp( - (Ej / kB) * (1/T1 - 1/T0) )
    delta_inverse_T = (1.0 / T1) - (1.0 / T0)
    scaling_vector = np.exp(-(ej_map / kb) * delta_inverse_T)

    # 4. Perturb Peaks
    perturbed_peaks = peaks * scaling_vector

    # 5. Add Synthetic Noise (scaled to peak intensity)
    max_val = np.max(spectra)
    noise = np.random.normal(0, noise_std * max_val, size=spectra.shape)

    # 6. Reconstruct Augmented Spectra
    augmented_spectra = perturbed_peaks + baseline + noise

    # 7. Write to Output HDF5 File
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, 'w') as hf_out:
        hf_out.create_dataset('wavelengths', data=wavelengths)
        hf_out.create_dataset('spectra', data=augmented_spectra)
        hf_out.create_dataset('original_spectra', data=spectra)
        hf_out.attrs['T0'] = T0
        hf_out.attrs['T1'] = T1

    print(f"Augmented spectra successfully saved to: {output_path}")

def _als_baseline_1d(y: np.ndarray, lam: float, p: float, niter: int) -> np.ndarray:
    """Helper function to perform ALS baseline fit on a single 1D array."""
    L = len(y)
    
    # Construct second-difference matrix D using spdiags
    e = np.ones(L)
    data = np.array([e, -2 * e, e])
    offsets = np.array([0, 1, 2])
    D = sparse.spdiags(data, offsets, L - 2, L).T  # Shape: (L, L - 2)
    
    w = np.ones(L)
    z = np.copy(y)
    
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z_sol = spsolve(Z, w * y)
        z = np.asarray(z_sol, dtype=np.float32)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z



if __name__ == "__main__":
    print('hi')