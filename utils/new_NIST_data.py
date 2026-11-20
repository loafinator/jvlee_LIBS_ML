from __future__ import annotations

print('new_NIST_data.py loading ...')

import os
import h5py
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Optional, Dict
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.optimize import least_squares


def clean_libs_spectrum(spectra_matrix: np.ndarray, window_size: int = 10, chunk: int = 5000) -> np.ndarray:
    """Applies rolling baseline removal and area normalization."""
    out = np.empty(spectra_matrix.shape, dtype=np.float32)
    for a in range(0, len(spectra_matrix), chunk):
        X = spectra_matrix[a:a + chunk].astype(np.float32)
        base = uniform_filter1d(
            minimum_filter1d(X, size=window_size, axis=1, mode='nearest'),
            size=window_size, axis=1, mode='nearest'
        )
        X = np.maximum(X - base, 0.0)
        area = X.sum(axis=1, keepdims=True)
        out[a:a + chunk] = np.divide(X, area, out=X, where=area > 0)
    return out


def load_single_element_library(input_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Reads single-element NIST CSVs and normalizes each to 100% pure element basis."""
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

        wavelength_cols.sort(key=lambda c: float(c))

        conc_cols = [c for c in df.columns if c.startswith("conc_") and c.endswith("_wt%")]
        active_element = None

        for c_col in conc_cols:
            if (df[c_col] > 0).any():
                formula = c_col.replace("conc_", "").replace("_wt%", "").replace("%", "").strip("_")
                if sum(1 for ch in formula if ch.isupper()) > 1:
                    continue
                active_element = formula

        if active_element is None:
            continue

        spectral_data = df[wavelength_cols].values.astype(np.float32)
        mean_spectrum = np.mean(spectral_data, axis=0)

        conc_val = df[f"conc_{active_element}_wt%"].iloc[0]
        if conc_val > 0:
            mean_spectrum = mean_spectrum / (conc_val / 100.0)

        element_spectra[active_element] = mean_spectrum

    return element_spectra, np.array([float(w) for w in wavelength_cols], dtype=np.float32)  # type: ignore


def calibrate_basis_to_experimental(
    basis_matrix: np.ndarray,       # [E, W] raw NIST basis already on exp_wl grid
    all_elements: list[str],        # e.g. ['Li', 'K', 'Cl', 'Ce', 'Gd', 'Sm', 'U']
    X_exp_train: np.ndarray,        # [N_train, W] non-lockbox experimental spectra (already cleaned)
    y_exp_train: np.ndarray,        # [N_train, E] experimental wt% matching all_elements
    n_knots: int = 16,
) -> np.ndarray:
    """
    Fits per-element sensitivity scalars and a smooth 16-knot wavelength response
    curve R(lambda) on non-lockbox experimental data so NIST basis spectra match
    experimental peak heights across UV, VIS, and NIR.
    """
    rng = np.random.default_rng(42)
    selected_indices = []
    dopant_cols = [all_elements.index(d) for d in ['Ce', 'Gd', 'Sm', 'U'] if d in all_elements]

    # Sample up to 1,000 shots per dopant (takes ALL non-lockbox shots for rare dopants like Gd)
    for col_i in dopant_cols:
        pos_idx = np.where(y_exp_train[:, col_i] > 0.05)[0]
        if len(pos_idx) > 0:
            take = min(1000, len(pos_idx))
            selected_indices.append(rng.choice(pos_idx, size=take, replace=False))

    # Also include up to 1,000 pure-host (blank) shots to anchor Li, K, Cl baseline
    host_only_idx = np.where(np.all(y_exp_train[:, dopant_cols] <= 0.05, axis=1))[0]
    if len(host_only_idx) > 0:
        selected_indices.append(rng.choice(host_only_idx, size=min(1000, len(host_only_idx)), replace=False))

    idx = np.unique(np.concatenate(selected_indices))
    C = (y_exp_train[idx] / 100.0).astype(np.float64)   # [N, E]
    X_target = X_exp_train[idx].astype(np.float64)       # [N, W]
    E, W = basis_matrix.shape

    # Normalize each element's raw basis to max=1.0 first (fixes 2.41e9 vs 4.68e3 scale gap)
    norm_basis = basis_matrix.astype(np.float64).copy()
    for e in range(E):
        peak = norm_basis[e].max()
        if peak > 0:
            norm_basis[e] /= peak

    knot_x = np.linspace(0, W - 1, n_knots)
    chan_x = np.arange(W, dtype=np.float64)
    targ_norm = X_target / np.maximum(np.linalg.norm(X_target, axis=1, keepdims=True), 1e-8)

    def residuals(params: np.ndarray) -> np.ndarray:
        log_elem_scale = params[:E]
        log_irf_knots = params[E:]

        elem_scale = np.exp(log_elem_scale)[:, None]                   # [E, 1]
        irf_curve = np.exp(np.interp(chan_x, knot_x, log_irf_knots))   # [W]
        cal_basis = norm_basis * elem_scale * irf_curve[None, :]       # [E, W]

        pred_raw = (C @ cal_basis).astype(np.float32)
        pred_clean = clean_libs_spectrum(pred_raw).astype(np.float64)
        pred_norm = pred_clean / np.maximum(np.linalg.norm(pred_clean, axis=1, keepdims=True), 1e-8)
        return (pred_norm - targ_norm).ravel()

    p0 = np.zeros(E + n_knots, dtype=np.float64)
    res = least_squares(residuals, p0, method="trf", bounds=(-10.0, 10.0), diff_step=1e-3, max_nfev=60, verbose=1)

    elem_scale = np.exp(res.x[:E])[:, None]
    irf_curve = np.exp(np.interp(chan_x, knot_x, res.x[E:]))
    calibrated_basis = (norm_basis * elem_scale * irf_curve[None, :]).astype(np.float32)

    print("\nCalibrated per-element sensitivity scalars:")
    for name, sc in zip(all_elements, elem_scale.ravel()):
        print(f"  [{name:>2}]: {sc:.4f}")

    return calibrated_basis


def synthetic_data_generator(
    nist_input_dir: str | Path,
    exp_h5_path: str | Path,
    output_dir: str | Path,
    num_samples: int = 200000,
    dopant_names: list[str] = ['Ce', 'Gd', 'Sm', 'U'],
    host_sigma: float = 0.015,
    max_dopant_wt: float = 6.5,
    max_total_dopant_salt_frac: float = 0.20,
    category_probs: Optional[Dict[int, float]] = None,
    seed: int = 42,
) -> tuple[Path, Path]:
    """
    Generates calibrated, baseline-cleaned 4-dopant synthetic spectra aligned
    to the experimental wavelength grid and non-lockbox experimental response.
    """
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(nist_input_dir).resolve()

    # 1. Load non-lockbox experimental data for grid alignment & basis calibration
    with h5py.File(exp_h5_path, 'r') as hf_exp:
        X_exp = hf_exp['X'][:]  # type: ignore
        y_exp = hf_exp['y'][:]  # type: ignore
        exp_wl = hf_exp['wavelengths'][:].astype(np.float32)  # type: ignore
        target_elements = [
            e.decode('utf-8') if isinstance(e, bytes) else e
            for e in hf_exp['target_elements'][:]  # type: ignore
        ]
        if 'is_lockbox' not in hf_exp:
            raise KeyError(f"Missing 'is_lockbox' in {exp_h5_path}.")
        is_lockbox = hf_exp['is_lockbox'][:].astype(bool)  # type: ignore

    train_mask = ~is_lockbox # type: ignore
    X_exp_train = X_exp[train_mask] # type: ignore
    y_exp_train = y_exp[train_mask] # type: ignore
    print(f"Loaded {len(X_exp_train)} non-lockbox experimental shots for calibration (targets={target_elements}).") # type: ignore

    # 2. Load NIST singles and resample the 7 basis curves onto exp_wl
    print("Extracting single-element basis spectra from NIST CSV library...")
    elem_library, nist_wl = load_single_element_library(input_path)

    raw_basis = np.zeros((len(target_elements), len(exp_wl)), dtype=np.float32)
    for row_idx, elem in enumerate(target_elements):
        if elem not in elem_library:
            raise KeyError(f"Element '{elem}' not found in {input_path}")
        raw_basis[row_idx, :] = np.interp(exp_wl, nist_wl, elem_library[elem], left=0.0, right=0.0)

    # 3. Calibrate NIST basis against non-lockbox experimental spectra
    print("Calibrating NIST basis sensitivities and R(lambda) to non-lockbox experimental data...")
    calibrated_basis = calibrate_basis_to_experimental(
        basis_matrix=raw_basis,
        all_elements=target_elements,
        X_exp_train=X_exp_train, # type: ignore
        y_exp_train=y_exp_train, # type: ignore
        n_knots=16,
    )

    # 4. Sample realistic 4-dopant LiCl-KCl compositions with exact mass balance
    if category_probs is None:
        category_probs = {
            0: 0.10,  # 10% Pure LiCl-KCl host
            1: 0.45,  # 45% Single dopant
            2: 0.30,  # 30% Binary mixtures
            3: 0.15,  # 15% Ternary mixtures
        }

    k_options = list(category_probs.keys())
    k_weights = np.array([category_probs[k] for k in k_options], dtype=np.float64)
    k_weights /= np.sum(k_weights)
    categories = rng.choice(k_options, size=num_samples, p=k_weights)

    num_dopants = len(dopant_names)
    dopant_cl_valency = {'Ce': 3.0, 'Gd': 3.0, 'Sm': 3.0, 'U': 3.0}
    dopant_molar_mass = {'Ce': 140.12, 'Gd': 157.25, 'Sm': 150.36, 'U': 238.03}
    M_CL = 35.453

    cl_per_metal_mass = np.array(
        [(dopant_cl_valency[d] * M_CL) / dopant_molar_mass[d] for d in dopant_names],
        dtype=np.float64,
    )

    dopant_metal_wt = np.zeros((num_samples, num_dopants), dtype=np.float64)
    for i in range(num_samples):
        k = int(categories[i])
        if k == 0:
            continue
        active_indices = rng.choice(num_dopants, size=k, replace=False)
        for d_idx in active_indices:
            if rng.random() < 0.5:
                wt_val = 10.0 ** rng.uniform(-3.0, np.log10(0.5))
            else:
                wt_val = rng.uniform(0.5, max_dopant_wt)
            dopant_metal_wt[i, d_idx] = wt_val / 100.0

    # Enforce exact mass balance when total dopant chloride salt exceeds max_total_dopant_salt_frac
    dopant_cl_wt = np.sum(dopant_metal_wt * cl_per_metal_mass, axis=1, keepdims=True)
    total_dopant_salt_wt = np.sum(dopant_metal_wt, axis=1, keepdims=True) + dopant_cl_wt
    over_cap = total_dopant_salt_wt > max_total_dopant_salt_frac
    if np.any(over_cap):
        scale = np.where(over_cap, max_total_dopant_salt_frac / np.maximum(total_dopant_salt_wt, 1e-12), 1.0)
        dopant_metal_wt *= scale
        dopant_cl_wt *= scale
        total_dopant_salt_wt *= scale

    remaining_matrix_wt = 1.0 - total_dopant_salt_wt

    li_rel_target = 0.0966 / (0.0966 + 0.2150)
    li_rel_perturbed = np.clip(rng.normal(loc=li_rel_target, scale=host_sigma, size=(num_samples, 1)), 0.20, 0.45)
    k_rel_perturbed = 1.0 - li_rel_perturbed

    m_li = li_rel_perturbed * 6.94
    m_k = k_rel_perturbed * 39.10
    m_cl_host = 35.453
    m_host_total = m_li + m_k + m_cl_host

    host_li_wt = remaining_matrix_wt * (m_li / m_host_total)
    host_k_wt = remaining_matrix_wt * (m_k / m_host_total)
    host_cl_wt = remaining_matrix_wt * (m_cl_host / m_host_total)
    total_cl_wt = host_cl_wt + dopant_cl_wt

    compositions_dict = {
        'Li': host_li_wt.flatten() * 100.0,
        'K': host_k_wt.flatten() * 100.0,
        'Cl': total_cl_wt.flatten() * 100.0,
    }
    for idx, name in enumerate(dopant_names):
        compositions_dict[name] = dopant_metal_wt[:, idx] * 100.0

    compositions_df = pd.DataFrame(compositions_dict)
    csv_out_path = output_dir / "calibrated_4dopant_compositions_wt_pct.csv"
    compositions_df.to_csv(csv_out_path, index=False)
    print(f"Saved calibrated compositions manifest to: {csv_out_path}")

    # 5. Generate calibrated spectra, apply clean_libs_spectrum, and save HDF5
    h5_out_path = output_dir / "calibrated_4dopant_synthetic_200k.h5"
    num_wavelengths = len(exp_wl)

    with h5py.File(h5_out_path, "w") as hf:
        hf.create_dataset("wavelengths", data=exp_wl)
        spectra_ds = hf.create_dataset(
            "spectra",
            shape=(num_samples, num_wavelengths),
            dtype=np.float32,
            chunks=(2000, num_wavelengths),
            compression="gzip",
            compression_opts=3,
        )

        chunk_size = 20000
        for start_idx in range(0, num_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, num_samples)
            comp_chunk = (compositions_df.iloc[start_idx:end_idx][target_elements].values / 100.0).astype(np.float32)
            raw_chunk = comp_chunk @ calibrated_basis
            spectra_ds[start_idx:end_idx, :] = clean_libs_spectrum(raw_chunk)
            print(f"  Processed and saved chunk {start_idx} to {end_idx} / {num_samples}")

    print(f"✅ Saved calibrated synthetic dataset to {h5_out_path}")
    return h5_out_path, csv_out_path


def final_combo(
    exp_h5_path: Path | str,
    syn_h5_path: Path | str,
    syn_csv_path: Path | str,
    merged_h5_out: Path | str,
):
    """Combines experimental HDF5 with pre-calibrated, preprocessed synthetic HDF5."""
    exp_h5_path = Path(exp_h5_path)
    syn_h5_path = Path(syn_h5_path)
    syn_csv_path = Path(syn_csv_path)
    merged_h5_out = Path(merged_h5_out)

    with h5py.File(exp_h5_path, 'r') as hf_exp:
        X_exp = hf_exp['X'][:]  # type: ignore
        y_exp = hf_exp['y'][:]  # type: ignore
        exp_wl = hf_exp['wavelengths'][:]  # type: ignore
        target_elements = [
            e.decode('utf-8') if isinstance(e, bytes) else e
            for e in hf_exp['target_elements'][:]  # type: ignore
        ]
        groups_exp = hf_exp['groups'][:] if 'groups' in hf_exp else np.zeros(len(X_exp), dtype=np.int32)  # type: ignore
        if 'is_lockbox' not in hf_exp:
            raise KeyError(f"Missing 'is_lockbox' in {exp_h5_path}.")
        is_lockbox_exp = hf_exp['is_lockbox'][:].astype(bool)  # type: ignore

    print(f"Loaded Experimental: X={X_exp.shape}, y={y_exp.shape}, targets={target_elements}") # type: ignore

    with h5py.File(syn_h5_path, 'r') as hf_syn:
        X_syn = hf_syn['spectra'][:]  # type: ignore
        syn_wl = hf_syn['wavelengths'][:]  # type: ignore

    if not np.allclose(syn_wl, exp_wl): # type: ignore
        raise ValueError("Synthetic and experimental wavelength grids do not match!")

    df_syn = pd.read_csv(syn_csv_path)
    y_syn = df_syn[target_elements].to_numpy(dtype=np.float32)

    X_merged = np.vstack([X_exp, X_syn]).astype(np.float32) # type: ignore
    y_merged = np.vstack([y_exp, y_syn]).astype(np.float32) # type: ignore

    is_synthetic = np.concatenate([
        np.zeros(len(X_exp), dtype=np.int32), # type: ignore
        np.ones(len(X_syn), dtype=np.int32), # type: ignore
    ])

    max_exp_group = int(np.max(groups_exp)) if len(groups_exp) > 0 else 0 # type: ignore
    groups_syn = np.arange(max_exp_group + 1, max_exp_group + 1 + len(X_syn), dtype=np.int32) # type: ignore
    groups_merged = np.concatenate([groups_exp, groups_syn]) # type: ignore

    is_lockbox_syn = np.zeros(len(X_syn), dtype=bool) # type: ignore
    is_lockbox_merged = np.concatenate([is_lockbox_exp, is_lockbox_syn])

    comp_kwargs = {'compression': 'gzip', 'compression_opts': 3}
    with h5py.File(merged_h5_out, 'w') as hf_out:
        hf_out.create_dataset('X', data=X_merged, **comp_kwargs)
        hf_out.create_dataset('y', data=y_merged, **comp_kwargs)
        hf_out.create_dataset('wavelengths', data=exp_wl.astype(np.float32)) # type: ignore
        hf_out.create_dataset('target_elements', data=np.array(target_elements, dtype=object), dtype=h5py.string_dtype(encoding='utf-8'))
        hf_out.create_dataset('groups', data=groups_merged)
        hf_out.create_dataset('is_lockbox', data=is_lockbox_merged, **comp_kwargs)

        meta = hf_out.create_group('meta')
        meta.create_dataset('is_synthetic', data=is_synthetic)

    print(f"✅ Successfully exported merged dataset to {merged_h5_out}")
    print(f"   Final Shapes: X={X_merged.shape}, y={y_merged.shape}")


if __name__ == "__main__":
    exp_path = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/processed-data_20261110b.h5")
    nist_singles_dir = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/NIST/NIST_singles/Enriched")
    syn_out_dir = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/NIST/NIST_combined")

    syn_h5, syn_csv = synthetic_data_generator(
        nist_input_dir=nist_singles_dir,
        exp_h5_path=exp_path,
        output_dir=syn_out_dir,
        num_samples=200000,
    )

    final_combo(
        exp_h5_path=exp_path,
        syn_h5_path=syn_h5,
        syn_csv_path=syn_csv,
        merged_h5_out=Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/exp-syn_processed-data_20261118.h5"),
    )