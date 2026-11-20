from __future__ import annotations

"""
jvlee_LIBS_ML > utils > debug.py

Function definitions for creating and logging to a log files. Used in most other
scripts as the primary method of printing so that things aren't only printed in 
the terminal, but also in an associated log file.

"""

print('debug.py loading...')

# region Imports
# region plain
import sys, os 
import logging
import time
import h5py
import numpy as np
import pandas as pd
import torch
# endregion

# region from
from logging import handlers
from datetime import datetime 
from pathlib import Path
from typing import Optional, IO
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
# endregion
# endregion

def log(logger,msg):
    if logger.handlers:
        logger.info(msg)
    else:
        print(msg)


def get_worker_logger(name):
    return logging.getLogger(f'worker.{name}')


def safe_open_log(filepath, max_retries=3, delay=0.5):
    """Try to open log file with retries for locked files."""
    Path(filepath).parent.parent.mkdir(parents=True,exist_ok=True)
    Path(filepath).parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(max_retries):
        try:
            return open(filepath, 'a', encoding='utf-8')
        except PermissionError:
            if attempt < max_retries - 1:
                print(f"Log file locked, retrying in {delay}s...", file=sys.__stderr__)
                time.sleep(delay)
            else:
                raise


class Logger(logging.Handler):
    """
    A logging.Handler that mirrors records to botht the terminal and a log file.
    Can be used as:
        - A direct Handler: logging.getLogger().addHandler(Logger(path))
        - A QueueListener destination for multiprocessing-safe logging
        - A sys.stdout drop-in(write/flush methods still present)
    """
    def __init__(self, filepath):
        super().__init__()
        self.terminal = sys.__stdout__
        self.log: Optional[IO[str]] = None
        try:
            self.log = safe_open_log(filepath=filepath)
            if self.log is None:
                raise RuntimeError(f"Failed to open log file: {filepath}")
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            separator = f'\n{"="*40}\n{timestamp}\n{"="*50}\n'
            self.log.write(separator)
            self.log.flush()
        except Exception as e:
            if self.terminal is not None:
                self.terminal.write(f"Failed to open log file: {e}\n")
            raise

    def emit(self, record):
        msg = self.format(record) + '\n'
        if self.terminal is not None:
            self.terminal.write(msg)
        if self.log is not None:
            self.log.write(msg)
            self.log.flush()

    def write(self, message):
        if self.terminal is not None:
            self.terminal.write(message)
        if self.log is not None:
            self.log.write(message)
            self.log.flush()

    def flush(self):
        if self.log is not None:
            try:
                self.log.flush()
            except (ValueError, AttributeError):
                pass  # File already closed

    def close(self):
        if self.log is not None:
            try:
                self.log.flush()
                self.log.close()
            except (ValueError, AttributeError):
                pass  # Already closed
            finally:
                self.log = None


def exp_syn_gap_check(
    h5_path: Path | str = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/exp-syn_processed-data_20261110b.h5",
    master_table: Path | str = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/master_line_table.csv",
):
    h5_path = Path(h5_path).resolve()
    master_table = Path(master_table).resolve()

    with h5py.File(h5_path, "r") as hf:
        X = np.asarray(hf["X"][:], dtype=np.float32) # type: ignore
        y = np.asarray(hf["y"][:], dtype=np.float32) # type: ignore
        wl = np.asarray(hf["wavelengths"][:]) # type: ignore
        is_synth = np.asarray(
            hf["meta/is_synthetic"][:] if "meta/is_synthetic" in hf else hf["is_synthetic"][:], # type: ignore
            dtype=int,
        )
        groups = np.asarray(hf["groups"][:] if "groups" in hf else hf["group_id"][:]) # type: ignore
        name_key = "target_elements" if "target_elements" in hf else "feature_cols"
        names = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in hf[name_key][:]] # type: ignore

    # L2-normalize in-place to match load_crossval_data() without duplicating RAM
    norms = np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    X /= norms
    scale_999 = float(np.percentile(X[is_synth == 0].max(axis=1), 99.9))
    X /= scale_999
    print(f"Exp 99.9th percentile max: {scale_999:.5f} | Scaled peak median: {np.median(X.max(axis=1)):.4f}")

    # 1. Check master_line_table intensity scaling across elements
    df = pd.read_csv(master_table)
    print("\n--- Max base_intensity in master_line_table by element ---")
    print(df.groupby("element")["base_intensity"].agg(["count", "max", "median"]).loc[
        [e for e in ["Li", "K", "Cl", "Ce", "Gd", "Sm", "U"] if e in df["element"].values]
    ])

    # 2. Compare Group 118 (real ~5.857 wt% Sm) vs high-Sm synthetic shots
    sm_idx = names.index("Sm")
    real_g118 = np.where((is_synth == 0) & (groups == 118))[0]
    syn_high_sm = np.where((is_synth == 1) & (y[:, sm_idx] > 4.5))[0]

    print(f"\n--- Sm Diagnostic: Real Group 118 (n={len(real_g118)}) vs Synth Sm > 4.5 wt% (n={len(syn_high_sm)}) ---")
    if len(real_g118) > 0 and len(syn_high_sm) > 0:
        mu_real = X[real_g118].mean(axis=0)
        mu_syn = X[syn_high_sm].mean(axis=0)
        cos_sim = np.dot(mu_real, mu_syn) / (np.linalg.norm(mu_real) * np.linalg.norm(mu_syn) + 1e-8)
        print(f"Full-spectrum cosine similarity (Real G118 vs Synth High-Sm): {cos_sim:.4f}")

        for line_nm in [356.83, 359.26, 442.43, 446.73]:
            win = np.abs(wl - line_nm) <= 0.25
            if win.any():
                r_pk = mu_real[win].max()
                s_pk = mu_syn[win].max()
                print(f"  Sm II {line_nm:6.2f} nm | Real peak: {r_pk:.4f} | Synth peak: {s_pk:.4f} | Ratio (Real/Synth): {r_pk / max(s_pk, 1e-8):.2f}")

    # 3. Check ALL experimental Sm groups > 0.5 wt%
    real_sm_all = np.where((is_synth == 0) & (y[:, sm_idx] > 0.5))[0]
    unique_sm_groups = np.unique(groups[real_sm_all])
    print(f"\nAll Real Sm > 0.5 wt% shots: n={len(real_sm_all)} across groups {unique_sm_groups}")
    for g_id in unique_sm_groups:
        g_mask = np.where((is_synth == 0) & (groups == g_id))[0]
        mu_g = X[g_mask].mean(axis=0)
        wt_g = y[g_mask[0], sm_idx]
        p356 = mu_g[np.abs(wl - 356.83) <= 0.25].max()
        p442 = mu_g[np.abs(wl - 442.43) <= 0.25].max()
        p484 = mu_g[np.abs(wl - 484.80) <= 0.25].max()
        print(f"  Group {g_id:4d} (n={len(g_mask):4d}, Sm={wt_g:.3f} wt%) | 356.8nm: {p356:.4f} | 442.4nm: {p442:.4f} | 484.8nm: {p484:.4f}")

    print(f"\n--- 4-Dopant UV / VIS / NIR Peak Ratio Diagnostic (Grid: {wl.min():.1f} - {wl.max():.1f} nm) ---")
    dopants = ["Ce", "Gd", "Sm", "U"]
    dop_indices = [names.index(d) for d in dopants]

    for el in dopants:
        el_idx = names.index(el)
        other_idx = [i for i in dop_indices if i != el_idx]

        # Prefer pure single-dopant shots (> 1.5 wt%) to avoid line blending; fallback to any > 2.0 wt%
        real_pure = (is_synth == 0) & (y[:, el_idx] > 1.5) & (y[:, other_idx].sum(axis=1) <= 1e-4)
        syn_pure = (is_synth == 1) & (y[:, el_idx] > 1.5) & (y[:, other_idx].sum(axis=1) <= 1e-4)

        real_mask = real_pure if real_pure.sum() >= 10 else ((is_synth == 0) & (y[:, el_idx] > 2.0))
        syn_mask = syn_pure if syn_pure.sum() >= 10 else ((is_synth == 1) & (y[:, el_idx] > 2.0))

        if not real_mask.any() or not syn_mask.any():
            print(f"\n[{el}] Insufficient high-conc shots (Real={real_mask.sum()}, Synth={syn_mask.sum()})")
            continue

        mu_r = X[real_mask].mean(axis=0)
        mu_s = X[syn_mask].mean(axis=0)
        cos_sim = np.dot(mu_r, mu_s) / (np.linalg.norm(mu_r) * np.linalg.norm(mu_s) + 1e-8)

        # Restrict line table lookup to wavelengths actually covered by the spectrometer
        sub_df = df[
            (df["element"] == el)
            & (df["wavelength_nm"] >= wl.min())
            & (df["wavelength_nm"] <= wl.max())
        ].sort_values("base_intensity", ascending=False)

        uv_df = sub_df[sub_df["wavelength_nm"] < 400.0]
        vis_df = sub_df[(sub_df["wavelength_nm"] >= 400.0) & (sub_df["wavelength_nm"] <= 700.0)]
        nir_df = sub_df[sub_df["wavelength_nm"] > 700.0]

        print(
            f"\n[{el}] Real (n={real_mask.sum()}, mean={y[real_mask, el_idx].mean():.2f} wt%) vs "
            f"Synth (n={syn_mask.sum()}, mean={y[syn_mask, el_idx].mean():.2f} wt%) | "
            f"Cosine Sim: {cos_sim:.4f} | Table lines (UV/VIS/NIR): {len(uv_df)}/{len(vis_df)}/{len(nir_df)}"
        )

        bands = [
            ("UV (<400nm)", uv_df["wavelength_nm"].head(3).tolist()),
            ("VIS (400-700)", vis_df["wavelength_nm"].head(3).tolist()),
            ("NIR (>700nm)", nir_df["wavelength_nm"].head(3).tolist()),
        ]

        for region, lines in bands:
            if not lines:
                print(f"  {region:<14} (no lines in master_line_table)")
                continue
            for line_nm in lines:
                win = np.abs(wl - line_nm) <= 0.25
                if win.any():
                    r_pk = mu_r[win].max()
                    s_pk = mu_s[win].max()
                    print(
                        f"  {region:<14} {line_nm:6.2f} nm | Real: {r_pk:.4f} | "
                        f"Synth: {s_pk:.4f} | Ratio (Real/Synth): {r_pk / max(s_pk, 1e-8):.2f}"
                    )

    # 4. Shuffled Domain Classifier ROC-AUC on top-32 PCA components
    rng = np.random.default_rng(42)
    idx_r = rng.choice(np.where(is_synth == 0)[0], size=15000, replace=False)
    idx_s = rng.choice(np.where(is_synth == 1)[0], size=15000, replace=False)
    idx_sub = rng.permutation(np.concatenate([idx_r, idx_s]))
    y_dom = is_synth[idx_sub]

    pca = PCA(n_components=32, random_state=42).fit_transform(X[idx_sub])
    clf = LogisticRegression(max_iter=500).fit(pca[:24000], y_dom[:24000])
    auc = roc_auc_score(y_dom[24000:], clf.predict_proba(pca[24000:])[:, 1])
    print(f"\nDomain Classifier ROC-AUC (shuffled, 0.5=identical, 1.0=separable): {auc:.4f}")

def check_exp_hosts():
    from pathlib import Path
    import h5py
    import numpy as np
    import pandas as pd

    H5_FILE = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/exp-syn_processed-data_20261110b.h5"
    MASTER_TABLE_PATH = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/master_line_table.csv"

    with h5py.File(H5_FILE, "r") as hf:
        y = hf["y"][:] # type: ignore
        name_key = "target_elements" if "target_elements" in hf else "feature_cols"
        names = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in hf[name_key][:]] # type: ignore

        if "meta/is_synthetic" in hf:
            is_synth = hf["meta/is_synthetic"][:] # type: ignore
        elif "is_synthetic" in hf:
            is_synth = hf["is_synthetic"][:] # type: ignore
        else:
            is_synth = np.zeros(len(y), dtype=np.int32) # type: ignore

        is_lockbox = hf["is_lockbox"][:].astype(bool) if "is_lockbox" in hf else np.zeros(len(y), dtype=bool) # type: ignore

    for label, mask in [
        ("EXPERIMENTAL (all)", is_synth == 0),
        ("EXPERIMENTAL (CV only)", (is_synth == 0) & (~is_lockbox)), # type: ignore
        ("SYNTHETIC", is_synth == 1),
    ]:
        sub = y[mask] # type: ignore
        print(f"\n=== {label} (n = {len(sub)}) ===") # type: ignore
        print(f"{'Elem':<6} | {'%>0':>7} | {'Min':>8} | {'Mean':>8} | {'Median':>8} | {'P95':>8} | {'Max':>8}")
        print("-" * 68)
        for j, el in enumerate(names):
            col = sub[:, j] # type: ignore
            pct_pos = 100.0 * np.mean(col > 0) # type: ignore
            print(
                f"{el:<6} | {pct_pos:6.1f}% | {col.min():8.4f} | {col.mean():8.4f} | " # type: ignore
                f"{np.median(col):8.4f} | {np.percentile(col, 95):8.4f} | {col.max():8.4f}" # type: ignore
            )

    # Also check line counts & max base_intensity per element in master_line_table.csv
    df = pd.read_csv(MASTER_TABLE_PATH)
    print("\n=== master_line_table.csv Line Summary ===")
    if "element" in df.columns:
        df["base_elem"] = df["element"].astype(str).str.strip().str.split().str[0]
        for el in ["Li", "K", "Cl", "Ce", "Gd", "Sm", "U"]:
            sub_df = df[df["base_elem"] == el]
            max_amp = sub_df["base_intensity"].max() if len(sub_df) else 0.0
            print(f"{el:<6} | matched rows: {len(sub_df):<6} | max base_intensity: {max_amp:.4g}")

if __name__ == "__main__":
    print('Hi')

    # exp_syn_gap_check(
    # h5_path= "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/exp-syn_processed-data_20261110b.h5",
    # master_table = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/master_line_table.csv",
    # )
    # exp_syn_gap_check(
    # h5_path= "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/exp-syn_processed-data_20261118.h5",
    # master_table = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/master_line_table.csv",
    # )

    check_exp_hosts()