from __future__ import annotations

"""
jvlee_LIBS_ML > LIBS > p3VAE > h5py_checker.py
"""

import h5py
from typing import cast
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------------------
# CHECK .h5 FOR NaN's
# ---------------------------------------------------------------------------------------
# h5_path = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/exp_syn_train_val_test_dataset_clean.h5"

# with h5py.File(h5_path, 'r') as hf:
#     for split in ['train', 'val', 'test']:
#         if split not in hf:
#             continue
#         print(f"\n--- Checking split: {split} ---")
#         spectra = hf[split]['spectra'][:]       # type: ignore
#         targets = hf[split]['metadata']['elem_comp_wt%'][:]       # type: ignore
#         nan_spectra = np.isnan(spectra).sum()
#         nan_targets = np.isnan(targets).sum()
#         print(f"Spectra shape: {spectra.shape} | NaNs in spectra: {nan_spectra}")       # type: ignore
#         print(f"Targets shape: {targets.shape} | Nans in targets: {nan_targets}")       # type: ignore

#         if nan_spectra > 0:
#             bad_rows = np.isnan(spectra).any(axis=1)
#             print(f"  -> {bad_rows.sum()} rows in '{split}/spectra' contatin NaNs")
#         if nan_targets > 0:
#             bad_rows = np.isnan(targets).any(axis=1)
#             print(f"  -> {bad_rows.sum()} rows in '{split}/metadata' contatin NaNs")


# ---------------------------------------------------------------------------------------
# COMPARE TWO FILES TO SEE IF STRUCTURE IS THE SAME
# ---------------------------------------------------------------------------------------
# def compare_h5_structure(file1_path: str, file2_path: str) -> bool:
#     """Recursively checks if two HDF5 files have identical group/dataset structures."""
    
#     def get_structure(group: h5py.Group) -> dict:
#         structure = {}
#         def visitor(name, obj):
#             if isinstance(obj, h5py.Dataset):
#                 structure[name] = ("dataset", obj.shape, obj.dtype)
#             elif isinstance(obj, h5py.Group):
#                 structure[name] = ("group",)
        
#         group.visititems(visitor)
#         return structure

#     with h5py.File(file1_path, 'r') as f1, h5py.File(file2_path, 'r') as f2:
#         struct1 = get_structure(f1)
#         struct2 = get_structure(f2)
        
#         if struct1 == struct2:
#             print("Both HDF5 files have the EXACT same structure, shapes, and dtypes.")
#             return True
#         else:
#             print("Structures differ!")
#             diff1 = set(struct1.keys()) - set(struct2.keys())
#             diff2 = set(struct2.keys()) - set(struct1.keys())
#             if diff1:
#                 print(f"Keys only in File 1: {diff1}")
#             if diff2:
#                 print(f"Keys only in File 2: {diff2}")
#             return False

# # Usage example:
# compare_h5_structure(
#     "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/exp_syn_train_val_test_dataset_clean.h5",
#     "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/exp_syn_train_val_test_dataset.h5",
#     # "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/NIST_combined/synthetic_spectra_200k.h5",
#     # "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/experimental.h5"
# )

# ---------------------------------------------------------------------------------------
# PRINT OUT ALL THE COLUMN HEADERS / METADATA KEYS OF A .H5
# ---------------------------------------------------------------------------------------
# with h5py.File('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/exp_syn_train_val_test_dataset_clean.h5', 'r') as f:
#     metadata_group = f['test/metadata']
    
#     # List all 219 dataset names inside metadata
#     field_names = list(metadata_group.keys()) # type: ignore
#     print("Available metadata fields (first 10):", field_names[:])
#     print("Total fields:", len(field_names))

# ---------------------------------------------------------------------------------------
# INSPECT SPECIFIC PATH & IDENTIFIER KEYS INSIDE METADATA
# ---------------------------------------------------------------------------------------

# path_keys = ['file_id', 'file_id.1', 'file_path', 'og_path']

# with h5py.File('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/experimental.h5', 'r') as f:
#     meta_group = cast(h5py.Group, f['metadata'])
    
#     for key in path_keys:
#         if key in meta_group:
#             val = meta_group[key][:] # type: ignore
#             # Decode byte strings if stored as binary
#             if val.dtype.kind == 'S': # type: ignore
#                 val = [v.decode('utf-8') for v in val] # type: ignore
#             print(f"--- Key: {key} (Length: {len(val)}) ---") # type: ignore
#             print("First 100 entries:", val[:100]) # type: ignore
#             print()

# ---------------------------------------------------------------------------------------
# DELETE DESIGNATED COLUMN(S) FROM .h5 FILE
# ---------------------------------------------------------------------------------------
def drop_h5_columns(
        file_path: Path, 
        cols_to_drop: list[str], 
        group_path: str = "metadata"
):
    file_path = Path(file_path)
    
    with h5py.File(file_path, "a") as f:  # Must open in read/write ("a" or "r+") mode
        if group_path not in f:
            print(f"Group '{group_path}' not found in {file_path.name}")
            return
            
        meta_grp = f[group_path]
        
        dropped = []
        for col in cols_to_drop:
            if col in meta_grp: # type: ignore
                del meta_grp[col] # type: ignore
                dropped.append(col)
                
        # Optional: Update metadata_cols attribute if it exists
        if "metadata_cols" in meta_grp.attrs:
            current_attrs = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in meta_grp.attrs["metadata_cols"] # type: ignore
            ]
            updated_attrs = [c for c in current_attrs if c not in cols_to_drop]
            meta_grp.attrs["metadata_cols"] = updated_attrs

        f.flush()
        print(f"Dropped {len(dropped)} column(s) from {file_path.name}: {dropped}")


# drop_h5_columns(
#     file_path="/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/experimental.h5", # type: ignore
#     cols_to_drop=["file_id.1", "file_path"],
#     group_path="metadata"
# )
# h5_path = Path('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/experimental.h5')
# with h5py.File(h5_path, "r") as f:
#     meta_grp = f["metadata"]
#     all_keys = list(meta_grp.keys()) # type: ignore

# target_col = "conc_BaCl2_wt%"
# if target_col in all_keys:
#     stop_idx = all_keys.index(target_col)
#     cols_to_drop = all_keys[:stop_idx]

#     print(f"Found '{target_col}' at index {stop_idx}. Dropping {len(cols_to_drop)} columns.")

#     drop_h5_columns(
#         file_path=h5_path,
#         cols_to_drop=cols_to_drop,
#         group_path='metadata'
#     )

# else:
#     print(f"Target column '{target_col}' not found in metadata group!")

# ---------------------------------------------------------------------------------------
# COMPARE 'FILE_ID' VS 'FILE_ID.1' AND CHECK IF 'FILE_PATH' IS EMPTY
# ---------------------------------------------------------------------------------------
# with h5py.File('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/experimental.h5', 'r') as f:
#     meta = f['metadata']
    
#     file_id = meta['file_id'][:] # type: ignore
#     file_id_1 = meta['file_id.1'][:] # type: ignore
#     file_path = meta['file_path'][:] # type: ignore
    
#     # Check if file_id and file_id.1 are identical
#     are_ids_equal = np.array_equal(file_id, file_id_1) # type: ignore
#     print(f"Are 'file_id' and 'file_id.1' identical? {are_ids_equal}")
    
#     # Check if file_path is completely 0
#     is_path_empty = np.all(file_path == 0)
#     print(f"Is 'file_path' entirely zeroes? {is_path_empty}")

# ---------------------------------------------------------------------------------------
# SAMPLE NON-ZERO 'FILE_PATH' ENTRIES AND COMPARE WITH 'OG_PATH' AND 'FILE_ID'
# ---------------------------------------------------------------------------------------
# with h5py.File('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/experimental.h5', 'r') as f:
#     meta = cast(h5py.Group, f['metadata'])
    
#     file_path = meta['file_path'][:] # type: ignore
#     og_path = meta['og_path'][:] # type: ignore
#     file_id = meta['file_id'][:] # type: ignore
    
#     # Identify boolean mask for non-zero entries
#     # (Handles numeric floats or string '0' / b'0')
#     if np.issubdtype(file_path.dtype, np.number): # type: ignore
#         is_nonzero = file_path != 0
#     else:
#         is_nonzero = (file_path != b'0') & (file_path != '0') & (file_path != b'')
    
#     nonzero_count = np.sum(is_nonzero)
#     print(f"Total non-zero entries in file_path: {nonzero_count} / {len(file_path)}") # type: ignore
    
#     # Print a sample of non-zero rows comparing all three sources
#     nonzero_indices = np.where(is_nonzero)[0][:10]  # First 10 indices
#     print("\n--- Sample Comparison (where file_path != 0) ---")
#     for idx in nonzero_indices:
#         fp_val = file_path[idx] # type: ignore
#         og_val = og_path[idx].decode('utf-8') if isinstance(og_path[idx], bytes) else og_path[idx] # type: ignore
#         id_val = file_id[idx].decode('utf-8') if isinstance(file_id[idx], bytes) else file_id[idx] # type: ignore
        
#         print(f"Index {idx}:")
#         print(f"  file_path: {fp_val}")
#         print(f"  og_path  : {og_val}")
#         print(f"  file_id  : {id_val}")
#         print("-" * 40)

# ---------------------------------------------------------------------------------------
# INSPECT WAVELENGTH GRID STATISTICS (MIN, MAX, STEP SIZE, TOTAL POINTS)
# ---------------------------------------------------------------------------------------
# with h5py.File('/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/test_master_base.h5', 'r') as f:
#     # 1. Read the wavelength array
#     wavelengths = f['wavelengths'][:]
    
#     # 2. Compute key grid statistics
#     min_wl = wavelengths[0]
#     max_wl = wavelengths[-1]
#     n_points = len(wavelengths)
#     step_size = wavelengths[1] - wavelengths[0] if n_points > 1 else 0
    
#     print(f"--- Wavelength Grid Summary ---")
#     print(f"Min Wavelength : {min_wl:.4f} nm")
#     print(f"Max Wavelength : {max_wl:.4f} nm")
#     print(f"Step Size      : {step_size:.4f} nm")
#     print(f"Total Points   : {n_points}")
#     print(f"First 5 values : {wavelengths[:5]}")
#     print(f"Last 5 values  : {wavelengths[-5:]}")

# ---------------------------------------------------------------------------------------
# IDENTIFY EMPTY OR JUNK 'DATE AND TIME:' COLUMNS IN PANDAS DATAFRAME
# ---------------------------------------------------------------------------------------
# date_cols = [c for c in base_meta.columns if c.startswith("Date and Time:")] # type: ignore
# junk_cols = date_cols + ["?", "Unnamed: 0", "blank"]

# print(f"Found {len(date_cols)} 'Date and Time:' columns.\n")

# date_summary = []
# for col in date_cols:
#     vals = base_meta[col] # type: ignore

#     # Handle missing, null, 0, or NaN entries
#     is_null_or_zero = vals.isna() | (vals == 0) | (vals == "0") | (vals == "")
#     zero_count = is_null_or_zero.sum()
#     total_count = len(vals)
#     pct_empty = (zero_count / total_count) * 100

#     date_summary.append(
#         {
#             "column": col,
#             "empty_count": zero_count,
#             "total": total_count,
#             "pct_empty": pct_empty,
#         }
#     )

# # Summary table
# df_summary = pd.DataFrame(date_summary)
# print(df_summary[["column", "empty_count", "total", "pct_empty"]])

# # Quick decision check
# completely_empty = df_summary[df_summary["pct_empty"] == 100]
# print(
#     f"\nTotal 'Date and Time' columns that are 100% empty/zero: {len(completely_empty)} / {len(date_cols)}"
# )


# ---------------------------------------------------------------------------------------
# IDENTIFY EMPTY OR JUNK 'DATE AND TIME:' COLUMNS IN .h5 FILE
# ---------------------------------------------------------------------------------------
# h5_path = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/experimental.h5"

# with h5py.File(h5_path, "r") as f:
#     meta_group = f["metadata"]

#     # 1. Identify all 'Date and Time:' datasets in the HDF5 metadata group
#     date_cols = [key for key in meta_group.keys() if key.startswith("Date and Time:")] # type: ignore
#     junk_cols = date_cols + ["?", "Unnamed: 0", "blank"]

#     print(f"Found {len(date_cols)} 'Date and Time:' columns in HDF5 metadata.\n")

#     date_summary = []
    
#     for col in date_cols:
#         # Load dataset slice directly into NumPy array
#         vals = meta_group[col][:] # type: ignore
#         total_count = len(vals) # type: ignore

#         # 2. Vectorized check across string/bytes/numeric dtypes
#         if vals.dtype.kind in ("S", "U"):  # Fixed-length byte strings or Unicode # type: ignore
#             # Decode bytes to string if needed, then check for empty representations
#             if vals.dtype.kind == "S": # type: ignore
#                 vals = np.char.decode(vals, "utf-8", errors="ignore") # type: ignore
            
#             is_null_or_zero = (
#                 (vals == "") | (vals == "0") | (vals == "None") | (vals == "nan")
#             )
#         else:  # Numeric types (float, int)
#             is_null_or_zero = np.isnan(vals) | (vals == 0) # type: ignore

#         zero_count = int(np.sum(is_null_or_zero))
#         pct_empty = (zero_count / total_count) * 100

#         date_summary.append(
#             {
#                 "column": col,
#                 "empty_count": zero_count,
#                 "total": total_count,
#                 "pct_empty": pct_empty,
#             }
#         )

# # Summary table using pandas just for clean table formatting
# df_summary = pd.DataFrame(date_summary)
# print(df_summary[["column", "empty_count", "total", "pct_empty"]])

# # Quick decision check
# completely_empty = df_summary[df_summary["pct_empty"] == 100]
# print(
#     f"\nTotal 'Date and Time' columns that are 100% empty/zero: {len(completely_empty)} / {len(date_cols)}"
# )

# cols_to_drop = completely_empty["column"].tolist()
# if cols_to_drop:
#     drop_h5_columns(file_path=h5_path,cols_to_drop=cols_to_drop,group_path='metadata') # type: ignore

# ---------------------------------------------------------------------------------------
# INSPECT SINGLE .h5 FILE'S WT% CONCENTRATION DATA
# ---------------------------------------------------------------------------------------
# h5_path = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/exp_syn_train_val_test_dataset.h5"

# with h5py.File(h5_path, "r") as f:
#     # 1. Load the matrix and element names
#     elem_comp = f["test/metadata/elem_comp_wt%"][:] # type: ignore
#     elem_names_bytes = f["test/metadata/elem_names"][:] # type: ignore
    
#     # Decode string bytes (e.g. b'Li' -> 'Li')
#     elem_names = [name.decode("utf-8") for name in elem_names_bytes] # type: ignore
    
#     # 2. Print high-level shapes and column map
#     print("=== DATASET OVERVIEW ===")
#     print(f"Matrix shape: {elem_comp.shape}") # type: ignore
#     print(f"Element count: {len(elem_names)}")
#     print(f"Element order: {elem_names}\n")

#     # 3. Wrap in Pandas DataFrame for clean summary stats
#     df = pd.DataFrame(elem_comp, columns=elem_names) # type: ignore

#     print("=== FIRST 5 SAMPLES (wt%) ===")
#     print(df.head())
#     print("\n")

#     print("=== SUMMARY STATISTICS ===")
#     # Shows min, max, mean, non-zero counts per element
#     stats = df.describe().T[["mean", "min", "max"]]
#     stats["non_zero_count"] = (df > 0.001).sum()
#     print(stats)
#     print("\n")

#     # 4. Check Mass Balance (Row Sums)
#     row_sums = df.sum(axis=1)
#     print("=== MASS BALANCE CHECK (Row Sums) ===")
#     print(f"Min row sum:  {row_sums.min():.2f}%")
#     print(f"Max row sum:  {row_sums.max():.2f}%")
#     print(f"Mean row sum: {row_sums.mean():.2f}%")

#     # Flag any rows that deviate significantly from 100%
#     non_100 = df[np.abs(row_sums - 100.0) > 0.1]
#     if len(non_100) > 0:
#         print(f"⚠️ Warning: {len(non_100)} rows do not sum to ~100% (e.g., pure solvent/blank runs or missing elements).")
#     else:
#         print("✅ All sample rows successfully sum to 100.0% mass balance!")

# with h5py.File("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/experimental.h5", "r") as f:
#     u = f["metadata/conc_UCl3_wt%"][:] # type: ignore
#     ce = f["metadata/conc_CeCl3_wt%"][:] # type: ignore
#     licl = f["metadata/frac_LiCl"][:] # type: ignore
#     kcl = f["metadata/frac_KCl"][:] # type: ignore
    
#     # Check row 50000 (a sample with uranium/cerium present)
#     idx = 50000
#     print(f"Sample {idx}:")
#     print(f"  frac_LiCl: {licl[idx]}") # type: ignore
#     print(f"  frac_KCl:  {kcl[idx]}") # type: ignore
#     print(f"  UCl3 wt%:  {u[idx]}") # type: ignore
#     print(f"  CeCl3 wt%: {ce[idx]}") # type: ignore
#     print(f"  Raw Sum of Chlorides: {licl[idx]*100 + kcl[idx]*100 + u[idx] + ce[idx]:.2f}%") # type: ignore

# ---------------------------------------------------------------------------------------
# LIST TOP-LEVEL ROOT KEYS FOR ALL DATASET .H5 FILES
# ---------------------------------------------------------------------------------------
# with h5py.File("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/combined_pairs_LIBS.h5", "r") as f:
#     print("Keys in combined_pairs_LIBS.h5:", list(f.keys()))

# with h5py.File("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/old_combined_LIBS.h5", "r") as f:
#     print("Keys in old_combined_LIBS.h5:", list(f.keys()))

# with h5py.File("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/training_ready_LIBS.h5", "r") as f:
#     print("Keys in training_ready_LIBS.h5:", list(f.keys()))

# with h5py.File("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/training_ready_pairs_LIBS.h5", "r") as f:
#     print("Keys in training_ready_pairs_LIBS.h5:", list(f.keys()))

# # with h5py.File("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/trn_val_split_LIBS.h5", "r") as f:
# #     print("Keys in trn_val_split_LIBS.h5:", list(f.keys()))

# with h5py.File("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/tv_split_pairs_LIBS.h5", "r") as f:
#     print("Keys in tv_split_pairs_LIBS.h5:", list(f.keys()))

# with h5py.File("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/experimental_tvsplit.h5", "r") as f:
#     print("Keys in experimental_tvsplit.h5:", list(f.keys()))

# with h5py.File("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/experimental.h5", "r") as f:
#     print("Keys in experimental.h5:", list(f.keys()))
# print('\n\n')

# with h5py.File("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/NIST_combined/synthetic_spectra_200k.h5", "r") as f:
#     print("Keys in synthetic_spectra_200k.h5:", list(f.keys()))
# print('\n\n')

# with h5py.File("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/elemental_experimental.h5", "r") as f:
#     print("Keys in elemental_experimental:", list(f.keys()))
# print('\n\n')


# ---------------------------------------------------------------------------------------
# PRINT OUT THE FILE STRUCTURE FOR ALL FILES IN 'file_paths'
# ---------------------------------------------------------------------------------------
file_paths = [
    # "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/experimental.h5",
    # "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/NIST_combined/synthetic_spectra_200k.h5",
    # "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/elemental_experimental.h5",
    # "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/exp_syn_train_val_test_dataset_clean.h5",
    # "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/exp_syn_train_val_test_dataset.h5"
    # "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/cts_noleak_xandy.h5"
    # "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/backups/combined_exp_syn_dataset.h5"
    "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/data/cts_noleak_crossval.h5"
]

# def print_h5_structure(g, indent=0):
#     """Recursively print the contents/structure of an HDF5 group or file."""
#     for key in g.keys():
#         item = g[key]
#         prefix = "  " * indent
#         if isinstance(item, h5py.Group):
#             print(f"{prefix}📁 [{key}] Group ({len(item)} items)")
#             print_h5_structure(item, indent + 1)
#         elif isinstance(item, h5py.Dataset):
#             print(f"{prefix}📄 [{key}] Dataset: shape={item.shape}, dtype={item.dtype}")

# # Iterate over your files
# for path in file_paths:
#     print(f"=== File: {path} ===")
#     with h5py.File(path, "r") as f:
#         print_h5_structure(f)
#     print("\n")

    # with h5py.File(path, 'r') as f:
    #     print(f"element names: {f['feature_cols'][:]}") # type: ignore


# ---------------------------------------------------------------------------------------
# INSPECT AND COMPARE SYNTHETIC DATA TO ELEMENTAL_EXPERIMENTAL
# ---------------------------------------------------------------------------------------
# exp_path = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/elemental_experimental.h5"
# syn_h5_path = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/NIST_combined/synthetic_spectra_200k.h5"
# syn_csv_path = "/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/NIST_combined/synthetic_compositions_wt_pct.csv"

# print("=" * 60)
# print("1. EXPERIMENTAL COMPOSITION METADATA")
# print("=" * 60)
# with h5py.File(exp_path, 'r') as hf:
#     exp_names = [name.decode('utf-8') if isinstance(name, bytes) else name for name in hf['metadata/elem_names'][:]] # type: ignore
#     exp_comp = hf['metadata/elem_comp_wt%'][:2] # type: ignore
#     exp_wl = hf['wavelengths'][:] # type: ignore

# print("Experimental Element Names (19):")
# print(exp_names)
# print("\nExperimental First 2 Rows (wt%):")
# print(pd.DataFrame(exp_comp, columns=exp_names)) # type: ignore

# print("\n" + "=" * 60)
# print("2. SYNTHETIC CSV COMPOSITION")
# print("=" * 60)
# df_syn = pd.read_csv(syn_csv_path, nrows=2)
# syn_names = list(df_syn.columns)
# print("Synthetic CSV Columns (19):")
# print(syn_names)
# print("\nSynthetic First 2 Rows (wt%):")
# print(df_syn)

# print("\n" + "=" * 60)
# print("3. WAVELENGTH MATRIX CHECK")
# print("=" * 60)
# with h5py.File(syn_h5_path, 'r') as hf:
#     syn_wl = hf['wavelengths'][:] # type: ignore

# wl_diff = np.max(np.abs(exp_wl - syn_wl)) # type: ignore
# print(f"Experimental Wavelength Range: {exp_wl.min():.3f} nm - {exp_wl.max():.3f} nm") # type: ignore
# print(f"Synthetic Wavelength Range:    {syn_wl.min():.3f} nm - {syn_wl.max():.3f} nm") # type: ignore
# print(f"Max Absolute Wavelength Difference: {wl_diff:.6e} nm")