import zipfile
import pandas as pd
from pathlib import Path

# Path to your ZIP storage directory
ZIP_DIR = Path("/lustre/home/leejv2/git_repos/jvlee_LIBS_ML/LIBS/NIST_trio_data")

# Find the first available ZIP archive
zip_files = sorted(list(ZIP_DIR.glob("*.zip")))

if not zip_files:
    print("❌ No ZIP files found in the output directory yet.")
else:
    target_zip = zip_files[0]
    print(f"📦 Inspecting Archive: {target_zip.name}\n")

    with zipfile.ZipFile(target_zip, 'r') as zf:
        file_list = zf.namelist()
        print(f"Total CSVs in this zip: {len(file_list)}")
        
        # Pick the first CSV inside the archive
        sample_filename = file_list[0]
        print(f"📄 Sample File: {sample_filename}\n" + "-"*50)

        # Read and preview directly into pandas without extracting to disk
        with zf.open(sample_filename) as csv_file:
            df = pd.read_csv(csv_file)

        print("\n--- Shape & Data Info ---")
        print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        
        print("\n--- First 10 Rows ---")
        print(df.head(10).to_string(index=False))

        print("\n--- Summary Statistics ---")
        print(df.describe())