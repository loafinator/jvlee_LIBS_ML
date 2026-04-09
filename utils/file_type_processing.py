"""
jvlee_LIBS_ML>utils>file_type_processing.py

Function definitions for file gathering and conversion. Current conversions defined
are all to .csv and form one of the following: .mpr, .mat, .asc, and .txt. This is 
used mainly in 'get_libs.py' to retrieve data from W: drive and converted it to a 
standard .csv format.

"""

print('file_type_processing.py loading...')

# region Imports
# region plain
import shutil
import logging
# endregion

# region as
import pandas as pd
# endregion

# region from
from pathlib import Path
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging import handlers
# endregion
# endregion


def get_file_genre(
        data_root: str | Path,
        save_root: str | Path,
        key: str | List[str]= None,
        skip_key: Optional[List[str]]=None,
        allowed_extensions: Optional[List[str]]=None,
        preserve_structure: bool=True,
        processed: int=0,
        skipped: int=0,
        failed: int=0,
) -> tuple[int, int, int]:
    data_root = Path(data_root).resolve()
    save_root = Path(save_root).resolve()
    save_root.mkdir(parents=True, exist_ok=True)

    # region Logger Setup
    logger = _get_worker_logger(Path(save_root).stem)
    # endregion


    if key is None:
        key = []
    keys = [key] if isinstance(key, str) else (key if key else [])

    if allowed_extensions is None:
        allowed_extensions = []

    if skip_key is None:
        skip_key = []

    seen =  set()
    for k in keys:
        for folder in data_root.rglob(f'*{k}*'):
            if folder in seen:
                continue
            seen.add(folder)

            if not folder.is_dir():
                skipped += 1
                log(logger=logger, msg = f'         Skipped (not a folder): {folder.name}')
                continue
            folder_path_str = str(folder).lower()
            if any(skip.lower() in folder_path_str for skip in skip_key):
                skipped += 1
                log(logger=logger, msg = f'         Skipped folder (skip_key): {folder.name}')
                continue

            for file in folder.rglob('*'):
                if not file.is_file():
                    continue

                file_path_str = str(file).lower()
                if any(skip.lower() in file_path_str for skip in skip_key):
                    skipped += 1
                    log(logger=logger, msg = f'         Skipped file (skip_key): {file.name}')
                    continue

                elif file.suffix.lower() not in allowed_extensions:
                    skipped += 1
                    log(logger=logger, msg = f'         Skipped (Not an allowed extension): {file.name}')
                    continue

                try:
                    if preserve_structure:
                        rel_path = file.relative_to(data_root)
                        new_path = save_root / rel_path
                    else:
                        new_path = save_root / file.name

                    new_path.parent.mkdir(parents=True, exist_ok=True)

                    if new_path.exists():
                        log(logger=logger, msg = f'         Skipping (already exists): {new_path}')
                        skipped += 1
                        continue

                    shutil.copy2(file,new_path)
                    log(logger=logger, msg = f'Saved {file.name}  |  Processed: {processed+1}')
                    processed += 1

                except Exception as e:
                    log(logger=logger, msg = f"  ❌ Error on {file.name}: {e}")
                    failed += 1

    log(logger=logger, msg = f"\nFinished!\n    Processed: {processed}  |  Skipped: {skipped}  |  Failed: {failed}")
    return processed, skipped, failed

def recursive_file_extension_converter(
        data_root: str | Path, 
        save_root: str | Path,
        target_extension: str | List[str] = '.mpr',
        end_extension: str = '.csv',
        skip_key: str = None,           # this should be taken care of in the get_file_genre, but
                                        # I didn't want to run it again at the moment, so I added
                                        # it here instead. 
        preserve_structure: bool=True,
        libs: bool=True,
        processed: int=0,
        skipped: int=0,
        failed: int=0,
) -> tuple[int, int, int]:
    data_root, save_root = Path(data_root).resolve(), Path(save_root).resolve()
    extensions = [target_extension] if isinstance(target_extension, str) else target_extension
    extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]

    # region Logger Setup
    logger = _get_worker_logger(Path(save_root).stem)
    # endregion


    if skip_key is None:
        skip_key = []

    for ext in extensions:
        target = f'*{ext}'

        for file in data_root.rglob(target):
            try:
                file_str = str(file)
                if any(key in file_str for key in skip_key):
                    log(logger=logger, msg = f'         Skipped (skip_key): {file.name}')
                    skipped += 1
                    continue

                if preserve_structure:
                    rel_path = file.relative_to(data_root)
                    new_path = (save_root / rel_path).with_suffix(end_extension)
                else:
                    new_path = (save_root / file.name).with_suffix(end_extension)

                new_path.parent.mkdir(parents=True, exist_ok=True)

                if new_path.exists():
                    log(logger=logger, msg = f'         Skipping (already exists): {new_path}')
                    skipped += 1
                    continue

                log(logger=logger, msg = f"[{processed+1}] Converting {file.name} to {end_extension}")
                if end_extension == '.csv':
                    if target == '*.mpr':
                        mpr_to_csv(str(file), str(new_path))
                        processed += 1
                    elif target == '*.mat':
                        mat_to_csv(str(file), str(new_path), libs=libs)
                        processed += 1
                    elif target == '*.asc':
                        asc_to_csv(str(file), str(new_path), libs=libs)
                        processed += 1
                    elif target == '*.txt':
                        txt_to_csv(str(file), str(new_path)) 
                        processed += 1

            except Exception as e:
                log(logger=logger, msg = f"  ❌ Error on {file.name}: {e}")
                failed += 1

    log(logger=logger, msg = f"\nFinished!\n    Processed: {processed}  |  Skipped: {skipped}  |  Failed: {failed}")
    return processed, skipped, failed

def mpr_to_csv(
        mpr_path: str | Path, 
        csv_path: str | Path,
) -> None:
    from galvani import BioLogic
    mpr_path = Path(mpr_path).resolve()
    csv_path = Path(csv_path).resolve()

    # region Logger Setup
    logger = _get_worker_logger(Path(csv_path).stem)
    # endregion

    try:
        mpr = BioLogic.MPRfile(mpr_path)
        log(logger=logger, msg = f"Found and loaded (mpr to csv): {mpr_path.name}")
        df = pd.DataFrame(mpr.data)
        # log(logger=logger, msg = df.head())  # preview first few rows
        df.to_csv(csv_path, index=False)
        log(logger=logger, msg = f'Saved successfully: {mpr_path.name}')
    except Exception as e:
        log(logger=logger, msg = f"  ❌ Galvani error: {e}")

def mat_to_csv(
        mat_path: str | Path,
        csv_path: str | Path,
        libs: bool=True,
) -> None:
    import scipy.io as sio 
    import h5py
    mat_path = Path(mat_path).resolve()
    csv_path = Path(csv_path).resolve()

    # region Logger Setup
    logger = _get_worker_logger(Path(csv_path).stem)
    # endregion


    try:
        mat = sio.loadmat(mat_path)
        log(logger=logger, msg = f'Found and loaded (mat to csv): {mat_path.name} ')
        data = {k: v.flatten() for k, v in mat.items() if not k.startswith('__')}
        # log(logger=logger, msg = f'Data extracted from {mat_path.name}')

        if libs:
            lambda_keys = sorted([k for k in data if 'lamb' in k.lower()])
            spectra_keys = sorted([k for k in data if 'spectra' in k.lower()])

            if not lambda_keys or not spectra_keys:
                log(logger=logger, msg = f'         Skipping (no lambda/spectra keys): {mat_path.name}')
                return

            for lk, sk in zip(lambda_keys, spectra_keys):
                wavelengths = data[lk]
                spectra_flat = data[sk]
                n_shots = len(spectra_flat) // len(wavelengths)
                spectra_2d = spectra_flat.reshape(n_shots, len(wavelengths))

                df = pd.DataFrame(spectra_2d,columns=wavelengths)
                df.index.name = 'shot'

                out_path = csv_path.with_stem(f'{csv_path.stem}_{lk}')
                df.to_csv(out_path, index=False)
                log(logger=logger, msg = f'Saved {out_path.name}  |  shape: {df.shape}')

        else:
            # for k, v in data.items():
            #     log(logger=logger, msg = f'  {k}: length {len(v)}')

            lengths = {len(v) for v in data.values()}
            if len(lengths) > 1:
                log(logger=logger, msg = f'  Warning: saving arrays separately (mismatched lengths): {lengths}')
                for k, v in data.items():
                    array_path = csv_path.with_stem(f'{csv_path.stem}_{k}')
                    pd.DataFrame({k: v}).to_csv(array_path, index=False)
                    log(logger=logger, msg = f'     Saved: {array_path.name}  |  shape: {array_path.shape}')
            else:
                df = pd.DataFrame(data)
                log(logger=logger, msg = df.head())
                df.to_csv(csv_path, index=False)
                log(logger=logger, msg = f'Saved: {mat_path.name}  |  shape: {df.shape}')
    except NotImplementedError as e:
        log(logger=logger, msg = f'  Warning ({e}): h5py needed')
        with h5py.File(mat_path, 'r') as f:
            data = {k: f[k][()].flatten() for k in f.keys() if not k.startswith('__')}
        # log(logger=logger, msg = f'Data extracted from {mat_path.name}')
        if libs:
            lambda_keys = sorted([k for k in data if 'lamb' in k.lower()])
            spectra_keys = sorted([k for k in data if 'spectra' in k.lower()])
            for lk, sk in zip(lambda_keys, spectra_keys):
                wavelengths = data[lk]
                spectra_flat = data[sk]
                n_shots = len(spectra_flat) // len(wavelengths)
                spectra_2d = spectra_flat.reshape(n_shots, len(wavelengths))
                df = pd.DataFrame(spectra_2d, columns=wavelengths)
                df.index.name = 'shot'
                out_path = csv_path.with_stem(f'{csv_path.stem}_{lk}')
                df.to_csv(out_path, index=False)
                log(logger=logger, msg = f'Saved {out_path.name}  |  shape: {df.shape}')
        else:
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)
            log(logger=logger, msg = f'Saved {mat_path.name}   |  shape: {mat_path.shape}')
    except Exception as e:
        log(logger=logger, msg = f'  ❌ Scipy Error: {e}')

def asc_to_csv(
        asc_path: str | Path, 
        csv_path: str | Path,
        libs: bool=True,
) -> None:
    asc_path = Path(asc_path).resolve()
    csv_path = Path(csv_path).resolve()

    # region Logger Setup
    logger = _get_worker_logger(Path(csv_path).stem)
    # endregion


    try:
        # region Separator and read data
        with open(asc_path, 'r') as f:
            first_line = f.readline().strip()
        if ',' in first_line:
            sep = ','
        elif '\t' in first_line:
            sep = '\t'
        else:
            sep = r'\s+'

        df = pd.read_csv(asc_path, sep=sep, header=None, comment=';', engine='python')
        # endregion

        if libs:
            if df.shape[1] == 2:
                # Single shot: wavelength | intensity
                df.columns = ['wavelength', 'intensity']
                df_T = df.set_index('wavelength').T
                df_T.index.name = None
                df_T.to_csv(csv_path, index=False)
                log(logger=logger, msg = f'Saved {csv_path.name} | shape: {df.shape}')

            elif df.shape[1] > 2:
                # Multiple shots: wavelength | shot_1 | shot_2 | ...
                n_shots = df.shape[1] - 1
                df.columns = ['wavelength'] + [f'shot_{i+1}' for i in range(n_shots)]
                df_T = df.set_index('wavelength').T
                df_T.index.name = None
                df_T.to_csv(csv_path, index=False)
                log(logger=logger, msg = f'Saved {csv_path.name} | shape: {df.shape}')

            else:
                log(logger=logger, msg = f'         Skipping (unexpected shape): {asc_path.name}  |  shape={df.shape}')
                return
            
        else:
            df.to_csv(csv_path, index=False)
            log(logger=logger, msg = f'{asc_path.name} successfully saved as {csv_path}')

    except Exception as e:
        log(logger=logger, msg = f'  ❌ ASC error on {asc_path.name}: {e}')

def txt_to_csv(
        txt_path: str | Path,
        csv_path: str | Path,
) -> None:
    txt_path = Path(txt_path).resolve()
    csv_path = Path(csv_path).resolve()

    # region Logger Setup
    logger = _get_worker_logger(Path(csv_path).stem)
    # endregion


    try:
        data_rows = []

        with open(txt_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("'"):
                    continue  # skip header lines
                parts = line.split('\t')
                try:
                    wl = float(parts[0])
                    intensities = [float(x) for x in parts[1:] if x.strip()]
                    if intensities:
                        data_rows.append((wl, intensities))
                except ValueError:
                    continue

        if not data_rows:
            log(logger=logger, msg = f'Warning: no data rows found in {txt_path.name}')
            return

        wavelengths = [r[0] for r in data_rows]
        # Each row is one wavelength, each column is one shot — need to transpose
        # so that rows = shots, columns = wavelengths (same as asc_to_csv output)
        n_shots = len(data_rows[0][1])
        spectra_2d = [[data_rows[wl_i][1][shot_i] 
                       for wl_i in range(len(wavelengths))] 
                       for shot_i in range(n_shots)]

        df = pd.DataFrame(spectra_2d, columns=wavelengths)
        df.to_csv(csv_path, index=False)
        log(logger=logger, msg = f'Saved {csv_path.name} | shape: {df.shape}')

    except Exception as e:
        log(logger=logger, msg = f'  ❌ txt error on {txt_path.name}: {e}')
    
def _copy_file(
        src_file: str | Path,
        dst_file: str | Path,
        overwrite: bool=False,
        ):

    # region Logger Setup
    logger = _get_worker_logger(Path(dst_file).stem)
    # endregion

    try:
        if dst_file.exists() and not overwrite:
            return 'skipped', src_file
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        return 'processed', src_file.name
    except Exception as e:
        return 'failed', f'{src_file.name} -> {e}'

def file_segregator(
        data_root: str | Path, 
        data_destination: str | Path,
        pattern: str = '*_CV_*',
        preserve_structure: bool = True,
        overwrite: bool = False,
        ) -> tuple[int, int, int]:
    
    data_root, data_destination = Path(data_root).resolve(), Path(data_destination).resolve()

    # region Logger Setup
    logger = _get_worker_logger(Path(data_destination).stem)
    # endregion

    
    if not data_root.is_dir():
        raise NotADirectoryError(f"data_root is not a directory: {data_root}")
    
    processed, skipped, failed = 0, 0, 0
    file_pairs = []

    data_destination.mkdir(parents=True, exist_ok=True)

    log(logger=logger, msg = f"""Searching in: {data_root}\nCopying to: {data_destination}\nPattern: {pattern}\nPreserve tree: {preserve_structure}""")
    

    for src_file in data_root.rglob(pattern):
        if not src_file.is_file():
            skipped += 1
            continue
        if data_destination in src_file.parents or src_file.is_relative_to(data_destination):
            skipped += 1
            continue
        if preserve_structure:
            dst_file = data_destination / src_file.relative_to(data_root)
        else:
            dst_file = data_destination / src_file.name 

        file_pairs.append((src_file, dst_file))

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(_copy_file, src, dst, overwrite):
                   src for src, dst in file_pairs}
        for future in as_completed(futures):
            status, name = future.result()
            if status == 'skipped': skipped += 1
            elif status == 'failed': failed += 1
            else: processed += 1

    log(logger=logger, msg = '\n' + '='*60)
    log(logger=logger, msg = f'Finished:')
    log(logger=logger, msg= f'  Processed: {processed}')
    log(logger=logger, msg= f'  Skipped: {skipped}')
    log(logger=logger, msg= f'  Failed: {failed}')
    log(logger=logger, msg= f'  Total Found: {processed + skipped + failed}')
    log(logger=logger, msg = '\n' + '='*60)

    return processed, skipped, failed

def log(logger, msg):
    if logger.hasHandlers():
        logger.info(msg)
    else:
        print(msg)

def _get_worker_logger(name):
    return logging.getLogger(f'worker.{name}')

if __name__ == '__main__':
    print('hi')
    # mpr_to_csv(mpr_path=r"W:\Phongikaroon Group\Dalsung Y\Electrochemical NEPU project\Data\Cd Exp\PURE SALT _1\500C\1_CV_200mV_C01.mpr",
    #            csv_path=r"W:\Phongikaroon Group\Dalsung Y\Electrochemical NEPU project\Data\Cd Exp\PURE SALT _1\500C\1_CV_200mV_C01.csv")
    
    # asc_to_csv(asc_path=r"W:\Phongikaroon Group\AndrewsH\Backup\Experiments & Calculations\SmCl3 Studies\SmCl3 CV and LIBS 1\Sm LIBS raw data\5_smcl3_run1.asc",
    #            csv_path=r"G:\My Drive\RLSL\Data\testing\5_smcl3_run1.csv")