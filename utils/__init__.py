"""
jvlee_LIBS_ML>utils>__init__.py
"""

from .file_type_processing import *
from .speak import *
from .data_prep import * 
from .plotting import *
from .debug import *
from .post_processing import *

import multiprocessing
if multiprocessing.current_process().name == 'MainProcess':
    from .xpu_setup import *

__all__: list[str] = [
    # region file_type_processing
    "mpr_to_csv",
    "recursive_file_extension_converter",
    "file_segregator",
    "get_file_genre",
    # endregion

    # region xpu_setup
    "mae",
    "init_xpu_trainer",
    # endregion

    # region speak
    "gen_speak",
    # endregion

    # region data_prep
    "enrich_file_with_metadata",
    "enrich_with_progress",
    "species_sort_key",
    "parent_concentration_data",
    "clean_single_technique_file",
    "trn_val_splitter_HDF5",
    "trn_val_splitter_CSV",
    "standardize_wavelength_grid",
    "combine_and_save_as_HDF5",
    "combine_and_save_as_CSV",
    "sanitize_path",
    "_long_path",
    "_worker_init",
    "_get_worker_logger",
    "hf_get",
    # endregion

    # region plotting
    "animate_training",
    "plot_residual_history",
    "plot_predicted_vs_actual",
    "plot_peak_error_heatmap",
    # endregion
    
    # region debug    
    "Logger",
    "log",
    "get_worker_logger",
    # endregion

    # region post processing
    "run_spectral_inference",
    "accuracy",
    # endregion
]