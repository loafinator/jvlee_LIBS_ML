"""
jvlee_LIBS_ML>utils>__init__.py
"""

# print('init: file_type_processing importing')
from .file_type_processing import *
# print('init: speak importing')
from .speak import *
# print('init: data_prep importing')
from .data_prep import * 
# print('init: plotting importing')
from .plotting import *
# print('init: debug importing')
from .debug import *
# print('all non-xpu imports done')
from .post_processing import *

import multiprocessing
# print(f'process name: {multiprocessing.current_process().name}')
if multiprocessing.current_process().name == 'MainProcess':
    # print('loading xpu_setup...')
    from .xpu_setup import *
    # print('xpu_setup loaded.')

__all__ = [
    # region file_type_processing
    "mpr_to_csv",
    "recursive_file_extension_converter",
    "file_segregator",
    "get_file_genre",
    # endregion

    # region xpu_setup
    # "init_xpu_d2l_classifier_trainer",
    # # "init_xpu_1d_cnn_trainer",
    "mae",
    # "init_xpu_mlp_trainer",
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
    "get_worker_logger"
    # endregion

    # region post processing
    "run_spectral_inference",
    "accuracy",
    # endregion
    
]