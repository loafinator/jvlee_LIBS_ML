"""
jvlee_LIBS_ML > LIBS > get_libs.py

NOTE: update all paths before running from new location(machine).

Retrieve LIBS data from desired location. The use case is the W: drive.
Cleans, labels, compiles, and splits all data in preperation for training
and validation.

"""

# region Imports
# region plain
import sys, os
import time
import logging
import multiprocessing
import h5py 
# endregion

# region as
import numpy as np
from pathlib import Path
# endregion

# region form
from logging.handlers import QueueHandler, QueueListener
# endregion

# region custom
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from utils import (
    recursive_file_extension_converter, 
    get_file_genre, 
    Logger, 
    enrich_with_progress, 
    gen_speak, 
    combine_and_save_as_HDF5, 
    trn_val_splitter_HDF5, 
    load_h5_dataset
)
# endregion
# endregion

log_path = Path(r"G:\My Drive\RLSL\Python\LIBS\get_libs_log.txt")


if __name__ == '__main__':
    start_time = time.perf_counter()

    my_logger = Logger(log_path)
    my_logger.setFormatter(logging.Formatter('%(asctime)s  |  %(message)s'))

    sys.stdout = my_logger

    log_q = multiprocessing.Queue()
    listener = QueueListener(log_q, my_logger, respect_handler_level=True)
    listener.start()

    try:    
        # region get_file_genre
        # get_file_genre(
        #     data_root=r"W:\Phongikaroon Group",
        #     save_root=r"G:\My Drive\RLSL\Data\user",
        #     key=['libs', 'spectra'],
        #     skip_key=['aqueous', 'water', 'cement', 'obsidian'],
        #     allowed_extensions=['.txt'],        #'.csv', '.mat', '.asc', 
        #     preserve_structure=True,
        #     log_path=log_path
        # )
        # gen_speak('Libs files successfully copied.')
        # print('\nAll LIBS files have been copied from W: drive')
        # endregion

        # region recursive_file_extension_converter
        # recursive_file_extension_converter(
        #     data_root=r"G:\My Drive\RLSL\Data\user",
        #     save_root=r"G:\My Drive\RLSL\Data\MATs_to_CSVs",
        #     target_extension= ['.mat', '.asc', '.txt'],
        #     end_extension='.csv',
        #     skip_key=['Laser Energy Test'],
        #     preserve_structure=True,
        #     libs=True,
        #     log_path=log_path
        # )
        # gen_speak('m a t files converted to c s v files successfully.')
        # print('\nFile extension converter complete\n')
        # endregion

        # region pre-filtering file types
        # data_root = Path(r"G:\My Drive\RLSL\Data\MATs_to_CSVs")
        # enriched_root=Path(r"G:\My Drive\RLSL\Data\enriched_CSVs")


        # files = list(data_root.rglob('*.csv'))
        # print(f'Found {len(files)} files')

        # delete_key = [
        #     'test_snr', 
        #     'snr',
        #     'import',
        #     'solid_normalized_spectra',
        #     'iron with rust',
        #     'sampleu2_t1_r'
        # ]
        #     # 'delay', 'energy', 'enrgy', 'flow', 'pressure', 'test_snr', 'static_', 'blank'
            
        #     # 'energy_import_lamba_',
        #     # 'flow_impoprt_lamba_',
        #     # 'pressure_import_lamba_',
        #     # 'delay_import_lamba_',
        #     # 'data_import_lamba_'
        # files = [f for f in files if not any(key in f.name.lower() for key in delete_key)]
        # # #gen_speak('Delete-key files pre-filtered.')
        # print(f'Found {len(files)} files after pre-filtering against delete key')
        # endregion

        # region enrich_with_progress
        # data_root = Path(r"G:\My Drive\RLSL\Data\MATs_to_CSVs")
        # enriched_root=Path(r"G:\My Drive\RLSL\Data\enriched_CSVs")
        # files = list(data_root.rglob('*.csv'))

        # region Enrichment Check
        # enriched_root = Path(r"G:\My Drive\RLSL\Data\testing\enriched")
        # files = [
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations - Copy\Sm-Gd Study\LIBS\blank_run7.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations\SmCl3 Studies\SmCl3 CV and LIBS 1\Sm LIBS data\SL_Values_472_lambda_SLA.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations\SmCl3 Studies\SmCl3 CV and LIBS 1\Sm LIBS data\SLA_1_run2.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations\SmCl3 Studies\SmCl3 CV and LIBS 1\Sm LIBS raw data\10_smcl3_run3.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations\SmCl3 Studies\SmCl3 LIBS Glovebox\Optimization\Energy\Qdelay =70\70Qdelay_run2.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations\SmCl3 Studies\SmCl3 LIBS Glovebox\Optimization\Energy\Qdelay =70\SL_Values_472_lambda_1_smcl3.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations\SmCl3 Studies\SmCl3 LIBS Glovebox\Optimization\Energy\Qdelay =140\140Qdelay_run4.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations\SmCl3 Studies\SmCl3 LIBS Glovebox\Optimization\Gate delay\10us\10us_delay_run1.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations\SmCl3 Studies\SmCl3 LIBS Glovebox\Optimization\Gate delay\6us\6us_delay_run2.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations\SmCl3 Studies\SmCl3 LIBS Glovebox\Optimization\Gate width\7us\7us_width_run4.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations\SmCl3 Studies\SmCl3 LIBS Glovebox\Optimization\Gate width\10us\10us_width_run4.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations\SmCl3 Studies\SmCl3 LIBS Glovebox\Optimization\Shot count\20 shots\20shots_run1.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations\SmCl3 Studies\SmCl3 LIBS Glovebox\Optimization\Shot count\100 shots\100shots_run4.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations\SmCl3 Studies\SmCl3 LIBS Glovebox\Sm LIBS raw data\ANS samples\SLA_run2.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations\SmCl3 Studies\SmCl3 LIBS Glovebox\Sm LIBS raw data\ANS Samples Trial 2\SLB_run4.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations\SmCl3 Studies\SmCl3 LIBS Glovebox\Sm LIBS raw data\blank sample\blank_run6.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations\SmCl3 Studies\SmCl3 LIBS Glovebox\Sm LIBS raw data\Trial 1\10.0-SmCl3_run5.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations\SmCl3 Studies\SmCl3 LIBS Glovebox\Sm LIBS raw data\Trial 2\0.5-SmCl3_t2_run5.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations - Copy\Pure Mg Study\LIBS\SampleM1_t1_run1.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations - Copy\Pure U Study\LIBS\SampleU1_t1_run2.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations - Copy\Pure U Study\LIBS\SampleU25_t1_run4.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations - Copy\Pure U Study\LIBS\SampleU2_t1_run6.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations - Copy\Sm-Gd Study\LIBS\0.5-SmCl3_t2_run4.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations - Copy\Sm-Gd Study\LIBS\10.0-SmCl3_t2_run8.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations - Copy\Sm-Gd Study\LIBS\Sample1_t1_run9.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations - Copy\Sm-Gd Study\LIBS\SampleT1_t1_run7.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations - Copy\U-Gd Study\LIBS\SampleU1_t1_run1.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations - Copy\U-Gd Study\LIBS\SampleU25_t1_run3.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations - Copy\U-Gd Study\LIBS\SampleUG12_t1_run9.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\backup 5_10_2019\Experiments & Calculations\Withdrawal Studies\Pure Mg Study\LIBS\SampleM1_t1_run3.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\backup 5_10_2019\Experiments & Calculations\Withdrawal Studies\Pure Mg Study\LIBS\SampleM4_t1_run9.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\backup 5_10_2019\Experiments & Calculations\Withdrawal Studies\Gd-Mg Study\LIBS\SampleGM2_t1_run1.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\backup 5_10_2019\Experiments & Calculations\Withdrawal Studies\Gd-Mg Study\LIBS\SampleGM4_t1_run10.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 2-17-2017\NEUP Keith Data\Data transfer\Sample Spectra\0.5 wt CeCl3 Samples\1 rep 1.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 2-17-2017\NEUP Keith Data\Data transfer\Sample Spectra\1 wt CeCl3 Samples\8 rep 1.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 2-17-2017\NEUP Keith Data\Data transfer\Sample Spectra\2 wt CeCl3 Samples\14 rep 3.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 2-17-2017\NEUP Keith Data\Data transfer\Sample Spectra\3 wt CeCl3 Samples\21 rep 2.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 2-17-2017\NEUP Keith Data\Data transfer\Sample Spectra\4 wt CeCl3 Samples\30 rep 2.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 2-17-2017\NEUP Keith Data\Data transfer\Sample Spectra\5 wt CeCl3 Samples\33 rep 5.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 2-17-2017\NEUP Keith Data\Data transfer\Sample Spectra\Analysis\Solid_Normalized_Spectra_lamba_S.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\CeCl3-LiCl-KCl-H2O Test\Static\Calibration Curves\200 shots 5 repeats\d2t1\500 ppm d2t1.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\CeCl3-LiCl-KCl-H2O Test\Static\Calibration Curves\200 shots 5 repeats\d2t1\3000 ppm d2t1.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\CeCl3-LiCl-KCl-H2O Test\Static\Calibration Curves\200 shots 5 repeats\d2t1\10000 ppm d2t1.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\CeCl3-LiCl-KCl-H2O Test\Static\Calibration Curves\200 shots 5 repeats\t1\10000 ppm t1.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\CeCl3-LiCl-KCl-H2O Test\Static\Calibration Curves\200 shots 5 repeats\t1\4000 ppm t1.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\CeCl3-LiCl-KCl-H2O Test\Static\Calibration Curves\200 shots 5 repeats\t1\500 ppm t1.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\CeCl3-LiCl-KCl-H2O Test\Static\Calibration Curves\200 shots 5 repeats\Matlab analysis\200 ppm\200 ppm d3t1.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\CeCl3-LiCl-KCl-H2O Test\Static\Calibration Curves\200 shots 5 repeats\Matlab analysis\200 ppm\Static_200ppm_Values_lamba.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\CeCl3-LiCl-KCl-H2O Test\Static\Calibration Curves\200 shots 5 repeats\Matlab analysis\200 ppm\Static_200ppm_Values_Li_lamba.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\CeCl3-LiCl-KCl-H2O Test\Static\Calibration Curves\200 shots 5 repeats\Matlab analysis\6000 ppm\6000 ppm t2.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\CeCl3-LiCl-KCl-H2O Test\Static\Calibration Curves\200 shots 5 repeats\Matlab analysis\6000 ppm\6000 ppm d3t1.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\CeCl3-LiCl-KCl-H2O Test\Static\Calibration Curves\200 shots 5 repeats\Matlab analysis\100 ppm\100 ppm d3t2.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\CeCl3-LiCl-KCl-H2O Test\Static\Calibration Curves\200 shots 5 repeats\Matlab analysis\100 ppm\100 ppm t5.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\CeCl3-LiCl-KCl-H2O Test\Static\Calibration Curves\200 shots 5 repeats\Matlab analysis\Static LIBS Data\100 ppm d3t3.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\CeCl3-LiCl-KCl-H2O Test\Static\Calibration Curves\200 shots 5 repeats\Matlab analysis\Static LIBS Data\200 ppm t5.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\CeCl3-LiCl-KCl-H2O Test\Static\Calibration Curves\200 shots 5 repeats\t2\4000 ppm t2.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\CeCl3-LiCl-KCl-H2O Test\Static\Calibration Curves\200 shots 4000 ppm x10 1 Hz\t3.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\CeCl3-LiCl-KCl-H2O Test\Static\Calibration Curves\500 shots 4000 ppm x ten\t7.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\CeCl3-LiCl-KCl-H2O Test\Static\Calibration Curves\500 shots no repeats\4000 ppm Ce.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\CeCl3-LiCl-KCl-H2O Test\Static\Delay Time\Internal Trigger\409.5 us delay.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\CeCl3-LiCl-KCl-H2O Test\Static\Repetability Test\Internal Trigger\Test 3.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\CeCl3-LiCl-KCl-H2O Test\Static\Repetability Test\External Trigger\repeatability test 2.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Fiber optic test\Cable 2 into Cable 1 50 to 100 direction 2.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Iron with rust\Spot 1 clean at 87 mJ.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Iron with rust\Spot 2 rust 87 mJ 10.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Iron with rust\Spot 2 rust 87 mJ 12-22.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Iron with rust\Spot 3 rust 45 mJ 15 shots.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\1st Exp 1 wt% CeCl3\Gas flow Study\5 mm flow KR 3.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\1st Exp 1 wt% CeCl3\Gas flow Study\Flow_12p5mm_SNR_Values_lamba.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\1st Exp 1 wt% CeCl3\Gate delay study\Study 2\2 us delay K 2.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\1st Exp 1 wt% CeCl3\Gate delay study\Study 2\Delay_1us_SNR_Values_lamba.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\1st Exp 1 wt% CeCl3\Gate delay study\15 mm flow\2 us delay K 3.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\1st Exp 1 wt% CeCl3\Gate delay study\15 mm flow\Delay1us_SNR_Values_lamba.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\1st Exp 1 wt% CeCl3\Laser energy study\40 mJ energy K 3.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\1st Exp 1 wt% CeCl3\Optics and flow optimization\15 units flow kinetic range 3.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\1st Exp 1 wt% CeCl3\Optics and flow optimization\Test_SNR_Values_lamba.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\2nd Exp 5 wt% CeCl3\2nd Run, Energy\65 mJ 15 psi 1.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\2nd Exp 5 wt% CeCl3\Energy Optimization\165 mJ 35 psi 9 mm flow 1.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\2nd Exp 5 wt% CeCl3\Gas flow Optimization\50 psi 10 mm flow 1.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\2nd Exp 5 wt% CeCl3\Gate Delay\5us delay 35 psi 9 mm flow 1.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\3rd Exp 5 wt% CeCl3\Energy\115 mJ Rep 2.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\3rd Exp 5 wt% CeCl3\Pressure\115 mJ Rep 1.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\3rd Exp 5 wt% CeCl3\Pressure\30 psi Rep 1.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\4th Exp 3wt% CeCl3\Gate Delay\0p25 us rep 3.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\5th Exp 3wt% CeCl3\Gate delay Optimization\8 us rep 2.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\5th Exp 3wt% CeCl3\Gate delay Optimization\25 us rep 1.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\6th EXP Pure LiCl-KCl\Rep 5.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\6th EXP Pure LiCl-KCl\Rep 13.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\7th EXP 0.1 wt% CeCl3\0.1 wt Rep 7.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\MS8 0.1 wt% CeCl3\Data\MS8 Rep 3.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\MS8 0.1 wt% CeCl3\Data\MS8_Data_lamba_MS8.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\MS9 0.5 wt% CeCl3\Data\MS9 Rep 4.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\MS10 EXP 1 wt% CeCl3\Data\MS10 Rep 3.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\MS11 EXP 1 wt% CeCl3\Data MS11\Rep 4.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\MS12 3wt% CeCl3\Data\MS12 17 Rep 2.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\MS12 3wt% CeCl3\Data\MS12 Rep 1.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\MS13 5wt% CeCl3\Data\MS13 Rep 4.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\MS14 5 wt% CeCl3\Data\MS14 Rep 3.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\MS14 5 wt% CeCl3\Data at 50 mJ\MS14 50 Rep 5.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\MS14 5 wt% CeCl3\Data ELT\MS14 ELT Rep 6.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\MS14 5 wt% CeCl3\Data long term\MS14 LT Rep 4.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\MS15\Data\MS15 Rep 4.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\MS16 0.5 wt% CeCl3\Data 300\MS16 300 Rep 3.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\MSU1 1 wt% UCl3\Data\MSU1 Rep 6.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\MSU2 1 wt% UCl3\Data\MSU2 Rep 2.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\MSU3\Data\MSU3 Rep 4.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\MSU3\Data\MSU3_Data_lamba_MSU3.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Molten Salt Aerosol\MSU4\Data\MSU4 Rep 12.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Solid Salt\DU1\DU1 Rep 14.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 9-12-2016\Spectra\Solid Salt\SSU1\Data 10 us\SSU1 Rep 4.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\KBrice\NEUP Keith Data\Data transfer\Sample Spectra\0.5 wt CeCl3 Samples\1 rep 2.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\KBrice\NEUP Keith Data\Data transfer\Sample Spectra\1 wt CeCl3 Samples\8 rep 6.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\KBrice\NEUP Keith Data\Data transfer\Sample Spectra\2 wt CeCl3 Samples\14 rep 4.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\KBrice\NEUP Keith Data\Data transfer\Sample Spectra\3 wt CeCl3 Samples\22 rep 3.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\KBrice\NEUP Keith Data\Data transfer\Sample Spectra\4 wt CeCl3 Samples\27 rep 4.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\KBrice\NEUP Keith Data\Data transfer\Sample Spectra\5 wt CeCl3 Samples\32 rep 5.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Killinger\2018-2019\VTR\Quasi-Reference Electrode Study\LIBS\Test_4.csv",
        #     r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Taha\LIBS-Gate-delay-txt-data\gtd1.7r2.csv",
        # ]
        # endregion
        
        # enrich_with_progress(
        #     files=files,
        #     enriched_root=enriched_root,
        #     technique_name='libs',
        #     drop_columns=None,
        #     rename_columns=None,
        #     composition_columns=None,
        #     min_wavelength=250.0, 
        #     max_wavelength=1000.0,      # 350 to 800 was good for UV-Vis but need bigger range
        #                               # for LIBS. Going to go with 250 - 1000
        #     allowed_extensions=None,
        #     required_columns=None,
        #     salt_states=None,
        #     experimental_variation_columns=None,
        #     include_experimental_variation_columns=True,
        #     log_q=log_q
        # )

        # # region Run Notes
        # NOTE the "G:\My Drive\RLSL\Data\enriched_CSVs\Data transfer_Sample Spectra_Analysis_Data_Import_SSCe4_lamba_4.csv"
        # files do not have any concentration data. I think that they probably should have Ce concentration or even SSCe
        # which I think means Stain Steel Cesium... But I am not going to re-run the whole thing right now because it freaking
        # took 6.85 hours to run. I might start it up again tonight if I really feel like it lol.
        # 
        # NOTE the Pure U studies don't have any U concentrations...... Checking if I should make it so that it really is
        # 100% U for the pure studies... Same for Pure Mg.
        #  
        # NOTE the maps have all been made for AWilliams and AndrewsH, none of the others
        # at this point, it kind of looks like Killinger might be either really great
        # or really bad bc its not a LiCl-KCl, but  that is a problem for tomorrow at the
        # earliest lol.
        # 
        # NOTE's on the last run:
            # 15 mm flow didn't get populated. Actually there is no flow column at all...
            # same with delay...
            # I should probably remove the Iron with rust runs.
            # No shot column either...
            # I think SSCe means Salt Sample Cesium....?????
            # is it possible to find a concentration map for SmCl3_t#?
            # the sm_Gd study_libs_samle#_gdcl3_data_import_lambda... should have the same concentration as the sample3_ti,
            #   just need to add it to the map.
            # same for T# samples
            # There is no concentration data in Taha Gate Delay txt files...
            #   same goes for the data import lambdas for the same files.
            # See if there is a map for:  "G:\My Drive\RLSL\Data\enriched_CSVs\Molten Salt Aerosol_1st Exp 1 wt% CeCl3_Gas flow Study_5 mm flow K 1.csv"
            # Need Energy column
            # Need Pressure column as well
            # Need a rep column
            # Is there a map for MS16? 
            # Need a Qdelay column
            # No concentration data for: "G:\My Drive\RLSL\Data\enriched_CSVs\Spectra_Solid Salt_DU1_DU1 Rep 12.csv"
            # No concentration data for: "G:\My Drive\RLSL\Data\enriched_CSVs\Static_Calibration Curves_200 shots 4000 ppm x10 1 Hz_t1.csv"
            # No concentration data for: "G:\My Drive\RLSL\Data\enriched_CSVs\VTR_Quasi-Reference Electrode Study_LIBS_Test_7.csv" 
            # No concentration data for:  "G:\My Drive\RLSL\Data\enriched_CSVs\Withdrawal Studies_Gd-Mg Study_LIBS_SampleGM1_t1_run1.csv"
                #  Enriching Spectra_Solid Salt_DU1_MSU4_Data_lamba_MSU4.csv
                # Loaded MSU4 Rep 23.csv → shape (301, 27448)
                # ❌ Failed to clean MSU1 Rep 10.csv: [Errno 22] Invalid argument
                # Skipped enrichment of MSU1 Rep 10.csv (techniques=UNKNOWN)
                # Enriching Spectra_Solid Salt_DU1_DU1 Rep 1.csv
                # ❌ Failed to clean MSU1 Rep 5.csv: [Errno 22] Invalid argument
                # Skipped enrichment of MSU1 Rep 5.csv (techniques=UNKNOWN)
                # Enriching Molten Salt Aerosol_MS13 5wt% CeCl3_Data_MS13 Rep 3.csv
                # ❌ Failed to clean MS14 50 Rep 3.csv: [Errno 22] Invalid argument
                # Skipped enrichment of MS14 50 Rep 3.csv (techniques=UNKNOWN)
                # ❌ Failed to clean MS14 LT Rep 3.csv: [Errno 22] Invalid argument
                # Skipped enrichment of MS14 LT Rep 3.csv (techniques=UNKNOWN)
                # Enriching Molten Salt Aerosol_MS13 5wt% CeCl3_Data_MS13 Rep 4.csv
                # Enriching Molten Salt Aerosol_MS13 5wt% CeCl3_Data_MS13 Rep 5.csv
                # ❌ Failed to clean MS14 LT Rep 5.csv: [Errno 22] Invalid argument
                # Skipped enrichment of MS14 LT Rep 5.csv (techniques=UNKNOWN)
                # Enriching Molten Salt Aerosol_MS13 5wt% CeCl3_Data_MS13 Rep 6.csv
                # ❌ Failed to clean MS14 50 Rep 6.csv: [Errno 22] Invalid argument
                # Skipped enrichment of MS14 50 Rep 6.csv (techniques=UNKNOWN)
                # Enriching Molten Salt Aerosol_MS13 5wt% CeCl3_Data_MS13 Rep 7.csv
                # ❌ Failed to clean MS13 Rep 2.csv: [Errno 22] Invalid argument
                # Skipped enrichment of MS13 Rep 2.csv (techniques=UNKNOWN)
                # Enriching Molten Salt Aerosol_MS13 5wt% CeCl3_Data_MS13 Rep 8.csv
                # ❌ Failed to clean MS13 Rep 1.csv: [Errno 22] Invalid argument
                # ❌ Failed to clean MS14 ELT Rep 4.csv: [Errno 22] Invalid argument
                # ❌ Failed to clean MS14 ELT Rep 3.csv: [Errno 22] Invalid argument
                # Skipped enrichment of MS13 Rep 1.csv (techniques=UNKNOWN)
                # Skipped enrichment of MS14 ELT Rep 3.csv (techniques=UNKNOWN)
                # Skipped enrichment of MS14 ELT Rep 4.csv (techniques=UNKNOWN)
                # Enriching Molten Salt Aerosol_MS12 3wt% CeCl3_Data_MS12_Data_lamba_MS12.csv
                # Enriching Molten Salt Aerosol_MS12 3wt% CeCl3_Data_MS12 12 Rep 1.csv
                # Enriching Molten Salt Aerosol_MS12 3wt% CeCl3_Data_MS12 12 Rep 2.csv
                # ❌ Failed to clean MS14 50 Rep 4.csv: [Errno 22] Invalid argument
                # Skipped enrichment of MS14 50 Rep 4.csv (techniques=UNKNOWN)
                # Enriching Molten Salt Aerosol_MS12 3wt% CeCl3_Data_MS12 12 Rep 3.csv
                # ❌ Failed to clean MS14 ELT Rep 6.csv: [Errno 22] Invalid argument
                # ❌ Failed to clean MS14 LT Rep 2.csv: [Errno 22] Invalid argument
                # Skipped enrichment of MS14 ELT Rep 6.csv (techniques=UNKNOWN)
                # Skipped enrichment of MS14 LT Rep 2.csv (techniques=UNKNOWN)
                # Enriching Molten Salt Aerosol_MS12 3wt% CeCl3_Data_MS12 15 Rep 2.csv
                # Enriching Molten Salt Aerosol_MS12 3wt% CeCl3_Data_MS12 15 Rep 1.csv
                # ❌ Failed to clean MS14 LT Rep 1.csv: [Errno 22] Invalid argument
                # Skipped enrichment of MS14 LT Rep 1.csv (techniques=UNKNOWN)
                # Enriching Molten Salt Aerosol_MS12 3wt% CeCl3_Data_MS12 15 Rep 3.csv
                # ❌ Failed to clean MS13 Rep 8.csv: [Errno 22] Invalid argument
                # Skipped enrichment of MS13 Rep 8.csv (techniques=UNKNOWN)
                # Enriching Molten Salt Aerosol_MS12 3wt% CeCl3_Data_MS12 17 Rep 1.csv
                # ❌ Failed to clean MS14 LT Rep 6.csv: [Errno 22] Invalid argument
                # Skipped enrichment of MS14 LT Rep 6.csv (techniques=UNKNOWN)
                # Enriching Molten Salt Aerosol_MS12 3wt% CeCl3_Data_MS12 17 Rep 2.csv
                # ❌ Failed to clean MS14 ELT Rep 7.csv: [Errno 22] Invalid argument
                # Skipped enrichment of MS14 ELT Rep 7.csv (techniques=UNKNOWN)
                # Enriching Molten Salt Aerosol_MS12 3wt% CeCl3_Data_MS12 17 Rep 3.csv
                # ❌ Failed to clean MS12 12 Rep 1.csv: [Errno 22] Invalid argument
                # Skipped enrichment of MS12 12 Rep 1.csv (techniques=UNKNOWN)
                # Enriching Molten Salt Aerosol_MS12 3wt% CeCl3_Data_MS12 20 Rep 1.csv
                # ❌ Failed to clean MS12 15 Rep 2.csv: [Errno 22] Invalid argument
                # Skipped enrichment of MS12 15 Rep 2.csv (techniques=UNKNOWN)
                # Enriching Molten Salt Aerosol_MS12 3wt% CeCl3_Data_MS12 20 Rep 2.csv
                # ❌ Failed to clean MS12 12 Rep 2.csv: [Errno 22] Invalid argument
                # Skipped enrichment of MS12 12 Rep 2.csv (techniques=UNKNOWN)
        # gen_speak('File enrichment complete')
        # print('\nFile metadata enrichment complete\n')
        # endregion
        # endregion

        # region combine_and_save
        # combined_path = combine_and_save_as_HDF5(
        #     individuals_root=Path(r"G:\My Drive\RLSL\Data\enriched_CSVs"),
        #     combined_root=Path(r"G:\My Drive\RLSL\Data\combined_CSVs"),
        #     technique='libs',
        #     compression='gzip',
        #     compression_level=3,
        #     log_path=log_path
        # )
        # gen_speak('All files have been combined.')
        # if combined_path:
        #     print(f'\nFiles have been combined at: {combined_path.resolve()}')
        # else:
        #     print('\nWarning: combine_and_save_as_HDF5 returned None - no files combined!')


        # combined_path = combine_and_save(
        #                     individuals_root=Path(r"G:\My Drive\RLSL\Data\enriched_CSVs"),
        #                     combined_root=Path(r"G:\My Drive\RLSL\Data\combined_CSVs"),
        #                     technique=None,
        #                 )
        # endregion

        # region Load h5
        
        

        # endregion

        # region trn_val_splitter

        h5_path = r"G:\My Drive\RLSL\Data\combined_CSVs\combined_LIBS.h5"
        h5_split_path = r"G:\My Drive\RLSL\Data\combined_CSVs\trn_val_split_LIBS.h5"
        # spectra, metadata, wavelengths = load_h5_dataset(
        #     h5_path=h5_path,
        #     meta_cols_wanted=None
        # )

        trn_val_splitter_HDF5(
            h5_path= h5_path,
            output_path= h5_split_path,
            test_size= 0.2,
            random_state= 42,
            log_path=log_path
        )

        # trn_val_splitter(
        #     df_path=Path(r"G:\My Drive\RLSL\Data\combined_CSVs\cleaned_combined_LIBS.csv"),
        #     trn_path=Path(r"G:\My Drive\RLSL\Data\combined_CSVs\training_data.csv"),
        #     val_path=Path(r"G:\My Drive\RLSL\Data\combined_CSVs\validation_data.csv"),
        #     test_size=0.2,
        #     random_state=42,
        #     shuffle=True,
        #     log_q = log_q
        # )
        # endregion
        
        # dp = Path(r"G:\My Drive\RLSL\Data\combined_CSVs\cleaned_combined_LIBS.csv").resolve()
        # df = pd.read_csv(dp)

        # has_temp = df['temperature_C'].notna() & (df['temperature_C'] != 0)
        # print(f'{has_temp.sum()} / {df.shape[0]} rows have temperature data')

        # # also worth seeing the distribution of what temperatures you actually have
        # print(f"\nTemperature value counts:")
        # print(df['temperature_C'].value_counts().sort_index())

        #df = r"G:\My Drive\Radiochemistry_&_Laser_Spectroscopy_Lab\Data\LIBS\everything_LIBS_from_Wdrive\enriched_CSVs\column_check_folder\1_smcl3_Data_Import_lambda_1_smcl3.csv"
        #df = pd.read_csv(df)
        #print(df.columns[:30].to_list())

        print('Complete, flushing and closing logger')
        end_time = time.perf_counter()
        print(f'Time elapsed: {(end_time - start_time)/60} minutes.')

    finally:
        listener.stop()
        sys.stdout = sys.__stdout__
        my_logger.close()
