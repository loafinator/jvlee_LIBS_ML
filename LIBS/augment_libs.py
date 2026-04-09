"""
Python > LIBS > augment_libs.py
"""


import sys, os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
log_path = r"G:\My Drive\RLSL\Python\LIBS\augment_libs_log.txt"

from utils import recursive_file_extension_converter, get_file_genre, Logger, recursive_col_trim, enrich_with_progress, gen_speak, combine_and_save, trn_val_splitter
import time
from pathlib import Path


if __name__ == '__main__':
    start_time = time.perf_counter()
    sys.stdout = Logger(log_path)



    print('Complete, flushing and closing logger')
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    end_time = time.perf_counter()
    print(f'Time elapsed: {(end_time - start_time)/60} minutes.')