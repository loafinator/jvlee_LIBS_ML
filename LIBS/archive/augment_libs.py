"""
jvlee_LIBS_ML > LIBS > augment_libs.py

Currently not in use.

"""

# region Imports
# region plain
import sys, os
import time
# endregion

# region from
from pathlib import Path
# endregion

# region custom
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
log_path = r"G:\My Drive\RLSL\Python\LIBS\augment_libs_log.txt"
from utils import recursive_file_extension_converter, get_file_genre, Logger, enrich_with_progress, gen_speak
# endregion
# endregion

if __name__ == '__main__':
    start_time = time.perf_counter()
    sys.stdout = Logger(log_path)



    print('Complete, flushing and closing logger')
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    end_time = time.perf_counter()
    print(f'Time elapsed: {(end_time - start_time)/60} minutes.')