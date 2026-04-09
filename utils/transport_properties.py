"""
jvlee_LIBS_ML>utils>transport_properties.py

Currently not in use.

"""

# region Imports
# region custom
from .data_prep import batch_csv_enrichment
# endregion
# endregion

print('transport_properties.py loading...')


if __name__=='__main__':
    
    batch_csv_enrichment(
        data_root= r"W:\Phongikaroon Group\AndrewsH",
        csv_root= r"G:\My Drive\Radiochemistry_&_Laser_Spectroscopy_Lab\Data\ML_practice_data_from_W_drive"
    )