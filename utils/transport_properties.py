from .data_prep import batch_csv_enrichment

print('transport_properties.py loading...')


if __name__=='__main__':
    
    batch_csv_enrichment(
        data_root= r"W:\Phongikaroon Group\AndrewsH",
        csv_root= r"G:\My Drive\Radiochemistry_&_Laser_Spectroscopy_Lab\Data\ML_practice_data_from_W_drive"
    )