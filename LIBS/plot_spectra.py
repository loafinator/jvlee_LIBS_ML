"""
python > LIBS > plot_spectra.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# spectra_path = r"G:\My Drive\RLSL\Data\enriched_CSVs\Experiments & Calculations - Copy_Pure Mg Study_LIBS_SampleM1_t1_run1.csv"
# spectra_path = r"G:\My Drive\RLSL\Data\enriched_CSVs\Data transfer_Sample Spectra_Analysis_Data_Import_SSCe6_lamba_4.csv"
# spectra_path = r"G:\My Drive\RLSL\Data\enriched_CSVs\Data transfer_Sample Spectra_3 wt CeCl3 Samples_Data_Import_lamba_4.csv"
# spectra_path = r"G:\My Drive\RLSL\Data\enriched_CSVs\Calibration Curves_200 shots 5 repeats_t4_8000 ppm t4.csv"
# spectra_path = r"G:\My Drive\RLSL\Data\enriched_CSVs\200 shots 5 repeats_Matlab analysis_Static LIBS Data_10000 ppm d2t1.csv"
# spectra_path = r"G:\My Drive\RLSL\Data\enriched_CSVs\Experiments & Calculations - Copy_Pure U Study_LIBS_SampleU2_t1_run1.csv"
# spectra_path = r"G:\My Drive\RLSL\Data\enriched_CSVs\Experiments & Calculations - Copy_Sm-Gd Study_LIBS_0.5-SmCl3_t2_run5.csv"
# spectra_path = r"G:\My Drive\RLSL\Data\enriched_CSVs\Experiments & Calculations - Copy_Sm-Gd Study_LIBS_blank_run7.csv"
# spectra_path = r"G:\My Drive\RLSL\Data\enriched_CSVs\MATs_to_CSVs_Taha_LIBS-Gate-delay-txt-data_SS1_GD_Data_Import_lamba_1.csv"
# spectra_path = r"G:\My Drive\RLSL\Data\enriched_CSVs\Molten Salt Aerosol_1st Exp 1 wt% CeCl3_Gas flow Study_12.5 mm flow K 1.csv"
# spectra_path = r"G:\My Drive\RLSL\Data\enriched_CSVs\Molten Salt Aerosol_1st Exp 1 wt% CeCl3_Laser energy study_60 mJ energy K 3.csv"
# spectra_path = r"G:\My Drive\RLSL\Data\enriched_CSVs\Molten Salt Aerosol_2nd Exp 5 wt% CeCl3_Energy Optimization_Flow_Import_lamba_2.csv"
# spectra_path = r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations - Copy\Pure Sm Studies\SmCl3 LIBS Glovebox\Optimization\Gate delay\15us\15us_delay_run3.csv"
# spectra_path = r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations - Copy\Pure Sm Studies\SmCl3 LIBS Glovebox\Optimization\Gate delay\26us\26us_delay_run4.csv"
# spectra_path = r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations - Copy\Pure Sm Studies\SmCl3 LIBS Glovebox\Optimization\Gate width\7us\7us_width_run1.csv"
# spectra_path = r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations - Copy\Pure Sm Studies\SmCl3 LIBS Glovebox\Optimization\Gate width\10us\10us_width_run4.csv"
# spectra_path = r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations - Copy\Pure Sm Studies\SmCl3 LIBS Glovebox\Optimization\Shot count\20 shots\20shots_run2.csv"
# spectra_path = r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations - Copy\Pure Sm Studies\SmCl3 LIBS Glovebox\Optimization\Shot count\100 shots\100shots_run4.csv"
# spectra_path = r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations - Copy\Pure Sm Studies\SmCl3 LIBS Glovebox\Optimization\Energy\Qdelay =85\85Qdelay_run2.csv"
# spectra_path = r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations - Copy\Pure Sm Studies\SmCl3 LIBS Glovebox\Optimization\Energy\Qdelay =140\140Qdelay_run5.csv"
# spectra_path = r"G:\My Drive\RLSL\Data\MATs_to_CSVs\Awilliams\DATA BACKUP\Backup 5_11_2016\NEUP LIBS Project\Preliminary Data and Information\Spectra\Molten Salt Aerosol\2nd Exp 5 wt% CeCl3\Energy Optimization\35 psi 8 mm flow 3.csv"
spectra_path = r"G:\My Drive\RLSL\Data\MATs_to_CSVs\AndrewsH\Backup\Experiments & Calculations - Copy\Pure U Study\LIBS\SampleU3_t1_run10.csv"

spectra_file = pd.read_csv(spectra_path, header=None)
spectra_data = spectra_file.iloc[:, 26:].apply(pd.to_numeric, errors='coerce')
wavelengths = spectra_data.iloc[0, :]
intensities = spectra_data.iloc[1, :]

title = 'U3_t1_run10_LIBS_Spectrum'
fig_save_root = r"G:\My Drive\RLSL\Projects\Spectroscopy\\"
fig_save_path = fig_save_root + title +'.png'

plt.figure(figsize=(10,5))
plt.plot(wavelengths, intensities)
# plt.plot(wavelengths, spectra_data.iloc[2, :])
# plt.plot(wavelengths, spectra_data.iloc[3, :])
# plt.plot(wavelengths, spectra_data.iloc[4, :])
# plt.plot(wavelengths, spectra_data.iloc[5, :])
plt.xlabel('Wavelength')
plt.ylabel('Intensity')
plt.title(title)
plt.savefig(fig_save_path, dpi=150, bbox_inches='tight')
# plt.show()