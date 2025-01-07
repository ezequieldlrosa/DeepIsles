# Author: Ezequiel de la Rosa (ezequieldlrosa@gmail.com)
# 03.04.2023

import os
import sys
PATH_DEEPSISLES = os.getcwd()  # path-to-ensemble-repo
sys.path.append(PATH_DEEPSISLES)
from src.isles22_ensemble import IslesEnsemble

# .nii/.nii.gz/.mha or DICOM folder
INPUT_FLAIR = os.path.join(PATH_DEEPSISLES, 'data', 'sub-strokecase0001_ses-0001_flair.nii.gz')  # path-to-FLAIR
INPUT_DWI = os.path.join(PATH_DEEPSISLES, 'data', 'sub-strokecase0001_ses-0001_dwi.nii.gz')      # pat-t-DWI
INPUT_ADC = os.path.join(PATH_DEEPSISLES, 'data', 'sub-strokecase0001_ses-0001_adc.nii.gz')      # path-to-ADC
OUTPUT_PATH = os.path.join(PATH_DEEPSISLES, 'example_test')                                      # path-to-output

stroke_segm = IslesEnsemble()

# Run the ensemble prediction
stroke_segm.predict_ensemble(ensemble_path=PATH_DEEPSISLES,
                             input_dwi_path=INPUT_DWI,
                             input_adc_path=INPUT_ADC,
                             input_flair_path=INPUT_FLAIR,
                             output_path=OUTPUT_PATH,
                             skull_strip=False,
                             fast=False,
                             save_team_outputs=False,
                             results_mni=False,
                             parallelize=True
                             )

