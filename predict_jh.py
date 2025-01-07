# Author: Ezequiel de la Rosa (ezequieldlrosa@gmail.com)
# 03.04.2023

import os
import sys
import time
import glob
import numpy as np
import pandas as pd
import nibabel as nib
ENSEMBLE_PATH = os.getcwd()  # path-to-ensemble-repo
print(ENSEMBLE_PATH)
#sys.path.append(ENSEMBLE_PATH)
#sys.path.append(os.path.join(ENSEMBLE_PATH, 'src'))
from src.isles22_ensemble import IslesEnsemble



input_case = '/home/edelarosa/Documents/datasets/example_dwi/16d87226-a1df402b-f84e9e07-2b9e3fa2-5b683e73'
    #input_case = '/mnt/hdda/edelarosa/jh_example/dataset01/raw_data/sub-0214eb85'
OUTPUT_PATH = '/home/edelarosa/Documents/datasets/example_dwi/test_me'
INPUT_FLAIR = os.path.join(input_case, 'flair')
INPUT_DWI = os.path.join(input_case, 'adc')
INPUT_ADC = os.path.join(input_case, 'dwi')


stroke_segm = IslesEnsemble()


# Run the ensemble prediction
stroke_segm.predict_ensemble(ensemble_path=ENSEMBLE_PATH,
                             input_dwi_path=INPUT_DWI,
                             input_adc_path=INPUT_ADC,
                             input_flair_path=INPUT_FLAIR,
                             output_path=OUTPUT_PATH,
                             fast=True,
                             save_team_outputs=True,
                             skull_strip=True,
                             results_mni=True)
print('sum:', np.sum(nib.load(os.path.join(OUTPUT_PATH, 'lesion_msk.nii.gz')).get_fdata()))



