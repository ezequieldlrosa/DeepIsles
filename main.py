# Author: Ezequiel de la Rosa (ezequieldlrosa@gmail.com)
# 03.04.2023

import os
import sys
import argparse
import subprocess

cwd = os.getcwd()
sys.path.append(cwd)
from src.isles22_ensemble import IslesEnsemble


def main():
    parser = argparse.ArgumentParser(description='Isles22 Ensemble Algorithm')
    parser.add_argument('--dwi_file_name', type=str, required=True, help='Name of DWI image (required)')
    parser.add_argument('--adc_file_name', type=str, required=True, help='Name of ADC image (required)')
    parser.add_argument('--flair_file_name', type=str, default=None, help='Name of FLAIR image (optional)')
    parser.add_argument('--fast', action='store_true', help='Run only the best isles22 algorithm')
    parser.add_argument('--save_team_outputs', action='store_true', help='Save individual team outputs')
    parser.add_argument('--skull_strip', action='store_true', help='Run skull stripping')
    parser.add_argument('--parallelize', action='store_false', help='Run inference in parallel')
    parser.add_argument('--results_mni', action='store_true', help='Save results in MNI space')

    args = parser.parse_args()

    if args.dwi_file_name is None:
        raise ValueError('Please provide a DWI image file name')

    if args.adc_file_name is None:
        raise ValueError('Please provide an ADC image file name')

    INPUT_FLAIR = None
    if args.flair_file_name:
        INPUT_FLAIR = os.path.join('/app', 'data', args.flair_file_name)  # path-to-FLAIR
    INPUT_DWI = os.path.join('/app', 'data', args.dwi_file_name)  # pat-t-DWI
    INPUT_ADC = os.path.join('/app', 'data', args.adc_file_name)  # path-to-ADC

    if INPUT_FLAIR is not None:
        if os.path.exists(INPUT_FLAIR) is False:
            raise FileNotFoundError(f'{INPUT_FLAIR} does not exist')

    if os.path.exists(INPUT_DWI) is False:
        raise FileNotFoundError(f'{INPUT_DWI} does not exist')

    if os.path.exists(INPUT_ADC) is False:
        raise FileNotFoundError(f'{INPUT_ADC} does not exist')

    OUTPUT_PATH = os.path.join('/app', 'data', 'results')  # path-to-output
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    stroke_segm = IslesEnsemble()
    stroke_segm.predict_ensemble(ensemble_path=cwd,
                                 input_dwi_path=INPUT_DWI,
                                 input_adc_path=INPUT_ADC,
                                 input_flair_path=INPUT_FLAIR,
                                 output_path=OUTPUT_PATH,
                                 fast=args.fast,
                                 save_team_outputs=args.save_team_outputs,
                                 skull_strip=args.skull_strip,
                                 parallelize=args.parallelize,
                                 results_mni=args.results_mni)

    subprocess.run(['chmod', '777', OUTPUT_PATH], check=True)


if __name__ == '__main__':
    main()