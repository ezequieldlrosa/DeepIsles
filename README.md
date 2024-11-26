![alt text](https://raw.githubusercontent.com/ezequieldlrosa/DeepIsles/main/logo.png)

# DeepIsles
## ISLES'22 Ischemic Stroke Lesion Segmentation  

## Introduction
Algorithm to predict ischemic stroke lesions from MRI data.

This repository contains an ensemble algorithm devised in the context of the ISLES'22 MICCAI Challenge (https://isles22.grand-challenge.org/).
The top-3 leading algorithms are used in a majority-voting scheme to predict outputs.

1. [Installation](#installation)
2. [Usage](#usage)
3. [Get started](#get-started)
4. [Citation](#citation)
5. [About DeepIsles algorithms ](#about-deepisles-algorithms)
6. [Questions](#questions)
7. [Acknowledgement](#acknowledgement)


## Installation
1.1) Clone this repository.

```bash
git clone https://github.com/ezequieldlrosa/DeepIsles.git
cd DeepIsles
```

1.2) Create a conda environment and install dependencies. 
**Note: Mandatory Python version 3.8.0 (!)**

```bash
conda create --name deepisles python=3.8.0 pip=23.3.1
conda activate deepisles
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install -c conda-forge openslide-python
conda install python=3.8.0 # important, since pytorch triggers the installation of later python versions
pip install -e ./src/SEALS/
pip install -e ./src/FACTORIZER/model/factorizer/
pip install -e ./src/HD-BET
pip install -r requirements.txt

```

If successfully installed all required packages, you can follow  the steps below to download and place the checkpoints.

1.3) Download the model weights from [here](https://zenodo.org/records/14026715) and decompress the file inside this repo.
Your directory should look like:
```

DeepIsles/
├── weights/
│   ├── SEALS/
│   │   └── (...)
│   ├── NVAUTO/
│   │   └── (...)
│   └── FACTORIZER/
│       └── (...)
```


## Usage
### Supported Formats
- **Input formats**:  `.dcm`, `.nii`, `.nii.gz`, `.mha`.
- **Processing**: The algorithm works directly in the native image space — no additional preprocessing required.

### Required Image Modalities
- **DWI (b=1000)**: Required
- **ADC**: Required
- **FLAIR**: Required for ensemble (optional for single algorithm outputs)


### Python

```bash
PATH_DEEPISLES = 'path-to-repo' 
import sys
sys.path.append(PATH_DEEPISLES)
from isles22_ensemble import IslesEnsemble

INPUT_FLAIR = 'path-to-flair.nii.gz'
INPUT_ADC = 'path-to-adc.nii.gz'
INPUT_DWI = 'path-to-dwi.nii.gz'
OUTPUT_PATH = 'path-to-output-folder'

stroke_segm = IslesEnsemble()
stroke_segm.predict_ensemble(ensemble_path=PATH_DEEPISLES,
                 input_dwi_path=INPUT_DWI,
                 input_adc_path=INPUT_ADC,
                 input_flair_path=INPUT_FLAIR,
                 output_path=OUTPUT_PATH)
```

### Docker

Requirements: 
- [Docker](https://docs.docker.com/engine/install/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Download](https://hub.docker.com/repository/docker/isleschallenge/deepisles/general) the Docker image or build it as  ```bash docker build -t deepisles .```

Docker usage: 
```bash
docker run --gpus all -v /*path*/DeepIsles/data:/app/data deepisles --dwi_file_name sub-strokecase0001_ses-0001_dwi.nii.gz --adc_file_name sub-strokecase0001_ses-0001_adc.nii.gz --flair_file_name sub-strokecase0001_ses-0001_flair.nii.gz
```

### Extra Parameters

- **`skull_strip`**: `True`/`False` (default: `False`) — Perform skull stripping on input images.
- **`fast`**: `True`/`False` (default: `False`) — Run a single model for faster execution.
- **`parallelize`**: `True`/`False` (default: `True`) — Up to 50% faster inference on GPUs with ≥12 GB memory.
- **`save_team_outputs`**: `True`/`False` (default: `False`) — Save outputs of individual models before ensembling.
- **`results_mni`**: `True`/`False` (default: `False`) — Save images and outputs in MNI.



## Get started 
Try DeepIsles out over the provided example data:
```bash
 python scripts/predict.py
```

The example scan belongs to the ISLES'22 dataset (Hernandez Petzsche et al., Sci Data 2022).

## Citation
If you use this repository, please cite the following publications:

1. **de la Rosa, E., Reyes, M., Liew, S. L., Hutton, A., Wiest, R., Kaesmacher, J., ... & Wiestler, B. (2024).**  
   *A Robust Ensemble Algorithm for Ischemic Stroke Lesion Segmentation: Generalizability and Clinical Utility Beyond the ISLES Challenge.*  
   arXiv preprint: [arXiv:2403.19425](https://arxiv.org/abs/2403.19425)

2. **Hernandez Petzsche, M. R., de la Rosa, E., Hanning, U., Wiest, R., Valenzuela, W., Reyes, M., ... & Kirschke, J. S. (2022).**  
   *ISLES 2022: A multi-center magnetic resonance imaging stroke lesion segmentation dataset.*  
   *Scientific Data, 9*(1), 762.



## About DeepIsles algorithms 
* Algorithm SEALS is based on [nnUnet](https://github.com/MIC-DKFZ/nnUNet). Git [repo](https://github.com/Tabrisrei/ISLES22_SEALS) 

* Algorithm NVAUTO is based on [MONAI](https://github.com/Project-MONAI/MONAI) Auto3dseg. Git [repo](https://github.com/mahfuzmohammad/isles22)

* Algorithm SWAN is based on [FACTORIZER](https://github.com/pashtari/factorizer). Git [repo](https://github.com/pashtari/factorizer-isles22)


## Questions
Please contact Ezequiel de la Rosa (ezequiel.delarosa@uzh.ch).

## Acknowledgement
- We thank all ISLES'22 challenge participants, collaborators and organizers for allowing this work to happen. We also thank all developers and maintaners of the repos herein used. 
- Skull-stripping is done with [HD-BET](https://github.com/MIC-DKFZ/HD-BET).
- The used FLAIR-MNI atlas is obtained from [this paper](https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2019.00208/full) (https://zenodo.org/records/3379848). 
