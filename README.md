![alt text](https://raw.githubusercontent.com/ezequieldlrosa/DeepIsles/main/logo.png)

# DeepISLES <br> State-of-the-art ischemic stroke lesion segmentation in MRI

## Introduction
DeepISLES is an out-of-the-box software tool for processing MRI scans and segmenting ischemic stroke lesions, developed in collaboration with leading teams from the [ISLES'22 MICCAI Challenge](https://isles22.grand-challenge.org/).

**Content:**
1. [Running DeepISLES](#running-deepisles)
5. [About DeepISLES Algorithms](#about-deepisles-algorithms)
6. [Citations](#citations)
7. [Questions](#questions)
8. [Acknowledgement](#acknowledgement)

## Running DeepISLES
![DeepISLES_Formats](https://raw.githubusercontent.com/ezequieldlrosa/DeepIsles/main/deepisles_formats.png)

DeepISLES is available in four different formats, catering to various use cases, from an easy-to-use web service to a command-line implementation for batch-processing large data sets. We provide an example MRI scan from the ISLES'22 dataset (Hernandez Petzsche et al., Sci Data 2022) in /data/:

- [Web-service](#web-service)
- [Standalone Software](#standalone-software)
- [Docker](#docker)
- [Source Git](#source-git)

## Web-service

DeepISLES is available as a [web-service](https://grand-challenge.org/algorithms/deepisles/).

### Usage

1. Create an account on Grand Challenge ([https://grand-challenge.org/](https://grand-challenge.org/)) and validate it.
2. Request access to [DeepISLES](https://grand-challenge.org/algorithms/deepisles/) ("Try out algorithm").
3. Drag-and-drop the MRI scans, wait until the job ends, and download your results!

Note: For the web-service, DICOM inputs must be provided for each MR sequence as a single .zip file.

## Standalone Software

A standalone version of DeepISLES is available, complete with a graphical user interface (GUI) that supports both single and batch processing. This version is hosted on [NITRC](https://www.nitrc.org/projects/deepisles/) ("Downloads" section on the left-hand side). Installation instructions for Linux and Windows systems (GPU required) are also provided there.

![DeepISLES_GUI](https://raw.githubusercontent.com/ezequieldlrosa/DeepIsles/main/deepisles_gui.png)


## [Docker](https://hub.docker.com/repository/docker/isleschallenge/deepisles)

For easy command-line usage, we provide a pre-built docker image.

### Requirements: 
- [Docker](https://docs.docker.com/engine/install/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Download the Docker image (``` docker pull isleschallenge/deepisles ```).

### Example Docker usage: 
```bash
docker run --gpus all -v */path_to_data*:/app/data isleschallenge/deepisles --dwi_file_name dwi.nii.gz --adc_file_name adc.nii.gz --flair_file_name flair.nii.gz
```

For DICOM usage, organize your data directory into subfolders (e.g., `/path_to_data/dwi_folder`, `/path_to_data/adc_folder`, `/path_to_data/flair_folder`), and pass the folder names as arguments by running:

```bash
docker run --gpus all -v */path_to_data*:/app/data isleschallenge/deepisles --dwi_file_name dwi_folder --adc_file_name adc_folder --flair_file_name flair_folder 
```

**Note**: Please replace `*/path_to_data*` with the path where you store your image data files, for example, `/mnt/media/data`. By default, the docker will save results into a (newly-created) subfolder named `results/` in that folder (e.g.  `/mnt/media/data/results`)

### Extra Parameters

- **`skull_strip`**: `True`/`False` (default: `False`) — Perform skull stripping on input images.
- **`fast`**: `True`/`False` (default: `False`) — Run a single model for faster execution.
- **`parallelize`**: `True`/`False` (default: `True`) — Up to 50% faster inference on GPUs with ≥12 GB memory.
- **`save_team_outputs`**: `True`/`False` (default: `False`) — Save outputs of individual models before ensembling.
- **`results_mni`**: `True`/`False` (default: `False`) — Save images and outputs in MNI.


## Source Git

### Installation
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

If you successfully installed all required packages, you can follow  the steps below to download and place the checkpoints.

1.3) Download the model weights from [here](https://zenodo.org/records/14026715) and decompress the file inside this repo.
From the terminal:
```
wget https://zenodo.org/records/14026715/files/stroke_ensemble_weights.7z?download=1
mv 'stroke_ensemble_weights.7z?download=1' weights.7z
7za x weights.7z 
```

Your directory should look like this:
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


### Usage
#### Supported Formats
- **Input formats**:  `.dcm`, `.nii`, `.nii.gz`, `.mha`.
- **Processing**: The algorithm works directly in the native image space — no additional preprocessing is required.

#### Required Image Modalities
- **DWI (b=1000)**: Required
- **ADC**: Required
- **FLAIR**: Required for ensemble (optional for single algorithm outputs)


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
```

### Get started 
Try DeepISLES out with the provided example data:
```bash
 python scripts/predict.py
```

## Citations
If you use this repository, please cite the following publications:

1. **de la Rosa, E., Reyes, M., Liew, S. L., Hutton, A., Wiest, R., Kaesmacher, J., ... & Wiestler, B. (2024).**  
   *A Robust Ensemble Algorithm for Ischemic Stroke Lesion Segmentation: Generalizability and Clinical Utility Beyond the ISLES Challenge.*  
   [arXiv:2403.19425](https://arxiv.org/abs/2403.19425)

2. **Hernandez Petzsche, M. R., de la Rosa, E., Hanning, U., Wiest, R., Valenzuela, W., Reyes, M., ... & Kirschke, J. S. (2022).**  
   *ISLES 2022: A multi-center magnetic resonance imaging stroke lesion segmentation dataset.*  
   [*Scientific Data, 9*(1), 762](https://www.nature.com/articles/s41597-022-01875-5)

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
