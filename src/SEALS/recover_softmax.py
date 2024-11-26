#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import argparse
from glob import glob

import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *

class ISLES22():
    def __init__(self, root):
        self.root = root
        self.data_dict = {}

    def load_data(self):
        dwi_folder = 'dwi'
        adc_folder = 'adc'
        # flair_folder  = 'flair-brain-mri'

        # self.dwi_path = glob(os.path.join(self.root, dwi_folder, '*.nii.gz'))[0]
        # self.adc_path = glob(os.path.join(self.root, adc_folder, '*.nii.gz'))[0]
        # self.flair_path = glob(os.path.join(self.root, flair_folder, '*.nii.gz'))[0]

        dwi_path = os.path.join(self.root, dwi_folder, dwi_folder + '.nii.gz')
        ss_dwi_path = os.path.join(self.root, dwi_folder, dwi_folder + '_ss.nii.gz')
        if os.path.exists(ss_dwi_path):
            self.dwi_path = ss_dwi_path
        elif os.path.exists(dwi_path):
            self.dwi_path = dwi_path

        adc_path = os.path.join(self.root, adc_folder, adc_folder + '.nii.gz')
        ss_adc_path = os.path.join(self.root, adc_folder, adc_folder + '_ss.nii.gz')

        if os.path.exists(ss_adc_path):
            self.adc_path = ss_adc_path
        elif os.path.exists(adc_path):
            self.adc_path = adc_path

def reimplement_resize(image_file, target_file, resample_method=sitk.sitkLinear):
    """
    Respacing file to target space size
    :param image_file: sitk.SimpleITK.Image
    :param target_spacing: np.array([H_space, W_space, D_space])
    :resample_method: SimpleITK resample method (e.g. SimpleITK.sitkLinear, SimpleITK.sitkNearestNeighbor)
    :return: resampled_image_file: sitk.SimpleITK.Image
    """
    # pdb.set_trace()
    if isinstance(image_file, str):
        image_file = sitk.ReadImage(image_file)
    elif isinstance(image_file, np.ndarray):
        image_file = sitk.GetImageFromArray(image_file)
    elif type(image_file) is not sitk.SimpleITK.Image:
        assert False, "Unknown data type to respaceing!"
    
    if isinstance(target_file, str):
        target_file = sitk.ReadImage(target_file)
    elif isinstance(target_file, np.ndarray):
        target_file = sitk.GetImageFromArray(target_file)
    elif type(target_file) is not sitk.SimpleITK.Image:
        assert False, "Unknown data type to respaceing!"

    # set target size
    target_origing   = image_file.GetOrigin()
    target_direction = image_file.GetDirection()
    target_spacing   = target_file.GetSpacing()
    target_size      = target_file.GetSize()

    # pdb.set_trace()
    # initialize resampler
    resampler_image = sitk.ResampleImageFilter()
    # set the parameters of image
    resampler_image.SetReferenceImage(image_file)  # set rasampled image meta data same to origin data
    resampler_image.SetOutputOrigin(target_origing)
    resampler_image.SetOutputDirection(target_direction)  # set target image space
    resampler_image.SetOutputSpacing(target_spacing)  # set target image space
    resampler_image.SetSize(target_size)  # set target image size
    if resample_method == sitk.sitkNearestNeighbor:
        resampler_image.SetOutputPixelType(sitk.sitkUInt8)
    else:
        resampler_image.SetOutputPixelType(sitk.sitkFloat32)
    resampler_image.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler_image.SetInterpolator(resample_method)

    # launch the resampler
    resampled_image_file = resampler_image.Execute(image_file)
    resampled_image_file = sitk.GetArrayFromImage(resampled_image_file)

    return resampled_image_file

def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        pkl_dict = pickle.load(f)
    return pkl_dict


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', help="root_path of 5 folds", required=True)
    parser.add_argument('-o', "--output_folder", required=True, help="folder for saving evaluation csv")
    parser.add_argument('-m', '--model_type', help='model_type, required.', required=True)
    parser.add_argument('-f', '--fold_index', default=0, help='evaluation_fold, required.', required=True)
    parser.add_argument('--raw_data_dir', type=str, required=True, help="Path to the raw data directory")
    args = parser.parse_args()

    raw_data_dir = args.raw_data_dir
    args = parser.parse_args()

    input_folder   = args.input_folder
    output_folder  = args.output_folder
    model_type     = args.model_type
    fold_index     = args.fold_index

    #raw_data_dir = '../input/images'
    dataset_ISLES22 = ISLES22(raw_data_dir)
    dataset_ISLES22.load_data()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files_npzs = sorted(glob(os.path.join(input_folder, model_type, fold_index, '*.npz')))
    
    for file_npz in files_npzs:
        file_pkl = file_npz.replace('npz', 'pkl')

        predict_softmax = np.load(file_npz)['softmax']

        pkl_dict = load_pkl(file_pkl)
        origin_size = pkl_dict['original_size_of_raw_data']
        origin_bbox = pkl_dict['crop_bbox']
        origin_array = np.zeros((2, origin_size[0], origin_size[1], origin_size[2]), dtype=np.float32)
        origin_array[0] = 1.0

        origin_array[:, 
                    origin_bbox[0][0]:origin_bbox[0][1], 
                    origin_bbox[1][0]:origin_bbox[1][1], 
                    origin_bbox[2][0]:origin_bbox[2][1]] = predict_softmax.astype(np.float32)

        target_array_shape = sitk.GetArrayFromImage(sitk.ReadImage(dataset_ISLES22.dwi_path)).shape
        target_array = np.zeros((2, target_array_shape[0], target_array_shape[1], target_array_shape[2]), dtype=np.float32)

        target_array[0] = reimplement_resize(origin_array[0], dataset_ISLES22.dwi_path)
        target_array[1] = reimplement_resize(origin_array[1], dataset_ISLES22.adc_path)

        target_file = os.path.join(output_folder, file_npz.split('/')[-1])
        
        np.savez(target_file, softmax=target_array)


