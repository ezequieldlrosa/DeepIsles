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

import shutil
from copy import deepcopy
from typing import Union, Tuple
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
from multiprocessing import Pool
from nnunet.postprocessing.connected_components import apply_postprocessing_to_folder, load_postprocessing

def save_segmentation_nifti_from_softmax(segmentation_softmax: Union[str, np.ndarray], 
                                         out_fname: str,
                                         order: int = 1,
                                         region_class_order: Tuple[Tuple[int]] = None,
                                         seg_postprogess_fn: callable = None, seg_postprocess_args: tuple = None,
                                         resampled_npz_fname: str = None,
                                         non_postprocessed_fname: str = None, force_separate_z: bool = None,
                                         interpolation_order_z: int = 0, verbose: bool = True):
    """
    This is a utility for writing segmentations to nifto and npz. It requires the data to have been preprocessed by
    GenericPreprocessor because it depends on the property dictionary output (dct) to know the geometry of the original
    data. segmentation_softmax does not have to have the same size in pixels as the original data, it will be
    resampled to match that. This is generally useful because the spacings our networks operate on are most of the time
    not the native spacings of the image data.
    If seg_postprogess_fn is not None then seg_postprogess_fnseg_postprogess_fn(segmentation, *seg_postprocess_args)
    will be called before nifto export
    There is a problem with python process communication that prevents us from communicating obejcts
    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
    communicated by the multiprocessing.Pipe object then the placeholder (\%i I think) does not allow for long
    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
    patching system python code.) We circumvent that problem here by saving softmax_pred to a npy file that will
    then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
    filename or np.ndarray for segmentation_softmax and will handle this automatically
    :param segmentation_softmax:
    :param out_fname:
    :param properties_dict:
    :param order:
    :param region_class_order:
    :param seg_postprogess_fn:
    :param seg_postprocess_args:
    :param resampled_npz_fname:
    :param non_postprocessed_fname:
    :param force_separate_z: if None then we dynamically decide how to resample along z, if True/False then always
    /never resample along z separately. Do not touch unless you know what you are doing
    :param interpolation_order_z: if separate z resampling is done then this is the order for resampling in z
    :param verbose:
    :return:
    """
    if verbose: print("force_separate_z:", force_separate_z, "interpolation order:", order)

    if isinstance(segmentation_softmax, str):
        assert isfile(segmentation_softmax), "If isinstance(segmentation_softmax, str) then " \
                                             "isfile(segmentation_softmax) must be True"
        del_file = deepcopy(segmentation_softmax)
        segmentation_softmax = np.load(segmentation_softmax)
        os.remove(del_file)

    # first resample, then put result into bbox of cropping, then save
    current_shape = segmentation_softmax.shape
    # current_spacing = dct.get('spacing_after_resampling')
    # original_spacing = dct.get('original_spacing')

    
    if verbose: print("no resampling necessary")
    seg_old_spacing = segmentation_softmax

    if resampled_npz_fname is not None:
        np.savez_compressed(resampled_npz_fname, softmax=seg_old_spacing.astype(np.float16))
        # this is needed for ensembling if the nonlinearity is sigmoid

    if region_class_order is None:
        seg_old_spacing = seg_old_spacing.argmax(0)
    else:
        seg_old_spacing_final = np.zeros(seg_old_spacing.shape[1:])
        for i, c in enumerate(region_class_order):
            seg_old_spacing_final[seg_old_spacing[i] > 0.5] = c
        seg_old_spacing = seg_old_spacing_final

    seg_old_size = seg_old_spacing

    if seg_postprogess_fn is not None:
        seg_old_size_postprocessed = seg_postprogess_fn(np.copy(seg_old_size), *seg_postprocess_args)
    else:
        seg_old_size_postprocessed = seg_old_size

    seg_resized_itk = sitk.GetImageFromArray(seg_old_size_postprocessed.astype(np.uint8))
    sitk.WriteImage(seg_resized_itk, out_fname)

    if (non_postprocessed_fname is not None) and (seg_postprogess_fn is not None):
        seg_resized_itk = sitk.GetImageFromArray(seg_old_size.astype(np.uint8))
        sitk.WriteImage(seg_resized_itk, non_postprocessed_fname)

def merge_files(files, out_file, override, store_npz):
    if override or not isfile(out_file):
        softmax = [np.load(f)['softmax'][None] for f in files]
        softmax = np.vstack(softmax)
        softmax = np.mean(softmax, 0)

        regions_class_order = None

        # Softmax probabilities are already at target spacing so this will not do any resampling (resampling parameters
        # don't matter here)
        save_segmentation_nifti_from_softmax(softmax, out_file, 3, regions_class_order, None, None,
                                             force_separate_z=None)
        if store_npz:
            np.savez_compressed(out_file[:-7] + ".npz", softmax=softmax)


def merge(folders, output_folder, threads, override=True, postprocessing_file=None, store_npz=False):
    maybe_mkdir_p(output_folder)

    if postprocessing_file is not None:
        output_folder_orig = deepcopy(output_folder)
        output_folder = join(output_folder, 'not_postprocessed')
        maybe_mkdir_p(output_folder)
    else:
        output_folder_orig = None

    patient_ids = [subfiles(i, suffix=".npz", join=False) for i in folders]
    patient_ids = [i for j in patient_ids for i in j]
    patient_ids = [i[:-4] for i in patient_ids]
    patient_ids = np.unique(patient_ids)
    # import pdb; pdb.set_trace()
    for f in folders:
        assert all([isfile(join(f, i + ".npz")) for i in patient_ids]), "Not all patient npz are available in " \
                                                                        "all folders"

    files = []
    out_files = []
    for p in patient_ids:
        files.append([join(f, p + ".npz") for f in folders])
        out_files.append(join(output_folder, p + ".nii.gz"))

    p = Pool(threads)
    p.starmap(merge_files, zip(files, out_files, [override] * len(out_files), [store_npz] * len(out_files)))
    p.close()
    p.join()

    if postprocessing_file is not None:
        for_which_classes, min_valid_obj_size = load_postprocessing(postprocessing_file)
        print('Postprocessing...')
        apply_postprocessing_to_folder(output_folder, output_folder_orig,
                                       for_which_classes, min_valid_obj_size, threads)
        shutil.copy(postprocessing_file, output_folder_orig)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="This script will merge predictions (that were prdicted with the "
                                                 "-npz option!). You need to specify a postprocessing file so that "
                                                 "we know here what postprocessing must be applied. Failing to do so "
                                                 "will disable postprocessing")
    parser.add_argument('-f', '--folders', nargs='+', help="list of folders to merge. All folders must contain npz "
                                                           "files", required=True)
    parser.add_argument('-o', '--output_folder', help="where to save the results", required=True, type=str)
    parser.add_argument('-t', '--threads', help="number of threads used to saving niftis", required=False, default=2,
                        type=int)
    parser.add_argument('-pp', '--postprocessing_file', help="path to the file where the postprocessing configuration "
                                                             "is stored. If this is not provided then no postprocessing "
                                                             "will be made. It is strongly recommended to provide the "
                                                             "postprocessing file!",
                        required=False, type=str, default=None)
    parser.add_argument('--npz', action="store_true", required=False, help="stores npz")

    args = parser.parse_args()

    folders = args.folders
    threads = args.threads
    output_folder = args.output_folder
    pp_file = args.postprocessing_file
    npz = args.npz

    merge(folders, output_folder, threads, override=True, postprocessing_file=pp_file, store_npz=npz)


if __name__ == "__main__":
    main()
