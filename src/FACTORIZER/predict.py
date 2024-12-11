from typing import Dict, List, Optional, Sequence
import os
import glob
import warnings
import logging
import numpy as np
import torch
from torch.utils.data._utils.collate import np_str_obj_array_pattern
from pytorch_lightning import seed_everything
import SimpleITK as sitk
from monai import transforms
from monai.data.utils import orientation_ras_lps
import factorizer as ft

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
seed_everything(42, workers=True)
torch.set_default_dtype(torch.float32)
cwd = os.path.dirname(__file__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def predict(dwi, adc, flair, model_path):
    # register flair to dwi
    flair = sitk.Elastix(dwi, flair)
    for file in glob.glob(cwd + "/TransformParameters.*"):
        os.remove(file)  # remove transform parameters files

    # init data module
    data_properties = {"test": [{"input": [dwi, adc, flair]}]}
    dm = ft.ISLESDataModule(
        data_properties=data_properties,
        spacing=[2.0, 2.0, 2.0],
        spatial_size=[64, 64, 64],
        num_workers=0,
        cache_num=1,
        cache_rate=1.0,
        batch_size=1,
        seed=42,
    )

    test_transform = transforms.Compose(
        [
            LoadFromSimpleITKd("input"),
            transforms.NormalizeIntensityd("input", nonzero=True, channel_wise=True),
            transforms.ToTensord("input"),
        ]
    )
    dm.test_transform = test_transform
    dm.setup("test")
    base_model_path = os.path.join(model_path, 'FACTORIZER')
    # load (ensemble) model
    net_class = ft.Ensemble
    net_params = {
        "models": [
            ft.SemanticSegmentation.load_from_checkpoint(
                os.path.join(base_model_path,
                             "logs/isles2022_dwi-adc-flair/fold0/resunet/version_0/checkpoints/epoch=1999-step=99999.ckpt")
            ),
            ft.SemanticSegmentation.load_from_checkpoint(
                os.path.join(base_model_path,
                             "logs/isles2022_dwi-adc-flair/fold1/resunet/version_0/checkpoints/epoch=1999-step=99999.ckpt")
            ),
            ft.SemanticSegmentation.load_from_checkpoint(
                os.path.join(base_model_path,
                             "logs/isles2022_dwi-adc-flair/fold2/resunet/version_0/checkpoints/epoch=1999-step=99999.ckpt")
            ),
            ft.SemanticSegmentation.load_from_checkpoint(
                os.path.join(base_model_path,
                             "logs/isles2022_dwi-adc-flair/fold3/resunet/version_0/checkpoints/epoch=1999-step=99999.ckpt")
            ),
            ft.SemanticSegmentation.load_from_checkpoint(
                os.path.join(base_model_path,
                             "logs/isles2022_dwi-adc-flair/fold4/resunet/version_0/checkpoints/epoch=1999-step=99999.ckpt")
            ),
            ft.SemanticSegmentation.load_from_checkpoint(
                os.path.join(base_model_path,
                             "logs/isles2022_dwi-adc-flair/fold0/swin-factorizer/version_0/checkpoints/epoch=1999-step=99999.ckpt")
            ),
            ft.SemanticSegmentation.load_from_checkpoint(
                os.path.join(base_model_path,
                             "logs/isles2022_dwi-adc-flair/fold1/swin-factorizer/version_0/checkpoints/epoch=1999-step=99999.ckpt")
            ),
            ft.SemanticSegmentation.load_from_checkpoint(
                os.path.join(base_model_path,
                             "logs/isles2022_dwi-adc-flair/fold2/swin-factorizer/version_0/checkpoints/epoch=1999-step=99999.ckpt")
            ),
            ft.SemanticSegmentation.load_from_checkpoint(
                os.path.join(base_model_path,
                             "logs/isles2022_dwi-adc-flair/fold3/swin-factorizer/version_0/checkpoints/epoch=1999-step=99999.ckpt")
            ),
            ft.SemanticSegmentation.load_from_checkpoint(
                os.path.join(base_model_path,
                             "logs/isles2022_dwi-adc-flair/fold4/swin-factorizer/version_0/checkpoints/epoch=1999-step=99999.ckpt")
            ),
        ],
        "weights": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    }
    network = (net_class, net_params)
    inferer = ft.ISLESInferer(
        spacing=[2.0, 2.0, 2.0],
        spatial_size=[64, 64, 64],
        overlap=0.5,
        post="class",
    )
    model = ft.SemanticSegmentation(
        network=network, inferer=inferer, loss=(None,), metrics={}
    ).to(device)

    # inference
    # with torch.inference_mode():
    with torch.no_grad():
        batch = [*dm.test_dataloader()][0]
        input_device = batch['input'].to(device)
        batch['input'] = input_device
        pred = model.inferer.get_postprocessed(batch, model)
        
    pred_device = pred['input'].to("cpu")
    pred['input'] = pred_device
    pred = ft.decollate_batch(pred)[0]["input"][0].T.numpy()
    return pred


class LoadFromSimpleITKd(transforms.MapTransform):
    """Get data from a SimpleITK object.
    The obtained data array will be in C order, for example, a 3D image NumPy
    array index order will be `CDWH`.

    Args:
        channel_dim: the channel dimension of the input image, default is None.
            This is used to set original_channel_dim in the metadata, EnsureChannelFirstD reads this field.
            If None, `original_channel_dim` will be either `no_channel` or `-1`.
                - Nifti file is usually "channel last", so there is no need to specify this argument.
        reverse_indexing: whether to use a reversed spatial indexing convention for the returned data array.
            If ``False``, the spatial indexing follows the numpy convention;
            otherwise, the spatial indexing convention is reversed to be compatible with ITK. Default is ``False``.
            This option does not affect the metadata.
        affine_lps_to_ras: whether to convert the affine matrix from "LPS" to "RAS". Defaults to ``True``.
            Set to ``True`` to be consistent with ``NibabelReader``, otherwise the affine matrix remains in the ITK convention.
        kwargs: additional args for `itk.imread` API. more details about available args:
            https://github.com/InsightSoftwareConsortium/ITK/blob/master/Wrapping/Generators/Python/itk/support/extras.py

    """

    def __init__(
        self,
        keys,
        allow_missing_keys=False,
        dtype=np.float32,
        channel_dim: Optional[int] = None,
        reverse_indexing: bool = False,
        affine_lps_to_ras: bool = True,
        **kwargs,
    ):
        super().__init__(keys, allow_missing_keys)
        self.dtype = dtype
        self.channel_dim = channel_dim
        self.reverse_indexing = reverse_indexing
        self.affine_lps_to_ras = affine_lps_to_ras
        self.kwargs = kwargs

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            array, meta = self.get_data(d[key])
            d[key] = array
            d[f"{key}_meta_dict"] = meta
        return d

    def get_data(self, img):
        """
        Extract data array and metadata from loaded image and return them.
        This function returns two objects, first is numpy array of image data, second is dict of metadata.
        It constructs `affine`, `original_affine`, and `spatial_shape` and stores them in meta dict.
        When loading a list of files, they are stacked together at a new dimension as the first dimension,
        and the metadata of the first image is used to represent the output metadata.

        Args:
            img: an ITK image object loaded from an image file or a list of ITK image objects.

        """
        img_array: List[np.ndarray] = []
        compatible_meta: Dict = {}

        if isinstance(img, sitk.Image):
            img = (img,)

        assert isinstance(img, Sequence)

        for i in img:
            data = self._get_array_data(i)
            img_array.append(data)
            header = self._get_meta_dict(i)
            header["original_affine"] = self._get_affine(i, self.affine_lps_to_ras)
            header["affine"] = header["original_affine"].copy()
            header["spatial_shape"] = self._get_spatial_shape(i)
            if self.channel_dim is None:  # default to "no_channel" or -1
                header["original_channel_dim"] = (
                    "no_channel"
                    if len(data.shape) == len(header["spatial_shape"])
                    else -1
                )
            else:
                header["original_channel_dim"] = self.channel_dim
            _copy_compatible_dict(header, compatible_meta)

        return _stack_images(img_array, compatible_meta), compatible_meta

    def _get_meta_dict(self, img) -> Dict:
        """
        Get all the metadata of the image and convert to dict type.

        Args:
            img: an ITK image object loaded from an image file.

        """
        meta_dict = {
            key: img.GetMetaData(key)
            for key in img.GetMetaDataKeys()
            if not key.startswith("ITK_")
        }

        meta_dict["spacing"] = np.asarray(img.GetSpacing(), dtype=self.dtype)
        return meta_dict

    def _get_affine(self, img, lps_to_ras: bool = True):
        """
        Get or construct the affine matrix of the image, it can be used to correct
        spacing, orientation or execute spatial transforms.

        Args:
            img: an ITK image object loaded from an image file.
            lps_to_ras: whether to convert the affine matrix from "LPS" to "RAS". Defaults to True.

        """

        direction = np.reshape(
            np.array(img.GetDirection(), dtype=self.dtype),
            [img.GetDimension()] * 2,
        )
        spacing = np.asarray(img.GetSpacing(), dtype=self.dtype)
        origin = np.asarray(img.GetOrigin(), dtype=self.dtype)

        sr = min(max(direction.shape[0], 1), 3)
        affine: np.ndarray = np.eye(sr + 1, dtype=self.dtype)
        affine[:sr, :sr] = direction[:sr, :sr] @ np.diag(spacing[:sr])
        affine[:sr, -1] = origin[:sr]
        if lps_to_ras:
            affine = orientation_ras_lps(affine)
        return affine

    def _get_spatial_shape(self, img):
        """
        Get the spatial shape of `img`.

        Args:
            img: an ITK image object loaded from an image file.

        """
        sr = img.GetDimension()
        sr = max(min(sr, 3), 1)
        _size = list(img.GetSize())
        if self.channel_dim is not None:
            _size.pop(self.channel_dim)
        return np.asarray(_size[:sr])

    def _get_array_data(self, img):
        """
        Get the raw array data of the image, converted to Numpy array.

        Following PyTorch conventions, the returned array data has contiguous channels,
        e.g. for an RGB image, all red channel image pixels are contiguous in memory.
        The last axis of the returned array is the channel axis.

        See also:

            - https://github.com/InsightSoftwareConsortium/ITK/blob/v5.2.1/Modules/Bridge/NumPy/wrapping/PyBuffer.i.in

        Args:
            img: an ITK image object loaded from an image file.

        """
        np_img = sitk.GetArrayViewFromImage(img)
        np_img = np_img.astype(dtype=self.dtype)
        if img.GetNumberOfComponentsPerPixel() == 1:  # handling spatial images
            return np_img if self.reverse_indexing else np_img.T
        # handling multi-channel images
        return np_img if self.reverse_indexing else np.moveaxis(np_img.T, 0, -1)


def _copy_compatible_dict(from_dict: Dict, to_dict: Dict):
    if not isinstance(to_dict, dict):
        raise ValueError(f"to_dict must be a Dict, got {type(to_dict)}.")
    if not to_dict:
        for key in from_dict:
            datum = from_dict[key]
            if (
                isinstance(datum, np.ndarray)
                and np_str_obj_array_pattern.search(datum.dtype.str) is not None
            ):
                continue
            to_dict[key] = datum
    else:
        affine_key, shape_key = "affine", "spatial_shape"
        if affine_key in from_dict and not np.allclose(
            from_dict[affine_key], to_dict[affine_key]
        ):
            raise RuntimeError(
                "affine matrix of all images should be the same for channel-wise concatenation. "
                f"Got {from_dict[affine_key]} and {to_dict[affine_key]}."
            )
        if shape_key in from_dict and not np.allclose(
            from_dict[shape_key], to_dict[shape_key]
        ):
            raise RuntimeError(
                "spatial_shape of all images should be the same for channel-wise concatenation. "
                f"Got {from_dict[shape_key]} and {to_dict[shape_key]}."
            )


def _stack_images(image_list: List, meta_dict: Dict):
    if len(image_list) <= 1:
        return image_list[0]
    if meta_dict.get("original_channel_dim", None) not in ("no_channel", None):
        channel_dim = int(meta_dict["original_channel_dim"])
        return np.concatenate(image_list, axis=channel_dim)
    # stack at a new first dim as the channel dim, if `'original_channel_dim'` is unspecified
    meta_dict["original_channel_dim"] = 0
    return np.stack(image_list, axis=0)

