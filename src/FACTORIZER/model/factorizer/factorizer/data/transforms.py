import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import (
    Transform,
    MapTransform,
    LoadImaged,
    Identityd,
    SaveImaged,
)
from monai.inferers import SlidingWindowInferer
from monai.data import decollate_batch

from .utils import move_to


class Renamed(Transform):
    def __call__(self, data):
        if "image" in data:
            data["input"] = data.pop("image")

        if "image_transforms" in data:
            data["input_transforms"] = data.pop("image_transforms")

        if "image_meta_dict" in data:
            data["input_meta_dict"] = data.pop("image_meta_dict")

        if "label" in data:
            data["target"] = data.pop("label")

        if "label_transforms" in data:
            data["target_transforms"] = data.pop("label_transforms")

        if "label_meta_dict" in data:
            data["target_meta_dict"] = data.pop("label_meta_dict")

        if "id" in data:
            data["input_meta_dict"]["filename_or_obj"] = data["id"]
        else:
            data["id"] = os.path.basename(
                data["input_meta_dict"]["filename_or_obj"]
            ).split(".")[0]

        return data


class ReadImaged(MapTransform):
    def __init__(self, keys, allow_missing_keys=False, *args, **kwargs):
        super().__init__(keys, allow_missing_keys)
        self._args = args
        self._kwargs = kwargs

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if isinstance(d[key], str):
                load_image = LoadImaged(key, *self._args, **self._kwargs)
                d = load_image(d)
            else:
                modalities = []
                for modality in d[key]:
                    d["modality"] = modality
                    if "modality_meta_dict" in d:
                        del d["modality_meta_dict"]

                    load_image = LoadImaged(
                        "modality", *self._args, **self._kwargs
                    )
                    d = load_image(d)
                    modalities.append(d["modality"])

                d[key] = np.stack(modalities)
                if d[key].shape[0] == 1:
                    d[key] = np.squeeze(d[key], axis=0)

                d[f"{key}_meta_dict"] = d["modality_meta_dict"]
                del d["modality"], d["modality_meta_dict"]

        return d


class Interpolate(object):
    def __init__(self, key, spacing, **kwargs):
        super().__init__()
        self.key = key
        self.meta_key = f"{key}_meta_dict"
        self.spacing = torch.tensor(spacing)
        self.orig_spacing = None
        self.orig_size = None
        self.new_size = None
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def transform(self, batch):
        self.orig_spacing = batch[self.meta_key]["spacing"][0].to("cpu")
        if torch.allclose(self.orig_spacing, self.spacing):
            out = batch
        else:
            out = copy.deepcopy(batch)
            self.orig_size = batch[self.meta_key]["spatial_shape"][0].to("cpu")
            self.new_size = self.orig_size * self.orig_spacing / self.spacing
            self.new_size = self.new_size.round().to(
                device="cpu", dtype=torch.int
            )
            out[self.key] = F.interpolate(
                batch[self.key],
                size=tuple(self.new_size.numpy()),
                **self.kwargs,
            )

        return out

    def inverse_transform(self, batch):
        if torch.allclose(self.orig_spacing, self.spacing):
            out = batch
        else:
            out = copy.deepcopy(batch)
            out[self.key] = F.interpolate(
                batch[self.key],
                size=tuple(self.orig_size.numpy()),
                **self.kwargs,
            )

        return out


class Inferer(object):
    def __init__(
        self,
        spacing,
        spatial_size,
        post=None,
        write_dir=None,
        output_dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.device = None

        # downsampling
        self.interpolate = Interpolate(
            "input", spacing=spacing, mode="trilinear"
        )

        # inference method
        self.inferer = SlidingWindowInferer(roi_size=spatial_size, **kwargs)

        # postprocessing transforms
        self.post = Identityd("input") if post is None else post

        # write to file
        self.output_dtype = output_dtype
        self.write_dir = write_dir

    def get_preprocessed(self, batch, model):
        batch = self.interpolate.transform(batch)
        return batch

    def get_inferred(self, batch, model):
        batch = self.get_preprocessed(batch, model)
        batch["input"] = self.inferer(batch["input"], model)
        return batch

    def get_postprocessed(self, batch, model):
        batch = self.get_inferred(batch, model)
        batch = self.interpolate.inverse_transform(batch)
        batch = self.post(batch)
        if "input_transforms" in batch:
            del batch["input_transforms"]

        if "target_transforms" in batch:
            del batch["target_transforms"]

        return batch

    def write(self, batch, write_dir):
        if write_dir is not None:
            batch = decollate_batch(batch)
            for sample in batch:
                save = SaveImaged(
                    "input",
                    output_dir=write_dir,
                    output_dtype=self.output_dtype,
                    output_postfix="pred",
                    print_log=True,
                )
                sample = move_to(sample, device="cpu")
                save(sample)

    def __call__(self, batch, model):
        batch = self.get_postprocessed(batch, model)
        self.write(batch, self.write_dir)
        return batch["input"]

