import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from monai import transforms
from monai.data import DataLoader
from monai.apps import CrossValidation

from ..data import (
    StandardDataset,
    Renamed,
    Inferer,
    load_properties,
)


###################################
# Transform
###################################


class BraTSOneHotEncoderd(transforms.MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    
    Decathlon version:
    label 0: background,
    label 1: edema (ED)
    label 2: necrotic and non-enhancing tumor (NCR&NET)
    label 3: enhancing tumor (ET)

    BraTS since 2017:
    label 0: background,
    label 1: necrotic and non-enhancing tumor (NCR/NET)
    label 2: edema (ED)
    label 4: enhancing tumor (ET)

    If nested = True, the classes are enhancing tumor (ET), tumor core (TC),
    and whole tumor (WT):

    TC: ET + NCR/NET
    WT: ET + NCR/NET + ED

    """

    def __init__(
        self, keys, nested=True, allow_missing_keys=False, version="decathlon"
    ):
        super().__init__(keys, allow_missing_keys)
        self.nested = nested
        self.version = version
        if version == "decathlon":
            self.labels = {"background": 0, "ED": 1, "NCR/NET": 2, "ET": 3}
        else:
            self.labels = {"background": 0, "ED": 2, "NCR/NET": 1, "ET": 4}

    def __call__(self, data):
        if self.nested:
            d = dict(data)
            for key in self.key_iterator(d):
                result = []
                # ET
                result.append(d[key] == self.labels["ET"])
                # TC: ET + NCR/NET
                result.append(
                    np.logical_or(
                        d[key] == self.labels["ET"],
                        d[key] == self.labels["NCR/NET"],
                    )
                )
                # WT: ET + NCR/NET + ED
                result.append(
                    np.logical_or(
                        np.logical_or(
                            d[key] == self.labels["ET"],
                            d[key] == self.labels["NCR/NET"],
                        ),
                        d[key] == self.labels["ED"],
                    )
                )
                d[key] = np.stack(result, axis=0).astype(np.float32)
        else:
            d = dict(data)
            for key in self.key_iterator(d):
                result = []
                # ED
                result.append(d[key] == self.labels["ED"])
                # NCR/NET
                result.append(d[key] == self.labels["NCR/NET"])
                # ET
                result.append(d[key] == self.labels["ET"])
                d[key] = np.stack(result, axis=0).astype(np.float32)

        return d


def brats_train_transform(spatial_size=(128, 128, 128), version="decathlon"):
    train_transform = [
        transforms.LoadImaged(["image", "label"]),
        transforms.AsChannelFirstd("image")
        if version == "decathlon"
        else transforms.Identityd("image"),
        BraTSOneHotEncoderd("label", nested=True, version=version),
        transforms.CropForegroundd(["image", "label"], source_key="image"),
        transforms.NormalizeIntensityd(
            "image", nonzero=True, channel_wise=True
        ),
        transforms.RandSpatialCropd(
            ["image", "label"], roi_size=spatial_size, random_size=False
        ),
        transforms.RandAffined(
            ["image", "label"],
            prob=0.15,
            spatial_size=spatial_size,
            rotate_range=[30 * np.pi / 180] * 3,
            scale_range=[0.3] * 3,
            mode=("bilinear", "bilinear"),
            as_tensor_output=False,
        ),
        transforms.RandFlipd(["image", "label"], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(["image", "label"], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(["image", "label"], prob=0.5, spatial_axis=2),
        transforms.RandGaussianNoised("image", prob=0.15, std=0.1),
        transforms.RandGaussianSmoothd(
            "image",
            prob=0.15,
            sigma_x=(0.5, 1.5),
            sigma_y=(0.5, 1.5),
            sigma_z=(0.5, 1.5),
        ),
        transforms.RandScaleIntensityd("image", prob=0.15, factors=0.3),
        transforms.RandShiftIntensityd("image", prob=0.15, offsets=0.1),
        transforms.RandAdjustContrastd("image", prob=0.15, gamma=(0.7, 1.5)),
        transforms.AsDiscreted("label", threshold=0.5),
        transforms.ToTensord(["image", "label"]),
        Renamed(),
    ]
    train_transform = transforms.Compose(train_transform)
    return train_transform


def brats_val_transform(version="decathlon"):
    val_transform = [
        transforms.LoadImaged(["image", "label"], allow_missing_keys=True),
        transforms.AsChannelFirstd("image")
        if version == "decathlon"
        else transforms.Identityd("image"),
        BraTSOneHotEncoderd(
            "label", nested=True, allow_missing_keys=True, version=version
        ),
        transforms.NormalizeIntensityd(
            "image", nonzero=True, channel_wise=True
        ),
        transforms.ToTensord(["image", "label"], allow_missing_keys=True),
        Renamed(),
    ]
    val_transform = transforms.Compose(val_transform)
    return val_transform


def brats_test_transform(version="decathlon"):
    return brats_val_transform(version=version)


def brats_vis_transform(version="decathlon"):
    vis_transform = [
        transforms.LoadImaged(["image", "label"], allow_missing_keys=True),
        transforms.AsChannelFirstd("image")
        if version == "decathlon"
        else transforms.Identityd("image"),
        BraTSOneHotEncoderd(
            "label", nested=False, allow_missing_keys=True, version=version
        ),
        transforms.NormalizeIntensityd("image", channel_wise=True),
        transforms.ToTensord(["image", "label"], allow_missing_keys=True),
        Renamed(),
    ]
    vis_transform = transforms.Compose(vis_transform)
    return vis_transform


###################################
# Data module
###################################


class BraTSDataModule(LightningDataModule):
    def __init__(
        self,
        data_properties,
        version="decathlon",
        spacing=(1.0, 1.0, 1.0),
        spatial_size=(128, 128, 128),
        num_splits=5,
        split=0,
        batch_size=2,
        num_workers=None,
        seed=42,
        **kwargs,
    ):
        self.data_properties = load_properties(data_properties)
        self.num_splits = num_splits
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.dataset_params = kwargs

        self.train_transform = brats_train_transform(spatial_size, version)
        self.val_transform = brats_val_transform(version)
        self.test_transform = brats_test_transform(version)
        self.vis_transform = brats_vis_transform(version)

        self.train_set = self.val_set = self.test_set = None

    def setup(self, stage):
        if stage in ("fit", "validate", None):
            # make training set
            train_cv = CrossValidation(
                StandardDataset,
                self.num_splits,
                self.seed,
                data_properties=self.data_properties,
                section="training",
                transform=self.train_transform,
                **self.dataset_params,
            )
            train_folds = [
                k for k in range(self.num_splits) if k != self.split
            ]
            self.train_set = train_cv.get_dataset(train_folds)

            # make validation set
            val_cv = CrossValidation(
                StandardDataset,
                self.num_splits,
                self.seed,
                data_properties=self.data_properties,
                section="validation",
                transform=self.val_transform,
                **self.dataset_params,
            )
            self.val_set = val_cv.get_dataset([self.split])

        if stage in ("test", "predict", None):
            # make test set
            self.test_set = StandardDataset(
                data_properties=self.data_properties,
                section="test",
                transform=self.test_transform,
                **self.dataset_params,
            )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return test_loader


# alias
BRATS = BraTSDataModule


###################################
# Inference
###################################


class BraTSInferer(Inferer):
    def __init__(
        self,
        version="decathlon",
        spacing=(1.0, 1.0, 1.0),
        spatial_size=(128, 128, 128),
        post=None,
        write_dir=None,
        output_dtype=None,
        **kwargs,
    ) -> None:

        if version == "decathlon":
            self.labels = {"background": 0, "ED": 1, "NCR/NET": 2, "ET": 3}
        else:
            self.labels = {"background": 0, "ED": 2, "NCR/NET": 1, "ET": 4}

        # postprocessing transforms
        if post == "logit-nested":
            post = transforms.Lambdad("input", lambda x: x)
            output_dtype = np.float32 if output_dtype is None else output_dtype
        elif post == "prob-nested":
            post = transforms.Lambdad("input", lambda x: x.sigmoid(dim=1))
            output_dtype = np.float32 if output_dtype is None else output_dtype
        elif post == "one-hot-nested":
            post = transforms.Lambdad(
                "input", lambda x: torch.where(x >= 0, 1.0, 0.0)
            )
            output_dtype = np.uint8 if output_dtype is None else output_dtype
        elif post == "class":

            def func(x):
                mask = torch.where(x >= 0, 1.0, 0.0)
                et = mask[:, 0, ...]
                tc = mask[:, 1, ...]
                wt = mask[:, 2, ...]
                out = torch.zeros_like(wt)
                out[et == 1] = self.labels["ET"]
                out[torch.logical_and(tc == 1, et == 0)] = self.labels[
                    "NCR/NET"
                ]
                out[torch.logical_and(wt == 1, tc == 0)] = self.labels["ED"]
                return out

            post = transforms.Lambdad("input", func)
            output_dtype = np.uint8 if output_dtype is None else output_dtype
        elif post == "one-hot":

            def func(x):
                mask = torch.where(x >= 0, 1.0, 0.0)
                et = mask[:, 0:1, ...]
                tc = mask[:, 1:2, ...]
                wt = mask[:, 2:3, ...]
                bg = (wt != 0).float()
                nt = torch.logical_and(tc == 1, et == 0).float()
                ed = torch.logical_and(wt == 1, tc == 0).float()
                return torch.cat((bg, ed, nt, et), dim=1)

            post = transforms.Lambdad("input", func)
            output_dtype = np.uint8 if output_dtype is None else output_dtype

        super().__init__(
            spacing=spacing,
            spatial_size=spatial_size,
            post=post,
            write_dir=write_dir,
            output_dtype=output_dtype,
            **kwargs,
        )
