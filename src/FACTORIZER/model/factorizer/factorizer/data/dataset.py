from typing import Callable, List, Optional, Sequence, Text, Union
import os
import json
import warnings

import numpy as np
from monai.transforms import LoadImaged, Randomizable
from monai.utils.misc import ensure_tuple
from monai.data import CacheDataset


def _append_paths(base_dir: Text, items: List[dict]) -> List[dict]:
    for item in items:
        if not isinstance(item, dict):
            raise TypeError(
                f"Every item in items must be a dict but got {type(item).__name__}."
            )

        if "image" in item:
            if isinstance(item["image"], str):
                item["image"] = [item["image"]]

            full_paths = []
            for k in item["image"]:
                full_paths.append(os.path.normpath(os.path.join(base_dir, k)))

            item["image"] = full_paths

        if "label" in item:
            if isinstance(item["label"], str):
                item["label"] = [item["label"]]

            full_paths = []
            for k in item["label"]:
                full_paths.append(os.path.normpath(os.path.join(base_dir, k)))

            item["label"] = full_paths

    return items


def load_properties(data_properties: Union[Text, dict]) -> dict:
    if isinstance(data_properties, Text):
        data_properties_file_path = Text(data_properties)
        base_dir = os.path.dirname(data_properties_file_path)
        with open(data_properties_file_path) as json_file:
            data_properties = json.load(json_file)
    elif isinstance(data_properties, dict):
        data_properties = dict(data_properties)
        base_dir = ""
    else:
        raise ValueError(
            "`data_properties` must be a dict or a path to a json file."
        )

    keys = (
        "name",
        "description",
        "reference",
        "licence",
        "tensorImageSize",
        "modality",
        "labels",
        "numTraining",
        "numTest",
        "training",
        "test",
    )
    properties = {}
    for key in keys:
        if key in data_properties:
            properties[key] = data_properties[key]
        else:
            warnings.warn(f"key {key} is not in the data properties file.")

    if "test" in properties:
        expected_data = []
        for case in properties["test"]:
            if isinstance(case, Text):
                expected_data.append({"image": case})
            elif isinstance(case, dict):
                expected_data.append(case)
            else:
                raise TypeError(
                    f"Every item in `test` must be a dict or str but got {type(case).__name__}."
                )

        properties["test"] = expected_data

    if "training" in properties:
        _append_paths(base_dir, properties["training"])

    if "validation" in properties:
        _append_paths(base_dir, properties["validation"])

    if "test" in properties:
        _append_paths(base_dir, properties["test"])

    return properties


class StandardDataset(Randomizable, CacheDataset):
    def __init__(
        self,
        data_properties: Union[Text, dict],
        section: Text,
        transform: Optional[Union[Sequence[Callable], Callable]] = None,
        **kwargs,
    ) -> None:

        self._properties = load_properties(data_properties)
        self.section = section

        if transform is None:
            transform = LoadImaged(["image", "label"])

        self.indices: np.ndarray = np.array([])
        self.datalist = self._generate_data_list(self._properties)

        CacheDataset.__init__(self, self.datalist, transform, **kwargs)

    def get_indices(self) -> np.ndarray:
        """Get the indices of datalist used in this dataset."""
        return self.indices

    def randomize(self, data: List[int]) -> None:
        self.R.shuffle(data)

    def get_properties(
        self, keys: Optional[Union[Sequence[Text], Text]] = None
    ):
        """
        Get the loaded properties of dataset with specified keys.
        If no keys specified, return all the loaded properties.

        """
        if keys is None:
            return self._properties
        if self._properties is not None:
            return {key: self._properties[key] for key in ensure_tuple(keys)}

        return {}

    def _generate_data_list(self, data_properties: dict) -> List[dict]:
        if self.section in ("training", "validation"):
            section = "training"
        else:
            section = self.section

        datalist = data_properties[section]
        return self._split_datalist(datalist)

    def _split_datalist(self, datalist: List[dict]) -> List[dict]:
        return datalist
