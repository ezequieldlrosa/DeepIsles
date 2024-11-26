from typing import Any, Iterable, Text, Tuple, Union, Sequence, Callable
from numbers import Number
import os
import sys
import inspect
from importlib import import_module
from contextlib import contextmanager
from itertools import chain, accumulate
from functools import partialmethod, partial, reduce
from operator import mul

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import torch
from torch import nn
from pytorch_lightning.callbacks import Callback
from monai.utils.dist import (
    evenly_divisible_all_gather,
    string_list_all_gather,
)


@contextmanager
def null_context():
    yield


class UniversalTuple(tuple):
    def __contains__(self, other):
        return True


class EncodeStrings(object):
    def __init__(self, embed_length=50) -> None:
        self.embed_length = embed_length

    def encode(self, strings):
        unicodes = []
        for s in strings:
            padded_s = "".join(["!"] * (self.embed_length - len(s))) + s
            unicodes.append(list(padded_s.encode("utf8")))

        return unicodes

    def decode(self, unicodes):
        strings = []
        for x in unicodes:
            strings.append(bytes(x).decode("utf8").lstrip("!"))

        return strings


def as_tuple(obj: Any) -> Tuple[Any, ...]:
    """Convert to tuple."""
    if not isinstance(obj, Sequence) or isinstance(obj, Text):
        obj = (obj,)

    return tuple(obj)


def pair(obj: Tuple) -> Tuple[Any, Any]:
    return (obj[0], obj[1] if len(obj) >= 2 else obj[0])


def prod(x: Sequence[Number]):
    return reduce(mul, x, 1)


def cumprod(x: Sequence[Number]):
    return list(accumulate(x, mul))


def has_args(obj, keywords: Union[str, Sequence[str]]) -> bool:
    """
    Return a boolean indicating whether the given callable `obj` has
    the `keywords` in its signature.
    """
    if not callable(obj):
        return False

    sig = inspect.signature(obj)
    return all(key in sig.parameters for key in as_tuple(keywords))


def partialclass(cls, *args, **kwds):
    new_cls = type(
        cls.__name__,
        (cls,),
        {"__init__": partialmethod(cls.__init__, *args, **kwds)},
    )
    return new_cls


def wrap_class(obj: Union[Sequence, Callable]):
    assert isinstance(
        obj, (Sequence, Callable)
    ), f"{obj} should be a sequence or callable."

    if isinstance(obj, Sequence) and isinstance(obj[0], Callable):
        args = []
        kwargs = {}
        for i, a in enumerate(obj):
            if i == 0:
                callable_obj = a
            elif isinstance(a, Sequence):
                args.extend(a)
            elif isinstance(a, dict):
                kwargs.update(a)

        return partial(callable_obj, *args, **kwargs)
    else:
        return obj


def is_wrappable_class(obj: Any):
    if isinstance(obj, Callable):
        out = True
    elif isinstance(obj, Sequence):
        flags = []
        for i, a in enumerate(obj):
            if i == 0:
                flags.append(isinstance(a, Callable))
            else:
                flags.append(isinstance(a, (Sequence, dict)))

        out = all(flags)
    else:
        out = False

    return out


def get_class(obj: Callable):
    if isinstance(obj, partial):
        return obj.func
    else:
        return obj


def dispatcher(func):
    if callable(func):
        func = func
    elif isinstance(func, str):
        *module_name, func_name = func.split(".")
        module_name = ".".join(module_name)
        if module_name:
            module = (
                locals().get(module_name)
                or globals().get(module_name)
                or import_module(module_name)
            )

            func = getattr(module, func_name)
        else:
            func = globals()[func]
    return func


def read_config(path):
    directory = os.path.dirname(path)
    sys.path.insert(0, directory)
    base = os.path.basename(path)
    file_name = os.path.splitext(base)[0]
    module = import_module(file_name, directory)
    config = getattr(module, "CONFIG")
    return config


def channels_first(obj):
    class ChannelFirst(obj):
        def forward(self, x, *args, **kwargs):
            # x: B × C × S1 × ... × SM
            shape = list(x.shape)
            out = x.flatten(2).transpose(1, 2)
            out = super().forward(out, *args, **kwargs)
            shape[1] = out.shape[-1]
            out = out.transpose(1, 2).reshape(shape)
            return out

    ChannelFirst.__name__ = f"{obj.__name__}nd"
    locals()[f"{obj.__name__}nd"] = ChannelFirst
    del ChannelFirst
    return locals()[f"{obj.__name__}nd"]


def local_norm(x, kernel_size):
    local_mean = ndi.uniform_filter(x, size=kernel_size)
    local_sqr_mean = ndi.uniform_filter(x ** 2, size=kernel_size)
    local_std = np.sqrt(
        np.clip(local_sqr_mean - local_mean ** 2, 1e-8, np.inf)
    )
    out = (x - local_mean) / local_std
    return out


def all_gather(data):
    results = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            results[k] = evenly_divisible_all_gather(v.contiguous())
        else:
            results[k] = string_list_all_gather(v)

    return results


def move_to(d, *args, **kwargs):
    if isinstance(d, (list, tuple, set, frozenset)):
        d = [move_to(v, *args, **kwargs) for v in d]
    elif isinstance(d, dict):
        d = {k: move_to(v, *args, **kwargs) for k, v in d.items()}
    elif isinstance(d, torch.Tensor):
        d = d.to(*args, **kwargs)

    return d


def collate(batch):
    first_sample = batch[0]
    out = {}
    for key, value in first_sample.items():
        if isinstance(value, (list, tuple)):
            out[key] = [*chain(*(sample[key] for sample in batch))]
        elif isinstance(value, torch.Tensor):
            out[key] = torch.cat([sample[key] for sample in batch], dim=0)
        else:
            out[key] = [sample[key] for sample in batch]

    return out


def decollate_channels(batch, keys=None):
    keys = batch.keys() if keys is None else keys
    out = {}
    for k, v in batch.items():
        if (
            k in keys
            and isinstance(v, torch.Tensor)
            and v.ndim == 2
            and v.shape[1] > 1
        ):
            for c in range(v.shape[1]):
                out[f"{k}_{c}"] = v[:, c : (c + 1)]
        else:
            out[k] = v

    return out


def reduce_channels(batch, keys=None):
    keys = batch.keys() if keys is None else keys
    out = {}
    for k, v in batch.items():
        if (
            k in keys
            and isinstance(v, torch.Tensor)
            and v.ndim == 2
            and v.shape[1] > 1
        ):
            out[f"{k}_avg"] = torch.nanmean(batch[k], dim=1, keepdim=True)

    return out


def reduce_batch(batch, keys=None):
    keys = batch.keys() if keys is None else keys
    out = {}
    for k, v in batch.items():
        if k in keys and isinstance(v, torch.Tensor):
            out[k] = torch.nanmean(v)

    return out


def remove_dublicates(data, key=None):
    seen_ids = {}  # {id: index}
    for i, s in enumerate(data[key]):
        if s not in seen_ids:
            seen_ids[s] = i

    result = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            result[k] = v[list(seen_ids.values())]
        else:
            result[k] = [v[i] for i in seen_ids.values()]

    return result


class ValEveryNSteps(Callback):
    def __init__(self, every_n_steps):
        self.every_n_steps = every_n_steps

    def on_batch_end(self, trainer, pl_module):
        if (
            trainer.global_step % self.every_n_steps == 0
            and trainer.global_step != 0
        ):
            trainer.validate(pl_module, trainer.datamodule.val_dataloader)


class SaveValResults(Callback):
    def __init__(self, save_path=None):
        self.save_path = save_path

    def on_validation_end(self, trainer, pl_module):
        if self.save_path is None:
            self.save_path = os.path.join(
                trainer.logger.log_dir, "results.csv"
            )

        val_results = move_to(pl_module.val_results, device="cpu")
        for k, v in val_results.items():
            if isinstance(v, torch.Tensor):
                val_results[k] = v.flatten()

        val_results = pd.DataFrame(val_results)
        val_results = val_results.replace([np.inf, -np.inf], np.nan)
        val_results.to_csv(self.save_path)


class TakeSnapshot(Callback):
    def __init__(self, epochs=None, save_dir=None):
        super(TakeSnapshot, self).__init__()
        self.epochs = () if epochs is None else epochs
        self.save_dir = save_dir

    def on_validation_end(self, trainer, pl_module):
        if self.save_dir is None:
            self.save_dir = os.path.join(trainer.logger.log_dir, "checkpoints")
        epoch = trainer.current_epoch
        if epoch in self.epochs:
            filepath = os.path.join(self.save_dir, f"epoch={epoch}.ckpt")
            trainer.save_checkpoint(filepath)
            print(f"\r Snapshot taken, epoch = {epoch}")

    def get_lr(self, trainer):
        optimizer = trainer.lr_schedulers[0]["scheduler"].optimizer
        for param_group in optimizer.param_groups:
            return param_group["lr"]


class Ensemble(nn.Module):
    def __init__(
        self, models, weights=None, reduction=torch.mean,
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = (
            torch.ones(len(models))
            if weights is None
            else torch.tensor(weights)
        )
        assert len(self.models) == len(self.weights)
        self.reduction = reduction

    def forward(self, x):
        self.weights = self.weights.to(dtype=x.dtype, device=x.device)
        out = [model(x) for model in self.models]
        out = torch.stack(out, dim=0)
        weights = self.weights.reshape(-1, *([1] * (out.ndim - 1)))
        out = weights * out
        out = self.reduction(out, dim=0)
        return out

