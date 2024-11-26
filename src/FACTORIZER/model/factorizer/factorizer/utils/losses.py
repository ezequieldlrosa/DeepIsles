from typing import Iterable

from torch import nn
import torch
import torch.nn.functional as F


class DeepSuprLoss(nn.Module):
    def __init__(self, loss, **kwargs):
        super().__init__()
        self.loss = loss(**kwargs)

    def forward(
        self, pred: Iterable[torch.Tensor], true: torch.Tensor
    ) -> torch.Tensor:
        t = 0
        weight_sum = 0.0
        loss = 0.0
        for y_hat in pred:
            weight = 1 / (2 ** t)
            if t > 0:
                y = F.interpolate(true, size=y_hat.shape[2:], mode="nearest")
            else:
                y = true

            loss += weight * self.loss(y_hat, y)
            weight_sum += weight
            t += 1

        return loss / weight_sum

