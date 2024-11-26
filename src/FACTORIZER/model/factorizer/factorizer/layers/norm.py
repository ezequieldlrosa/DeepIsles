import torch
from torch import nn


class LayerNorm(nn.Module):
    """"Layer norm for channels-first inputs."""

    def __init__(self, dim, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(dim, **kwargs)

    def forward(self, x):
        # x: B × C × S1 × S2 × ... × Sp
        out = torch.einsum("b c ... -> b ... c", x)
        out = self.norm(out)
        out = torch.einsum("b ... c -> b c ...", out)
        return out
