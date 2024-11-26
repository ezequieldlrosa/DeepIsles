from torch import nn
from torch.nn.modules.utils import _pair

from .linear import Linear


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=None,
        hidden_dim=None,
        ratio=2,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        output_dim = input_dim if output_dim is None else output_dim
        hidden_dim = (
            int(ratio * input_dim) if hidden_dim is None else hidden_dim
        )
        dropout = _pair(dropout)

        self.block = nn.Sequential(
            Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout[0]),
            Linear(hidden_dim, output_dim),
            nn.Dropout(dropout[1]),
        )

    def forward(self, x):
        return self.block(x)
