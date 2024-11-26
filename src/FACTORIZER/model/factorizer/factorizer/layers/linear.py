from torch import nn


class Linear(nn.Module):
    """"Linear layer for channels-first inputs."""

    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype=None,
    ):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=2)
        self.linear = nn.Conv1d(
            in_features,
            out_features,
            kernel_size=1,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        shape = x.shape
        out = self.flatten(x)
        out = self.linear(out)
        out = out.reshape(shape[0], -1, *shape[2:])
        return out
