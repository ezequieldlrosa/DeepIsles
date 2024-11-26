from torch import nn


class DepthWiseP2P(nn.Module):
    "Depth-wise patch-to-patch transform."

    def __init__(self, size):
        super().__init__()
        # patches already flattened in the matricization step
        num_pixels = size[-1]  # last dim is #pixels in a patch
        self.linear = nn.Linear(num_pixels, num_pixels)

    def forward(self, x):
        # x: (batch × patches) × channels × pixels
        return self.linear(x)

