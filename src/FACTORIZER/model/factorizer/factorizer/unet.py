import torch
from torch import nn
from torchvision.ops import StochasticDepth

from .layers import Linear
from .utils.helpers import as_tuple, prod, wrap_class


class DoubleConv(nn.Module):
    """(Conv -- Drop -- Norm -- Act) ** 2."""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        conv=(nn.Conv3d, {"kernel_size": 3, "padding": 1}),
        norm=(nn.GroupNorm, (8,)),
        act=nn.LeakyReLU,
        drop=(nn.Dropout, {"p": 0.0}),
        stride=1,
        **kwargs,
    ):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        conv = wrap_class(conv)
        drop = wrap_class(drop)
        norm = wrap_class(norm)
        act = wrap_class(act)

        self.block1 = nn.Sequential(
            conv(in_channels, mid_channels, stride=stride),
            drop(),
            norm(mid_channels),
            act(),
        )

        self.block2 = nn.Sequential(
            conv(mid_channels, out_channels, stride=1),
            drop(),
            norm(out_channels),
            act(),
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out


class BasicBlock(nn.Module):
    """Basic ResNet block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        conv=(nn.Conv3d, {"kernel_size": 3, "padding": 1}),
        norm=(nn.GroupNorm, (8,)),
        act=nn.LeakyReLU,
        drop=(nn.Dropout, {"p": 0.0}),
        stride=1,
        **kwargs,
    ):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        conv1 = wrap_class(conv)
        conv2 = wrap_class(conv)
        drop = wrap_class(drop)
        norm = wrap_class(norm)
        act = wrap_class(act)

        self.conv1 = conv1(in_channels, mid_channels, stride=stride)
        self.drop1 = drop()
        self.norm1 = norm(mid_channels)
        self.conv2 = conv2(mid_channels, out_channels)
        self.drop2 = drop()
        self.norm2 = norm(out_channels)
        self.act = act()

        if prod(as_tuple(stride)) != 1 or in_channels != out_channels:
            self.shortcut = conv[0](
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)

        out = self.conv1(x)
        out = self.drop1(out)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.drop2(out)
        out = self.norm2(out)

        out += shortcut

        out = self.act(out)

        return out


class BottleneckBlock(nn.Module):
    """ResNet bottleneck block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        alpha=0.5,
        conv=(nn.Conv3d, {"kernel_size": 3, "padding": 1}),
        norm=(nn.GroupNorm, (8,)),
        act=nn.LeakyReLU,
        drop=(nn.Dropout, {"p": 0.0}),
        stride=1,
        **kwargs,
    ):
        super().__init__()
        mid_channels = int(in_channels * alpha)

        conv2 = wrap_class(conv)
        drop = wrap_class(drop)
        norm = wrap_class(norm)
        act = wrap_class(act)

        self.conv1 = conv[0](in_channels, mid_channels, kernel_size=1)
        self.drop1 = drop()
        self.norm1 = norm(mid_channels)
        self.conv2 = conv2(mid_channels, mid_channels, stride=stride)
        self.drop2 = drop()
        self.norm2 = norm(mid_channels)
        self.conv3 = conv[0](mid_channels, out_channels, kernel_size=1)
        self.drop3 = drop()
        self.norm3 = norm(mid_channels)
        self.act = act()

        if prod(as_tuple(stride)) != 1 or in_channels != out_channels:
            self.shortcut = conv[0](
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)

        out = self.conv1(x)
        out = self.drop1(out)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.drop2(out)
        out = self.norm2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.drop3(out)
        out = self.norm3(out)

        out += shortcut
        out = self.act(out)

        return out


class PreActivationBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        conv=(nn.Conv3d, {"kernel_size": 3, "padding": 1}),
        norm=(nn.GroupNorm, (8,)),
        act=nn.LeakyReLU,
        drop=(nn.Dropout, {"p": 0.0}),
        stride=1,
        **kwargs,
    ):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        conv1 = wrap_class(conv)
        conv2 = wrap_class(conv)
        drop = wrap_class(drop)
        norm = wrap_class(norm)
        act = wrap_class(act)

        self.norm1 = norm(in_channels)
        self.act = act()
        self.conv1 = conv1(in_channels, mid_channels, stride=stride)
        self.drop1 = drop()
        self.norm2 = norm(mid_channels)
        self.conv2 = conv2(mid_channels, out_channels)
        self.drop2 = drop()

        if prod(as_tuple(stride)) != 1 or in_channels != out_channels:
            self.shortcut = conv[0](
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )

    def forward(self, x):
        out = self.norm1(x)
        out = self.act(out)
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.drop1(out)

        out = self.norm2(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.drop2(out)

        out += shortcut
        return out


class CovNeXtBlock(nn.Module):
    """ ConvNeXt Block"""

    def __init__(
        self,
        in_channels,
        out_channels,
        mlp_ratio=4,
        stochastic_depth_prob=0.0,
        layer_scale=1e-6,
        conv=(nn.Conv3d, {"kernel_size": 7, "padding": 3}),
        norm=nn.LayerNorm,
        act=nn.GELU,
        **kwargs,
    ):
        super().__init__()

        conv = wrap_class(conv)
        norm = wrap_class(norm)
        act = wrap_class(act)

        if in_channels != out_channels:
            self.adapter = Linear(in_channels, out_channels, bias=False)

        hidden_channels = int(mlp_ratio * in_channels)
        self.dwconv = conv(out_channels, out_channels, groups=out_channels)
        self.norm = norm(out_channels)
        self.pwconv1 = nn.Linear(out_channels, hidden_channels)
        self.act = act()
        self.pwconv2 = nn.Linear(hidden_channels, out_channels)
        self.gamma = (
            nn.Parameter(layer_scale * torch.ones(out_channels))
            if layer_scale > 0
            else None
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, x):
        x = self.adapter(x) if hasattr(self, "adapter") else x
        out = self.dwconv(x)
        out = torch.einsum("b c ... -> b ... c", out)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        if self.gamma is not None:
            out = self.gamma * out
        out = torch.einsum("b ... c -> b c ...", out)
        out = self.stochastic_depth(out)
        out += x
        return x


class Same(object):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def __getitem__(self, *args, **kwargs):
        return self.block


class UNetEncoderBlock(nn.Module):
    """U-Net encoder block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        depth=1,
        stride=2,
        downsample=(nn.Conv3d, {"kernel_size": 2}),
        block=Same(PreActivationBlock),
    ):
        super().__init__()
        if prod(as_tuple(stride)) == 1:
            wrapped_block = wrap_class(block[0])
            self.blocks = [wrapped_block(in_channels, out_channels)]

        elif downsample == "on-block":
            wrapped_block = wrap_class(block[0])
            self.blocks = [
                wrapped_block(in_channels, out_channels, stride=stride)
            ]

        else:
            downsample = wrap_class(downsample)
            wrapped_block = wrap_class(block[0])
            self.blocks = [
                downsample(in_channels, out_channels, stride=stride),
                wrapped_block(out_channels, out_channels),
            ]

        for i in range(1, depth):
            wrapped_block = wrap_class(block[i])
            self.blocks.append(wrapped_block(out_channels, out_channels))

        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        x = self.blocks(x)
        return x


def layer_iterator(depth_list):
    for layer, depth in enumerate(depth_list):
        for sublayer in range(depth):
            yield layer, sublayer


class UNetEncoder(nn.Module):
    """U-Net encoder."""

    def __init__(
        self,
        in_channels,
        depth=(1, 1, 1, 1, 1),
        width=(32, 64, 128, 256, 512),
        strides=(1, 2, 2, 2, 2),
        downsample=None,
        block=Same(PreActivationBlock),
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                UNetEncoderBlock(
                    in_channels,
                    width[0],
                    depth[0],
                    strides[0],
                    downsample,
                    {
                        j: block[i, j]
                        for i, j in layer_iterator(depth)
                        if i == 0
                    },
                ),
            ]
        )
        for i in range(len(width) - 1):
            self.blocks.append(
                UNetEncoderBlock(
                    width[i],
                    width[i + 1],
                    depth[i + 1],
                    strides[i + 1],
                    downsample,
                    {
                        j: block[h, j]
                        for h, j in layer_iterator(depth)
                        if h == i + 1
                    },
                )
            )

    def forward(self, x):
        out = [self.blocks[0](x)]
        for block in self.blocks[1:]:
            out.append(block(out[-1]))

        return out


class UNetDecoderBlock(nn.Module):
    """U-Net decoder block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        depth=1,
        stride=2,
        upsample=(nn.ConvTranspose3d, {"kernel_size": 2}),
        block=Same(PreActivationBlock),
    ):
        super().__init__()
        upsample = wrap_class(upsample)
        self.upsample = upsample(in_channels, out_channels, stride=stride)

        wrapped_block = wrap_class(block[0])
        self.blocks = [wrapped_block(2 * out_channels, out_channels)]
        for i in range(1, depth):
            wrapped_block = wrap_class(block[i])
            self.blocks.append(wrapped_block(out_channels, out_channels))

        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.blocks(x)
        return x


class UNetDecoder(nn.Module):
    """U-Net decoder."""

    def __init__(
        self,
        in_channels,
        depth=(1, 1, 1, 1),
        width=(256, 128, 64, 32),
        strides=(2, 2, 2, 2),
        upsample=None,
        block=Same(PreActivationBlock),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.blocks = nn.ModuleList(
            [
                UNetDecoderBlock(
                    in_channels,
                    width[0],
                    depth[0],
                    strides[0],
                    upsample,
                    {
                        j: block[i, j]
                        for i, j in layer_iterator(depth)
                        if i == 0
                    },
                )
            ]
        )
        for i in range(len(width) - 1):
            self.blocks.append(
                UNetDecoderBlock(
                    width[i],
                    width[i + 1],
                    depth[i + 1],
                    strides[i + 1],
                    upsample,
                    {
                        j: block[h, j]
                        for h, j in layer_iterator(depth)
                        if h == i + 1
                    },
                )
            )

    def forward(self, x):
        out = x
        for i, block in enumerate(self.blocks):
            i0 = -1 - i
            i1 = -2 - i
            out[i1] = block(out[i0], out[i1])

        return out


class UNet(nn.Module):
    """Generic U-Net architecture."""

    def __init__(
        self,
        in_channels,
        out_channels,
        spatial_dims=3,
        stem_width=None,
        encoder_depth=(1, 1, 1, 1, 1),
        encoder_width=(32, 64, 128, 256, 512),
        strides=(1, 2, 2, 2, 2),
        decoder_depth=(1, 1, 1, 1),
        stem=None,
        downsample=None,
        block=None,
        upsample=None,
        head=None,
        num_deep_supr=1,
    ):
        super().__init__()
        conv = getattr(nn, f"Conv{spatial_dims}d")
        tconv = getattr(nn, f"ConvTranspose{spatial_dims}d")
        if stem is None:
            stem = (conv, {"kernel_size": 3, "padding": 1, "bias": False})

        if downsample is None:
            downsample = (conv, {"kernel_size": 2})

        if block is None:
            block = (
                PreActivationBlock,
                {"conv": (conv, {"kernel_size": 3, "padding": 1})},
            )
            block = Same(block)

        if upsample is None:
            upsample = (tconv, {"kernel_size": 2})

        if head is None:
            head = (conv, {"kernel_size": 1})

        stem = wrap_class(stem)
        head = wrap_class(head)

        if stem_width is None:
            stem_width = encoder_width[0]

        self.stem = stem(in_channels, stem_width)
        self.encoder = UNetEncoder(
            stem_width,
            encoder_depth,
            encoder_width,
            strides,
            downsample,
            block,
        )
        self.decoder = UNetDecoder(
            encoder_width[-1],
            decoder_depth,
            encoder_width[-2::-1],
            strides[:0:-1],
            upsample,
            {
                (i, j): block[i + len(encoder_depth), j]
                for i, j in layer_iterator(decoder_depth)
            },
        )

        self.num_deep_supr = num_deep_supr
        self.heads = nn.ModuleList([])
        for j in range(num_deep_supr):
            self.heads.append(head(encoder_width[j], out_channels))

    def forward_features(self, x):
        out = self.stem(x)
        out = self.encoder(out)
        out = self.decoder(out)
        return out

    def forward(self, x):
        y = self.forward_features(x)

        out = []
        for j, head in enumerate(self.heads):
            out.append(head(y[j]))

        return out

