import copy

import torch
from torch import nn

from .utils.helpers import wrap_class, cumprod
from .layers import LayerNorm, Linear, MLP
from .factorization import Matricize, SVD
from .unet import UNet


class PosEmbed(nn.Module):
    """"Learnable positional embedding."""

    def __init__(self, input_dim, spatial_size, dropout=0.0, **kwargs):
        super().__init__()
        self.pos = nn.Parameter(torch.empty(1, input_dim, *spatial_size))
        nn.init.normal_(self.pos, std=1.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: B × C × S1 × S2 × ... × Sp
        out = x + self.pos
        out = self.dropout(out)
        return out


class DimWisePosEmbed(nn.Module):
    """"Learnable positional embedding."""

    def __init__(self, input_dim, spatial_size, dropout=0.0, **kwargs):
        super().__init__()
        self.pos = []
        for dim, size in enumerate(spatial_size):
            pe_spatial_size = [
                size if j == dim else 1 for j in range(len(spatial_size))
            ]
            pe = nn.Parameter(torch.empty(1, input_dim, *pe_spatial_size))
            nn.init.normal_(pe, std=1.0)
            self.pos.append(pe)

        self.pos = nn.ParameterList(self.pos)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: B × C × S1 × S2 × ... × Sp
        out = x
        for p in self.pos:
            out = out + p

        out = self.dropout(out)
        return out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class FactorizerSubblock(nn.Module):
    """Generic Factorizer Sublock."""

    def __init__(
        self,
        input_dim,
        output_dim,
        spatial_size,
        tensorize=(Matricize, {"num_heads": 1, "grid_size": 1}),
        act=nn.Identity,
        factorize=SVD,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        tensorize = wrap_class(tensorize)
        act = wrap_class(act)
        factorize = wrap_class(factorize)

        self.in_proj = Linear(input_dim, output_dim, bias=False)

        self.tensorize = tensorize((None, output_dim, *spatial_size))

        self.act = act()

        tensorized_size = self.tensorize.output_size[2:]
        tensorized_size = tuple(
            s for s in tensorized_size if s != 1
        )  # squeeze size
        self.tensorized_size = tensorized_size
        self.factorize = factorize(tensorized_size, **kwargs)

        self.out_proj = Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: B × C × S1 × S2 × ... × Sp

        # input linear projection
        out = self.in_proj(x)

        # tensorize (fold/unfold)
        out = self.tensorize(out)

        # apply activation function
        out = self.act(out)

        # save the size
        shape = out.shape

        # remove 1-dim modes (squeeze)
        out = out.reshape(*shape[:2], *self.tensorized_size)

        # tensor factorization)
        out = self.factorize(out)

        # back to the original size (unsqueeze)
        out = out.reshape(shape)

        # detensorize (unfold/fold)
        out = self.tensorize.inverse_forward(out)

        # output linear projection
        out = self.out_proj(out)

        # dropout
        out = self.dropout(out)
        return out


class FactorizerSubblockV2(nn.Module):
    """Generic Factorizer Sublock V2."""

    def __init__(
        self,
        input_dim,
        output_dim,
        spatial_size,
        inner_dim=None,
        ratio=2,
        tensorize=(Matricize, {"num_heads": 1, "grid_size": 1}),
        act=nn.Identity,
        factorize=SVD,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        inner_dim = int(ratio * input_dim) if inner_dim is None else inner_dim
        tensorize = wrap_class(tensorize)
        act = wrap_class(act)
        factorize = wrap_class(factorize)

        self.tensorize = tensorize((None, input_dim, *spatial_size))
        self.act = act()
        tensorized_size = self.tensorize.output_size[2:]
        tensorized_size = tuple(
            s for s in tensorized_size if s != 1
        )  # squeeze size
        self.tensorized_size = tensorized_size
        self.factorize = factorize(tensorized_size, **kwargs)

        self.mlp = MLP(input_dim, output_dim, inner_dim, ratio, dropout)

    def forward(self, x):
        # x: B × C × S1 × S2 × ... × Sp

        # tensorize (fold/unfold)
        out = self.tensorize(x)

        # apply activation function
        out = self.act(out)

        # save the size
        shape = out.shape

        # remove 1-dim modes (squeeze)
        out = out.reshape(*shape[:2], *self.tensorized_size)

        # tensor factorization)
        out = self.factorize(out)

        # back to the original size (unsqueeze)
        out = out.reshape(shape)

        # detensorize (unfold/fold)
        out = self.tensorize.inverse_forward(out)

        # MLP
        out = self.mlp(out)
        return out


class FactorizerBlock(nn.Module):
    """Generic Factorizer Block."""

    def __init__(
        self,
        input_dim,
        output_dim,
        spatial_size,
        adapter=None,
        pos_embed=False,
        dropout=0.0,
        **subblocks,
    ):
        super().__init__()
        if input_dim != output_dim:
            if adapter is None:
                self.adapter = nn.Sequential(
                    Linear(input_dim, output_dim, bias=False),
                    nn.Dropout(dropout),
                )
            else:
                adapter = wrap_class(adapter)
                self.adapter = adapter(input_dim, output_dim)

        if pos_embed:
            self.pos_embed = PosEmbed(input_dim, spatial_size, dropout=dropout)

        self.blocks = nn.ModuleDict()
        for key, value in subblocks.items():
            subblock = wrap_class(value)
            block = Residual(
                nn.Sequential(
                    LayerNorm(output_dim),
                    subblock(
                        output_dim, output_dim, spatial_size=spatial_size
                    ),
                )
            )
            self.blocks[key] = block

    def forward(self, x):
        # x: B × C × S1 × S2 × ... × Sp
        out = self.adapter(x) if hasattr(self, "adapter") else x
        out = self.pos_embed(out) if hasattr(self, "pos_embed") else out

        for block in self.blocks.values():
            out = block(out)

        return out


class FactorizerBlockV2(nn.Module):
    """Generic Factorizer V2 Block."""

    def __init__(
        self,
        input_dim,
        output_dim,
        spatial_size,
        adapter=None,
        pos_embed=DimWisePosEmbed,
        dropout=0.0,
        **subblocks,
    ):
        super().__init__()
        if input_dim != output_dim:
            if adapter is None:
                self.adapter = nn.Sequential(
                    Linear(input_dim, output_dim, bias=False),
                    nn.Dropout(dropout),
                )
            else:
                adapter = wrap_class(adapter)
                self.adapter = adapter(input_dim, output_dim)

        pos_embed = wrap_class(pos_embed)
        self.pos_embed = pos_embed(output_dim, spatial_size, dropout=dropout)

        self.blocks = nn.ModuleDict()
        for key, value in subblocks.items():
            subblock = wrap_class(value)
            block = Residual(
                nn.Sequential(
                    LayerNorm(output_dim),
                    subblock(
                        output_dim, output_dim, spatial_size=spatial_size
                    ),
                )
            )
            self.blocks[key] = block

    def forward(self, x):
        # x: B × C × S1 × S2 × ... × Sp
        out = self.adapter(x) if hasattr(self, "adapter") else x
        out = self.pos_embed(out) if hasattr(self, "pos_embed") else out

        for block in self.blocks.values():
            out = block(out)

        return out


class SegmentationFactorizer(UNet):
    """Segmentation Factorizer (SeF)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        spatial_size,
        stem_width=None,
        encoder_depth=(1, 1, 1, 1, 1),
        encoder_width=(32, 64, 128, 256, 512),
        strides=(1, 2, 2, 2, 2),
        decoder_depth=(1, 1, 1, 1),
        stem=None,
        downsample=None,
        upsample=None,
        head=None,
        pos_embed=True,
        num_deep_supr=1,
        **kwargs,
    ):

        self.spatial_size = spatial_size
        spatial_dims = len(spatial_size)
        blocks = self._get_blocks(
            spatial_size,
            encoder_depth,
            decoder_depth,
            strides,
            pos_embed,
            **kwargs,
        )
        super().__init__(
            in_channels,
            out_channels,
            spatial_dims=spatial_dims,
            stem_width=stem_width,
            encoder_depth=encoder_depth,
            encoder_width=encoder_width,
            strides=strides,
            decoder_depth=decoder_depth,
            stem=stem,
            downsample=downsample,
            block=blocks,
            upsample=upsample,
            head=head,
            num_deep_supr=num_deep_supr,
        )

    def _get_blocks(
        self,
        spatial_size,
        encoder_depth,
        decoder_depth,
        strides,
        pos_embed,
        **kwargs,
    ):
        net_depth = encoder_depth + decoder_depth
        num_layers = len(net_depth)
        cumstrides = cumprod(strides)
        blocks = {}
        for layer, depth in enumerate(net_depth):
            for sublayer in range(depth):
                # get spatial size at current level
                level = min(layer, num_layers - 1 - layer)
                current_spatial_size = tuple(
                    s // cumstrides[level] for s in spatial_size
                )

                bridge = (layer, sublayer) == (num_layers // 2, 0)
                params = {
                    "spatial_size": current_spatial_size,
                    "pos_embed": pos_embed and bridge,
                    **kwargs,
                }
                blocks[(layer, sublayer)] = (FactorizerBlock, params)

        return blocks


class SegmentationFactorizerV2(UNet):
    """Segmentation Factorizer V2 (SeFV2)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        spatial_size,
        stem_width=None,
        encoder_depth=(1, 1, 1, 1, 1),
        encoder_width=(32, 64, 128, 256, 512),
        strides=(1, 2, 2, 2, 2),
        decoder_depth=(1, 1, 1, 1),
        stem=None,
        downsample=None,
        upsample=None,
        head=None,
        pos_embed=DimWisePosEmbed,
        num_deep_supr=1,
        **kwargs,
    ):

        self.spatial_size = spatial_size
        spatial_dims = len(spatial_size)
        blocks = self._get_blocks(
            spatial_size,
            encoder_depth,
            decoder_depth,
            strides,
            pos_embed,
            **kwargs,
        )
        super().__init__(
            in_channels,
            out_channels,
            spatial_dims=spatial_dims,
            stem_width=stem_width,
            encoder_depth=encoder_depth,
            encoder_width=encoder_width,
            strides=strides,
            decoder_depth=decoder_depth,
            stem=stem,
            downsample=downsample,
            block=blocks,
            upsample=upsample,
            head=head,
            num_deep_supr=num_deep_supr,
        )

    def _get_blocks(
        self,
        spatial_size,
        encoder_depth,
        decoder_depth,
        strides,
        pos_embed,
        **kwargs,
    ):
        net_depth = encoder_depth + decoder_depth
        num_layers = len(net_depth)
        cumstrides = cumprod(strides)
        blocks = {}
        for layer, depth in enumerate(net_depth):
            for sublayer in range(depth):
                # get spatial size at current level
                level = min(layer, num_layers - 1 - layer)
                current_spatial_size = tuple(
                    s // cumstrides[level] for s in spatial_size
                )
                params = {
                    "spatial_size": current_spatial_size,
                    "pos_embed": pos_embed,
                    **kwargs,
                }
                blocks[(layer, sublayer)] = (FactorizerBlockV2, params)

        return blocks


def ablate(obj, name, indicator):
    class Ablated(obj):
        def _get_blocks(self, *args, **kwargs):
            blocks = super()._get_blocks(*args, **kwargs)
            for (layer, sublayer), (cl, params) in blocks.items():
                params_new = copy.deepcopy(params)
                if indicator(layer, sublayer):
                    params_new[name[0]][1][name[1]] = nn.Identity
                    blocks[(layer, sublayer)] = (cl, params_new)

            return blocks

    Ablated.__name__ = obj.__name__
    locals()[obj.__name__] = Ablated
    del Ablated
    return locals()[obj.__name__]
