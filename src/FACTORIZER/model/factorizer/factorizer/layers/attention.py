import math

import torch
from torch import nn
from performer_pytorch import SelfAttention


class ScaledDotProductAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.softmax = nn.Softmax(dim=-2)

    def attention_weights(self, query, key, mask):
        _, E, _ = query.shape
        query = query / math.sqrt(E)
        # t(* × E × N) @ (* × E × M) -> (* × N × M)
        attn = key.transpose(-2, -1) @ query
        if mask is not None:
            attn.masked_fill_(mask[None], -math.inf)

        attn = self.softmax(attn)
        return attn

    def forward(self, query, key=None, value=None, mask=None):
        # query: * × E × M, key: * × E × N, value: * × D × N
        key = query if key is None else key
        value = key if value is None else value

        attn = self.attention_weights(query, key, mask)
        # (* × D × N) @ (* × N × M) -> (* × D × M)
        output = value @ attn
        return output


class FastScaledDotProductAttention(nn.Module):
    """Fast scaled dot product attention based on Performer."""

    def __init__(self, size, kernel_fun=nn.ReLU(), **kwargs):
        super().__init__()
        self.dim, _ = size
        self.kernel_fun = kernel_fun

        proj = torch.randn(self.dim, self.dim)
        orth_proj, _ = torch.qr(proj)

        self.register_buffer("orth_proj", orth_proj)

    def forward(self, query, key=None, value=None, **kwargs):
        # query: * × E × M, key: * × E × N, value: * × D × N
        key = query if key is None else key
        value = key if value is None else value

        query = query / math.sqrt(self.dim)

        phi_q = self.kernel_fun(
            torch.einsum("... c m, c e -> ... e m", query, self.orth_proj)
        )
        phi_k = self.kernel_fun(
            torch.einsum("... c n, c e -> ... e n", key, self.orth_proj)
        )

        # normalize
        phi_k_rowsum = phi_k.sum(-1)  # * × E
        scale = torch.einsum("... e m, ... e -> ... m", phi_q, phi_k_rowsum)
        scale = 1 / (scale + 1e-8)
        phi_q = torch.einsum("... e m, ... m -> ... e m", phi_q, scale)

        kv = torch.einsum("... e n, ... d n -> ... e d", phi_k, value)
        output = torch.einsum("... e m, ... e d -> ... d m", phi_q, kv)
        return output


class FastSelfAttention(SelfAttention):
    def __init__(
        self,
        input_dim,
        output_dim=None,
        spatial_size=None,
        num_heads=None,
        head_dim=None,
        *args,
        **kwargs
    ):
        output_dim = input_dim if output_dim is None else output_dim
        assert (num_heads, head_dim) != (
            None,
            None,
        ), "'num_heads' or 'head_dim' must be specified."

        if num_heads is None:
            num_heads = input_dim // head_dim

        super().__init__(
            input_dim, heads=num_heads, dim_head=head_dim, *args, **kwargs
        )

    def forward(self, x, *args, **kwargs):
        # x: B × C × S1 × ... × SM
        shape = list(x.shape)
        out = x.flatten(2).transpose(1, 2)
        out = super().forward(out, *args, **kwargs)
        shape[1] = out.shape[-1]
        out = out.transpose(1, 2).reshape(shape)
        return out

