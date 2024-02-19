import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn, Tensor
from zeta.structs import ViTransformerWrapper, Encoder
from zeta.nn import PerceiverLayer


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


# residual


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


# rotary positional embedding
# https://arxiv.org/abs/2104.09864


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(
            max_seq_len, device=device, dtype=self.inv_freq.dtype
        )
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


# classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU for gating the feedforward
# https://arxiv.org/abs/2002.05202


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


# parallel attention and feedforward with residual
# discovered by Wang et al + EleutherAI from GPT-J fame


class ParallelTransformerBlock(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.norm = LayerNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (
            attn_inner_dim,
            dim_head,
            dim_head,
            (ff_inner_dim * 2),
        )

        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.fused_attn_ff_proj = nn.Linear(
            dim, sum(self.fused_dims), bias=False
        )
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)

        self.ff_out = nn.Sequential(
            SwiGLU(), nn.Linear(ff_inner_dim, dim, bias=False)
        )

        # for caching causal mask and rotary embeddings

        self.register_buffer("mask", None, persistent=False)
        self.register_buffer("pos_emb", None, persistent=False)

    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones(
            (n, n), device=device, dtype=torch.bool
        ).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def forward(self, x):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device, h = x.shape[1], x.device, self.heads

        # pre layernorm

        x = self.norm(x)

        # attention queries, keys, values, and feedforward inner

        q, k, v, ff = self.fused_attn_ff_proj(x).split(
            self.fused_dims, dim=-1
        )

        # split heads
        # they use multi-query single-key-value attention, yet another Noam Shazeer paper
        # they found no performance loss past a certain scale, and more efficient decoding obviously
        # https://arxiv.org/abs/1911.02150

        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        # rotary embeddings

        positions = self.get_rotary_embedding(n, device)
        q, k = map(
            lambda t: apply_rotary_pos_emb(positions, t), (q, k)
        )

        # scale

        q = q * self.scale

        # similarity

        sim = einsum("b h i d, b j d -> b h i j", q, k)

        # causal mask

        causal_mask = self.get_mask(n, device)
        sim = sim.masked_fill(
            causal_mask, -torch.finfo(sim.dtype).max
        )

        # attention

        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.attn_out(out) + self.ff_out(ff)


# transformer
class VisualAdapter(nn.Module):
    """
    VisualAdapter module for adapting visual information using ViT and Perceiver resampler.

    Args:
        dim (int): Dimension of the model.
        depth (int): Number of layers in the model.
        heads (int): Number of attention heads.
        dropout (int): Dropout rate.
        dim_head (int): Dimension of each attention head.
        ff_mult (int): Multiplier for the feed-forward network dimension.
        image_size (int): Size of the input image.
        patch_size (int): Size of each image patch.
        *args: Variable length arguments.
        **kwargs: Keyword arguments.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dropout: int,
        dim_head: int,
        ff_mult: int,
        image_size: int,
        patch_size: int,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dropout = dropout
        self.dim_head = dim_head
        self.ff_mult = ff_mult

        self.vit = ViTransformerWrapper(
            image_size=image_size,
            patch_size=patch_size,
            attn_layers=Encoder(
                dim=dim, depth=depth, heads=heads, *args, **kwargs
            ),
            *args,
            **kwargs,
        )

        # Perceiver resampler
        self.resampler = PerceiverLayer(
            dim,
            heads,
            1,
            dim_head,
            dropout,
            dropout,
            ff_mult,
            *args,
            **kwargs,
        )

        # TODO: Add latent queries for the resampler

    def forward(self, text: Tensor, img: Tensor, *args, **kwargs):
        """
        Forward pass of the VisualAdapter module.

        Args:
            text (Tensor): Input text tensor.
            img (Tensor): Input image tensor.
            *args: Variable length arguments.
            **kwargs: Keyword arguments.

        Returns:
            Tensor: Output tensor after applying the Perceiver resampler.
        """
        b, s, d = text.shape

        learnable_queries = nn.Parameter(
            torch.randn(b, s, self.dim),
        )

        img = self.vit(img, return_embeddings=True)

        # Perceiver resampler
        resampler = self.resampler(learnable_queries, text, img)

        return resampler


# x = torch.randn(1, 32, 512)
# img = torch.randn(1, 3, 224, 224)

# model = VisualAdapter(
#     dim=512,
#     depth=6,
#     heads=8,
#     dropout=0.1,
#     dim_head=64,
#     ff_mult=4,
#     image_size=224,
#     patch_size=16,
# )

# out = model(x, img)
# print(out.shape)


class TinyPalm2(nn.Module):
    """
    Initialize a TinyPalm2 model.

    Args:
        dim (int): The input dimension.
        depth (int): The number of transformer blocks.
        dim_head (int, optional): The dimension of each attention head. Defaults to 64.
        heads (int, optional): The number of attention heads. Defaults to 8.
        ff_mult (int, optional): The multiplier for the feed-forward layer dimension. Defaults to 4.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Examples:
        >>> model = TinyPalm2(dim=512, depth=6)

    """

    def __init__(
        self,
        dim: int,
        depth: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.dim_head = dim_head
        self.heads = heads
        self.ff_mult = ff_mult

        self.net = nn.Sequential(
            *[
                Residual(
                    ParallelTransformerBlock(
                        dim=dim,
                        dim_head=dim_head,
                        heads=heads,
                        ff_mult=ff_mult,
                        *args,
                        **kwargs,
                    )
                )
                for _ in range(depth)
            ],
            LayerNorm(dim),
            nn.Linear(dim, dim, bias=False),
            *args,
            **kwargs,
        )

        # They used embedding weight not tied projection out to logits
        self.net[-1].weight = self.net[0].weight

        nn.init.normal_(self.net[0].weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the TinyPalm2 model.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        out = self.net(x)

        return out


def LargePaLM(
    *, dim, num_tokens, depth, dim_head=64, heads=8, ff_mult=4
):
    """
    LargePaLM model architecture.

    Args:
        dim (int): Dimension of the embedding and hidden states.
        num_tokens (int): Number of tokens in the vocabulary.
        depth (int): Number of transformer blocks.
        dim_head (int, optional): Dimension of each head in the multi-head attention. Defaults to 64.
        heads (int, optional): Number of attention heads. Defaults to 8.
        ff_mult (int, optional): Multiplier for the feed-forward network dimension. Defaults to 4.

    Returns:
        nn.Sequential: LargePaLM model.
    """
    net = nn.Sequential(
        nn.Embedding(num_tokens, dim),
        *[
            Residual(
                ParallelTransformerBlock(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    ff_mult=ff_mult,
                )
            )
            for _ in range(depth)
        ],
        LayerNorm(dim),
        nn.Linear(dim, num_tokens, bias=False),
    )

    # they used embedding weight tied projection out to logits, not common, but works
    net[-1].weight = net[0].weight

    nn.init.normal_(net[0].weight, std=0.02)
    return net
