import math
import torch
import torch.nn.functional as F
from typing import Optional
from torch import Tensor, nn
from einops import rearrange


try:
    from torch.nn.functional import scaled_dot_product_attention
except ImportError:
    print(
        "torch.nn.functional.scaled_dot_product_attention not found. Make sure you are using PyTorch >= 2.0.0."
        "Alternatively, install Flash Attention https://github.com/HazyResearch/flash-attention"
    )

    def scaled_dot_product_attention(
        Q: Tensor, K: Tensor, V: Tensor, attn_mask: Tensor = None, 
        dropout_p: float = 0.0, is_causal: bool = False, scale: float = None
    ):
        """Simple Scaled Dot-Product Attention in PyTorch without Flash Attention"""
        if scale is None:
            scale = math.sqrt(Q.size(-1))  # scale factor
        # compute the attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        # apply causal masking if required
        if is_causal:
            mask = torch.triu(torch.ones_like(attn_scores), diagonal=1)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        # apply attention mask if provided
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float("-inf"))
        # compute attention probabilities
        attn_probs = F.softmax(attn_scores, dim=-1)
        # apply dropout
        attn_probs = F.dropout(attn_probs, p=dropout_p)
        # compute the weighted sum of values
        return torch.matmul(attn_probs, V)


def flash_attn_wrapper(self, func, *args, **kwargs):
    """Wrapper for flash attention to automatically cast to fp16 if needed"""
    if self.force_flash_attn and args[0].is_cuda:
        original_dtype = args[0].dtype
        args = [arg.half() for arg in args if isinstance(arg, torch.Tensor)]
        out = func(*args, **kwargs)
        return out.to(original_dtype)
    else:
        return func(*args, **kwargs)


class Normalization(nn.Module):
    def __init__(self, embed_dim, normalization="batch"):
        super(Normalization, self).__init__()

        normalizer_class = {"batch": nn.BatchNorm1d, "instance": nn.InstanceNorm1d}.get(
            normalization, None
        )

        self.normalizer = normalizer_class(embed_dim, affine=True)

    def init_parameters(self):
        for name, param in self.named_parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return x


class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)
    

class MultiHeadAttention(nn.Module):
    """PyTorch native implementation of Flash Multi-Head Attention with automatic mixed precision support.
    Uses PyTorch's native `scaled_dot_product_attention` implementation, available from 2.0

    Note:
        If `scaled_dot_product_attention` is not available, use custom implementation of `scaled_dot_product_attention` without Flash Attention.
        In case you want to use Flash Attention, you may have a look at the MHA module under `rl4co.bench4co.nn.flash_attention.MHA`.

    Args:
        embed_dim: total dimension of the model
        num_heads: number of heads
        bias: whether to use bias
        attention_dropout: dropout rate for attention weights
        causal: whether to apply causal mask to attention scores
        device: torch device
        dtype: torch dtype
        force_flash_attn: whether to force flash attention. If True, then we automatically cast to fp16
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        attention_dropout: float = 0.0,
        causal: bool = False,
        device=None,
        dtype=None,
        force_flash_attn: bool = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.force_flash_attn = force_flash_attn
        self.attention_dropout = attention_dropout

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert (
            self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, x, key_padding_mask=None):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        """
        # Project query, key, value
        q, k, v = rearrange(
            self.Wqkv(x), "b s (three h d) -> three b h s d", three=3, h=self.num_heads
        ).unbind(dim=0)

        # Scaled dot product attention
        out = self.flash_attn_wrapper(
            scaled_dot_product_attention,
            q,
            k,
            v,
            attn_mask=key_padding_mask,
            dropout_p=self.attention_dropout,
        )
        return self.out_proj(rearrange(out, "b h s d -> b s (h d)"))

    flash_attn_wrapper = flash_attn_wrapper


class MultiHeadAttentionLayer(nn.Sequential):
    """Multi-Head Attention Layer with normalization and feed-forward layer

    Args:
        num_heads: number of heads in the MHA
        embed_dim: dimension of the embeddings
        feed_forward_hidden: dimension of the hidden layer in the feed-forward layer
        normalization: type of normalization to use (batch, layer, none)
        force_flash_attn: whether to force FlashAttention (move to half precision)
    """

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        feed_forward_hidden: int = 512,
        normalization: Optional[str] = "batch",
        force_flash_attn: bool = False,
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    embed_dim, num_heads, force_flash_attn=force_flash_attn
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim),
                )
                if feed_forward_hidden > 0
                else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization),
        )