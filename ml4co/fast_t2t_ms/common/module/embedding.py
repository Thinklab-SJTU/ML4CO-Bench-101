import math
import mindspore as ms
from mindspore import nn, ops, Tensor


class PositionEmbeddingSine(nn.Cell):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self,
        embed_dim: int,
        tau: int = 10000,
        normalize: bool = True,
        scale: float = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.tau = tau
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def construct(self, node_coords: Tensor) -> Tensor:
        """
        node_coords (B, N, 2)
        node_coords_x / node_coords_y (B, N)
        x_embed / y_embed (B, N, self.embed_dim)
        embed (B, N, 2*self.embed_dim)
        """
        # check dim of node_coords and get x and y of it
        dim_2 = False
        if node_coords.ndim == 2:
            dim_2 = True
            node_coords = ops.expand_dims(node_coords, 0)
        node_coords_x = node_coords[:, :, 0]
        node_coords_y = node_coords[:, :, 1]

        # deal with normalize of node_coords_x/y
        if self.normalize:
            node_coords_x = node_coords_x * self.scale
            node_coords_y = node_coords_y * self.scale

        # get dim_t
        dim_t = ops.arange(self.embed_dim, dtype=ms.float32)
        dim_t = 2.0 * ops.floor(dim_t / 2) / self.embed_dim
        dim_t = self.tau ** dim_t

        # (B, N) -> (B, N, self.embed_dim)
        x_embed = ops.expand_dims(node_coords_x, -1) / dim_t
        y_embed = ops.expand_dims(node_coords_y, -1) / dim_t

        # sin for odd and cos for even
        x_embed = ops.stack(
            (x_embed[:, :, 0::2].sin(), x_embed[:, :, 1::2].cos()), axis=3
        )
        x_embed = x_embed.reshape(x_embed.shape[0], x_embed.shape[1], -1)
        y_embed = ops.stack(
            (y_embed[:, :, 0::2].sin(), y_embed[:, :, 1::2].cos()), axis=3
        )
        y_embed = y_embed.reshape(y_embed.shape[0], y_embed.shape[1], -1)

        # merge
        embed = ops.cat((x_embed, y_embed), axis=2)

        # check dim
        if dim_2:
            embed = embed[0]
        return embed


class ScalarEmbeddingSine1D(nn.Cell):
    def __init__(self, embed_dim: int, tau: int = 10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.tau = tau

    def construct(self, x: Tensor) -> Tensor:
        """
        x: (V)
        embed: (V, self.embed_dim)
        """
        # get dim_t
        dim_t = ops.arange(self.embed_dim, dtype=ms.float32)
        dim_t = 2.0 * ops.floor(dim_t / 2) / self.embed_dim
        dim_t = self.tau ** dim_t

        # (N) -> (N, self.embed_dim)
        embed = ops.expand_dims(x, -1) / dim_t

        # sin for odd and cos for even
        embed = ops.stack((embed[:, 0::2].sin(), embed[:, 1::2].cos()), axis=2)
        return embed.reshape(embed.shape[0], -1)


class ScalarEmbeddingSine2D(nn.Cell):
    def __init__(self, embedding_dim: int, tau: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.tau = tau

    def construct(self, x: Tensor) -> Tensor:
        """
        x: (B, V)
        embed: (B, V, self.embedding_dim)
        """
        # get dim_t
        dim_t = ops.arange(self.embedding_dim, dtype=ms.float32)
        dim_t = 2.0 * ops.floor(dim_t / 2) / self.embedding_dim
        dim_t = self.tau ** dim_t

        # (B, V) -> (B, V, self.embedding_dim)
        embed = ops.expand_dims(x, -1) / dim_t

        # sin for odd and cos for even
        embed = ops.stack(
            (embed[:, :, 0::2].sin(), embed[:, :, 1::2].cos()), axis=3
        )
        return embed.reshape(embed.shape[0], embed.shape[1], -1)


class ScalarEmbeddingSine3D(nn.Cell):
    def __init__(self, embed_dim: int, tau: int = 10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.tau = tau

    def construct(self, x: Tensor) -> Tensor:
        """
        x: (B, V, V)
        embed: (B, V, V, self.embed_dim)
        """
        # get dim_t
        dim_t = ops.arange(self.embed_dim, dtype=ms.float32)
        dim_t = 2.0 * ops.floor(dim_t / 2) / self.embed_dim
        dim_t = self.tau ** dim_t

        # (B, H, W) -> (B, H, W, self.embed_dim)
        embed = ops.expand_dims(x, -1) / dim_t

        # sin for odd and cos for even
        embed = ops.stack(
            (embed[:, :, :, 0::2].sin(), embed[:, :, :, 1::2].cos()), axis=4
        )
        return embed.reshape(
            embed.shape[0], embed.shape[1], embed.shape[2], -1
        )


def sinusoidal_embedding(x: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """
    Create sinusoidal embeddings.

    :param x: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = ops.exp(
        -math.log(max_period)
        * ops.arange(0, half, dtype=ms.float32)
        / half
    )
    args = ops.expand_dims(x.astype(ms.float32), -1) * ops.expand_dims(freqs, 0)
    embedding = ops.cat([ops.cos(args), ops.sin(args)], axis=-1)
    if dim % 2:
        embedding = ops.cat(
            [embedding, ops.zeros_like(embedding[:, :1])], axis=-1
        )
    return embedding
