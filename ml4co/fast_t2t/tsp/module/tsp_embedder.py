from typing import Sequence
from torch import nn, Tensor
from ml4co.fast_t2t.common import (
    PositionEmbeddingSine, ScalarEmbeddingSine1D, sinusoidal_embedding
)


class TSPEmbedder(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        # Super Args
        super(TSPEmbedder, self).__init__()

        # Store Args
        self.hidden_dim = hidden_dim

        # Node embedding for TSP
        self.node_embed = nn.Sequential(
            PositionEmbeddingSine(hidden_dim // 2),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Edge embedding for TSP
        self.edge_embed = nn.Sequential(
            ScalarEmbeddingSine1D(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Time embedding for TSP
        time_embed_dim = hidden_dim // 2
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, x: Tensor, e: Tensor, t: Tensor) -> Sequence[Tensor]:
        """
        Input:
            x: (V, 2) Points coordinates
            e: (E, H) Solution at time step t
            t: (1,) Time embeddings
        Output:
            x: (V, H) Node embeddings
            e: (E, H) Edge embeddings
            t: (H,) Time embeddings
        """
        x = self.node_embed(x) # (V, H)
        e = self.edge_embed(e) # (E, H)
        t = sinusoidal_embedding(t, dim=self.hidden_dim) # (H，)
        t = self.time_embed(t) # (H//2,)
        return x, e, t