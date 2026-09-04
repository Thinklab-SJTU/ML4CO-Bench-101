from torch import nn, Tensor
from .tsp_encoder import TSPEncoder
from .tsp_embedder import TSPEmbedder
from ml4co.fast_t2t.common import GroupNorm32


class TSPModel(nn.Module):
    def __init__(
        self, 
        hidden_dim: int = 256,
        num_layers: int = 12, 
    ):
        # Super Args
        super(TSPModel, self).__init__()

        # Embedder
        self.embedder = TSPEmbedder(hidden_dim=hidden_dim)

        # Encoder (GCN-based)
        self.encoder = TSPEncoder(
            hidden_dim=hidden_dim, num_layers=num_layers
        )

        # Final output projection
        self.out = nn.Sequential(
            GroupNorm32(32, hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 2, kernel_size=1, bias=True)
        )
    
    def forward(
        self, points: Tensor, st: Tensor, 
        t: Tensor, edge_index: Tensor = None
    ) -> Tensor:
        """
        Input:
            points: (V, 2) Points coordinates
            st: (E,) Solution at time step t
            t: (1,) Time embeddings
            edge_index: (2, E) Edge index
        Output:
            logits: (E, 2) Predicted logits
        """
        # [Embedder] x: (V, H), t: (H)
        x, e, t = self.embedder.forward(x=points, e=st, t=t)

        # [Encoder] e: (E, H)
        e = self.encoder.forward(x=x, e=e, t=t, edge_index=edge_index)

        # [Output] logits: (E, 2)
        nodes_num, hidden_dim = x.shape
        edges_num = edge_index.shape[1]
        e = e.reshape(1, nodes_num, -1, hidden_dim).permute((0, 3, 1, 2))
        e: Tensor = self.out(e)
        logits = e.reshape(-1, edges_num).permute((1, 0))

        # Return
        return logits
