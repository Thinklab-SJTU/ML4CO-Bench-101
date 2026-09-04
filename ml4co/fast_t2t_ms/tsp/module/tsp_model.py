from mindspore import nn, Tensor, ops
from .tsp_encoder import TSPEncoder
from .tsp_embedder import TSPEmbedder
from ml4co.fast_t2t_ms.common import GroupNorm32


class TSPModel(nn.Cell):
    def __init__(self, hidden_dim: int = 256, num_layers: int = 12):
        # Super Args
        super().__init__()

        # Embedder
        self.embedder = TSPEmbedder(hidden_dim=hidden_dim)

        # Encoder (GCN-based)
        self.encoder = TSPEncoder(hidden_dim=hidden_dim, num_layers=num_layers)

        # Final output projection
        self.out = nn.SequentialCell(
            GroupNorm32(32, hidden_dim),
            nn.ReLU(),
            nn.Conv2d(
                hidden_dim, 2, kernel_size=1, pad_mode="valid", has_bias=True
            ),
        )

    def construct(
        self,
        points: Tensor,
        st: Tensor,
        t: Tensor,
        edge_index: Tensor = None,
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
        # [Embedder] x: (V, H), t: (H//2,)
        x, e, t = self.embedder(points, st, t)

        # [Encoder] e: (E, H)
        e = self.encoder(x, e, t, edge_index)

        # [Output] logits: (E, 2)
        # reshape to (1, H, N, knn) for Conv2d, then flatten back to (E, 2)
        nodes_num, hidden_dim = x.shape
        edges_num = edge_index.shape[1]
        knn = edges_num // nodes_num
        e = e.reshape(1, nodes_num, knn, hidden_dim)
        e = ops.transpose(e, (0, 3, 1, 2))
        e = self.out(e)
        logits = e.reshape(2, edges_num)
        logits = ops.transpose(logits, (1, 0))

        # Return
        return logits
