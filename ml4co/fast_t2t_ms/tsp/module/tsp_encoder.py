from mindspore import nn, Tensor
from ml4co.fast_t2t_ms.common import GCNSparseLayer, zero_module


class TSPEncoder(nn.Cell):
    def __init__(self, hidden_dim: int = 256, num_layers: int = 12):
        # Super args
        super().__init__()

        # GCN layers
        self.gcn_layers = nn.CellList(
            [GCNSparseLayer(hidden_dim) for _ in range(num_layers)]
        )

        # Time layers
        self.time_embed_layers = nn.CellList(
            [
                nn.SequentialCell(
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )

        # Per layer output projection
        self.per_layer_out = nn.CellList(
            [
                nn.SequentialCell(
                    nn.LayerNorm((hidden_dim,)),
                    nn.SiLU(),
                    zero_module(nn.Linear(hidden_dim, hidden_dim)),
                )
                for _ in range(num_layers)
            ]
        )

    def construct(
        self, x: Tensor, e: Tensor, t: Tensor, edge_index: Tensor
    ) -> Tensor:
        """
        Input:
            x: (V, H) Node embeddings
            e: (E, H) Edge embeddings
            t: (H//2,) Time embeddings
            edge_index: (2, E) Edge index
        Output:
            e: (E, H) Encoded edge embeddings
        """
        # Apply layers
        for gcn_layer, time_layer, out_layer in zip(
            self.gcn_layers, self.time_embed_layers, self.per_layer_out
        ):
            x_in, e_in = x, e
            x_out, e_out = gcn_layer(x_in, e_in, edge_index)
            x = x_in + x_out
            e = e_in + out_layer(e_out + time_layer(t))

        # Return
        return e
