import mindspore as ms
from typing import Sequence
from mindspore import nn, ops, Tensor


class GroupNorm32(nn.GroupNorm):
    def construct(self, x: Tensor) -> Tensor:
        return super().construct(x.astype(ms.float32)).astype(x.dtype)


def zero_module(module: nn.Cell):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.get_parameters():
        p.set_data(ops.zeros_like(p))
    return module


class GCNSparseLayer(nn.Cell):
    """
    Sparse GCN layer. Aggregation uses ``tensor_scatter_add`` instead of
    ``torch_sparse`` (not available on MindSpore).
    """

    def __init__(self, hidden_dim: int):
        # Super Args
        super().__init__()

        # Linear Layers
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Normalization for nodes and edges
        self.norm_x = nn.LayerNorm((hidden_dim,))
        self.norm_e = nn.LayerNorm((hidden_dim,))

    def construct(self, x: Tensor, e: Tensor, edge_index: Tensor) -> Sequence[Tensor]:
        """
        Args:
            x: (V, H) Node features
            e: (E, H) Edge features
            edge_index: (2, E) Edge index
        Returns:
            x_out: (V, H) Updated node features
            e_out: (E, H) Updated edge features
        """
        nodes_num = x.shape[0]  # Total number of nodes

        # Linear transformation for node embeddings
        Ux = self.U(x)  # (V, H)

        # Aggregate neighbor information for edges
        Vx = self.V(x[edge_index[1]])  # (E, H)

        # Message passing from nodes to edges
        Ax = self.A(x)  # (V, H), source
        Bx = self.B(x)  # (V, H), target

        # Update edge features
        Ce = self.C(e)  # (E, H)
        e = Ax[edge_index[1]] + Bx[edge_index[0]] + Ce  # (E, H)

        # Sigmoid gates for edge features
        gates = ops.sigmoid(e)  # (E, H)

        # Aggregate messages for node embeddings
        x = Ux + self.aggregate(Vx, gates, edge_index, nodes_num)  # (V, H)

        # Apply normalization and activation
        x_out = ops.relu(self.norm_x(x))  # (V, H)
        e_out = ops.relu(self.norm_e(e))  # (E, H)

        return x_out, e_out

    def aggregate(
        self, Vx: Tensor, gates: Tensor, edge_index: Tensor, nodes_num: int
    ) -> Tensor:
        # Scatter-sum gated messages onto destination nodes (row = edge_index[0])
        values = Vx * gates  # (E, H)
        out = ops.zeros((nodes_num, values.shape[1]), values.dtype)
        indices = ops.expand_dims(edge_index[0].astype(ms.int32), -1)
        return ops.tensor_scatter_add(out, indices, values)


class GCNDenseLayer(nn.Cell):
    def __init__(self, hidden_dim: int):
        # Super Args
        super().__init__()

        # Linear Layers
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Normalization for nodes and edges
        self.norm_x = nn.LayerNorm((hidden_dim,))
        self.norm_e = nn.LayerNorm((hidden_dim,))

    def construct(self, x: Tensor, e: Tensor) -> Sequence[Tensor]:
        """
        Args:
            x: (B, V, H) Node features
            e: (B, V, V, H) Edge features
        Returns:
            Updated x and e after one layer of GNN.
        """
        batch_size, nodes_num, hidden_dim = x.shape

        # Linear transformation for node embeddings
        Ux = self.U(x)  # (B, V, H)

        # Aggregate neighbor information for edges
        Vx = self.V(x)  # (B, V, H)
        Vx = ops.broadcast_to(
            ops.expand_dims(Vx, 1), (batch_size, nodes_num, nodes_num, hidden_dim)
        )  # (B, V, V, H)

        # Message passing from nodes to edges
        Ax = self.A(x)  # (B, V, H), source
        Bx = self.B(x)  # (B, V, H), target

        # Update edge features
        Ce = self.C(e)  # (B, V, V, H)
        e = ops.expand_dims(Ax, 1) + ops.expand_dims(Bx, 2) + Ce  # (B, V, V, H)

        # Sigmoid gates for edge features
        gates = ops.sigmoid(e)  # (B, V, V, H)

        # Aggregate messages for node embeddings
        x = Ux + ops.reduce_sum(Vx * gates, 2)  # (B, V, H)

        # Apply normalization and activation
        x = x.reshape(batch_size * nodes_num, hidden_dim)  # (B*V, H)
        x = ops.relu(self.norm_x(x)).reshape(batch_size, nodes_num, hidden_dim)  # (B, V, H)
        e = e.reshape(batch_size * nodes_num * nodes_num, hidden_dim)
        e = ops.relu(self.norm_e(e)).reshape(
            batch_size, nodes_num, nodes_num, hidden_dim
        )  # (B, V, V, H)

        return x, e
