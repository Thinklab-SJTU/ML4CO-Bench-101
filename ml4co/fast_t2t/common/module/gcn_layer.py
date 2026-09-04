import torch
import torch.nn.functional as F
from typing import Sequence
from torch import Tensor, nn
from torch_sparse import SparseTensor
from torch_sparse import sum as sparse_sum


class GroupNorm32(nn.GroupNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


def zero_module(module: nn.Module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class GCNSparseLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        # Super Args
        super(GCNSparseLayer, self).__init__()

        # Linear Layers
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        # Normalization for nodes and edges
        self.norm_x = nn.LayerNorm(hidden_dim, elementwise_affine=True)
        self.norm_e = nn.LayerNorm(hidden_dim, elementwise_affine=True)

    def forward(self, x: Tensor, e: Tensor, edge_index: Tensor) -> Sequence[Tensor]:
        """
        Args:
            x: (V, H) Node features
            e: (E, H) Edge features
            edge_index: (2, E) Edge index
        Returns:
            x_out: (V, H) Updated node features
            e_out: (E, H) Updated edge features
        """
        nodes_num = x.shape[0] # Total number of nodes
        
        # Linear transformation for node embeddings
        Ux: Tensor = self.U(x) # (V, H)
        
        # Aggregate neighbor information for edges
        Vx = self.V(x[edge_index[1]]) # (E, H)

        # Message passing from nodes to edges
        Ax = self.A(x) # (V, H), source
        Bx = self.B(x) # (V, H), target
        
        # Update edge features
        Ce = self.C(e) # (E, H)
        e = Ax[edge_index[1]] + Bx[edge_index[0]] + Ce # (E, H)
            
        # Sigmoid gates for edge features
        gates = torch.sigmoid(e) # (E, H)
        
        # Aggregate messages for node embeddings
        x = Ux + self.aggregate(Vx, gates, edge_index, nodes_num) # (V, H)
        
        # Apply normalization and activation
        x_out = F.relu(self.norm_x(x)) # (V, H)
        e_out = F.relu(self.norm_e(e)) # (E, H)
        
        return x_out, e_out

    def aggregate(
        self, Vx: Tensor, gates: Tensor, edge_index: Tensor, nodes_num: int
    ) -> Tensor:
        sparseVh = SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            value=Vx * gates,
            sparse_sizes=(nodes_num, nodes_num)
        )
        return sparse_sum(sparseVh, dim=1)


class GCNDenseLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        # Super Args
        super(GCNDenseLayer, self).__init__()

        # Linear Layers
        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        # Normalization for nodes and edges
        self.norm_x = nn.LayerNorm(hidden_dim, elementwise_affine=True)
        self.norm_e = nn.LayerNorm(hidden_dim, elementwise_affine=True)
        
    def forward(self, x: Tensor, e: Tensor) -> Sequence[Tensor]:
        """
        Args:
            x: (B, V, H) Node features; 
            e: (B, V, V, H) Edge features
            graph: (B, V, V) Graph adjacency matrices
        Returns:
            Updated x and e after one layer of GNN.
        """
        batch_size, nodes_num, hidden_dim = x.shape
        
        # Linear transformation for node embeddings
        Ux: Tensor = self.U(x) # (B, V, H)
        
        # Aggregate neighbor information for edges
        Vx: Tensor = self.V(x) # (B, V, H)
        Vx = Vx.unsqueeze(1).expand(-1, nodes_num, -1, -1) # (B, V, V, H)
        
        # Message passing from nodes to edges
        Ax: Tensor = self.A(x) # (B, V, H), source
        Bx: Tensor = self.B(x) # (B, V, H), target
        
        # Update edge features
        Ce = self.C(e) # (B, V, V, H)
        e = Ax.unsqueeze(dim=1) + Bx.unsqueeze(dim=2) + Ce # (B, V, V, H)
            
        # Sigmoid gates for edge features
        gates = torch.sigmoid(e) # (B, V, V, H)
        
        # Aggregate messages for node embeddings
        x = Ux + torch.sum(Vx * gates, dim=2) # (B, V, H)

        # Apply normalization and activation
        x = x.view(batch_size * nodes_num, hidden_dim) # (B*V, H)
        x = F.relu(self.norm_x(x)).view(batch_size, nodes_num, hidden_dim) # (B, V, H)
        e = e.view(batch_size * nodes_num * nodes_num, hidden_dim)
        e = F.relu(self.norm_e(e)).view(batch_size, nodes_num, nodes_num, hidden_dim) # (B, V, V, H)
        
        return x, e