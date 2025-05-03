from typing import Sequence
from torch import nn, Tensor


class COExpanderEmbedder(nn.Module):
    def __init__(self, hidden_dim: int, sparse: bool, time_flag: bool):
        super(COExpanderEmbedder, self).__init__()
        
        # dims
        self.hidden_dim = hidden_dim

        # sparse
        self.sparse = sparse
        
        # time embedder
        if time_flag:
            self.time_embedder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def forward(
        self, nodes_feature: Tensor, x: Tensor, edges_feature: Tensor, 
        e: Tensor, mask: Tensor, t: Tensor
    ) -> Sequence[Tensor]:
        """
        Args:
            [sparse]
                nodes_feature: (V,) or (V, C)
                x: (V,)
                edges_feature: (E,) or (E, C) 
                e: (E,) or (E, C)
                mask: (V,) focus on x
                t: (1,)
            [dense]
                nodes_feature: (B, V) or (B, V, C)
                x: (B, V) 
                edges_feature: (B, V, V) 
                e: (B, V, V) 
                mask: (B, V) focus on x
                t: (1,)
        Return:
            [sparse]
                x: (V, H)
                e: (E, H)
                t: (H)
            [dense]
                x: (B, V, H) 
                e: (B, V, V, H) 
                t: (H) 
        """
        if self.sparse:
            return self.sparse_forward(nodes_feature, x, edges_feature, e, mask, t)
        else:
            return self.dense_forward(nodes_feature, x, edges_feature, e, mask, t)
        
    def sparse_forward(
        self, nodes_feature: Tensor, x: Tensor, edges_feature: Tensor, 
        e: Tensor, mask: Tensor, t: Tensor
    ) -> Sequence[Tensor]:
        raise NotImplementedError(
            "``sparse_forward`` is required to implemented in subclasses."
        )
    
    def dense_forward(
        self, nodes_feature: Tensor, x: Tensor, edges_feature: Tensor, 
        e: Tensor, mask: Tensor, t: Tensor
    ) -> Sequence[Tensor]:
        raise NotImplementedError(
            "``dense_forward`` is required to implemented in subclasses."
        )
    