from typing import Sequence
from torch import Tensor, nn
from co_expander.model.embedder.base import COExpanderEmbedder
from co_expander.model.embedder.time import timestep_embedding
from co_expander.model.embedder.utils import (
    PositionEmbeddingSine, ScalarEmbeddingSine1D, ScalarEmbeddingSine3D
)


class TSPEmbedder(COExpanderEmbedder):
    def __init__(self, hidden_dim: int, sparse: bool, time_flag: bool):
        super(TSPEmbedder, self).__init__(hidden_dim, sparse, time_flag)
        
        if self.sparse:
            # node embedder
            self.node_embed = nn.Sequential(
                PositionEmbeddingSine(hidden_dim // 2),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
            # edge embedder
            self.edge_embed = nn.Sequential(
                ScalarEmbeddingSine1D(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            # mask embedder
            self.mask_embed = nn.Embedding(2, hidden_dim)
            
        else:
            # node embedder
            self.node_embed = nn.Sequential(
                PositionEmbeddingSine(hidden_dim // 2),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
            # edge embedder
            self.edge_embed = nn.Sequential(
                ScalarEmbeddingSine3D(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim)
            )

            # mask embedder
            self.mask_embed = nn.Embedding(2, hidden_dim)

    def sparse_forward(
        self, nodes_feature: Tensor, x: Tensor, edges_feature: Tensor, 
        e: Tensor, mask: Tensor, t: Tensor
    ) -> Sequence[Tensor]:
        """
        Args:
            nodes_feature: (V, 2) 
            x: (V,) [not use]
            edges_feature: (E,) [not use]
            e: (E,) 
            mask: (V,) focus on x
            t: (1,)
        Return:
            x: (V, H)
            e: (E, H)
            t: (H)
        """   
        # embedding
        x = self.node_embed(nodes_feature) # (V, H)
        e = self.edge_embed(e) # (E, H)
        mask = self.mask_embed(mask.long()) # (E, H)
        e = e + mask # (V, H)
        if t is not None:
            t = self.time_embedder(
                timestep_embedding(t, self.hidden_dim)
            ) # (H,)
        return x, e, t
    
    def dense_forward(
        self, nodes_feature: Tensor, x: Tensor, edges_feature: Tensor, 
        e: Tensor, mask: Tensor, t: Tensor
    ) -> Sequence[Tensor]:
        """
        Args:
            nodes_feature: (B, V, 2) 
            x: (B, V) [not use]
            edges_feature: (B, V, V) [not use]
            e: (B, V, V)
            mask: (B, V) focus on x
            t: (1,)
        Return:
            x: (B, V, H)
            e: (B, V, V, H)
            t: (H)
        """
        # embedding
        x = self.node_embed(nodes_feature) # (B, V, H)
        e = self.edge_embed(e) # (B, V, V, H)
        mask = self.mask_embed(mask.long()) # (B, V, V, H)
        e = e + mask  # (B, V, H)
        if t is not None:
            t = self.time_embedder(
                timestep_embedding(t, self.hidden_dim)
            ) # (H,)
        return x, e, t