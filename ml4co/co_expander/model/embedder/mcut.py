from typing import Sequence
from torch import Tensor, nn
from co_expander.model.embedder.base import COExpanderEmbedder
from co_expander.model.embedder.time import timestep_embedding
from co_expander.model.embedder.utils import (
    ScalarEmbeddingSine1D, ScalarEmbeddingSine2D, ScalarEmbeddingSine3D
)


class MCutEmbedder(COExpanderEmbedder):
    def __init__(self, hidden_dim: int, sparse: bool, time_flag: bool):
        super(MCutEmbedder, self).__init__(hidden_dim, sparse, time_flag)
        
        if self.sparse:
            # node embedder
            self.node_embed = nn.Sequential(
                ScalarEmbeddingSine1D(hidden_dim),
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
                ScalarEmbeddingSine2D(hidden_dim),
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
            nodes_feature: (V,) [not use]
            x: (V,)
            edges_feature: (E,)
            e: (E,) [not use]
            mask: (V,) focus on x
            t: (1,)
        Return:
            x: (V, H)
            e: (E, H)
            t: (H)
        """   
        # embedding
        x = self.node_embed(x) # (V, H)
        mask = self.mask_embed(mask.long()) # (V, H)
        x = x + mask # (V, H)
        e = self.edge_embed(edges_feature) # (E, H)
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
            nodes_feature: (B, V) [not use]
            x: (B, V)
            edges_feature: (B, V, V)
            e: (B, V, V) [not use]
            mask: (B, V) focus on x
            t: (1,)
        Return:
            x: (B, V, H)
            e: (B, V, V, H)
            t: (H)
        """
        # embedding
        x = self.node_embed(x) # (B, V, H)
        mask = self.mask_embed(mask.long()) # (B, V, H)
        x = x + mask  # (B, V, H)
        e = self.edge_embed(edges_feature) # (B, V, V, H)
        if t is not None:
            t = self.time_embedder(
                timestep_embedding(t, self.hidden_dim)
            ) # (H,)
        return x, e, t