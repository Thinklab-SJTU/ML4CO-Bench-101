from typing import Sequence
from torch import Tensor, nn
from co_expander.model.embedder.base import COExpanderEmbedder
from co_expander.model.embedder.time import timestep_embedding
from co_expander.model.embedder.utils import (
    PositionEmbeddingSine, ScalarEmbeddingSine1D, 
    ScalarEmbeddingSine2D, ScalarEmbeddingSine3D
)


class CVRPEmbedder(COExpanderEmbedder):
    def __init__(self, hidden_dim: int, sparse: bool, time_flag: bool):
        super(CVRPEmbedder, self).__init__(hidden_dim, sparse, time_flag)
        
        # node embedder (position)
        self.pos_embed = PositionEmbeddingSine(hidden_dim // 2)

        # node embedder (demand)
        self.demand_embed = ScalarEmbeddingSine2D(hidden_dim)

        # node embedder (is depot)
        self.is_depot_embed = nn.Embedding(2, hidden_dim)

        # node embedder (merge)
        self.node_embed = nn.Linear(hidden_dim, hidden_dim)

        # edge embedder
        if self.sparse:
            # distance martix embedding
            self.distance_embed = nn.Sequential(
                ScalarEmbeddingSine1D(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            # diffusion label embedding
            self.label_embed = nn.Sequential(
                ScalarEmbeddingSine1D(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            # edge feature merge embedding
            self.edge_embed = nn.Linear(hidden_dim, hidden_dim)
            
            # mask embedding
            self.mask_embed = nn.Embedding(2, hidden_dim)
            
        else:
            # distance martix embedding
            self.distance_embed = nn.Sequential(
                ScalarEmbeddingSine3D(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            # diffusion label embedding
            self.label_embed = nn.Sequential(
                ScalarEmbeddingSine3D(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim)
            )
            
            # edge feature merge embedding
            self.edge_embed = nn.Linear(hidden_dim, hidden_dim) 
        
            # mask embedding
            self.mask_embed = nn.Embedding(2, hidden_dim)
            
    def dense_forward(
        self, nodes_feature: Tensor, x: Tensor, edges_feature: Tensor, 
        e: Tensor, mask: Tensor, t: Tensor
    ) -> Sequence[Tensor]:
        """
        Args:
            x: (V, 4); e: (E, 2); t: (1,)
        Return:
            x: (V, H); e: (E, H); t: (H)
        Note:
            4 dimensions: [x, y, demand, is depot] 
        """
        # node embedding (position)
        x_pos_embed: Tensor = self.pos_embed(nodes_feature[:, :, :2]) # (B, V, H)

        # node embedding (demand)
        x_demand_embed = self.demand_embed(nodes_feature[:, :, 2]) # (B, V, H)

        # node embedding (is_depot)
        x_is_depot_embed = self.is_depot_embed(nodes_feature[:, :, 3].long()) # (B, V, H)

        # node embedding (merge)
        x_feature = x_pos_embed + x_demand_embed + x_is_depot_embed # (B, V, H)
        x = self.node_embed(x_feature) # (B, V, H)
        
        # edge embedding
        e1 = self.distance_embed(edges_feature) # (B, V, V, H) 
        e2 = self.label_embed(e) # (B, V, V, H) 
        e = self.edge_embed(e1 + e2) # (B, V, V, H) 
        mask = self.mask_embed(mask.long()) # (B, V, V, H)
        e = e + mask  # (B, V, H)
        
        # time embedding
        if t is not None:
            t = self.time_embedder(
                timestep_embedding(t, self.hidden_dim)
            ) # (H,)

        # return
        return x, e, t