from torch import Tensor
from goal.model.adapter.base import (
    GOALNodeAdapter, GOALEdgeAdapter, GOALOutputAdapter
)


class ATSPNodeAdapter(GOALNodeAdapter):
    def __init__(self, hidden_dim: int, low_dim: int = 8):
        super(ATSPNodeAdapter, self).__init__(
            hidden_dim=hidden_dim, problem_dim=2, low_dim=low_dim
        )
        
    def forward(
        self, nodes_feature: Tensor, random_embedding: Tensor
    ) -> Tensor:
        node_emb = random_embedding
        node_orig_dest_emb = self.input_node_projection(self.init_feature)
        node_emb[:, 0, :] = random_embedding[:, 0, :] + node_orig_dest_emb[0]
        node_emb[:, -1, :] = random_embedding[:, -1, :] + node_orig_dest_emb[1]
        return node_emb
    
    
class ATSPEdgeAdapter(GOALEdgeAdapter):
    def __init__(
        self, hidden_dim: int, low_dim: int = 4, activation: str = "relu"
    ):
        super(ATSPEdgeAdapter, self).__init__(
            hidden_dim=hidden_dim, problem_dim=2, 
            low_dim=low_dim, activation=activation
        )
        
    def forward(self, edges_feature: Tensor) -> Tensor:
        if edges_feature is None:
            return None
        edge_proto_emb = edges_feature @ self.init_feature
        edge_emb = self.input_edge_projection(edge_proto_emb)
        if self.activation is not None:
            edge_emb = self.activation(edge_emb)
        return edge_emb
            
            
class ATSPOutputAdapter(GOALOutputAdapter):
    def __init__(self, hidden_dim: int):
        super(ATSPOutputAdapter, self).__init__(
            hidden_dim=hidden_dim, problem_dim=1
        )
            
    def forward(self, state: Tensor) -> Tensor:
        scores: Tensor = self.out(state)
        return scores.squeeze(-1)