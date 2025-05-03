from torch import Tensor
from goal.model.adapter.base import (
    GOALNodeAdapter, GOALEdgeAdapter, GOALOutputAdapter
)


class MVCNodeAdapter(GOALNodeAdapter):
    def __init__(self, hidden_dim: int, low_dim: int = 8):
        super(MVCNodeAdapter, self).__init__(
            hidden_dim=hidden_dim, problem_dim=-1, low_dim=low_dim
        )
        
    def forward(
        self, nodes_feature: Tensor, random_embedding: Tensor
    ) -> Tensor:
        return random_embedding
    
    
class MVCEdgeAdapter(GOALEdgeAdapter):
    def __init__(
        self, hidden_dim: int, low_dim: int = 4, activation: str = "relu"
    ):
        super(MVCEdgeAdapter, self).__init__(
            hidden_dim=hidden_dim, problem_dim=1, 
            low_dim=low_dim, activation=activation
        )
        
    def forward(self, edges_feature: Tensor) -> Tensor:
        if edges_feature is None:
            return None
        edges_feature = edges_feature.unsqueeze(dim=-1)
        edge_proto_emb = edges_feature @ self.init_feature
        edge_emb = self.input_edge_projection(edge_proto_emb)
        if self.activation is not None:
            edge_emb = self.activation(edge_emb)
        return edge_emb
            
            
class MVCOutputAdapter(GOALOutputAdapter):
    def __init__(self, hidden_dim: int):
        super(MVCOutputAdapter, self).__init__(
            hidden_dim=hidden_dim, problem_dim=1
        )
            
    def forward(self, state: Tensor) -> Tensor:
        scores: Tensor = self.out(state)
        return scores.squeeze(-1)