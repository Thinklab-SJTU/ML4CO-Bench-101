from torch import Tensor
from goal.model.adapter.base import (
    GOALNodeAdapter, GOALEdgeAdapter, GOALOutputAdapter
)


class CVRPNodeAdapter(GOALNodeAdapter):
    def __init__(self, hidden_dim: int, low_dim: int = 8):
        super(CVRPNodeAdapter, self).__init__(
            hidden_dim=hidden_dim, problem_dim=4, low_dim=low_dim
        )
        
    def forward(
        self, nodes_feature: Tensor, random_embedding: Tensor
    ) -> Tensor:
        node_proto_emb = nodes_feature @ self.init_feature[:-2]
        node_proto_emb[:, 0, :] = node_proto_emb[:, 0, :] + self.init_feature[-2]
        node_proto_emb[:, -1, :] = node_proto_emb[:, -1, :] + self.init_feature[-1]
        node_emb = random_embedding + self.input_node_projection(node_proto_emb)
        return node_emb
    
    
class CVRPEdgeAdapter(GOALEdgeAdapter):
    def __init__(
        self, hidden_dim: int, low_dim: int = 4, activation: str = "relu"
    ):
        super(CVRPEdgeAdapter, self).__init__(
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
            
            
class CVRPOutputAdapter(GOALOutputAdapter):
    def __init__(self, hidden_dim: int):
        super(CVRPOutputAdapter, self).__init__(
            hidden_dim=hidden_dim, problem_dim=2
        )
            
    def forward(self, state: Tensor) -> Tensor:
        scores: Tensor = self.out(state)
        return scores.squeeze(-1)