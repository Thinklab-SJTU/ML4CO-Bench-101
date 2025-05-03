import torch
from torch import Tensor, nn
from tensordict import TensorDict
from symnco.model.embedder.base import SymNCOInitEmbedder
from symnco.model.embedder.base import SymNCOContextEmbedder


class CVRPInitEmbedder(SymNCOInitEmbedder):
    def __init__(self, hidden_dim: int):
        super(CVRPInitEmbedder, self).__init__(hidden_dim)
        
        self.init_embed = nn.Linear(3, hidden_dim)
        self.init_embed_depot = nn.Linear(2, hidden_dim)
    
    def forward(self, td: TensorDict) -> Tensor:
        # [batch, 1, 2]-> [batch, 1, embedding_dim]
        depot, cities = td["locs"][:, :1, :], td["locs"][:, 1:, :]
        depot_embedding = self.init_embed_depot(depot)
        
        # [batch, n_city, 2, batch, n_city, 1]  -> [batch, n_city, embedding_dim]
        node_embeddings = self.init_embed(
            torch.cat((cities, td["demand"][..., None]), -1)
        )
        
        # [batch, n_city+1, embedding_dim]
        out = torch.cat((depot_embedding, node_embeddings), -2)
        return out
    
    
class CVRPContextEmbedder(SymNCOContextEmbedder):
    def __init__(self, hidden_dim: int):
        super(CVRPContextEmbedder, self).__init__(hidden_dim, hidden_dim+1)

    def _state_embedding(self, embeddings: torch.Tensor, td: TensorDict):
        state_embedding = td["vehicle_capacity"] - td["used_capacity"]
        return state_embedding