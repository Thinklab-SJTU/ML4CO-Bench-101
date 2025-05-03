import torch
from torch import Tensor, nn
from tensordict import TensorDict
from symnco.utils.ops import gather_by_index
from symnco.model.embedder.base import SymNCOInitEmbedder
from symnco.model.embedder.base import SymNCOContextEmbedder
from symnco.model.embedder.utils import PositionEmbeddingSine


class TSPInitEmbedder(SymNCOInitEmbedder):
    def __init__(self, hidden_dim: int):
        super(TSPInitEmbedder, self).__init__(hidden_dim)
        
        self.node_embed = nn.Sequential(
            PositionEmbeddingSine(hidden_dim // 2),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, td: TensorDict) -> Tensor:
        x = self.node_embed(td["locs"]) # (B, V, H)
        return x
    
    
class TSPContextEmbedder(SymNCOContextEmbedder):
    def __init__(self, hidden_dim: int):
        super(TSPContextEmbedder, self).__init__(hidden_dim, 2*hidden_dim)
        self.W_placeholder = nn.Parameter(
            torch.Tensor(2*self.hidden_dim).uniform_(-1, 1)
        )

    def forward(self, embeddings: torch.Tensor, td: TensorDict):
        batch_size = embeddings.size(0)
        # By default, node_dim = -1 (we only have one node embedding per node)
        node_dim = (
            (-1,) if td["first_node"].dim() == 1 else (td["first_node"].size(-1), -1)
        )
        if td["i"][(0,) * td["i"].dim()].item() < 1:  # get first item fast
            context_embedding = self.W_placeholder[None, :].expand(
                batch_size, self.W_placeholder.size(-1)
            )
        else:
            context_embedding = gather_by_index(
                embeddings,
                torch.stack([td["first_node"], td["current_node"]], -1).view(
                    batch_size, -1
                ),
            ).view(batch_size, *node_dim)
        return self.project_context(context_embedding)