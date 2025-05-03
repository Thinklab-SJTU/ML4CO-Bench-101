import torch
from typing import Any
from torch import nn, Tensor
from tensordict import TensorDict
from symnco.utils.ops import gather_by_index


class SymNCOInitEmbedder(nn.Module):
    def __init__(self, hidden_dim: int):
        super(SymNCOInitEmbedder, self).__init__()
        self.hidden_dim = hidden_dim
    
    def forward(self, td: TensorDict) -> Any:
        raise NotImplementedError(
            "``forward`` is required to implemented in subclasses."
        )
      
        
class SymNCOContextEmbedder(nn.Module):
    def __init__(
        self, hidden_dim: int, step_context_dim: int = None, linear_bias: bool = False
    ):
        super(SymNCOContextEmbedder, self).__init__()
        self.hidden_dim = hidden_dim
        step_context_dim = (
            step_context_dim if step_context_dim is not None else hidden_dim
        )
        self.project_context = nn.Linear(
            step_context_dim, hidden_dim, bias=linear_bias
        )

    def _cur_node_embedding(self, embeddings: Tensor, td: TensorDict):
        cur_node_embedding = gather_by_index(embeddings, td["current_node"])
        return cur_node_embedding

    def _state_embedding(self, embeddings: Tensor, td: TensorDict):
        raise NotImplementedError("Implement for each environment")

    def forward(self, embeddings: Tensor, td: TensorDict):
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        state_embedding = self._state_embedding(embeddings, td)
        context_embedding = torch.cat([cur_node_embedding, state_embedding], -1)
        return self.project_context(context_embedding)


class SymNCODynamicEmbedder(nn.Module):
    """Static embedding for general problems.
    This is used for problems that do not have any dynamic information, except for the
    information regarding the current action (e.g. the current node in TSP). See context embedding for more details.
    """

    def __init__(self):
        super(SymNCODynamicEmbedder, self).__init__()

    def forward(self, td: TensorDict):
        return 0, 0, 0