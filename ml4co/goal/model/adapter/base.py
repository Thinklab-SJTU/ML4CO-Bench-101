import torch
from torch import nn, Tensor


class GOALNodeAdapter(nn.Module):
    def __init__(self, hidden_dim: int, problem_dim: int, low_dim: int):
        super(GOALNodeAdapter, self).__init__()
        self.hidden_dim = hidden_dim
        self.problem_dim = problem_dim
        self.low_dim = low_dim
        if problem_dim != -1:
            self.init_feature = nn.Parameter(torch.ones(problem_dim, low_dim))
            self.input_node_projection = nn.Linear(low_dim, hidden_dim)
        
    def forward(
        self, nodes_feature: Tensor, random_embedding: Tensor
    ) -> Tensor:
        raise NotImplementedError(
            "``forward`` is required to implemented in subclasses."
        )


class GOALEdgeAdapter(nn.Module):
    def __init__(
        self, hidden_dim: int, problem_dim: int, low_dim: int, activation: str
    ):
        super(GOALEdgeAdapter, self).__init__()
        # basic
        self.hidden_dim = hidden_dim
        self.problem_dim = problem_dim
        self.low_dim = low_dim
        
        # trainalbe
        self.init_feature = nn.Parameter(torch.ones(problem_dim, low_dim))
        self.input_edge_projection = nn.Linear(low_dim, hidden_dim)
        
        # activation
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "gelu":
            self.activation = torch.nn.GELU()
        else:
            self.activation = None
            
    def forward(self, edges_feature: Tensor) -> Tensor:
        raise NotImplementedError(
            "``forward`` is required to implemented in subclasses."
        )


class GOALOutputAdapter(nn.Module):
    def __init__(self, hidden_dim: int, problem_dim: int):
        super(GOALOutputAdapter, self).__init__()
        # basic
        self.hidden_dim = hidden_dim
        self.problem_dim = problem_dim
        
        # trainalbe
        self.out = nn.Linear(hidden_dim, problem_dim)
            
    def forward(self, state: Tensor) -> Tensor:
        raise NotImplementedError(
            "``forward`` is required to implemented in subclasses."
        )