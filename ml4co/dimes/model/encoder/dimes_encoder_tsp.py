from .dense import DIMESDenseEncoder
from .sparse import DIMESSparseEncoder

def get_tsp_dimes_encoder(
    sparse: bool,
    num_nodes: int,
    num_layers: int = 3,
    hidden_dim: int = 64,
    device: str = "cuda"
):
    if sparse:
        return DIMESSparseEncoder(num_nodes, hidden_dim, device)
    else:
        return DIMESDenseEncoder(num_nodes, num_layers, hidden_dim, device)