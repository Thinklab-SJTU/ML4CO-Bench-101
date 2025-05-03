import torch
import numpy as np
import torch.utils.data
from torch import Tensor
from typing import Sequence
from ml4co_kit import check_dim, to_tensor


def atsp_data_process(
    dists: np.ndarray, ref_tour: np.ndarray, mode: str
) -> Sequence[Tensor]:
    # check dimension
    check_dim(dists, 2)
    check_dim(ref_tour, 1)
    
    # edges feature
    if mode != "solve":
        order_dists = dists[ref_tour, :][:, ref_tour]
        e = to_tensor(order_dists).float()
        e = torch.cat([e.unsqueeze(dim=-1), e.T.unsqueeze(dim=-1)], dim=-1)
        raw_data = order_dists
    else:
        raw_data = dists
        e = to_tensor(dists).float()
        e = torch.cat([e, e[0].unsqueeze(0)], dim=0)
        e = torch.cat([e, e[:, 0].unsqueeze(1)], dim=1)
        e = torch.cat([e.unsqueeze(dim=-1), e.T.unsqueeze(dim=-1)], dim=-1)
    
    return (
        None, # nodes feature, ATSP does not use it
        e.float(), # (V, V, 2): edges feature, distance matrix
        raw_data
    )