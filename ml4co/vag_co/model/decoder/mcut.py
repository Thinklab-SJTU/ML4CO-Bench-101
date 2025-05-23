import numpy as np
from torch import Tensor
from ml4co_kit import to_numpy, mcut_rlsa_local_search
from vag_co.model.decoder.base import VAGCODecoder


class MCutDecoder(VAGCODecoder):
    def __init__(
        self, 
        decoding_type: str = "greedy", 
        local_search_type: str = None,
        rlsa_tau: float = 0.01,
        rlsa_d: int = 5,        
        rlsa_k: int = 1000,
        rlsa_t: int = 1000,
        rlsa_device: str = "cuda",
        rlsa_seed: int = 1234
    ) -> None:
        super(MCutDecoder, self).__init__()
        
        # basic
        self.decoding_type = decoding_type
        self.local_search_type = local_search_type

        # rlsa
        self.rlsa_tau = rlsa_tau
        self.rlsa_d = rlsa_d
        self.rlsa_k = rlsa_k
        self.rlsa_t = rlsa_t
        self.rlsa_device = rlsa_device
        self.rlsa_seed = rlsa_seed
        
    def _decode(
        self, heatmap: Tensor, graph: Tensor, edge_index: Tensor
    ) -> np.ndarray:
        # tensor -> numpy array
        heatmap = to_numpy(heatmap)
        
        # decoding   
        if self.decoding_type == "greedy":
            sol: np.ndarray = (heatmap > 0.5)
            sol = sol.astype(np.int32)
        else:
            raise NotImplementedError()

        # local search
        if self.local_search_type == "rlsa":
            sol = mcut_rlsa_local_search(
                init_sol=sol,
                graph=to_numpy(graph),
                edge_index=to_numpy(edge_index),
                rlsa_kth_dim="both",
                rlsa_tau=self.rlsa_tau,
                rlsa_d=self.rlsa_d,
                rlsa_k=self.rlsa_k,
                rlsa_t=self.rlsa_t,
                rlsa_device=self.rlsa_device,
                seed=self.rlsa_seed
            )
        elif self.local_search_type is not None:
            raise NotImplementedError()
        
        return sol.astype(np.int32)
