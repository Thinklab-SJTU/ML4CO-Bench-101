import numpy as np
from torch import Tensor
from ml4co_kit import (
    to_numpy, mcl_greedy_decoder, 
    mcl_beam_decoder, mcl_rlsa_local_search
)
from gnn4co.model.decoder.base import GNN4CODecoder


class MClDecoder(GNN4CODecoder):
    def __init__(
        self,
        decoding_type: str = "greedy", 
        local_search_type: str = None,
        beam_size = 16,
        rlsa_tau: float = 0.01,
        rlsa_d: int = 5,        
        rlsa_k: int = 1000,
        rlsa_t: int = 1000,
        rlsa_beta: float = 1.02,
        rlsa_alpha: float = 0.3,
        rlsa_device: str = "cuda",
        rlsa_seed: int = 1234
    ) -> None:
        super(MClDecoder, self).__init__()
        
        # basic
        self.decoding_type = decoding_type
        self.local_search_type = local_search_type

        # beam
        self.beam_size = beam_size
        
        # rlsa
        self.rlsa_tau = rlsa_tau
        self.rlsa_d = rlsa_d
        self.rlsa_k = rlsa_k
        self.rlsa_t = rlsa_t
        self.rlsa_beta = rlsa_beta
        self.rlsa_alpha = rlsa_alpha
        self.rlsa_device = rlsa_device
        self.rlsa_seed = rlsa_seed
        
    def _node_sparse_decode(
        self, heatmap: Tensor, graph: Tensor
    ) -> np.ndarray:
        # tensor -> numpy array
        heatmap = to_numpy(heatmap)
        np_graph = to_numpy(graph)
        
        # decoding
        if self.decoding_type == "greedy":
            sol = mcl_greedy_decoder(
                heatmap=heatmap, graph=np_graph
            )
        elif self.decoding_type == "beam":
            sol = mcl_beam_decoder(
                heatmap=heatmap, graph=np_graph, beam_size=self.beam_size
            )
        else:
            raise NotImplementedError()

        # local search
        if self.local_search_type == "rlsa":
            sol = mcl_rlsa_local_search(
                init_sol=sol,
                graph=np_graph,
                rlsa_kth_dim="both",
                rlsa_tau=self.rlsa_tau,
                rlsa_d=self.rlsa_d,
                rlsa_k=self.rlsa_k,
                rlsa_t=self.rlsa_t,
                rlsa_alpha=self.rlsa_alpha,
                rlsa_beta=self.rlsa_beta,
                rlsa_device=self.rlsa_device,
                seed=self.rlsa_seed
            )
        elif self.local_search_type is not None:
            raise NotImplementedError()
        
        return sol.astype(np.int32)