import numpy as np
from torch import Tensor
from ml4co_kit import to_numpy, atsp_greedy_decoder, atsp_2opt_local_search
from gnn4co.model.decoder.base import GNN4CODecoder


class ATSPDecoder(GNN4CODecoder):
    def __init__(
        self,
        decoding_type: str = "greedy", 
        local_search_type: str = None
    ) -> None:
        super(ATSPDecoder, self).__init__()
        self.decoding_type = decoding_type
        self.local_search_type = local_search_type
        
    def _edge_dense_decode(
        self, heatmap: Tensor, x: Tensor, graph: Tensor
    ) -> np.ndarray:
        # tensor -> numpy array
        heatmap = to_numpy(heatmap)
        
        # decoding
        if self.decoding_type == "greedy":
            sol = atsp_greedy_decoder(-heatmap)
        else:
            raise NotImplementedError()
        
        # local search
        if self.local_search_type == "2opt":
            sol = atsp_2opt_local_search(
                init_tours=sol, dists=to_numpy(graph)
            )
        elif self.local_search_type is not None:
            raise NotImplementedError()
        
        # return
        return sol 