import numpy as np
from torch import Tensor
import scipy.special as ssp
from ml4co_kit import TSPEvaluator
from ml4co_kit import tsp_mcts_decoder
from ml4co_kit import to_numpy, tsp_2opt_local_search
from ml4co_kit import tsp_mcts_local_search, np_sparse_to_dense
from .base import DIMESDecoder
from ctypes import CDLL, c_int, c_double, byref
from dimes.model.encoder.sparse import tsp_greedy, tsp_softmax, Graph

DENSE_GREEDY_SOLVER = "ml4co/dimes/model/decoder/dense_greedy.so"

class TSPDecoder(DIMESDecoder):
    def __init__(
        self, 
        decoding_type: str = "greedy", 
        local_search_type: str = None,
        mcts_time_limit: float = 1.0,
        mcts_max_depth: int = 10,
        mcts_type_2opt: int = 2
    ) -> None:
        self.decoding_type = decoding_type
        self.local_search_type = local_search_type
        self.mcts_time_limit = mcts_time_limit
        self.mcts_max_depth = mcts_max_depth
        self.mcts_type_2opt = mcts_type_2opt
    
    def _edge_sparse_decode(
        self, heatmap: Tensor, x: Tensor, e: Tensor, edge_index: Tensor, nodes_num: int
    ) -> np.ndarray:
        # device
        device = heatmap.device
        
        # decoding
        if self.decoding_type == "greedy":
            _, sol = tsp_greedy(Graph(x, edge_index, e), heatmap, nodes_num)
            sol = to_numpy(sol)
            zero_index = sol.tolist().index(0)
            new_tour = np.concatenate([sol[zero_index:], sol[:zero_index]])
            new_tour = np.append(new_tour, 0)
            sol = new_tour

        if self.decoding_type == "sampling":
            tau = 0.01
            _, sol = tsp_softmax(Graph(x, edge_index, e), heatmap / tau, 4096)
            sol = to_numpy(sol)
            zero_index = sol.tolist().index(0)
            new_tour = np.concatenate([sol[zero_index:], sol[:zero_index]])
            new_tour = np.append(new_tour, 0)
            sol = new_tour

        elif self.decoding_type == "random_mcts":
            heatmap = to_numpy(heatmap)
            x = to_numpy(x)
            edge_index = to_numpy(edge_index)
            heatmap = np_sparse_to_dense(nodes_num, edge_index, edge_attr=heatmap)
            heatmap = make_for_mcts(heatmap)
            # get solutions
            sols = list()
            for _ in range(16):
                nodes_num = heatmap.shape[-1]
                tour = np.arange(1, nodes_num)
                np.random.shuffle(tour)
                tour = np.insert(tour, [0, len(tour)], [0, 0])
                sol = tsp_mcts_local_search(
                    init_tours=tour, heatmap=heatmap, points=x, time_limit=self.mcts_time_limit, 
                    max_depth=self.mcts_max_depth, type_2opt=self.mcts_type_2opt, continue_flag=2
                )
                sols.append(sol)
            sol = np.array(sols)
            
            # select best
            eval = TSPEvaluator(x)
            costs = [eval.evaluate(_sol) for _sol in sol]
            best_idx = np.argmin(np.array(costs))
            sol = sol[best_idx]
        
        # local search
        if self.local_search_type == "mcts":
            heatmap = to_numpy(heatmap)
            x = to_numpy(x)
            edge_index = to_numpy(edge_index)
            heatmap = np_sparse_to_dense(nodes_num, edge_index, edge_attr=heatmap)
            heatmap = make_for_mcts(heatmap)
            sol = tsp_mcts_local_search(
                init_tours=sol, heatmap=heatmap, points=x, time_limit=self.mcts_time_limit, 
                max_depth=self.mcts_max_depth, type_2opt=self.mcts_type_2opt, continue_flag=2
            )

        elif self.local_search_type == "2opt":
            sol = tsp_2opt_local_search(init_tours=sol, points=x, device=device)
        
        # return
        return sol

    def _edge_dense_decode(
        self, heatmap: Tensor, x: Tensor, graph: Tensor, 
    ) -> np.ndarray:
        # tensor -> numpy array
        device = heatmap.device
        x = to_numpy(x)
        heatmap = to_numpy(heatmap)
        
        # decoding
        if self.decoding_type == "greedy":
            sol = solve_nearest_neighbor(heatmap)
            zero_index = sol.index(0)
            new_tour = np.concatenate([sol[zero_index:], sol[:zero_index]])
            new_tour = np.append(new_tour, 0)
            sol = to_numpy(new_tour)

        elif self.decoding_type == "sampling" or self.decoding_type == "mcts":
            self.local_search_type = None # equal to ``sampling + mcts''
            heatmap = make_for_mcts(heatmap)
            sol = tsp_mcts_decoder(
                heatmap=heatmap, points=x, time_limit=self.mcts_time_limit, 
                max_depth=self.mcts_max_depth, type_2opt=self.mcts_type_2opt
            )

        elif self.decoding_type == "random_mcts":
            heatmap = make_for_mcts(heatmap)
            # get solutions
            sols = list()
            for _ in range(16):
                nodes_num = heatmap.shape[-1]
                tour = np.arange(1, nodes_num)
                np.random.shuffle(tour)
                tour = np.insert(tour, [0, len(tour)], [0, 0])
                sol = tsp_mcts_local_search(
                    init_tours=tour, heatmap=heatmap, points=x, time_limit=self.mcts_time_limit, 
                    max_depth=self.mcts_max_depth, type_2opt=self.mcts_type_2opt, continue_flag=2
                )
                sols.append(sol)
            sol = np.array(sols)
            
            # select best
            eval = TSPEvaluator(x)
            costs = [eval.evaluate(_sol) for _sol in sol]
            best_idx = np.argmin(np.array(costs))
            sol = sol[best_idx]
            
        # local search
        if self.local_search_type == "mcts":
            heatmap = make_for_mcts(heatmap)
            sol = tsp_mcts_local_search(
                init_tours=sol, heatmap=heatmap, points=x, time_limit=self.mcts_time_limit, 
                max_depth=self.mcts_max_depth, type_2opt=self.mcts_type_2opt, continue_flag=2
            )

        elif self.local_search_type == "2opt":
            sol = tsp_2opt_local_search(init_tours=sol, points=x, device=device)
        
        # return
        return sol

def make_for_mcts(heatmap: np.ndarray):
    sorted_vector = np.sort(heatmap, axis=-1)[:, -5].reshape(-1, 1)
    heatmap[(heatmap - sorted_vector) < 0] -= 1e9
    orig_matrix = heatmap
    start = 1.0
    minimum = 0.0
    while minimum < 1e-4: # adjust temperature
        heatmap = ssp.softmax(orig_matrix * start, axis=-1)
        minimum = heatmap[heatmap > 0].min()
        start *= 0.5
    return heatmap


def solve_nearest_neighbor(heatmap):
    """
    return: tour list

    Note that the returned loop is unclosed and the starting node does not have to be 0.
    """

    def _make_problem(dist):
        return (c_double * (n * n))(*dist.reshape(n * n).tolist()) 

    lib_tsp = CDLL(DENSE_GREEDY_SOLVER)
    n = heatmap.shape[-1]
    path = (c_int * n)(*(list(range(n))))
    cost = c_double(0)
    scaler = 1e6
    
    lib_tsp.nearest_neighbor(n, _make_problem(-heatmap * scaler), path, byref(cost))

    path = list(path)

    return path