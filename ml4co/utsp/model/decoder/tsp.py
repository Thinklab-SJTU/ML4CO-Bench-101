import numpy as np
from torch import Tensor
from typing import Union, List
from ml4co_kit import TSPEvaluator, TSPSolver
from ml4co_kit import (
    tsp_greedy_decoder, tsp_mcts_decoder, to_numpy, 
    tsp_2opt_local_search, tsp_mcts_local_search
)


class UTSPDecoder(object):
    def __init__(
        self, 
        decoding_type: str = "greedy", 
        local_search_type: str = None,
        mcts_time_limit: float = 1.0,
        mcts_max_depth: int = 10,
        mcts_type_2opt: int = 2
    ) -> None:
        super(UTSPDecoder, self).__init__()
        self.decoding_type = decoding_type
        self.local_search_type = local_search_type
        self.mcts_time_limit = mcts_time_limit
        self.mcts_max_depth = mcts_max_depth
        self.mcts_type_2opt = mcts_type_2opt
    
    def decode(
        self, heatmap: Tensor, x: Tensor, return_cost: bool = False
    ) -> Union[List[np.ndarray], np.floating]:
        solutions = list()
        for idx in range(heatmap.shape[0]):
            solutions.append(self._decode(heatmap=heatmap[idx], x=x[idx]))
        if return_cost:
            tsp_solver = TSPSolver()
            tsp_solver.from_data(
                points=to_numpy(x), tours=solutions
            )
            costs = tsp_solver.evaluate()
            return costs
        else:
            return solutions
        
    def _decode(self, heatmap: Tensor, x: Tensor) -> np.ndarray:
        # tensor -> numpy array
        device = heatmap.device
        x = to_numpy(x)
        heatmap = to_numpy(heatmap)
        heatmap = (heatmap + heatmap.T) / 2
        heatmap = np.clip(heatmap, a_min=1e-14, a_max=1-1e-14)
        
        # decoding
        if self.decoding_type == "greedy":
            sol = tsp_greedy_decoder(heatmap)
        elif self.decoding_type == "mcts":
            sol = tsp_mcts_decoder(
                heatmap=heatmap, points=x, time_limit=self.mcts_time_limit, 
                max_depth=self.mcts_max_depth, type_2opt=self.mcts_type_2opt
            )
        elif self.decoding_type == "random_mcts":
            # get solutions
            sols = list()
            for _ in range(16):
                nodes_num = heatmap.shape[-1]
                tour = np.arange(1, nodes_num)
                np.random.shuffle(tour)
                tour = np.insert(tour, [0, len(tour)], [0, 0])
                sol = tsp_mcts_local_search(
                    init_tours=tour, 
                    heatmap=heatmap, 
                    points=x, 
                    time_limit=self.mcts_time_limit, 
                    max_depth=self.mcts_max_depth, 
                    type_2opt=self.mcts_type_2opt, 
                    continue_flag=2
                )
                sols.append(sol)
            sol = np.array(sols)
            
            # select best
            eval = TSPEvaluator(x)
            costs = [eval.evaluate(_sol) for _sol in sol]
            best_idx = np.argmin(np.array(costs))
            sol = sol[best_idx]
        else:
            raise NotImplementedError()
        
        # local search
        if self.local_search_type == "mcts":
            sol = tsp_mcts_local_search(
                init_tours=sol, 
                heatmap=heatmap, 
                points=x, 
                time_limit=self.mcts_time_limit, 
                max_depth=self.mcts_max_depth, 
                type_2opt=self.mcts_type_2opt, 
                continue_flag=2
            )
        elif self.local_search_type == "2opt":
            sol = tsp_2opt_local_search(
                init_tours=sol, points=x, device=device
            )
        elif self.local_search_type is not None:
            raise NotImplementedError()
        
        # return
        return sol