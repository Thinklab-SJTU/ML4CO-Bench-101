import torch
import numpy as np
from ml4co_kit import (
    TSPSolver, iterative_execution, SOLVER_TYPE, tsp_2opt_local_search, 
    Timer, iterative_execution_for_file
)
from symnco.model import SymNCOModel


class SymNCOTSPSolver(TSPSolver):
    def __init__(
        self, 
        model: SymNCOModel, 
        seed: int = 1234, 
        use_2opt: bool = False
    ):
        super(SymNCOTSPSolver, self).__init__(solver_type=SOLVER_TYPE.ML4TSP)
        self.model = model
        self.model.eval()
        self.model.env.mode = "solve"
        torch.manual_seed(seed=seed)
        self.use_2opt = use_2opt
        
    def solve(
        self, batch_size: int = 1, show_time: bool = False
    ):
        # timer
        timer = Timer()
        timer.start()
        
        # solve
        msg = f"Solving solutions using SymNCOTSPSolver"
        samples_num = len(self.points)
        solutions_list = list()
        for idx in iterative_execution(range, samples_num // batch_size, msg, show_time):
            # begin index and end index
            begin_idx = idx * batch_size
            end_idx = begin_idx + batch_size
            
            # sparser
            td_data = self.model.env.process_data(
                batch_size=batch_size, points=self.points[begin_idx:end_idx],
            )

            # gain solutions
            with torch.no_grad():
                actions = self.model.inference_edge_process(td_data)
                solutions = list()
                for tour in actions:
                    zero_index = tour.tolist().index(0)
                    new_tour = np.concatenate([tour[zero_index:], tour[:zero_index]])
                    new_tour = np.append(new_tour, 0)
                    solutions.append(new_tour)    
                    
            # solution list
            solutions_list += solutions
        
        # 2opt
        if self.use_2opt:
            solutions = np.array(solutions_list)
            solutions_list = list()
            for sol, x in iterative_execution_for_file(
                zip(solutions, self.points), "Local Search", show_time
            ):
                ls_tour = tsp_2opt_local_search(sol, x)
                solutions_list.append(ls_tour)
        
        # timer
        timer.end()
        timer.show_time()
        
        # restore solution
        self.from_data(tours=solutions_list, ref=False)
        
        return self.tours
