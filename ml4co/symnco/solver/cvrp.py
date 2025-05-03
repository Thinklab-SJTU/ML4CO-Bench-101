import torch
import numpy as np
from ml4co_kit import (
    CVRPSolver, iterative_execution, SOLVER_TYPE, cvrp_classic_local_search,
    Timer, iterative_execution_for_file
)
from symnco.model import SymNCOModel


class SymNCOCVRPSolver(CVRPSolver):
    def __init__(
        self, 
        model: SymNCOModel, 
        seed: int = 1234,
        use_ls: bool = False
    ):
        super(SymNCOCVRPSolver, self).__init__(solver_type=SOLVER_TYPE.ML4CVRP)
        self.model = model
        self.model.eval()
        self.model.env.mode = "solve"
        torch.manual_seed(seed=seed)
        self.use_ls = use_ls
        
    def solve(
        self, batch_size: int = 1, show_time: bool = False
    ):
        # timer
        timer = Timer()
        timer.start()
        
        # solve
        msg = f"Solving solutions using SymNCOCVRPSolver"
        samples_num = len(self.points)
        solutions_list = list()
        for idx in iterative_execution(range, samples_num // batch_size, msg, show_time):
            # begin index and end index
            begin_idx = idx * batch_size
            end_idx = begin_idx + batch_size
            
            # sparser
            td_data = self.model.env.process_data(
                batch_size=batch_size, 
                depots=self.depots[begin_idx:end_idx],
                points=self.points[begin_idx:end_idx],
                demands=self.demands[begin_idx:end_idx],
                capacities=self.capacities[begin_idx:end_idx],
            )

            # gain solutions
            with torch.no_grad():
                actions = self.model.inference_edge_process(td_data)
                solutions: np.ndarray = np.concatenate(
                    [np.zeros(shape=(batch_size, 1)), actions, np.zeros(shape=(batch_size, 1))], axis=1
                ).astype(np.int32)

            # solution list
            solutions_list += solutions.tolist()
        
        # local search
        if self.use_ls:
            ls_list = list()
            depots = self.depots
            points = self.points
            demands = self.demands / np.expand_dims(self.capacities, axis=1)
            for sol, depot, _points, _demands in iterative_execution_for_file(
                zip(solutions_list, depots, points, demands), "Local Search", show_time
            ):
                ls_tour = cvrp_classic_local_search(sol, depot, _points, _demands)
                ls_list.append(ls_tour)
            solutions_list = ls_list
        
        # timer
        timer.end()
        timer.show_time()
        
        # restore solution
        self.from_data(tours=solutions_list, ref=False)
        
        return self.tours