import torch
import numpy as np
import torch.nn.functional as F
from ml4co_kit import TSPSolver
from ml4co_kit import (
    TSPSolver, iterative_execution, SOLVER_TYPE, 
    tsp_mcts_decoder, to_tensor, Timer, to_numpy
)


class TSPSoftDistSolver(TSPSolver):
    def __init__(
        self, tau: float, time_limit: float = 0.05, 
        seed: int = 1234, device: str = "cuda", 
    ):
        super(TSPSoftDistSolver, self).__init__(solver_type=SOLVER_TYPE.ML4TSP)
        np.random.seed(seed=seed)
        self.tau = tau
        self.time_limit = time_limit
        self.device = device

    def solve(self, show_time: bool = False):
        # timer
        timer = Timer()
        timer.start()
        
        # softdist
        batch_coords = to_tensor(self.points).to(self.device).float()
        coord_diff = batch_coords[:, :, None, :] - batch_coords[:, None, :, :]
        distance_matrix = torch.sqrt(torch.sum(coord_diff ** 2, dim=-1))
        eye = torch.eye(distance_matrix.size(1), device=self.device).unsqueeze(0)
        distance_matrix = torch.where(
            eye == 1, torch.tensor(float('inf'), dtype=torch.float, device=self.device), distance_matrix
        )
        batch_heatmap = F.softmax(- distance_matrix / self.tau, dim=2)
        batch_heatmap = to_numpy(batch_heatmap)
        
        # solve
        msg = f"Solving solutions using TSPSoftDistSolver"
        samples_num = len(self.points)
        solutions_list = list()
        for idx in iterative_execution(range, samples_num, msg, show_time):
            tour = tsp_mcts_decoder(
                heatmap=batch_heatmap[idx], points=self.points[idx], time_limit=self.time_limit
            )
            solutions_list.append(tour)
                
        # timer
        timer.end()
        timer.show_time()
        
        # restore solution
        self.from_data(tours=solutions_list, ref=False)
        
        return self.tours
    