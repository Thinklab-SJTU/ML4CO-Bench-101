import torch
import numpy as np
from ml4co_kit import (
    ATSPSolver, iterative_execution, SOLVER_TYPE, Timer, atsp_2opt_local_search
)
from matnet.model import MatNetATSPModel


class MatNetATSPSolver(ATSPSolver):
    def __init__(
        self, 
        model: MatNetATSPModel, 
        seed: int = 1234,
        beam_size: int = -1,
        use_2opt: bool = False
    ):
        super(MatNetATSPSolver, self).__init__(solver_type=SOLVER_TYPE.ML4ATSP)
        self.model = model
        self.model.eval()
        torch.manual_seed(seed=seed)
        self.beam_size = beam_size
        self.use_2opt = use_2opt
        
    def solve(
        self,
        batch_size: int = 1, 
        show_time: bool = False
    ):
        # timer
        timer = Timer()
        timer.start()
        
        # solve
        msg = f"Solving solutions using MatNetATSPSolver"
        samples_num = len(self.dists)
        solutions_list = list()
        for idx in iterative_execution(range, samples_num // batch_size, msg, show_time):
            # begin index and end index
            begin_idx = idx * batch_size
            end_idx = begin_idx + batch_size
            
            # solve
            with torch.no_grad():
                solutions = self.model.solve(
                    dists=self.dists[begin_idx:end_idx],
                    beam_size=self.beam_size
                )
            
            # get real solution
            real_solutions = list()
            for tour in solutions:
                tour: np.ndarray
                zero_index = tour.tolist().index(0)
                new_tour = np.concatenate([tour[zero_index:], tour[:zero_index]])
                new_tour = np.append(new_tour, 0)
                real_solutions.append(new_tour)    
            
            # solution list
            solutions_list += real_solutions

        # numpy
        solutions = np.array(solutions_list)
        
        # 2opt
        if self.use_2opt:
            ls_sol_list = list()
            for tour, dists in zip(solutions, self.dists):
                ls_sol_list.append(
                    atsp_2opt_local_search(tour, dists)
                )
            solutions = np.array(ls_sol_list)
            
        # timer
        timer.end()
        timer.show_time()
          
        # restore solution
        self.from_data(tours=solutions, ref=False)
        
        return self.tours