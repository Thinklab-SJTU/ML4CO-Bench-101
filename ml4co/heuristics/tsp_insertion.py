import numpy as np
from ml4co_kit import TSPSolver
from ml4co_kit import (
    TSPSolver, iterative_execution, SOLVER_TYPE, 
    tsp_insertion_decoder, tsp_2opt_local_search, Timer
)


class TSPInsertionSolver(TSPSolver):
    def __init__(
        self, 
        seed: int = 1234, 
        use_2opt: bool = False
    ):
        super(TSPInsertionSolver, self).__init__(solver_type=SOLVER_TYPE.ML4TSP)
        np.random.seed(seed=seed)
        self.use_2opt = use_2opt

    def solve(self, show_time: bool = False):
        # timer
        timer = Timer()
        timer.start()
        
        # solve
        msg = f"Solving solutions using TSPInsertionSolver"
        samples_num = len(self.points)
        solutions_list = list()
        for idx in iterative_execution(range, samples_num, msg, show_time):
            tour = tsp_insertion_decoder(self.points[idx])
            if self.use_2opt:
                tour = tsp_2opt_local_search(
                    init_tours=tour, points=self.points[idx]
                )
            solutions_list.append(tour)
                
        # timer
        timer.end()
        timer.show_time()
        
        # restore solution
        self.from_data(tours=solutions_list, ref=False)
        
        return self.tours