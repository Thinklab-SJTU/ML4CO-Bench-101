import torch
import numpy as np
from ml4co_kit import CVRPSolver, iterative_execution, SOLVER_TYPE, Timer
from co_expander.model import COExpanderCMModel


class COExpanderCVRPSolver(CVRPSolver):
    def __init__(self, model: COExpanderCMModel, seed: int = 1234):
        super(COExpanderCVRPSolver, self).__init__(solver_type=SOLVER_TYPE.ML4CVRP)
        self.model = model
        self.model.eval()
        self.model.env.mode = "solve"
        torch.manual_seed(seed=seed)
        
    def solve(
        self,
        batch_size: int = 1, 
        sampling_num: int = 1,
        show_time: bool = False
    ):
        # timer
        timer = Timer()
        timer.start()
        
        # solve
        msg = f"Solving solutions using COExpanderCVRPSolver"
        samples_num = len(self.points)
        solutions_list = list()
        demands = self.demands / np.expand_dims(self.capacities, axis=1)
        for idx in iterative_execution(range, samples_num // batch_size, msg, show_time):
            # begin index and end index
            begin_idx = idx * batch_size
            end_idx = begin_idx + batch_size
            
            # data process
            data = self.model.env.data_processor.cvrp_batch_data_process(
                depots=self.depots[begin_idx:end_idx],
                points=self.points[begin_idx:end_idx], 
                demands=demands[begin_idx:end_idx],
                ref_tours=self.ref_tours[begin_idx:end_idx], 
                sampling_num=sampling_num
            )

            # gain determined variables
            if self.model.env.sparse:
                with torch.no_grad():
                    vars = self.model.inference_edge_sparse_process(*data)
                    solutions = self.model.decoder.sparse_decode(vars, *data)
            else:
                with torch.no_grad():
                    vars = self.model.inference_edge_dense_process(*data)
                    solutions = self.model.decoder.dense_decode(vars, *data)
                    
            # solution list
            solutions_list += solutions

        # timer
        timer.end()
        timer.show_time()
        
        # restore solution
        self.from_data(tours=solutions_list, ref=False)
        
        return self.tours