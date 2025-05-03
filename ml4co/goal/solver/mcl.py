import torch
import numpy as np
from ml4co_kit import MClSolver, iterative_execution, SOLVER_TYPE, Timer
from goal.model import GOALModel


class GOALMClSolver(MClSolver):
    def __init__(self, model: GOALModel, beam_size: int = -1, seed: int = 1234):
        super(GOALMClSolver, self).__init__(solver_type=SOLVER_TYPE.ML4MCl)
        self.model = model
        self.model.eval()
        self.model.env.mode = "solve"
        self.beam_size = beam_size
        torch.manual_seed(seed=seed)
        
    def solve(
        self, batch_size: int = 1, sampling_num: int = 1, show_time: bool = False
    ):
        # timer
        timer = Timer()
        timer.start()
        
        # solve
        msg = f"Solving solutions using GOALMClSolver"
        samples_num = len(self.graph_data)
        for idx in iterative_execution(range, samples_num // batch_size, msg, show_time):
            # begin index and end index
            begin_idx = idx * batch_size
            end_idx = begin_idx + batch_size
            
            # data processor
            data = self.model.env.data_processor.mcl_batch_data_process(
                graph_data=self.graph_data[begin_idx:end_idx], 
                sampling_num=sampling_num,
                mode=self.model.env.mode
            )
            
            # solve
            with torch.no_grad():
                solutions = self.model.decoder.decode(
                    *data, model=self.model.model, 
                    beam_size=self.beam_size, return_cost=False
                )

            # best solution
            for _idx in range(batch_size):
                current_solutions = solutions[_idx * sampling_num : (_idx+1) * sampling_num]
                sel_nodes_num = np.array([
                    np.nonzero(current_solutions[_] + 1)[0].shape[0] for _ in range(sampling_num)
                ])
                best_idx = np.argmax(sel_nodes_num)
                best_nodes_num = np.max(sel_nodes_num)
                best_solution = current_solutions[best_idx]
                rewrite_solution = np.zeros_like(best_solution)
                rewrite_solution[best_solution[:best_nodes_num]] = 1
                self.graph_data[_idx+begin_idx].nodes_label = rewrite_solution  
                
        # timer
        timer.end()
        timer.show_time()