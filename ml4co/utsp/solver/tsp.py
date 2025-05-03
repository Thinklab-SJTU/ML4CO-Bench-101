import torch
from ml4co_kit import TSPSolver, iterative_execution, SOLVER_TYPE, Timer
from utsp.model import UTSPModel


class UTSPSolver(TSPSolver):
    def __init__(self, model: UTSPModel, seed: int = 1234):
        super(UTSPSolver, self).__init__(solver_type=SOLVER_TYPE.ML4TSP)
        self.model: UTSPModel = model
        self.model.eval()
        self.model.env.mode = "solve"
        torch.manual_seed(seed=seed)
        
    def solve(self, batch_size: int = 1, show_time: bool = False):
        # timer
        timer = Timer(apply=show_time)
        timer.start()
        
        # solve
        msg = f"Solving solutions using UTSPTSPSolver"
        samples_num = len(self.points)
        solutions_list = list()
        for idx in iterative_execution(range, samples_num // batch_size, msg, show_time):
            # begin index and end index
            begin_idx = idx * batch_size
            end_idx = begin_idx + batch_size
            
            # data processor
            x, graph = self.model.env.tsp_batch_data_process(
                points=self.points[begin_idx:end_idx], 
            )

            # inference to get heatmap and decoding
            with torch.no_grad():
                heatmap = self.model.inference_process(x=x, graph=graph)
                solutions = self.model.decoder.decode(heatmap=heatmap, x=x)
                    
            # solution list
            solutions_list += solutions

        # timer
        timer.end()
        timer.show_time()
        
        # restore solution
        self.from_data(tours=solutions_list, ref=False)
        
        return self.tours