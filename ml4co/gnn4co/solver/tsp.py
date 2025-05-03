import torch
from ml4co_kit import TSPSolver, iterative_execution, SOLVER_TYPE, Timer
from gnn4co.model import GNN4COModel


class GNN4COTSPSolver(TSPSolver):
    def __init__(self, model: GNN4COModel, seed: int = 1234):
        super(GNN4COTSPSolver, self).__init__(solver_type=SOLVER_TYPE.ML4TSP)
        self.model: GNN4COModel = model
        self.model.eval()
        self.model.env.mode = "solve"
        torch.manual_seed(seed=seed)
        
    def solve(
        self, batch_size: int = 1, sampling_num: int = 1, show_time: bool = False
    ):
        # timer
        timer = Timer(apply=show_time)
        timer.start()
        
        # solve
        msg = f"Solving solutions using GNN4COTSPSolver"
        samples_num = len(self.points)
        solutions_list = list()
        self._normalize_points()
        for idx in iterative_execution(range, samples_num // batch_size, msg, show_time):
            # begin index and end index
            begin_idx = idx * batch_size
            end_idx = begin_idx + batch_size
            
            # data processor
            data = self.model.env.data_processor.tsp_batch_data_process(
                points=self.points[begin_idx:end_idx], 
                ref_tours=self.ref_tours[begin_idx:end_idx], 
                sampling_num=sampling_num
            )

            # inference to get heatmap and decoding
            if self.model.env.sparse:
                with torch.no_grad():
                    heatmap = self.model.inference_edge_sparse_process(*data)
                    solutions = self.model.decoder.sparse_decode(heatmap, *data)
            else:
                with torch.no_grad():
                    heatmap = self.model.inference_edge_dense_process(*data)
                    solutions = self.model.decoder.dense_decode(heatmap, *data)
                    
            # solution list
            solutions_list += solutions

        # timer
        timer.end()
        timer.show_time()
        
        # restore solution
        self.from_data(tours=solutions_list, ref=False)
        
        return self.tours