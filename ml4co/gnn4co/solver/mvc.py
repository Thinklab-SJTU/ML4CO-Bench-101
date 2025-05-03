import torch
import numpy as np
from ml4co_kit import MVCSolver, iterative_execution, SOLVER_TYPE, Timer
from gnn4co.model import GNN4COModel


class GNN4COMVCSolver(MVCSolver):
    def __init__(self, model: GNN4COModel, seed: int = 1234):
        super(GNN4COMVCSolver, self).__init__(solver_type=SOLVER_TYPE.ML4MVC)
        self.model = model
        self.model.eval()
        self.model.env.mode = "solve"
        torch.manual_seed(seed=seed)
        
    def solve(
        self, batch_size: int = 1, sampling_num: int = 1, show_time: bool = False
    ):
        # timer
        timer = Timer()
        timer.start()
        
        # solve
        msg = f"Solving solutions using GNN4COMVCSolver"
        samples_num = len(self.graph_data)
        for idx in iterative_execution(range, samples_num // batch_size, msg, show_time):
            # begin index and end index
            begin_idx = idx * batch_size
            end_idx = begin_idx + batch_size
            
            # data process
            data = self.model.env.data_processor.mvc_batch_data_process(
                graph_data=self.graph_data[begin_idx:end_idx], sampling_num=sampling_num
            )
            
            # inference and decoding
            if self.model.env.sparse:
                with torch.no_grad():
                    vars = self.model.inference_node_sparse_process(*data)
                    solutions = self.model.decoder.sparse_decode(vars, *data)
            else:
                with torch.no_grad():
                    vars = self.model.inference_node_dense_process(*data)
                    solutions = self.model.decoder.dense_decode(vars, *data)

            # best solution
            for _idx in range(batch_size):
                current_solutions = solutions[_idx * sampling_num : (_idx+1) * sampling_num]
                sel_nodes_num_list = [(current_solutions[_]).sum() for _ in range(sampling_num)]
                best_idx = np.argmin(np.array(sel_nodes_num_list))
                self.graph_data[_idx+begin_idx].nodes_label = current_solutions[best_idx]

        # timer
        timer.end()
        timer.show_time()
        
        # return
        return self.graph_data