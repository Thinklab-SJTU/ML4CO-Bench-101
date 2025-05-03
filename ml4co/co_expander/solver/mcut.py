import torch
import numpy as np
from ml4co_kit import MCutSolver, iterative_execution, SOLVER_TYPE, Timer
from co_expander.model import COExpanderCMModel


class COExpanderMCutSolver(MCutSolver):
    def __init__(self, model: COExpanderCMModel, seed: int = 1234):
        super(COExpanderMCutSolver, self).__init__(solver_type=SOLVER_TYPE.ML4MCut)
        self.model = model
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
        msg = f"Solving solutions using COExpanderMCutSolver"
        samples_num = len(self.graph_data)
        for idx in iterative_execution(range, samples_num // batch_size, msg, show_time):
            # begin index and end index
            begin_idx = idx * batch_size
            end_idx = begin_idx + batch_size
            
            # data process
            data = self.model.env.data_processor.mcut_batch_data_process(
                graph_data=self.graph_data[begin_idx:end_idx], sampling_num=sampling_num
            )
            
            # gain determined variables
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
                current_edge_index = self.graph_data[_idx+begin_idx].edge_index
                energy_list = list()
                for s_idx in range(sampling_num):
                    src = current_solutions[s_idx][current_edge_index[0]] * 2 - 1
                    tgt = current_solutions[s_idx][current_edge_index[1]] * 2 - 1
                    energy = src * tgt
                    energy = (energy == -1).sum() / 2
                    energy_list.append(energy)
                best_idx = np.argmax(np.array(energy_list))
                self.graph_data[_idx+begin_idx].nodes_label = current_solutions[best_idx]

        # timer
        timer.end()
        timer.show_time()
        
        # return
        return self.graph_data