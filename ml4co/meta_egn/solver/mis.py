import torch
import numpy as np
import random
from ml4co_kit import MISSolver, iterative_execution, SOLVER_TYPE, Timer
from meta_egn.model import MetaEGNModel
from meta_egn.env import MetaEGNDataset


class MetaEGNMISSolver(MISSolver):
    def __init__(self, model: MetaEGNModel, seed: int = 1234, processed_path: str = None):
        super(MetaEGNMISSolver, self).__init__(solver_type=SOLVER_TYPE.ML4MIS)
        self.model: MetaEGNModel = model
        self.model.env.mode = "solve"
        self.processed_path = processed_path
        self.set_seed(seed=seed)

    def set_seed(self, seed: int = 1234):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        
    def solve(self, batch_size: int = 1, show_time: bool = False):
        # timer
        timer = Timer(apply=show_time)
        timer.start()        

        # solve
        msg = f"Solving solutions using MetaEGNMISSolver"
        samples_num = len(self.graph_data)

        pyg_data = MetaEGNDataset.process_solving(self.graph_data, save_path=self.processed_path)

        for idx in iterative_execution(range, samples_num // batch_size, msg, show_time):            
            # data processor
            pyg_graph = pyg_data[idx]

            # inference to get heatmap and decoding
            solution = self.model.model_inference(pyg_graph)
        
            # restore solution
            self.graph_data[idx].nodes_label = solution

        # timer
        timer.end()
        timer.show_time()
        
        # return
        return self.graph_data