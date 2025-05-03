import torch
from torch import optim
from ml4co_kit import TSPSolver, iterative_execution, SOLVER_TYPE, Timer
from ml4co.dimes.model import DIMESModel
from ml4co.dimes.model.encoder.dense import meta_update_dense
from ml4co.dimes.model.encoder.sparse import meta_update_sparse, Graph


class DIMESTSPSolver(TSPSolver):
    def __init__(self, model: DIMESModel, seed: int = 1234):
        super(DIMESTSPSolver, self).__init__(solver_type=SOLVER_TYPE.ML4TSP)
        self.model: DIMESModel = model
        self.model.model.eval()
        self.model.env.mode = "solve"
        torch.manual_seed(seed=seed)
        
    def solve(
        self,
        batch_size: int = 1, 
        sampling_num: int = 1,
        AS_steps: int = 0, # set `AS_steps > 0` to enable active search
        AS_sampling_num: int = 500,
        AS_learning_rate: float = 0.1,
        show_time: bool = False
    ):
        # timer
        timer = Timer(apply=show_time)
        timer.start()
        
        # solve
        msg = f"Solving solutions using DIMESTSPSolver"
        samples_num = len(self.points)
        solutions_list = list()
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
                    _, x, e, edge_index, _, _, _, _ = data
                    graph = Graph(x=x, edge_index=edge_index, edge_attr=e.unsqueeze(-1))
                    emb0 = self.model.model.emb_net(graph.x, graph.edge_index, graph.edge_attr)
                tmp_opt = lambda par: optim.AdamW(par, lr=AS_learning_rate)
                emb1, psi_net, _, _ = meta_update_sparse(emb0, self.model.model.par_net, graph, \
                                                          tmp_opt, AS_steps, AS_sampling_num)
                with torch.no_grad():
                    psi_net.eval()
                    heatmap = psi_net(emb1.detach())
                    solutions = self.model.decoder.sparse_decode(heatmap, *data)
            else:
                self.model.model.set_batch_size(batch_size)
                with torch.no_grad():
                    _, x, adj, _, _ = data
                    heatmap = self.model.model(x, adj)
                tmp_opt = lambda par: optim.AdamW(par, lr=AS_learning_rate)
                heatmap = meta_update_dense(adj, heatmap, tmp_opt, steps=AS_steps, samples=AS_sampling_num)
                with torch.no_grad():
                    solutions = self.model.decoder.dense_decode(heatmap, *data)
                    
            # solution list
            solutions_list += solutions

        # timer
        timer.end()
        timer.show_time()
        
        # restore solution
        self.from_data(tours=solutions_list, ref=False)
        
        return self.tours