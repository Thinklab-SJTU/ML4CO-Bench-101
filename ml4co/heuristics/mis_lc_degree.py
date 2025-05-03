from ml4co_kit import MISSolver
from ml4co_kit import (
    MISSolver, iterative_execution, SOLVER_TYPE, 
    Timer, mis_lc_degree_decoder, mis_rlsa_local_search
)


class MISLCDegreeSolver(MISSolver):
    def __init__(
        self, 
        use_rlsa: bool = False,
        rlsa_tau: float = 0.01, 
        rlsa_d: int = 2, 
        rlsa_k: int = 200, 
        rlsa_t: int = 500, 
        rlsa_alpha: float = 0.3,
        rlsa_beta: float = 1.02,
        rlsa_device: str = "cuda", 
        rlsa_seed: int = 1234
    ):
        super(MISLCDegreeSolver, self).__init__(solver_type=SOLVER_TYPE.ML4MIS)
        self.use_rlsa = use_rlsa
        self.rlsa_tau = rlsa_tau
        self.rlsa_d = rlsa_d
        self.rlsa_k = rlsa_k
        self.rlsa_t = rlsa_t
        self.rlsa_alpha = rlsa_alpha
        self.rlsa_beta = rlsa_beta
        self.rlsa_device = rlsa_device
        self.rlsa_seed = rlsa_seed
        
    def solve(self, show_time: bool = False):
        # timer
        timer = Timer()
        timer.start()
        
        # solve
        msg = f"Solving solutions using MISDegreeSolver"
        samples_num = len(self.graph_data)
        solutions_list = list()
        for idx in iterative_execution(range, samples_num, msg, show_time):
            graph = self.graph_data[idx].to_matrix()
            sol = mis_lc_degree_decoder(graph=graph)
            if self.use_rlsa:
                sol = mis_rlsa_local_search(
                    init_sol=sol,
                    graph=self.graph_data[idx].to_matrix(),
                    rlsa_tau=self.rlsa_tau,
                    rlsa_d=self.rlsa_d,
                    rlsa_k=self.rlsa_k,
                    rlsa_t=self.rlsa_t,
                    rlsa_alpha=self.rlsa_alpha,
                    rlsa_beta=self.rlsa_beta,
                    rlsa_device=self.rlsa_device,
                    seed=self.rlsa_seed
                )
            solutions_list.append(sol)
            
        # timer
        timer.end()
        timer.show_time()
        
        # restore solution
        self.from_graph_data(nodes_label=solutions_list, ref=False, cover=False)
        
        return self.graph_data