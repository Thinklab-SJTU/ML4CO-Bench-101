from ml4co_kit import MISSolver
from ml4co_kit import (
    MISSolver, iterative_execution, SOLVER_TYPE, Timer, mis_rlsa_decoder
)


class MISRLSASolver(MISSolver):
    def __init__(
        self, 
        rlsa_tau: float = 0.01, 
        rlsa_d: int = 2, 
        rlsa_k: int = 200, 
        rlsa_t: int = 500, 
        rlsa_alpha: float = 0.2,
        rlsa_beta: float = 1.02,
        rlsa_device: str = "cuda", 
        rlsa_seed: int = 1234
    ):
        super(MISRLSASolver, self).__init__(solver_type=SOLVER_TYPE.ML4MIS)
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
        msg = f"Solving solutions using MISRLSASolver"
        samples_num = len(self.graph_data)
        solutions_list = list()
        for idx in iterative_execution(range, samples_num, msg, show_time):
            solutions_list.append(mis_rlsa_decoder(
                graph=self.graph_data[idx].to_matrix(),
                rlsa_tau=self.rlsa_tau,
                rlsa_d=self.rlsa_d,
                rlsa_k=self.rlsa_k,
                rlsa_t=self.rlsa_t,
                rlsa_alpha=self.rlsa_alpha,
                rlsa_beta=self.rlsa_beta,
                rlsa_device=self.rlsa_device,
                seed=self.rlsa_seed
            ))
            
        # timer
        timer.end()
        timer.show_time()
        
        # restore solution
        self.from_graph_data(nodes_label=solutions_list, ref=False, cover=False)
        
        return self.graph_data