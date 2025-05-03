from ml4co_kit import MCutSolver
from ml4co_kit import (
    MCutSolver, iterative_execution, SOLVER_TYPE, Timer, mcut_rlsa_decoder
)


class MCutRLSASolver(MCutSolver):
    def __init__(
        self, 
        rlsa_tau: float = 1.5, 
        rlsa_d: int = 20, 
        rlsa_k: int = 200, 
        rlsa_t: int = 500, 
        rlsa_device: str = "cuda", 
        rlsa_seed: int = 1234
    ):
        super(MCutRLSASolver, self).__init__(solver_type=SOLVER_TYPE.ML4MCut)
        self.rlsa_tau = rlsa_tau
        self.rlsa_d = rlsa_d
        self.rlsa_k = rlsa_k
        self.rlsa_t = rlsa_t
        self.rlsa_device = rlsa_device
        self.rlsa_seed = rlsa_seed
        
    def solve(self, show_time: bool = False):
        # timer
        timer = Timer()
        timer.start()
        
        # solve
        msg = f"Solving solutions using MCutRLSASolver"
        samples_num = len(self.graph_data)
        solutions_list = list()
        for idx in iterative_execution(range, samples_num, msg, show_time):
            solutions_list.append(
                mcut_rlsa_decoder(
                    graph=self.graph_data[idx].to_matrix(),
                    rlsa_tau=self.rlsa_tau,
                    rlsa_d=self.rlsa_d,
                    rlsa_k=self.rlsa_k,
                    rlsa_t=self.rlsa_t,
                    rlsa_device=self.rlsa_device,
                    seed=self.rlsa_seed
                )
            )
            
        # timer
        timer.end()
        timer.show_time()
        
        # restore solution
        self.from_graph_data(nodes_label=solutions_list, ref=False, cover=False)
        
        return self.graph_data