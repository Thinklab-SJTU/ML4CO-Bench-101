import os
import sys
import numpy as np
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.gnn4co import *

# Problem Settings
PROBLEMS = [
    ["eil51", 51],
    ["berlin52", 52],
    ["st70", 70],
    ["eil76", 76],
    ["pr76", 76],
    ["rat99", 99],
    ["kroA100", 100],
    ["kroB100", 100],
    ["kroC100", 100],
    ["kroD100", 100],
    ["kroE100", 100],
    ["rd100", 100],
    ["eil101", 101],
    ["lin105", 105],
    ["pr107", 107],
    ["pr124", 124],
    ["bier127", 127],
    ["ch130", 130],
    ["pr136", 136],
    ["pr144", 144],
    ["ch150", 150],
    ["kroA150", 150],
    ["kroB150", 150],
    ["pr152", 152],
    ["u159", 159],
    ["rat195", 195],
    ["d198", 198],
    ["kroA200", 200],
    ["kroB200", 200],
    ["ts225", 225],
    ["tsp225", 225],
    ["pr226", 226],
    ["gil262", 262],
    ["pr264", 264],
    ["a280", 280],
    ["pr299", 299],
    ["lin318", 318],
    ["rd400", 400],
    ["fl417", 417],
    ["pr439", 439],
    ["pcb442", 442],
    ["d493", 493],
    ["u574", 574],
    ["rat575", 575],
    ["p654", 654],
    ["d657", 657],
    ["u724", 724],
    ["rat783", 783],
    ["pr1002", 1002]
]

# Solving Settings
SOLVING_SETTINGS = ("greedy", "mcts")

# Local Search Settings
USE_2OPT = True

# Test File & Pretrained Settings
WEIGHT_PATH_DICT = {
    50: "weights/gnn4co/gnn4co_tsp50_dense.pt",
    100: "weights/gnn4co/gnn4co_tsp100_dense.pt",
    500: "weights/gnn4co/gnn4co_tsp500_sparse.pt",
    1000: "weights/gnn4co/gnn4co_tsp1k_sparse.pt",
}

# Main
if __name__ == "__main__":
    # init
    gaps = list()
    costs = list()
    
    # solve
    for problem in PROBLEMS:
        # nodes num and sparse factor
        name, nodes_num = problem
        if nodes_num <= 150:
            MCTS_TIME_LIMIT = 0.05
            NODES_NUM = 100
            SPARSE_FACTOR = -1
            MCTS_TYPE_FLAG = 2   
        elif nodes_num <= 600:
            MCTS_TIME_LIMIT = 1.0
            NODES_NUM = 500
            SPARSE_FACTOR = 50
            MCTS_TYPE_FLAG = 1
        else:
            MCTS_TIME_LIMIT = 1.0
            NODES_NUM = 1000
            SPARSE_FACTOR = 100
            MCTS_TYPE_FLAG = 1
        txt_file = f"test_dataset/tsp/tsplib/{name}.txt"
        
        # solve
        solver = GNN4COTSPSolver(
            model=GNN4COModel(
                env=GNN4COEnv(
                    task="TSP", sparse_factor=SPARSE_FACTOR, device="cuda"
                ),
                encoder=TSPGNNEncoder(sparse=SPARSE_FACTOR>0),
                decoder=TSPDecoder(
                    decoding_type=SOLVING_SETTINGS[0], 
                    local_search_type=SOLVING_SETTINGS[1],
                    mcts_time_limit=MCTS_TIME_LIMIT,
                    mcts_type_2opt=MCTS_TYPE_FLAG
                ),
                weight_path=WEIGHT_PATH_DICT[NODES_NUM]
            )
        )
        solver.from_txt(txt_file, ref=True, show_time=False)
        solver.solve(show_time=False)
        result = solver.evaluate(calculate_gap=True)
        print(f"{name}: ", result)
        costs.append(result[0])
        gaps.append(result[2])
    print(np.mean(np.array(costs)))
    print(np.mean(np.array(gaps)))