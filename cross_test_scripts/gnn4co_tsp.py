import os
import sys
root_folder = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.gnn4co import *

# Problem Settings
TEST_TYPE = "Cluster-50"
SPARSE_FACTOR = -1 # 50 for TSP-500 and TSP-10K, 100 for TSP-1K

# Solving Settings
SOLVING_SETTINGS = ("random_mcts", None) # ("greedy", "2opt"), ("greedy", "mcts"), ("random_mcts", None), ("mcts", None)

# Local Search Settings
MCTS_TIME_LIMIT = 0.05 # 0.05 for TSP50 and TSP100; 1 for TSP500 and TSP1K; 5 for TSP10K
MCTS_TYPE_FLAG = 2 # 2 for TSP50 and TSP100; 1 for TSP500, TSP1K and TSP10K, 1 for random_mcts

# Test File & Pretrained Settings
TEST_FILE_DICT = {
    "Cluster-50": "cross_dataset/tsp/tsp50_cluster_3.730.txt",
    "Gaussian-50": "cross_dataset/tsp/tsp50_gaussian_23.840.txt",
    "Cluster-100": "cross_dataset/tsp/tsp100_cluster_5.526.txt",
    "Gaussian-100": "cross_dataset/tsp/tsp100_gaussian_34.031.txt",
}
WEIGHT_PATH_DICT = {
    "Cluster-50": "weights/gnn4co/gnn4co_tsp50_dense.pt",
    "Gaussian-50": "weights/gnn4co/gnn4co_tsp50_dense.pt",
    "Cluster-100": "weights/gnn4co/gnn4co_tsp100_dense.pt",
    "Gaussian-100": "weights/gnn4co/gnn4co_tsp100_dense.pt"
}

# Main
if __name__ == "__main__":
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
            weight_path=WEIGHT_PATH_DICT[TEST_TYPE]
        )
    )
    solver.from_txt(TEST_FILE_DICT[TEST_TYPE], ref=True, show_time=True, normalize=True)
    solver.solve(show_time=True)
    print(solver.evaluate(calculate_gap=True))