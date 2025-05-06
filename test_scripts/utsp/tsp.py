import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.utsp import *


# Problem Settings
NODES_NUM = 50

# Solving Settings
SOLVING_SETTINGS = ("greedy", None) # ("greedy", "2opt"), ("greedy", "mcts"), ("random_mcts", None), ("mcts", None)

# Local Search Settings
MCTS_TIME_LIMIT = 0.05 # 0.05 for TSP50 and TSP100; 1 for TSP500 and TSP1K; 5 for TSP10K
MCTS_TYPE_FLAG = 2 # 2 for TSP50 and TSP100; 1 for TSP500, TSP1K and TSP10K, 1 for random_mcts

# Test File & Pretrained Settings
TEST_FILE_DICT = {
    50: "test_dataset/tsp/tsp50_concorde_5.688.txt",
    100: "test_dataset/tsp/tsp100_concorde_7.756.txt",
    500: "test_dataset/tsp/tsp500_concorde_16.546.txt",
    1000: "test_dataset/tsp/tsp1000_concorde_23.118.txt"
}
WEIGHT_PATH_DICT = {
    50: "weights/utsp/utsp50.pt",
    100: "weights/utsp/utsp100.pt",
    500: "weights/utsp/utsp500.pt",
    1000: "weights/utsp/utsp1k.pt"
}

# Main
if __name__ == "__main__":
    solver = UTSPSolver(
        model=UTSPModel(
            env=UTSPEnv(
                generator=None, mode="solve", device="cuda",
            ),
            encoder=SAGEncoder(output_channels=NODES_NUM),
            decoder=UTSPDecoder(
                decoding_type=SOLVING_SETTINGS[0], 
                local_search_type=SOLVING_SETTINGS[1],
                mcts_time_limit=MCTS_TIME_LIMIT,
                mcts_type_2opt=MCTS_TYPE_FLAG
            ),
            weight_path=WEIGHT_PATH_DICT[NODES_NUM]
        )
    )
    solver.from_txt(TEST_FILE_DICT[NODES_NUM], ref=True, show_time=True)
    solver.solve(show_time=True)
    print(solver.evaluate(calculate_gap=True))