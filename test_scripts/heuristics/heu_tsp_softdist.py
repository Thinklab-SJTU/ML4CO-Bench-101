import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.heuristics.tsp_softdist import TSPSoftDistSolver

# settings
NODES_NUM = 500
TAU = 0.0066 # 0.0066 for TSP500; 0.0051 for TSP1K; 0.0018 for TSP10K
MCTS_TIME_LIMIT = 1 # 0.05 for TSP50 and TSP100; 1 for TSP500 and TSP1K; 5 for TSP10K
TEST_FILE_DICT = {
    500: "test_dataset/tsp/tsp500_concorde_16.546.txt",
    1000: "test_dataset/tsp/tsp1000_concorde_23.118.txt",
    10000: "test_dataset/tsp/tsp10000_lkh_500_71.755.txt"
}

# main
if __name__ == "__main__":
    solver = TSPSoftDistSolver(time_limit=MCTS_TIME_LIMIT, tau=TAU)
    solver.from_txt(TEST_FILE_DICT[NODES_NUM], ref=True, show_time=True)
    solver.solve(show_time=True)
    print(solver.evaluate(calculate_gap=True))