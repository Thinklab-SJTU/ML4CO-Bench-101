import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.gnn4co import *

# Problem Settings
NODES_NUM = 50

# Solving Settings
SOLVING_SETTINGS = ("greedy", "2opt")

# Test File & Pretrained Settings
TEST_FILE_DICT = {
    50: "test_dataset/atsp/atsp50_uniform_lkh_1000_1.5545.txt",
    100: "test_dataset/atsp/atsp100_uniform_lkh_1000_1.5660.txt",
    200: "test_dataset/atsp/atsp200_uniform_lkh_1000_1.5647.txt",
    500: "test_dataset/atsp/atsp500_uniform_lkh_1000_1.5734.txt"
}
WEIGHT_PATH_DICT = {
    50: "weights/gnn4co/gnn4co_atsp50_dense.pt",
    100: "weights/gnn4co/gnn4co_atsp100_dense.pt",
    200: "weights/gnn4co/gnn4co_atsp200_dense.pt",
    500: "weights/gnn4co/gnn4co_atsp500_dense.pt"
}

# Main
if __name__ == "__main__":
    solver = GNN4COATSPSolver(
        model=GNN4COModel(
            env=GNN4COEnv(
                task="ATSP", mode="solve", sparse_factor=-1, device="cuda",
            ),
            encoder=GNNEncoder(
                task="ATSP",
                sparse=False,
                block_layers=[2, 4, 4, 2],
                hidden_dim=256
            ),
            decoder=ATSPDecoder(
                decoding_type=SOLVING_SETTINGS[0], 
                local_search_type=SOLVING_SETTINGS[1],    
            ),
            weight_path=WEIGHT_PATH_DICT[NODES_NUM]
        )
    )
    solver.from_txt(TEST_FILE_DICT[NODES_NUM], ref=True, show_time=True)
    solver.solve(batch_size=1, show_time=True)
    print(solver.evaluate(calculate_gap=True))