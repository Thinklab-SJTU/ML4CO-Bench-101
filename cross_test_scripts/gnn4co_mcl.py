import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.gnn4co import *

# Problem Settings
TEST_TYPE = "HK-SMALL"

# Solving Settings
SOLVING_SETTINGS = ("beam", "rlsa") # ("greedy", "rlsa") / ("beam", None) / ("beam", "rlsa")

# Local Search Settings
RLSA_SETTINGS_DICT = {
    "HK-SMALL": (0.01, 2, 200, 50, 1.02, 0.3),
    "HK-LARGE": (0.01, 2, 200, 200, 1.02, 0.3),
    "WS-SMALL": (0.01, 2, 200, 50, 1.02, 0.3),
    "WS-LARGE": (0.01, 2, 200, 200, 1.02, 0.3),
}

# Test File & Pretrained Settings
TEST_FILE_DICT = {
    "HK-SMALL": "cross_dataset/mcl/mcl_hk-small_6.792.txt",
    "HK-LARGE": "cross_dataset/mcl/mcl_hk-large_6.774.txt",
    "WS-SMALL": "cross_dataset/mcl/mcl_ws-small_7.164.txt",
    "WS-LARGE": "cross_dataset/mcl/mcl_ws-large_5.978.txt",
}
WEIGHT_PATH_DICT = {
    "HK-SMALL": "weights/gnn4co/gnn4co_mcl_rb-small_sparse.pt",
    "HK-LARGE": "weights/gnn4co/gnn4co_mcl_rb-large_sparse.pt",
    "WS-SMALL": "weights/gnn4co/gnn4co_mcl_rb-small_sparse.pt",
    "WS-LARGE": "weights/gnn4co/gnn4co_mcl_rb-large_sparse.pt",
}

# Main
if __name__ == "__main__":
    rlsa_settings = RLSA_SETTINGS_DICT[TEST_TYPE]
    solver = GNN4COMClSolver(
        model=GNN4COModel(
            env=GNN4COEnv(
                task="MCl", mode="solve", sparse_factor=1, device="cuda",
            ),
            encoder=GNNEncoder(
                task="MCl",
                sparse=True,
                block_layers=[2, 4, 4, 2],
                hidden_dim=256
            ),
            decoder=MClDecoder(
                decoding_type=SOLVING_SETTINGS[0], 
                local_search_type=SOLVING_SETTINGS[1],
                rlsa_tau=rlsa_settings[0],
                rlsa_d=rlsa_settings[1],
                rlsa_k=rlsa_settings[2],
                rlsa_t=rlsa_settings[3],
                rlsa_beta=rlsa_settings[4],
                rlsa_alpha=rlsa_settings[5]
            ),
            weight_path=WEIGHT_PATH_DICT[TEST_TYPE]
        )
    )
    solver.from_txt(TEST_FILE_DICT[TEST_TYPE], ref=True, show_time=True)
    solver.solve(show_time=True)
    print(solver.evaluate(calculate_gap=True))