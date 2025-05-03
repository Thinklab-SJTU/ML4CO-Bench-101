import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.gnn4co import *

# Problem Settings
TEST_TYPE = "RB-LARGE"

# Solving Settings
SOLVING_SETTINGS = ("greedy", None)

# Local Search Settings
RLSA_SETTINGS_DICT = {
    "RB-SMALL": (0.01, 5, 1000, 500, 1.02, 0.3),
    "RB-LARGE": (0.01, 5, 1000, 1000, 1.02, 0.15),
    "RB-GIANT": (0.01, 5, 1000, 1000, 1.02, 0.1),
    "ER-700-800": (0.2, 10, 1000, 1000, 1.001, 0.2),
    "ER-1400-1600": (0.2, 10, 1000, 1000, 1.001, 0.1),
    "SATLIB": (0.2, 5, 1000, 1000, 1.02, 0.3),
}

# Test File & Pretrained Settings
TEST_FILE_DICT = {
    "RB-SMALL": "test_dataset/meta_gnco/query_dataset/mis/mis_ws_small_query.txt",
    "RB-LARGE": "test_dataset/meta_gnco/query_dataset/mis/mis_ba_large_query.txt",
    "RB-GIANT": "test_dataset/mis/mis_rb-giant_kamis-60s_49.260.txt",
    "ER-700-800": "test_dataset/mis/mis_er-700-800_kamis-60s_44.969.txt",
    "ER-1400-1600": "test_dataset/mis/mis_er-1400-1600_kamis-60s_50.938.txt",
    "SATLIB": "test_dataset/mis/mis_satlib_kamis-60s_425.954.txt"
}
WEIGHT_PATH_DICT = {
    "RB-SMALL": "weights/gnn4co/gnn4co_mis_rb-small_sparse.pt",
    "RB-LARGE": "weights/gnn4co/gnn4co_mis_rb-large_sparse.pt",
    "RB-GIANT": "weights/gnn4co/gnn4co_mis_rb-large_sparse.pt",
    "ER-700-800": "weights/gnn4co/gnn4co_mis_er-700-800_sparse.pt",
    "ER-1400-1600": "weights/gnn4co/gnn4co_mis_er-700-800_sparse.pt",
    "SATLIB": "weights/gnn4co/gnn4co_mis_satlib_sparse.pt"
}

# Main
if __name__ == "__main__":
    rlsa_settings = RLSA_SETTINGS_DICT[TEST_TYPE]
    solver = GNN4COMISSolver(
        model=GNN4COModel(
            env=GNN4COEnv(
                task="MIS", mode="solve", sparse_factor=1, device="cuda",
            ),
            encoder=GNNEncoder(
                task="MIS",
                sparse=True,
                block_layers=[2, 4, 4, 2],
                hidden_dim=256
            ),
            decoder=MISDecoder(
                decoding_type=SOLVING_SETTINGS[0], 
                local_search_type=SOLVING_SETTINGS[1],
                rlsa_tau=rlsa_settings[0],
                rlsa_d=rlsa_settings[1],
                rlsa_k=rlsa_settings[2],
                rlsa_t=rlsa_settings[3],
                rlsa_beta=rlsa_settings[4],
                rlsa_alpha=rlsa_settings[5],
                rlsa_device="cuda"
            ),
            weight_path=WEIGHT_PATH_DICT[TEST_TYPE]
        )
    )
    solver.from_txt(TEST_FILE_DICT[TEST_TYPE], ref=True, show_time=True)
    solver.solve(show_time=True)
    print(solver.evaluate(calculate_gap=True))