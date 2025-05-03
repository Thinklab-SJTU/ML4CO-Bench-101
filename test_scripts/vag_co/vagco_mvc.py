import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.vag_co import *

# Problem Settings
TEST_TYPE = "RB-SMALL"

# Solving Settings
SOLVING_SETTINGS = ("greedy", "rlsa")

# Local Search Settings
RLSA_SETTINGS_DICT = {
    "RB-SMALL": (0.01, 2, 1000, 1000, 1.02, 0.2),
    "RB-LARGE": (0.01, 2, 1000, 1000, 1.02, 0.2),
    "RB-GIANT": (0.01, 2, 1000, 1000, 1.02, 0.2),
    "TWITTER": (0.01, 2, 1000, 200, 4.0, 0.2),
    "COLLAB": (0.01, 2, 1000, 300, 1.02, 0.2)
}

# Test File & Pretrained Settings
TEST_FILE_DICT = {
    "RB-SMALL": "test_dataset/mvc/mvc_rb-small_gurobi-60s_205.764.txt",
    "RB-LARGE": "test_dataset/mvc/mvc_rb-large_gurobi-300s_968.228.txt",
    "RB-GIANT": "test_dataset/mvc/mvc_rb-giant_gurobi-3600s_2396.780.txt",
    "TWITTER": "test_dataset/mvc/mvc_twitter_gurobi-60s_85.251.txt",
    "COLLAB": "test_dataset/mvc/mvc_collab_gurobi-60s_65.086.txt"
}
WEIGHT_PATH_DICT = {
    "RB-SMALL": "weights/vag_co/vagco_mvc_rb-small_sparse.pt",
    "RB-LARGE": "weights/vag_co/vagco_mvc_rb-large_sparse.pt",
    "RB-GIANT": "weights/vag_co/vagco_mvc_rb-large_sparse.pt",
    "TWITTER": "weights/vag_co/vagco_mvc_rb-small_sparse.pt",
    "COLLAB": "weights/vag_co/vagco_mvc_rb-small_sparse.pt"
}

# Main
if __name__ == "__main__":
    rlsa_settings = RLSA_SETTINGS_DICT[TEST_TYPE]
    solver = VAGCOMVCSolver(
        model=VAGCOModel(
            env=VAGCOEnv(
                task="MVC", generator=None, mode="solve", device="cuda"
            ),
            encoder=GNNEncoder(
                task="MVC",
                sparse=True,
                block_layers=[1, 2, 2, 1],
                hidden_dim=256
            ),
            decoder=MVCDecoder(
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