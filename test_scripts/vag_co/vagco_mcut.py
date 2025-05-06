import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.vag_co import *

# Problem Settings
TEST_TYPE = "BA-GIANT"

# Solving Settings
SOLVING_SETTINGS = ("greedy", None) # ("greedy", "rlsa")

# Local Search Settings
RLSA_SETTINGS_DICT = {
    "BA-SMALL": (1, 20, 1000, 1000),
    "BA-LARGE": (1, 50, 1000, 1000),
    "BA-GIANT": (1, 100, 1000, 1000)
}

# Test File & Pretrained Settings
TEST_FILE_DICT = {
    "BA-SMALL": "test_dataset/mcut/mcut_ba-small_gurobi-60s_727.844.txt",
    "BA-LARGE": "test_dataset/mcut/mcut_ba-large_gurobi-300s_2936.886.txt",
    "BA-GIANT": "test_dataset/mcut/mcut_ba-giant_gurobi-3600s_7217.900.txt"
}
WEIGHT_PATH_DICT = {
    "BA-SMALL": "weights/vag_co/vagco_mcut_ba-small_sparse.pt",
    "BA-LARGE": "weights/vag_co/vagco_mcut_ba-large_sparse.pt",
    "BA-GIANT": "weights/vag_co/vagco_mcut_ba-large_sparse.pt"
}

# Main
if __name__ == "__main__":
    rlsa_settings = RLSA_SETTINGS_DICT[TEST_TYPE]
    solver = VAGCOMCutSolver(
        model=VAGCOModel(
            env=VAGCOEnv(
                task="MCut", generator=None, mode="solve", device="cuda"
            ),
            encoder=GNNEncoder(
                task="MCut",
                sparse=True,
                block_layers=[1, 2, 2, 1],
                hidden_dim=256
            ),
            decoder=MCutDecoder(
                decoding_type=SOLVING_SETTINGS[0], 
                local_search_type=SOLVING_SETTINGS[1],
                rlsa_tau=rlsa_settings[0],
                rlsa_d=rlsa_settings[1],
                rlsa_k=rlsa_settings[2],
                rlsa_t=rlsa_settings[3]
            ),
            weight_path=WEIGHT_PATH_DICT[TEST_TYPE]
        )
    )
    solver.from_txt(TEST_FILE_DICT[TEST_TYPE], ref=True, show_time=True)
    solver.solve(show_time=True)
    print(solver.evaluate(calculate_gap=True))