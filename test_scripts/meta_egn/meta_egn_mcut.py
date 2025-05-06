import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.meta_egn import *

# basic settings
TEST_TYPE = "BA-SMALL"

# Local Search Settings
USE_RLSA = False
RLSA_SETTINGS_DICT = {
    "BA-SMALL": (1, 20, 1000, 1000, 0.3),
    "BA-LARGE": (1, 50, 1000, 1000, 0.3),
    "BA-GIANT": (1, 100, 1000, 1000, 0.3)
}

# Test File & Pretrained Settings
TEST_FILE_DICT = {
    "BA-SMALL": "test_dataset/mcut/mcut_ba-small_gurobi-60s_727.844.txt",
    "BA-LARGE": "test_dataset/mcut/mcut_ba-large_gurobi-300s_2936.886.txt",
    "BA-GIANT": "test_dataset/mcut/mcut_ba-giant_gurobi-3600s_7217.900.txt"
}
WEIGHT_PATH_DICT = {
    "BA-SMALL": "weights/meta_egn/meta_egn_mcut_ba-small_sparse.pt",
    "BA-LARGE": "weights/meta_egn/meta_egn_mcut_ba-large_sparse.pt",
    "BA-GIANT": "weights/meta_egn/meta_egn_mcut_ba-large_sparse.pt",
}

# Main
if __name__ == "__main__":
    rlsa_settings = RLSA_SETTINGS_DICT[TEST_TYPE]
    processed_path = TEST_FILE_DICT[TEST_TYPE][:-4] + ".dat"
    print(f"Testing on {TEST_TYPE} dataset... USE_RLSA = {USE_RLSA}")
    solver = MetaEGNMCutSolver(
        model=MetaEGNModel(
            env=MetaEGNEnv(task="MCut", mode="solve", device="cuda"),
            encoder=GNNEncoder(
                num_layers=4, hidden_dim_1=64, hidden_dim_2=1, devide="cuda"
            ),
            decoder=MetaEGNDecoder(
                decode_kwargs={
                    "use_rlsa": USE_RLSA,
                    "rlsa_tau": rlsa_settings[0],
                    "rlsa_d": rlsa_settings[1],
                    "rlsa_k": rlsa_settings[2],
                    "rlsa_t": rlsa_settings[3]
                }
            ),
            weight_path=WEIGHT_PATH_DICT[TEST_TYPE],
        ),
        processed_path=processed_path
    )
    solver.from_txt(TEST_FILE_DICT[TEST_TYPE], ref=True, show_time=True)
    solver.solve(show_time=True)
    print(solver.evaluate(calculate_gap=True))