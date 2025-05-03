import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.meta_egn import *

# Problem Settings
TEST_TYPE = "RB-SMALL"

# Local Search Settings
USE_RLSA = False
RLSA_SETTINGS_DICT = {
    "RB-SMALL": (0.01, 2, 200, 50, 1.02, 0.3),
    "RB-LARGE": (0.01, 2, 200, 200, 1.02, 0.3),
    "RB-GIANT": (0.01, 2, 1000, 1000, 1.02, 0.3),
    "TWITTER": (0.01, 2, 200, 50, 4.0, 0.3),
    "COLLAB": (0.01, 2, 200, 20, 1.001, 0.3)
}

# Test File & Pretrained Settings
TEST_FILE_DICT = {
    "RB-SMALL": "test_dataset/mcl/mcl_rb-small_gurobi-60s_19.082.txt",
    "RB-LARGE": "test_dataset/mcl/mcl_rb-large_gurobi-300s_40.182.txt",
    "RB-GIANT": "test_dataset/mcl/mcl_rb-giant_gurobi-3600s_81.520.txt",
    "TWITTER": "test_dataset/mcl/mcl_twitter_gurobi-60s_14.210.txt",
    "COLLAB": "test_dataset/mcl/mcl_collab_gurobi-60s_42.113.txt"
}

WEIGHT_PATH_DICT = {
    "RB-SMALL": "weights/meta_egn/meta_egn_mcl_rb-small_sparse.pt",
    "TWITTER": "weights/meta_egn/meta_egn_mcl_rb-small_sparse.pt",
    "COLLAB": "weights/meta_egn/meta_egn_mcl_rb-small_sparse.pt",
    "RB-LARGE": "weights/meta_egn/meta_egn_mcl_rb-large_sparse.pt",
    "RB-GIANT": "weights/meta_egn/meta_egn_mcl_rb-large_sparse.pt",
}

# Main
if __name__ == "__main__":
    rlsa_settings = RLSA_SETTINGS_DICT[TEST_TYPE]
    processed_path = TEST_FILE_DICT[TEST_TYPE][:-4] + ".dat"
    solver = MetaEGNMClSolver(
        model=MetaEGNModel(
            env=MetaEGNEnv(task="MCl", mode="solve", device="cuda"),
            encoder=GNNEncoder(
                num_layers=4, hidden_dim_1=64, hidden_dim_2=1, devide="cuda"
            ),
            decoder=MetaEGNDecoder(
                decode_kwargs={
                    "use_rlsa": USE_RLSA,
                    "rlsa_tau": rlsa_settings[0],
                    "rlsa_d": rlsa_settings[1],
                    "rlsa_k": rlsa_settings[2],
                    "rlsa_t": rlsa_settings[3],
                    "rlsa_beta": rlsa_settings[4],
                    "rlsa_alpha": rlsa_settings[5],
                }
            ),
            weight_path=WEIGHT_PATH_DICT[TEST_TYPE],
        ),
        processed_path=processed_path
    )
    solver.from_txt(TEST_FILE_DICT[TEST_TYPE], ref=True, show_time=True)
    solver.solve(show_time=True)
    print(solver.evaluate(calculate_gap=True))