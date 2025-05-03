import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.co_expander import *

# Problem Settings
TEST_TYPE = "RB-SMALL"

# COExpander Settings
INFERENCE_STEP = 20 # Is
DETERMINATE_STEP = 1 # Ds
SAMPLING_NUM = 1 # S
BEAM_SIZE = -1 # Beam Search

# Local Search Settings
USE_RLSA = True
RLSA_SETTINGS_DICT = {
    "RB-SMALL": (0.01, 2, 200, 50, 1.02, 0.3),
    "RB-LARGE": (0.01, 2, 200, 200, 1.02, 0.3),
    "RB-GIANT": (0.01, 2, 1000, 1000, 1.02, 0.3),
    "TWITTER": (0.01, 2, 200, 50, 4.0, 0.3),
    "COLLAB": (0.01, 2, 200, 20, 1.001, 0.3)
}

# test files and prtrained files
TEST_FILE_DICT = {
    "RB-SMALL": "test_dataset/mcl/mcl_rb-small_gurobi-60s_19.082.txt",
    "RB-LARGE": "test_dataset/mcl/mcl_rb-large_gurobi-300s_40.182.txt",
    "RB-GIANT": "test_dataset/mcl/mcl_rb-giant_gurobi-3600s_81.520.txt",
    "TWITTER": "test_dataset/mcl/mcl_twitter_gurobi-60s_14.210.txt",
    "COLLAB": "test_dataset/mcl/mcl_collab_gurobi-60s_42.113.txt"
}
WEIGHT_PATH_DICT = {
    "RB-SMALL": "weights/co_expander/coexpander_mcl_rb-small_sparse.pt",
    "RB-LARGE": "weights/co_expander/coexpander_mcl_rb-large_sparse.pt",
    "RB-GIANT": "weights/co_expander/coexpander_mcl_rb-large_sparse.pt",
    "TWITTER": "weights/co_expander/coexpander_mcl_rb-small_sparse.pt",
    "COLLAB": "weights/co_expander/coexpander_mcl_rb-small_sparse.pt"
}


# Main
if __name__ == "__main__":
    rlsa_settings = RLSA_SETTINGS_DICT[TEST_TYPE]
    solver = COExpanderMClSolver(
        model=COExpanderCMModel(
            env=COExpanderEnv(
                task="MCl", mode="solve", sparse_factor=1, device="cuda",
            ),
            encoder=GNNEncoder(
                task="MCl",
                sparse=True,
                block_layers=[2, 4, 4, 2],
                hidden_dim=256
            ),
            decoder=COExpanderDecoder(
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
            inference_steps=INFERENCE_STEP,
            determinate_steps=DETERMINATE_STEP,
            beam_size=BEAM_SIZE,
        )
    )
    solver.from_txt(TEST_FILE_DICT[TEST_TYPE], ref=True, show_time=True)
    solver.solve(sampling_num=SAMPLING_NUM, show_time=True)
    print(solver.evaluate(calculate_gap=True))