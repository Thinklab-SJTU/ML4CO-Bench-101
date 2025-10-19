import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.co_expander import *

# Problem Settings
TEST_TYPE = "BA-SMALL"

# COExpander Settings
INFERENCE_STEP = 1 # Is
DETERMINATE_STEP = 20 # Ds
SAMPLING_NUM = 1 # S
BEAM_SIZE = -1 # Beam Search
FINETUNE = True

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
    True: {
        "BA-SMALL": "weights/co_expander/coexpander_mcut_ba-small_sparse_finetune.pt",
        "BA-LARGE": "weights/co_expander/coexpander_mcut_ba-large_sparse_finetune.pt",
        "BA-GIANT": "weights/co_expander/coexpander_mcut_ba-large_sparse_finetune.pt"
    },
    False: {
        "BA-SMALL": "weights/co_expander/coexpander_mcut_ba-small_sparse.pt",
        "BA-LARGE": "weights/co_expander/coexpander_mcut_ba-large_sparse.pt",
        "BA-GIANT": "weights/co_expander/coexpander_mcut_ba-large_sparse.pt"
    }
}

# main
if __name__ == "__main__":
    rlsa_settings = RLSA_SETTINGS_DICT[TEST_TYPE]
    solver = COExpanderMCutSolver(
        model=COExpanderCMModel(
            env=COExpanderEnv(
                task="MCut", mode="solve", sparse_factor=1, device="cuda",
            ),
            encoder=GNNEncoder(
                task="MCut",
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
                    "rlsa_t": rlsa_settings[3]
                }
            ),
            weight_path=WEIGHT_PATH_DICT[FINETUNE][TEST_TYPE],
            inference_steps=INFERENCE_STEP,
            determinate_steps=DETERMINATE_STEP,
            beam_size=BEAM_SIZE,
        )
    )
    solver.from_txt(TEST_FILE_DICT[TEST_TYPE], ref=True, show_time=True)
    solver.solve(sampling_num=SAMPLING_NUM, show_time=True)
    print(solver.evaluate(calculate_gap=True))