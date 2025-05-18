
import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.dimes import DIMESModel, DIMESEnv, TSPDecoder, get_tsp_dimes_encoder


# settings
NODES_NUM = 50
SPARSE_FACTOR = -1
SAVE_PATH_DICT = {
    50: "ml4co/dimes/weights/tsp50-epoch-{}-{:.4f}.pt",
    100: "ml4co/dimes/weights/tsp100-epoch-{}-{:.4f}.pt",
    500: "ml4co/dimes/weights/tsp500-epoch-{}-{:.4f}.pt",
    1000: "ml4co/dimes/weights/tsp1000-epoch-{}-{:.4f}.pt",
    10000: "ml4co/dimes/weights/tsp10000-epoch-{}-{:.4f}.pt"
}

if __name__ == "__main__":
    model=DIMESModel(
        env=DIMESEnv(
            task="TSP", 
            mode="train", 
            train_data_size=128000,
            val_data_size=128,
            train_batch_size=4,
            val_batch_size=1,
            sparse_factor=SPARSE_FACTOR,
            device="cuda",
            train_folder="path/to/train/folder",
            val_path="path/to/val/file",
            store_data=True # set False if your server's memory is not large enough.
        ),
        encoder=get_tsp_dimes_encoder(
            sparse=SPARSE_FACTOR>0, 
            hidden_dim=32 if SPARSE_FACTOR>0 else 64,
            num_nodes=NODES_NUM,
            device="cuda"
        ),
        decoder=TSPDecoder(),
        train_outer_steps=1000,
        train_inner_steps=15,
        train_inner_samples=500,
        val_inner_steps=0, # active search
        inner_lr=1e-2,
        outer_lr=1e-3,
        inner_weight_decay=1e-4,
        outer_weight_decay=1e-5,
        save_n_epochs=10,
        save_path=SAVE_PATH_DICT[NODES_NUM],
        # weight_path="path/to/pretrained/file"
    )

    model.model_train()