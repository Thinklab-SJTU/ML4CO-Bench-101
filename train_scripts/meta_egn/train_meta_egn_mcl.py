import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co.meta_egn import MetaEGNModel, MetaEGNEnv, GNNEncoder, MetaEGNDecoder


if __name__ == "__main__":
    model=MetaEGNModel(
        env=MetaEGNEnv(
            task="MCl", 
            mode="train", 
            train_data_size=12800,
            val_data_size=500,
            train_batch_size=1,
            val_batch_size=1,
            device="cuda",
            train_folder="path/to/train/folder",
            val_path="path/to/val/file",
            processed_save_path="path/to/processed/save/path"
        ),
        encoder=GNNEncoder(
            num_layers=4,
            hidden_dim_1=64,
            hidden_dim_2=1,
            devide="cuda"
        ),
        decoder=MetaEGNDecoder(decode_kwargs={"use_rlsa": False}),
        train_outer_steps=12800,
        train_inner_steps=1,
        train_inner_samples=96,
        inner_lr=5e-5,
        outer_lr=1e-3,
        penalty_coeff=0.5,
        save_n_epochs=10,
        save_path="path/to/checkpoint/save/path",
        # weight_path="path/to/pretrained/file"
    )

    model.model_train()