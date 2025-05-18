import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co_kit import Trainer
from  ml4co.co_expander import COExpanderCMModel, GNNEncoder, COExpanderEnv, COExpanderDecoder


if __name__ == "__main__":
    model = COExpanderCMModel(
        env=COExpanderEnv(
            task="MCl",
            mode="train",
            train_data_size=12800,
            val_data_size=500,
            train_batch_size=4,
            val_batch_size=1,
            num_workers=4,
            sparse_factor=1,
            device="cuda",
            train_folder="path/to/train/folder",
            val_path="path/to/val/file",
            store_data=True # set False if your server's memory is not large enough.
        ),
        encoder=GNNEncoder(
            task="MCl",
            sparse=True,
            block_layers=[2, 4, 4, 2],
            hidden_dim=256
        ),
        decoder=COExpanderDecoder(),
        learning_rate=0.0002,
        prompt_prob=0.0, # set prompt prob as zero
        # weight_path="path/to/pretrained/file"
    )

    trainer = Trainer(model=model, devices=[0], max_epochs=50)
    trainer.model_train()