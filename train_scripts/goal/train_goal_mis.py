import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co_kit import Trainer
from ml4co.goal import GOALModel, GOALEnv, GATEncoder, GOALDecoder


if __name__ == "__main__":
    model=GOALModel(
        env=GOALEnv(
            task="MIS", 
            mode="train", 
            train_data_size=128000,
            val_data_size=100,
            train_batch_size=4,
            val_batch_size=10,
            device="cuda",
            train_folder="path/to/train/folder",
            val_path="path/to/val/file",
            store_data=True # set False if your server's memory is not large enough.
        ),
        encoder=GATEncoder(task="MIS"),
        decoder=GOALDecoder(),
        learning_rate=0.0002,
        # weight_path="path/to/pretrained/file"
    )

    trainer = Trainer(model=model, devices=[0], max_epochs=50)
    trainer.model_train()