import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co_kit import Trainer
from ml4co.gnn4co import GNN4COModel, GNN4COEnv, TSPGNNEncoder, TSPDecoder


if __name__ == "__main__":
    model=GNN4COModel(
        env=GNN4COEnv(
            task="TSP", 
            mode="train", 
            train_data_size=128000,
            val_data_size=1280,
            train_batch_size=64,
            val_batch_size=1,
            sparse_factor=-1, 
            device="cuda",
            train_folder="path/to/train/folder",
            val_path="path/to/val/file",
            store_data=True # set False if your server's memory is not large enough.
        ),
        encoder=TSPGNNEncoder(sparse=False),
        decoder=TSPDecoder(),
        # weight_path="path/to/pretrained/file"
    )

    trainer = Trainer(model=model, devices=[0], max_epochs=50)
    trainer.model_train()