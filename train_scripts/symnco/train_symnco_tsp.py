import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co_kit import Trainer, TSPDataGenerator
from ml4co.symnco import SymNCOModel, GATEncoder, SymNCOTSPEnv, SymNCODecoder


# settings
NODES_NUM = 50
CAPACITY_DICT = {50: 40, 100: 50, 200: 80, 500: 100}


if __name__ == "__main__":
    capacity = CAPACITY_DICT[NODES_NUM]
    model = SymNCOModel(
        env=SymNCOTSPEnv(
            generator=TSPDataGenerator(
                only_instance_for_us=True,
                nodes_num=NODES_NUM
            ),
            mode="train",
            device="cuda",
            num_workers=4,
            train_data_size=128000,
            train_batch_size=32,
            val_batch_size=10,
            val_data_size=1280,
            val_path="path/to/val/file"
        ),
        encoder=GATEncoder(
            task="TSP",
            num_heads=8,
            hidden_dim=128,
            num_layers=3
        ),
        decoder=SymNCODecoder(
            task="TSP",
            hidden_dim=128
        ),
        decode_type="sampling",
        learning_rate=0.0002,
        # weight_path="path/to/pretrained/file"
    )

    trainer = Trainer(model=model, devices=[0])
    trainer.model_train()