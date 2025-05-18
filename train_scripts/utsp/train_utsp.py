import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co_kit import Trainer, TSPDataGenerator
from ml4co.utsp import UTSPModel, UTSPEnv, SAGEncoder, UTSPDecoder


# settings
NODES_NUM = 50

if __name__ == "__main__":
    model=UTSPModel(
        env=UTSPEnv(
            generator=TSPDataGenerator(
                only_instance_for_us=True,
                nodes_num=NODES_NUM
            ),
            mode="train", 
            train_data_size=128000,
            val_data_size=1280,
            train_batch_size=32,
            val_batch_size=1,
            device="cuda",
            val_path="path/to/val/file"
        ),
        encoder=SAGEncoder(output_channels=NODES_NUM),
        decoder=UTSPDecoder(),
        learning_rate=0.001,
        # weight_path="path/to/pretrained/file"
    )

    trainer = Trainer(model=model, devices=[0], max_epochs=50)
    trainer.model_train()