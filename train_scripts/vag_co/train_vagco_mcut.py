import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co_kit import Trainer, MCutDataGenerator
from ml4co.vag_co import VAGCOModel, VAGCOEnv, GNNEncoder, MCutDecoder


if __name__ == "__main__":
    model=VAGCOModel(
        env=VAGCOEnv(
            task="MCut", 
            generator=MCutDataGenerator(
                only_instance_for_us=True, 
                data_type="ba",
                nodes_num_min=200,
                nodes_num_max=300,
                ba_conn_degree=4
            ),
            mode="train", 
            train_data_size=32000,
            val_data_size=128,
            train_batch_size=8,
            val_batch_size=1,
            device="cuda",
            val_path="path/to/val/file",
        ),
        encoder=GNNEncoder(task="MCut", sparse=True, block_layers=[1, 2, 2, 1]),
        decoder=MCutDecoder(),
        # weight_path="path/to/pretrained/file"
    )

    trainer = Trainer(model=model, devices=[0], max_epochs=50)
    trainer.model_train()