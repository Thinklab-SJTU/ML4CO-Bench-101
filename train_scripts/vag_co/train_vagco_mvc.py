import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sub_folder = os.path.join(root_folder, "ml4co")
sys.path.append(sub_folder)
from ml4co_kit import Trainer, MVCDataGenerator
from ml4co.vag_co import VAGCOModel, VAGCOEnv, GNNEncoder, MVCDecoder


if __name__ == "__main__":
    model=VAGCOModel(
        env=VAGCOEnv(
            task="MVC", 
            generator=MVCDataGenerator(
                only_instance_for_us=True, 
                data_type="rb",
                nodes_num_min=200,
                nodes_num_max=300,
                rb_k_scale=(5, 12),
                rb_n_scale=(20, 25),
                rb_p_scale=(0.3, 1),
            ),
            mode="train", 
            train_data_size=128000,
            val_data_size=128,
            train_batch_size=8,
            val_batch_size=1,
            device="cuda",
            val_path="path/to/val/file",
        ),
        encoder=GNNEncoder(task="MVC", sparse=True, block_layers=[1, 2, 2, 1]),
        decoder=MVCDecoder(),
        # weight_path="path/to/pretrained/file"
    )

    trainer = Trainer(model=model, devices=[0], max_epochs=50)
    trainer.model_train()