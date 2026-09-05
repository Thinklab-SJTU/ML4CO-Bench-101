import os
import sys

root_folder = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(root_folder)
from ml4co.fast_t2t_ms.skip_ortools import apply as skip_ortools
skip_ortools()
from ml4co_kit.learning.extra_backends.mindspore import MSTrainer
from ml4co.fast_t2t_ms import *
from train_scripts.val_dataset import TSP100_VAL_PATH
from train_scripts.train_dataset import TSP100_TRAIN_FOLDER


# Dataset Settings
DATASET_NUM_WORKERS = 2
VAL_PATH = TSP100_VAL_PATH
TRAIN_FOLDER = TSP100_TRAIN_FOLDER


# Training Settings
BATCH_SIZE = 4
EPOCH_STEPS = 5000
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
WEIGHT_PATH = None
MAX_EPOCHS = 100
DEVICE_TARGET = "Ascend"
DEVICE_ID = 0


# Main
if __name__ == "__main__":
    # Create Train Dataset
    train_dataset = TSPDataset(
        num_per_epoch=BATCH_SIZE * EPOCH_STEPS,
        data_folder=TRAIN_FOLDER,
    )

    # Create Val Dataset
    val_dataset = TSPDataset(
        num_per_epoch=1280,
        data_path=VAL_PATH,
    )

    # Create Environment (device must match MSTrainer.device_target)
    env = TSPEnv(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        mode="train",
        train_batch_size=BATCH_SIZE,
        val_batch_size=BATCH_SIZE,
        num_workers=DATASET_NUM_WORKERS,
        device=DEVICE_TARGET,
        device_id=DEVICE_ID,
    )
    
    # Create Model
    model = TSPModel(hidden_dim=256, num_layers=12)

    # Create PL Model
    pl_model = TSPPLModel(
        env=env, 
        model=model, 
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        weight_path=WEIGHT_PATH,
    )

    # Create Trainer
    ckpt_filename = "epoch={epoch}-{val/ag_1:.4f}-{val/ag_5:.4f}-{val/am_1:.4f}-{val/am_5:.4f}"
    trainer = MSTrainer(
        model=pl_model,
        device_target=DEVICE_TARGET,
        device_id=DEVICE_ID,
        max_epochs=MAX_EPOCHS,
        fp16=False,
        ckpt_filename=ckpt_filename,
        mode="max",
        save_top_k=-1
    )

    # Train
    trainer.model_train()
