import torch
from torch import Tensor
from ml4co_kit import BaseEnv, GeneratorBase
from tensordict.tensordict import TensorDict
from torch.utils.data import DataLoader, Dataset
from symnco.env.torchrl.env_base import EnvBase


class FakeDataset(Dataset):
    def __init__(self, data_size: int):
        self.data_size = data_size

    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx: int):
        return torch.tensor([idx])
    
    
class SymNCOEnv(BaseEnv, EnvBase):
    batch_locked = False
    def __init__(
        self,
        task: str,
        generator: GeneratorBase = None,
        mode: str = None,
        train_data_size: int = 128000,
        val_data_size: int = 128,
        train_batch_size: int = 4,
        val_batch_size: int = 4,
        num_workers: int = 4,
        device: str = "cpu",
        val_path: str = None,
    ):
        # super args
        super(SymNCOEnv, self).__init__(
            name="SymNCOEnv",
            mode=mode,
            val_path=val_path,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers
        )
        super(BaseEnv, self).__init__(device=device)

        # other args
        self.task = task
        self.generator = generator
        self.train_dataset = FakeDataset(train_data_size)
        self.val_dataset = FakeDataset(val_data_size)
        
        # random seed
        random_seed = torch.empty((), dtype=torch.int64).random_().item()
        self._set_seed(random_seed)
    
    def _set_seed(self, seed: int):
        """Set the seed for the environment"""
        rng = torch.manual_seed(seed)
        self.rng = rng
        
    def train_dataloader(self):
        train_dataloader=DataLoader(
            self.train_dataset, 
            batch_size=self.train_batch_size, 
            shuffle=True,
            num_workers=self.num_workers, 
            pin_memory=True,
            persistent_workers=True, 
            drop_last=True
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader=DataLoader(
            self.val_dataset, 
            batch_size=self.val_batch_size, 
            shuffle=False
        )
        return val_dataloader
    
    def test_dataloader(self):
        test_dataloader=DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False
        )
        return test_dataloader
    
    def step(self, td: TensorDict) -> TensorDict:
        td = self._step(td)
        return {"next": td}

    def get_reward(self, td: TensorDict, actions: Tensor) -> TensorDict:
        self._check_solution_validity(td, actions)
        return self._get_reward(td, actions)

    def generate_train_data(self, batch_size: int) -> TensorDict:
        raise NotImplementedError(
            "``generate_train_data`` is required to implemented in subclasses."
        )

    def generate_val_data(self, batch_size: int) -> TensorDict:
        raise NotImplementedError(
            "``generate_val_data`` is required to implemented in subclasses."
        )

    def _step(self, td: TensorDict) -> TensorDict:
        raise NotImplementedError(
            "``_step`` is required to implemented in subclasses."
        )

    def reset(self, td: TensorDict, batch_size: int = None) -> TensorDict:
        """Reset function to call at the beginning of each episode"""
        if batch_size is None:
            batch_size = self.batch_size if td is None else td.batch_size
        if td is None or td.is_empty():
            td = self.generator(batch_size=batch_size)
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        self.to(td.device)
        return super().reset(td, batch_size=batch_size)

    def _reset(self, td: TensorDict = None, batch_size: int = None) -> TensorDict:
        raise NotImplementedError(
            "``_reset`` is required to implemented in subclasses."
        )

    def _check_solution_validity(self, td: TensorDict, actions: torch.Tensor) -> None:
        raise NotImplementedError(
            "``check_solution_validity`` is required to implemented in subclasses."
        )

    def _get_reward(self, td: TensorDict, actions: Tensor) -> TensorDict:
        raise NotImplementedError(
            "``_get_reward`` is required to implemented in subclasses."
        )
        
    def process_data(self) -> TensorDict:
        raise NotImplementedError(
            "``process_data`` is required to implemented in subclasses."
        )