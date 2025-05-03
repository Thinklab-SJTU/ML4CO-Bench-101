import torch
from typing import Any
from ml4co_kit import *
from torch.utils.data import DataLoader, Dataset
from vag_co.env.sparser import VAGCOSparser


class FakeDataset(Dataset):
    def __init__(self, data_size: int):
        self.data_size = data_size

    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx: int):
        return torch.tensor([idx])


class VAGCOEnv(BaseEnv):
    def __init__(
        self,
        task: str,
        generator: NodeGeneratorBase,
        mode: str = None,
        train_data_size: int = 128000,
        val_data_size: int = 128,
        train_batch_size: int = 4,
        val_batch_size: int = 4,
        num_workers: int = 4,
        device: str = "cpu",
        val_path: str = None
    ):
        super().__init__(
            name="VAGCOEnv",
            mode=mode,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            device=device
        )
        
        # basic
        self.task = task
        
        # train data folder and val path
        self.val_path = val_path
        
        # ml4co-kit solver
        self.mcl_solver = MClSolver()
        self.mcut_solver = MCutSolver()
        self.mis_solver = MISSolver()
        self.mvc_solver = MVCSolver()
        
        # dataset (Fake)
        self.train_dataset = FakeDataset(train_data_size)
        self.val_dataset = FakeDataset(val_data_size)
          
        # data_processor (sparser)
        self.data_processor = VAGCOSparser(self.device)
        
        # val
        self.val_data_cache = None
        
        # train data generator
        self.generator = generator
        
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

    #################################
    #       Generate Val Data       #           
    #################################

    def generate_val_data(self, val_idx: int) -> Any:
        begin_idx = val_idx * self.val_batch_size
        end_idx = begin_idx + self.val_batch_size
        if self.task == "MCl":
            return self.generate_val_data_mcl(begin_idx, end_idx)
        elif self.task == "MCut":
            return self.generate_val_data_mcut(begin_idx, end_idx)
        elif self.task == "MIS":
            return self.generate_val_data_mis(begin_idx, end_idx)
        elif self.task == "MVC":
            return self.generate_val_data_mvc(begin_idx, end_idx)
        else:
            raise NotImplementedError()

    def generate_val_data_mcl(self, begin_idx: int, end_idx: int) -> Any:
        if self.val_data_cache is None:
            self.mcl_solver.from_txt(self.val_path, ref=True)
            self.val_data_cache = {
                "graph_data": self.mcl_solver.graph_data
            }
        return self.data_processor.mcl_batch_data_process(
            graph_data=self.val_data_cache["graph_data"][begin_idx:end_idx]
        )

    def generate_val_data_mcut(self, begin_idx: int, end_idx: int) -> Any:
        if self.val_data_cache is None:
            self.mcut_solver.from_txt(self.val_path, ref=True)
            self.val_data_cache = {
                "graph_data": self.mcut_solver.graph_data
            }
        return self.data_processor.mcut_batch_data_process(
            graph_data=self.val_data_cache["graph_data"][begin_idx:end_idx]
        )
    
    def generate_val_data_mis(self, begin_idx: int, end_idx: int) -> Any:
        if self.val_data_cache is None:
            self.mis_solver.from_txt(self.val_path, ref=True)
            self.val_data_cache = {
                "graph_data": self.mis_solver.graph_data
            }
        return self.data_processor.mis_batch_data_process(
            graph_data=self.val_data_cache["graph_data"][begin_idx:end_idx]
        )
    
    def generate_val_data_mvc(self, begin_idx: int, end_idx: int) -> Any:
        if self.val_data_cache is None:
            self.mvc_solver.from_txt(self.val_path, ref=True)
            self.val_data_cache = {
                "graph_data": self.mvc_solver.graph_data
            }
        return self.data_processor.mvc_batch_data_process(
            graph_data=self.val_data_cache["graph_data"][begin_idx:end_idx]
        )

    #################################
    #      Generate Train Data      #
    #################################
    
    def generate_train_data(self, batch_size: int) -> Any:
        graph_data = self.generator.generate_only_instance_for_us(batch_size)
        if self.task == "MCl":
            return self.data_processor.mcl_batch_data_process(graph_data)
        elif self.task == "MCut":
            return self.data_processor.mcut_batch_data_process(graph_data)
        elif self.task == "MIS":
            return self.data_processor.mis_batch_data_process(graph_data)
        elif self.task == "MVC":
            return self.data_processor.mvc_batch_data_process(graph_data)