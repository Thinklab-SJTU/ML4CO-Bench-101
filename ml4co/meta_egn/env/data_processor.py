import os
import torch
from tqdm import tqdm
from pathlib import Path
from typing import Union
from ml4co_kit import MClSolver, MISSolver, MVCSolver, MCutSolver
from torch_geometric.data import InMemoryDataset, Data, Batch, DataLoader
from meta_egn.model.encoder.utils import get_diracs


class MetaEGNDataset(InMemoryDataset):
    def __init__(
        self,
        mode: str,
        ml4co_kit_solver: Union[MClSolver, MISSolver, MVCSolver, MCutSolver],
        txt_data_path: str = None,
        pt_store_path: str = None,
        data_size: int = None
    ):
        self.txt_data_path = txt_data_path
        self.pt_data_path = Path(pt_store_path)
        self.solver = ml4co_kit_solver
        self.data_size = data_size
        self.mode = mode
        super(MetaEGNDataset, self).__init__(root=self.pt_data_path)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def process(self):
        # for each task, it's a separate dataset
        data_list = []
        self.solver.from_txt(self.txt_data_path, ref=True, show_time=True)
        if self.data_size > len(self.solver.graph_data):
            print(f"{self.mode}_data_size set as {self.data_size}, but only {len(self.solver.graph_data)} available. \
                  {len(self.solver.graph_data)} instances to be loaded.")
        for task_index in tqdm(range(len(self.solver.graph_data[:self.data_size])), desc=f"preparing {self.mode} data"):
            graph = self.solver.graph_data[task_index]
            edges = torch.tensor(graph.edge_index)
            x = torch.zeros(graph.nodes_num).reshape(-1, 1)
            tmp_data_list = []
            tmp_data = Data(x=x, edge_index=edges)
            tmp_data_list.append(tmp_data)
            tmp_data_loader = DataLoader(tmp_data_list, batch_size=1)
            for data in tmp_data_loader:
                new_data = get_diracs(data, 1, effective_volume_range=0.15, receptive_field=5, device="cpu")
                final_data = Data(x=new_data.x, edge_index=new_data.edge_index, train_batch=new_data.batch)
                data_list.append(final_data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
    @classmethod
    def process_solving(cls, kit_graph_list, save_path):
        if os.path.exists(save_path):
            batched_data_list = torch.load(save_path)
            print("Processed data loaded for testing.")
        else:
            batched_data_list = []
            for graph in tqdm(kit_graph_list, desc=f"Processing test data for MetaEGNSolver"):
                edges = torch.tensor(graph.edge_index)
                x = torch.zeros(graph.nodes_num).reshape(-1, 1)
                tmp_data_list = []
                tmp_data = Data(x=x, edge_index=edges)
                tmp_data_list.append(tmp_data)
                tmp_data_loader = DataLoader(tmp_data_list, batch_size=1)
                for data in tmp_data_loader:
                    new_data = get_diracs(data, 1, effective_volume_range=0.15, receptive_field=5, device="cpu")
                    final_data = Data(x=new_data.x, edge_index=new_data.edge_index, train_batch=new_data.batch)
                    batched_data_list.append(Batch.from_data_list([final_data]))
            torch.save(batched_data_list, save_path)
            print(f"Processed data saved at {save_path} for reusing.")

        return batched_data_list