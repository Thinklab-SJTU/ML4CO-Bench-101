import torch
from torch import Tensor
from ml4co_kit import to_tensor, MVCGraphData


def mvc_sparse_process(mvc_graph: MVCGraphData):
    nodes_num = mvc_graph.nodes_num
    edge_index = to_tensor(mvc_graph.edge_index)
    edges_num = edge_index.shape[-1]
    x: Tensor = torch.randint(low=0, high=1, size=(nodes_num,))
    e = torch.ones(size=(edges_num,))
    if mvc_graph.ref_nodes_label is not None:
        ground_truth = to_tensor(mvc_graph.ref_nodes_label)
    else:
        ground_truth = torch.zeros(size=(nodes_num,))
    graph = to_tensor(mvc_graph.to_matrix()).long()
    
    return (
        x.float(), # (V,): nodes feature
        e.float(), # (E,): edges feature
        edge_index.long(), # (2, E): Index of edge endpoints
        graph, # (V, V): Adjacency matrix
        ground_truth.long(), # (V,): Ground truth
        nodes_num, # Number of nodes
        edges_num # Number of edges
    )