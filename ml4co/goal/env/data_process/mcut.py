import torch
from ml4co_kit import to_tensor, MCutGraphData


def mcut_data_process(mcut_graph: MCutGraphData):
    mcut_graph.add_self_loop()
    nodes_num = mcut_graph.nodes_num
    edge_index = to_tensor(mcut_graph.edge_index)
    edges_num = edge_index.shape[-1]
    x = torch.zeros(size=(nodes_num,))
    edges_feature = torch.ones(size=(edges_num,))
    if mcut_graph.ref_nodes_label is not None:
        ground_truth = to_tensor(mcut_graph.ref_nodes_label)
    else:
        ground_truth = torch.zeros(size=(nodes_num,))
    graph = to_tensor(mcut_graph.to_matrix()).long()
    mask = torch.zeros(size=(nodes_num,))
    
    return (
        None, # (V,): nodes feature, MCut does not use it
        x.float(), # (V,): Decision variable, whether to select a node or not
        edges_feature.float(), # (E,): edges feature
        None, # (E,): Decision variable, MCut does not use it
        edge_index.long(), # (2, E): Index of edge endpoints
        graph, # (V, V): Adjacency matrix
        mask.bool(), # (V,): Mask
        ground_truth.long(), # (V,): Ground truth
        nodes_num, # Number of nodes
        edges_num # Number of edges
    )