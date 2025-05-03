import torch
from ml4co_kit import to_tensor, MCutGraphData


def mcut_sparse_process(mcut_graph: MCutGraphData):
    # add or remove self loop (optional)
    # mcut_graph.remove_self_loop()
    # mcut_graph.add_self_loop()
    
    # data
    nodes_num = mcut_graph.nodes_num
    edge_index = to_tensor(mcut_graph.edge_index)
    edges_num = edge_index.shape[-1]
    x = torch.zeros(size=(nodes_num,))
    edges_feature = torch.ones(size=(edges_num,))
    
    # ground truth
    if mcut_graph.ref_nodes_label is not None:
        ground_truth = to_tensor(mcut_graph.ref_nodes_label)
    else:
        ground_truth = torch.zeros(size=(nodes_num,))
    
    # adj matrix and mask
    graph = to_tensor(mcut_graph.to_matrix()).long()
    mask = torch.zeros(size=(nodes_num,))
    
    # raw_data
    raw_data = mcut_graph
    
    return (
        None, # nodes' feature, not used for MCut
        x.float(), # (V,): nodes' decision variable, 0 or 1 / set A or set B
        edges_feature.float(), # (E,): edges' feature
        None, # (E,): edges' decision variable, not used for MCut
        edge_index.long(), # (2, E): index of edge endpoints
        graph, # (V, V): adjacency matrix
        mask.bool(), # (V,): mask
        ground_truth.long(), # (V,): ground truth
        nodes_num, # number of nodes
        edges_num, # number of edges
        raw_data, # original data
    )