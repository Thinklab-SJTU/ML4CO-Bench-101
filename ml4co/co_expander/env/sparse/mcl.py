import torch
from ml4co_kit import to_tensor, MClGraphData


def mcl_sparse_process(mcl_graph: MClGraphData):
    # add or remove self loop (optional)
    # mcl_graph.remove_self_loop()
    # mcl_graph.add_self_loop()
    
    # data
    nodes_num = mcl_graph.nodes_num
    edge_index = to_tensor(mcl_graph.edge_index)
    edges_num = edge_index.shape[-1]
    x = torch.zeros(size=(nodes_num,))
    edges_feature = torch.ones(size=(edges_num,))
    
    # ground truth
    if mcl_graph.ref_nodes_label is not None:
        ground_truth = to_tensor(mcl_graph.ref_nodes_label)
    else:
        ground_truth = torch.zeros(size=(nodes_num,))
    
    # adj matrix and mask
    graph = to_tensor(mcl_graph.to_matrix()).long()
    mask = torch.zeros(size=(nodes_num,))
    
    # raw_data
    raw_data = mcl_graph
    
    return (
        None, # nodes' feature, not used for MCl
        x.float(), # (V,): nodes' decision variable, whether to select a node or not
        edges_feature.float(), # (E,): edges' feature
        None, # (E,): edges' decision variable, not used for MCl
        edge_index.long(), # (2, E): index of edge endpoints
        graph, # (V, V): adjacency matrix
        mask.bool(), # (V,): mask
        ground_truth.long(), # (V,): ground truth
        nodes_num, # number of nodes
        edges_num, # number of edges
        raw_data, # original data
    )