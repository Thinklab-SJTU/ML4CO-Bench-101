import torch
from ml4co_kit import to_tensor, MVCGraphData


def mvc_sparse_process(mvc_graph: MVCGraphData):
    # add or remove self loop (optional)
    # mvc_graph.remove_self_loop()
    # mvc_graph.add_self_loop()
    
    # data
    nodes_num = mvc_graph.nodes_num
    edge_index = to_tensor(mvc_graph.edge_index)
    edges_num = edge_index.shape[-1]
    x = torch.zeros(size=(nodes_num,))
    edges_feature = torch.ones(size=(edges_num,))
    
    # ground truth
    if mvc_graph.ref_nodes_label is not None:
        ground_truth = to_tensor(mvc_graph.ref_nodes_label)
    else:
        ground_truth = torch.zeros(size=(nodes_num,))
        
    # adj matrix and mask
    graph = to_tensor(mvc_graph.to_matrix()).long()
    mask = torch.zeros(size=(nodes_num,))
    
    # raw_data
    raw_data = mvc_graph
    
    return (
        None, # nodes' feature, not used for MVC
        x.float(), # (V,): nodes' decision variable, whether to select a node or not
        edges_feature.float(), # (E,): edges' feature
        None, # (E,): edges' decision variable, not used for MVC
        edge_index.long(), # (2, E): index of edge endpoints
        graph, # (V, V): adjacency matrix
        mask.bool(), # (V,): mask
        ground_truth.long(), # (V,): ground truth
        nodes_num, # number of nodes
        edges_num, # number of edges
        raw_data, # original data
    )