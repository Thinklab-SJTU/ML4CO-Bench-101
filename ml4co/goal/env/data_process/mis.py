import torch
from ml4co_kit import to_tensor, MISGraphData


def mis_data_process(mis_graph: MISGraphData, max_nn: int):
    # add self loop
    mis_graph.add_self_loop()
    
    # nodes num
    nodes_num = mis_graph.nodes_num
    
    # get matrix (edge feature)
    e = torch.ones(size=(max_nn, max_nn))
    e[:nodes_num, :nodes_num] = to_tensor(mis_graph.to_matrix())
    e[torch.arange(nodes_num), torch.arange(nodes_num)] = 1
    
    # solution
    nodes_label = torch.zeros(size=(max_nn,))
    if mis_graph.ref_nodes_label is not None:
        nodes_label[:nodes_num] = to_tensor(mis_graph.ref_nodes_label)
    
    # raw data
    raw_data = mis_graph
    
    return (
        nodes_label.long(), # (V,)
        e.float(), # (V, V): edges feature, adj matrix
        raw_data
    )