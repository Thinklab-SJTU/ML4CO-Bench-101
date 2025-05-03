import torch
from ml4co_kit import to_tensor, MVCGraphData


def mvc_data_process(mvc_graph: MVCGraphData, max_nn: int):
    # add self loop
    mvc_graph.add_self_loop()
    
    # nodes num
    nodes_num = mvc_graph.nodes_num
    
    # get matrix (edge feature)
    e = torch.zeros(size=(max_nn, max_nn))
    e[:nodes_num, :nodes_num] = to_tensor(mvc_graph.to_matrix())
    e[torch.arange(nodes_num), torch.arange(nodes_num)] = 1
    
    # solution
    nodes_label = torch.zeros(size=(max_nn,))
    if mvc_graph.ref_nodes_label is not None:
        nodes_label[:nodes_num] = to_tensor(mvc_graph.ref_nodes_label)
    
    # raw data
    raw_data = mvc_graph
    
    return (
        nodes_label.long(), # (V,)
        e.float(), # (V, V): edges feature, adj matrix
        raw_data
    )