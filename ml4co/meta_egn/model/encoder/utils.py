import torch
import numpy as np
from torch_scatter import scatter_add
from torch_geometric.data import Batch
from torch_geometric.utils import is_undirected, to_undirected, add_remaining_self_loops, degree


def propagate(x, edge_index):
    row, col = edge_index
    out = scatter_add(x[col], row, dim=0)
    return out


# get the k-hop neighbors of node sample
def get_mask(x, edge_index, hops):
    for k in range(hops):
        x = propagate(x, edge_index)
    mask = (x > 0).float()
    return mask


def get_diracs(
    data, 
    N: int = 1, 
    device: str = "cuda", 
    n_diracs: int = 1,
    replace: bool = True, 
    receptive_field: int = 7,
    effective_volume_range: float = 0.1,
    max_iterations: int = 20,
    complement: bool = False
) -> Batch:
    if not is_undirected(data.edge_index):
        data.edge_index = to_undirected(data.edge_index)
        
    original_batch_index = data.batch
    original_edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.batch.shape[0])[0]
    batch_index = original_batch_index
    
    graphcount = data.num_graphs
    
    r, c = original_edge_index
    
    global_offset = 0
    all_nodecounts = scatter_add(torch.ones_like(batch_index, device=device), batch_index,0)
    recfield_vols = torch.zeros(graphcount, device=device)
    total_vols = torch.zeros(graphcount, device=device)
    
    for j in range(n_diracs):
        diracmatrix = torch.zeros(0,device=device)
        locationmatrix = torch.zeros(0,device=device).long()
        for k in range(graphcount):
            # get edges of current graph, remember to subtract offset
            graph_nodes = all_nodecounts[k]
            if graph_nodes == 0:
                print("all nodecounts: ", all_nodecounts)
            graph_edges = (batch_index[r] == k)
            graph_edge_index = original_edge_index[:, graph_edges] - global_offset           
            gr, gc = graph_edge_index

            # get dirac
            randInt = np.random.choice(range(graph_nodes), N, replace=replace)
            node_sample = torch.zeros(N * graph_nodes, device=device)
            offs  = torch.arange(N, device=device) * graph_nodes
            dirac_locations = (offs + torch.from_numpy(randInt).to(device))
            node_sample[dirac_locations] = 1

            # calculate receptive field volume and compare to total volume
            mask = get_mask(node_sample, graph_edge_index.detach(), receptive_field).float()
            deg_graph = degree(gr, (graph_nodes.item()))
            total_volume = deg_graph.sum()
            recfield_volume = (mask * deg_graph).sum()
            volume_range = recfield_volume / total_volume
            total_vols[k] = total_volume
            recfield_vols[k] = recfield_volume

            # if receptive field volume is less than x% of total volume, resample
            for iteration in range(max_iterations):  
                randInt = np.random.choice(range(graph_nodes), N, replace=replace)
                node_sample = torch.zeros(N * graph_nodes,device=device)
                offs  = torch.arange(N, device=device) * graph_nodes
                dirac_locations = (offs + torch.from_numpy(randInt).to(device))
                node_sample[dirac_locations] = 1

                mask = get_mask(node_sample, graph_edge_index, receptive_field).float()
                recfield_volume = (mask * deg_graph).sum()
                volume_range = recfield_volume / total_volume

                if volume_range > effective_volume_range:
                    recfield_vols[k] = recfield_volume
                    total_vols[k] = total_volume
                    break
            dirac_locations2 = torch.from_numpy(randInt).to(device) + global_offset
            global_offset += graph_nodes

            diracmatrix = torch.cat((diracmatrix, node_sample), 0)
            locationmatrix = torch.cat((locationmatrix, dirac_locations2), 0)
    
    locationmatrix = diracmatrix.nonzero()
    
    if complement:
        return Batch(
            batch=batch_index, x=diracmatrix, edge_index=original_edge_index, y=data.y,
            locations=locationmatrix, volume_range=volume_range, recfield_vol=recfield_vols,
            total_vol=total_vols, complement_edge_index=data.complement_edge_index
        )
    else:
        return Batch(
            batch=batch_index, x=diracmatrix, edge_index=original_edge_index,
            y=data.y, locations=locationmatrix, volume_range=volume_range,
            recfield_vol=recfield_vols, total_vol=total_vols
        )