from tqdm import trange
import numpy as np
import torch
import random
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.nn as gnn
import os.path as osp
import time
from ml4co_kit import to_numpy
import torch_sampling as pysa

__TSP_VERSION__ = 1.1

# assert with custom exception class
def assert_(cond, cls, *args, **kwargs):
    if not cond:
        raise cls(*args, **kwargs)

class Dict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

# preprocess instance graphs
class Graph:
    def __init__(self, x, edge_index, edge_attr):
        self.n_nodes = x.size(0)
        self.n_edges = edge_index.size(-1)
        self.x = x
        self.edge_index, self.edge_attr = pyg.utils.sort_edge_index(edge_index, edge_attr, num_nodes = self.n_nodes)
    @property
    def data(self):
        if not hasattr(self, '_data'):
            self._data = pyg.data.Data(x = self.x, edge_index = self.edge_index, edge_attr = self.edge_attr)
        return self._data
    @property
    def degs(self):
        if not hasattr(self, '_degs'):
            self._degs = self.edge_index[0].unique_consecutive(return_counts = True)[-1]
        return self._degs
    @classmethod
    def knn(cls, x, k = None, to_undirected = False):
        n_nodes = x.size(0)
        assert k is not None    
        edge_index = gnn.knn_graph(x, k, flow = 'target_to_source').to(x.device)
        edge_attr = F.pairwise_distance(x[edge_index[0]], x[edge_index[1]], keepdim = True)
        if to_undirected:
            edge_index, edge_attr = pyg.utils.to_undirected(edge_index, edge_attr)
        return cls(x = x, edge_index = edge_index, edge_attr = edge_attr)
    @classmethod
    def gen(cls, device, n_nodes_min, n_nodes_max, knn_k = None, to_undirected = False):
        n_nodes = random.randint(n_nodes_min, n_nodes_max)
        x = torch.rand(n_nodes, 2, device = device)
        return cls.knn(x = x, k = knn_k, to_undirected = to_undirected)
    @classmethod
    def gen_batch(cls, batch_size, *args, **kwargs):
        graph_list = [cls.gen(*args, **kwargs) for i in range(batch_size)]
        return graph_list, cls.to_pyg_batch(graph_list)
    @staticmethod
    def to_pyg_batch(graph_list):
        return pyg.data.Batch.from_data_list([graph.data for graph in graph_list])

# edge_index of complete graphs
class EICG:
    def __init__(self, device):
        self.device = device
        self.ei_dict = dict()
    def make(self, n_nodes):
        return torch.tensor([
            [u for u in range(n_nodes) for v in range(n_nodes) if u != v],
            [v for u in range(n_nodes) for v in range(n_nodes) if u != v]
        ], dtype = torch.long, device = self.device)
    def get(self, n_nodes):
        if n_nodes not in self.ei_dict:
            self.ei_dict[n_nodes] = self.make(n_nodes)
        return self.ei_dict[n_nodes]
    def fill_mat(self, n_nodes, values):
        edge_index = self.get(n_nodes)
        mat = torch.zeros((n_nodes, n_nodes), dtype = values.dtype, device = values.device)
        mat[edge_index[0], edge_index[1]] = values
        return mat

def torch_add_grad(x, grad):
    if x.grad is None:
        x.grad = grad
    else:
        x.grad.add_(grad)
    return x

# calculate distance matrix from coordinates
def x_to_adj(x, p = 2.):
    return torch.cdist(x.float(), x.float(), p = p)

def num_abbr(n):
    if n < 1000:
        return f'{n}'
    elif n % 1000 == 0:
        return f'{n // 1000}k'
    else:
        return f'{n / 1000}k'

# generate save_name
def tsp_save_name(args, save_name = None):
    if not save_name:
        timestamp = int(time.time() * 100)
        save_name = f'dimes-tsp{num_abbr(args.n_nodes)}-knn{args.knn_k}@{timestamp}'
    return osp.join(args.output_dir, save_name)

# interfaces for torch_sampling
@torch.no_grad()
def tsp_greedy(graph, par, sample_size, best=True):
    ys, tours = pysa.tsp_greedy(graph.x, graph.degs, graph.edge_index[1], par, sample_size)
    if best:
        y, i = ys.min(dim = 0)
        return y.detach().clone(), tours[i].detach().clone()
    else:
        return ys.detach().clone(), tours.detach().clone()

@torch.no_grad()
def tsp_softmax(graph, par, sample_size, y_bl=None, best=True):
    ys, tours = pysa.tsp_softmax(graph.x, graph.degs, graph.edge_index[1], par, sample_size, np.nan if y_bl is None else y_bl)
    if best:
        y, i = ys.min(dim = 0)
        return y.detach().clone(), tours[i].detach().clone()
    else:
        return ys.detach().clone(), tours.detach().clone()

@torch.no_grad()
def tsp_softmax_grad(graph, par, sample_size, y_bl=None):
    ys, grad = pysa.tsp_softmax_grad(graph.x, graph.degs, graph.edge_index[1], par, sample_size, np.nan if y_bl is None else y_bl)
    return ys.detach().clone(), grad.detach().clone()

# calculate cost from coordinates
def tsp_calc_cost_x(sol, x, p = 2.):
    sol_sizes = sol.size()
    sol = torch.cat([sol, sol[:, 0].unsqueeze(1)], dim = 1)
    return F.pairwise_distance(
        x.gather(dim = 1, index = sol[:, : -1].unsqueeze(2).expand(*sol_sizes, 2)).flatten(0, 1),
        x.gather(dim = 1, index = sol[:, 1 : ].unsqueeze(2).expand(*sol_sizes, 2)).flatten(0, 1),
        p = p).view(*sol_sizes).sum(dim = 1)

# calculate cost from distance matrix
def tsp_calc_cost_adj(sol, adj):
    batch_size, n_nodes = sol.size()
    sol = torch.cat([sol, sol[:, 0].unsqueeze(1)], dim = 1)
    bi = torch.arange(batch_size).unsqueeze(1).expand(batch_size, n_nodes).flatten()
    return adj[bi, sol[:, : -1].flatten(), sol[:, 1 : ].flatten()].view(batch_size, n_nodes).sum(dim = 1)

# convert theta to heatmap
def tsp_make_heatmap(n_nodes, par, edge_index, fname):
    idx = torch.argsort(edge_index[0])
    par = par[idx]
    edge_index = edge_index[:, idx]
    cnt = torch.unique_consecutive(edge_index[0], return_inverse = False, return_counts = True)[1].tolist()
    par = torch.split(par, cnt)
    edge_index = torch.split(edge_index, cnt, dim = 1)
    assert len(par) == len(edge_index)
    probs = torch.zeros((n_nodes, n_nodes), dtype = par[0].dtype, device = par[0].device)
    tbar = trange(len(par), desc = fname + ' computing')
    for i in tbar:
        probs[edge_index[i][0, 0], edge_index[i][1]] = F.softmax(par[i])
        if i == len(par) - 1:
            tbar.set_description(fname + ' saving')
    # return np.savetxt(fname, probs.cpu().detach().numpy(), fmt = '%.6f', delimiter = ' ', header = f'{n_nodes}', comments = '')
    return to_numpy(probs)


# REINFORCE
def meta_update_sparse(emb0, phi_net, graph, opt_fn, steps, sample_size, greedy_size=None):
    emb = emb0.detach().clone().requires_grad_()
    psi_net = phi_net.clone()
    psi_net.train()
    opt = opt_fn([emb, *psi_net.trainables()])
    tbar = range(1, steps + 1)
    y_means = []
    if greedy_size is not None:
        y_grs = [tsp_greedy(graph, par, greedy_size)[0].min().item()]
    else:
        y_grs = [None]
    for t in tbar:
        opt.zero_grad()
        par = psi_net(emb)
        ys, par_grad = tsp_softmax_grad(graph, par, sample_size, y_bl = None)
        y_means.append(ys.mean().item())
        if greedy_size is not None:
            y_grs.append(tsp_greedy(graph, par, greedy_size)[0].min().item())
        # tbar.set_description(f'meta_step={t} cost_mean={y_means[-1]:.4f}')
        #if not par_grad.isnan().any():
        par.backward(par_grad)
        opt.step()
        del par, ys, par_grad
    return emb, psi_net, y_means, y_grs

# REINFORCE gradient
def net_approx_grads(emb, psi_net, graph, sample_size):
    emb = emb.detach().clone().requires_grad_()
    if emb.grad is not None:
        emb.grad.zero_()
    par = psi_net(emb)
    _, par_grad = tsp_softmax_grad(graph, par.detach(), sample_size, y_bl = None)
    par.backward(par_grad)
    emb_grad = emb.grad.detach().clone()
    phi_grads = []
    for psi in psi_net.trainables():
        phi_grads.append(psi.grad.detach().clone())
    return emb_grad, phi_grads


def net_infer_greedy(args, net, graph):
    with torch.no_grad():
        emb0 = net.emb_net(graph.x, graph.edge_index, graph.edge_attr)
    emb1, psi_net, _, _ = meta_update_sparse(emb0, net.par_net, graph, args.inner_opt_fn, args.te_tune_steps, sample_size = args.te_tune_sample_size, greedy_size = None) #, verbose = verbose, plot = plot, save_name = save_name)
    with torch.no_grad():
        psi_net.eval()
        par1 = psi_net(emb1.detach())
        y, tour = tsp_greedy(graph, par1, args.te_sample_size)
        return emb1, psi_net, par1, y, tour

def net_infer_sampling(args, net, graph):
    with torch.no_grad():
        emb0 = net.emb_net(graph.x, graph.edge_index, graph.edge_attr)
    emb1, psi_net, _, _ = meta_update_sparse(emb0, net.par_net, graph, args.inner_opt_fn, args.te_tune_steps, sample_size = args.te_tune_sample_size, greedy_size = None) #, verbose = verbose, plot = plot, save_name = save_name)
    with torch.no_grad():
        psi_net.eval()
        par1 = psi_net(emb1.detach())
        y, tour = tsp_softmax(graph, par1 / args.te_sample_tau, args.te_sample_size, best = True)
        return emb1, psi_net, par1, y, tour
