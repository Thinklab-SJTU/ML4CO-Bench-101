import gc
import torch
from tqdm import trange
from typing import Union
from torch import optim, Tensor
from dimes.env.env import DIMESEnv
from dimes.model.decoder.base import DIMESDecoder
from ml4co.dimes.model.encoder.dense.gnn_encoder_dense import DIMESDenseEncoder
from ml4co.dimes.model.encoder.sparse.gnn_encoder_sparse import DIMESSparseEncoder
from ml4co.dimes.model.encoder.dense import meta_update_dense
from ml4co.dimes.model.encoder.sparse import *


class DIMESModel(object):
    def __init__(
        self,
        env: DIMESEnv,
        encoder: Union[DIMESDenseEncoder, DIMESSparseEncoder],
        decoder: DIMESDecoder,
        train_outer_steps: int = 100,
        train_inner_steps: int = 10,
        train_inner_samples: int = 100,
        val_inner_steps: int = 10,
        inner_lr: float = 1e-1,
        outer_lr: float = 1e-3,
        inner_weight_decay: float = 0.,
        outer_weight_decay: float = 5e-4,
        save_n_epochs: int = 10,
        save_path: str = "path/to/save/weights",
        weight_path: str = None
    ):
        self.env: DIMESEnv = env
        self.model: Union[DIMESDenseEncoder, DIMESSparseEncoder] = encoder
        self.decoder: DIMESDecoder = decoder
        self.train_outer_steps =  train_outer_steps
        self.train_inner_steps = train_inner_steps
        self.train_inner_samples = train_inner_samples
        self.val_inner_steps = val_inner_steps
        self.save_n_epochs = save_n_epochs
        self.save_path = save_path
        self.inner_optimizer = lambda par: optim.AdamW(par, lr=inner_lr, weight_decay=inner_weight_decay)
        self.outer_optimizer = lambda par: optim.AdamW(par, lr=outer_lr, weight_decay=outer_weight_decay)
        
        # load pretrained weights if needed
        if weight_path is not None:
            state_dict = torch.load(weight_path, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.env.device)

    def model_train(self):
        if self.env.sparse:
            self.train_sparse()
        else:
            self.train_dense()

    def train_sparse(self):
        self.model: DIMESSparseEncoder
        self.model.train()
        opt = self.outer_optimizer(self.model.parameters())
        tbar = trange(1, self.train_outer_steps + 1)
        best_val_cost = 1e5
        bs = self.env.train_batch_size
        n_nodes = self.model.num_nodes
        n_edges = n_nodes * self.env.sparse_factor
        for step in tbar:
            _, x, e, edge_index, _, _, _, _ = self.env.generate_train_data(self.env.train_batch_size)
            graph_list = [Graph(
                            x=x[i*n_nodes: (i+1)*n_nodes, :], 
                            edge_index=edge_index[:, i*n_edges: (i+1)*n_edges],
                            edge_attr=e.unsqueeze(-1)[i*n_edges: (i+1)*n_edges, :]
                        ) for i in range(bs)]
            batch = Graph.to_pyg_batch(graph_list)
            emb0_batch = self.model.emb_net(batch.x, batch.edge_index, batch.edge_attr)
            emb0_list = emb0_batch.split([graph.n_edges for graph in graph_list], dim = 0)
            emb0_grads = []
            # fine-tune emb and mlp
            phi_grad_lists = []
            for phi in self.model.par_net.trainables():
                phi_grad_lists.append([])
            for i, (graph, emb0) in enumerate(zip(graph_list, emb0_list)):
                emb1, psi_net, ys, _ = meta_update_sparse(emb0, self.model.par_net, graph, self.inner_optimizer, \
                                                          self.train_inner_steps, self.train_inner_samples, greedy_size=None)
                emb0_grad, phi_grads = net_approx_grads(emb1, psi_net, graph, sample_size = self.train_inner_samples)
                emb0_grads.append(emb0_grad)
                for phi_grad_list, phi_grad in zip(phi_grad_lists, phi_grads):
                    phi_grad_list.append(phi_grad)
            opt.zero_grad()
            emb0_batch.backward((torch.cat(emb0_grads, dim = 0) / self.env.train_batch_size).detach()) # mean
            for phi, phi_grad_list in zip(self.model.par_net.trainables(), phi_grad_lists):
                torch_add_grad(phi, torch.stack(phi_grad_list, dim = 0).mean(dim = 0).detach())
            opt.step()
            del graph_list, batch, emb0_batch, emb0_list, emb0_grads, phi_grad_lists, \
                phi, graph, emb0, emb1, psi_net, ys, _, emb0_grad, phi_grads, phi_grad_list
            gc.collect()

            if step % self.save_n_epochs == 0:
                val_costs = []
                self.model.eval()
                # self.model.set_batch_size(self.env.val_batch_size)
                for idx in trange(len(self.env.val_dataset), desc="validation"):
                    val_data = self.env.generate_val_data(idx)
                    _, x, e, edge_index, _, _, _, _ = val_data
                    graph = Graph(x=x, edge_index=edge_index, edge_attr=e.unsqueeze(-1))
                    with torch.no_grad():
                        emb0 = self.model.emb_net(graph.x, graph.edge_index, graph.edge_attr)
                    emb1, psi_net, _, _ = meta_update_sparse(emb0, self.model.par_net, graph, self.inner_optimizer, \
                                                            self.val_inner_steps, sample_size=self.train_inner_samples)
                    with torch.no_grad():
                        psi_net.eval()
                        heatmap = psi_net(emb1.detach())
                        cost = self.decode(heatmap, val_data)
                        val_costs.append(cost)
                cost_avg = sum(val_costs) / len(val_costs)
                tbar.set_description(f"validation cost: {cost_avg:.5f}")
                print("\n")
                if cost_avg < best_val_cost:
                    torch.save(self.model.state_dict(), self.save_path.format(step, cost_avg))
                    best_val_cost = cost_avg
                self.model.train()

    def train_dense(self):
        self.model.train()
        self.model.set_batch_size(self.env.train_batch_size)
        opt = self.outer_optimizer(self.model.parameters())
        tbar = trange(1, self.train_outer_steps + 1)
        best_val_cost = 1e5
        for step in tbar:
            opt.zero_grad()
            _, x, adj, _, _ = self.env.generate_train_data(self.env.train_batch_size)
            par0 = self.model(x, adj)
            par0: Tensor
            par1 = meta_update_dense(
                adj, 
                par0, 
                self.inner_optimizer, 
                self.train_inner_steps, 
                self.train_inner_samples
            )
            par0.backward(par1.grad / self.env.train_batch_size)
            opt.step()
            if step % self.save_n_epochs == 0:
                val_costs = []
                self.model.eval()
                self.model.set_batch_size(self.env.val_batch_size)
                for idx in trange(len(self.env.val_dataset), desc="validation"):
                    val_data = self.env.generate_val_data(idx)
                    _, x, adj, _, _ = val_data
                    with torch.no_grad():
                        par0 = self.model.forward(x, adj)
                    heatmap = meta_update_dense(adj, par0, self.inner_optimizer, steps=1, samples=100)
                    with torch.no_grad():
                        cost = self.decode(heatmap, val_data)
                        val_costs.append(cost)
                cost_avg = sum(val_costs) / len(val_costs)
                tbar.set_description(f"validation cost: {cost_avg:.5f}")
                print("\n")
                if cost_avg < best_val_cost:
                    torch.save(self.model.state_dict(), self.save_path.format(step, cost_avg))
                    best_val_cost = cost_avg
                self.model.train()
                self.model.set_batch_size(self.env.train_batch_size)

    
    def decode(self, heatmap, batch_data):
        if self.env.sparse:
            costs_avg = self.decoder.sparse_decode(heatmap, *batch_data, return_cost=True)
        else:
            costs_avg = self.decoder.dense_decode(heatmap, *batch_data, return_cost=True)
        return costs_avg