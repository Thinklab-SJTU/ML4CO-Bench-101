import numpy as np
import torch
from torch_geometric.utils import remove_self_loops
from torch_scatter import scatter_add
from .gnn_encoder import GNNEncoder
from .utils import *


class MetaLearner(object):
    def __init__(
        self, 
        model: GNNEncoder,
        loss_function,
        inner_lr: float = 1e-3,
        outer_lr: float = 1e-2,
        device: str = "cuda",
        inner_alg: str = 'gradient',
        outer_alg: str = 'adam',
        task: str = None
    ):
        self.model = model
        self.device = device
        self.model.to(self.device)
        assert outer_alg == 'sgd' or 'adam'
        self.inner_opt = torch.optim.Adam(self.model.parameters(), lr=inner_lr, eps=1e-3)
        if outer_alg == 'adam':
            self.outer_opt = torch.optim.Adam(self.model.parameters(), lr=outer_lr, eps=1e-3)
        else:
            self.outer_opt = torch.optim.SGD(self.model.parameters(), lr=outer_lr)
        self.loss_function = loss_function
        assert inner_alg == 'gradient' # sqp unsupported in this version
        self.inner_alg = inner_alg
        self.task = task

    def get_params(self):
        return torch.cat([param.data.view(-1) for param in self.model.parameters()], 0).clone()

    def set_params(self, param_vals):
        offset = 0
        for param in self.model.parameters():
            param.data.copy_(param_vals[offset:offset + param.nelement()].view(param.size()))
            offset += param.nelement()
            
    def set_outer_lr(self, lr):
        for param_group in self.outer_opt.param_groups:
            param_group['lr'] = lr
            
    def set_inner_lr(self, lr):
        for param_group in self.inner_opt.param_groups:
            param_group['lr'] = lr

    def regularization_loss(self, w_0, lam=0.0):
        """
        Add a regularization loss onto the weights
        The proximal term regularizes around the point w_0
        Strength of regularization is lambda
        lambda can either be scalar (type float) or ndarray (numpy.ndarray)
        """
        regu_loss = 0.0
        offset = 0
        regu_lam = lam if type(lam) == float or np.float64 else to_tensor(lam)
        if w_0.dtype == torch.float16:
            try:
                regu_lam = regu_lam.half()
            except:
                regu_lam = np.float16(regu_lam)
        for param in self.model.parameters():
            delta = param.view(-1) - w_0[offset:offset + param.nelement()].view(-1)
            if type(regu_lam) == float or np.float64:
                regu_loss += 0.5 * regu_lam * torch.sum(delta ** 2)
            else:
                param_lam = regu_lam[offset:offset + param.nelement()].view(-1)
                param_delta = delta * param_lam
                regu_loss += 0.5 * torch.sum(param_delta ** 2)
            offset += param.nelement()
        return regu_loss

    def get_loss(self, x, edge_index, batch, penalty_coefficient, device):
        probs = self.model.forward(x, edge_index, batch)
        if self.task == "MCl":
            loss = self.erdosloss_mcl(probs, edge_index, batch, penalty_coefficient, device)
        elif self.task == "MVC":
            loss = self.erdosloss_mvc(probs, edge_index, batch, penalty_coefficient, device)
        elif self.task == "MIS":
            loss = self.erdosloss_mis(probs, edge_index, batch, penalty_coefficient, device)
        elif self.task == "MCut":
            loss = self.erdosloss_mcut(probs, edge_index, batch, penalty_coefficient, device)
        else:
            raise NotImplementedError()
        return loss

    def predict(self, x, edge_index, batch, return_numpy=False):
        x.to(self.device)
        edge_index.to(self.device)
        batch.to(self.device)
        probs = self.model.forward(x, edge_index, batch)
        if return_numpy:
            probs = to_numpy(probs)
        return probs

    def learn_on_data(self, x, edge_index, batch, penalty_coefficient, device, num_steps=10,
                      add_regularization=False, w_0=None, lam=0.0):
        assert self.inner_alg == 'gradient'# or 'sqp' or 'adam'
        train_loss = []
        if self.inner_alg == 'gradient':
            for i in range(num_steps):
                self.inner_opt.zero_grad()
                tloss = self.get_loss(x, edge_index, batch, penalty_coefficient, device)
                loss = tloss + self.regularization_loss(w_0, lam) if add_regularization else tloss
                loss.backward()
                self.inner_opt.step()
                train_loss.append(to_numpy(tloss))
        return train_loss
    
    def learn_task(self, x, edge_index, batch, penalty_coefficient, device, num_steps=10, add_regularization=False, w_0=None, lam=0.0):

        return self.learn_on_data(x, edge_index, batch, penalty_coefficient, device, num_steps, add_regularization, w_0, lam)

    def move_toward_target(self, target, lam=2.0):
        """
        Move slowly towards the target parameter value
        Default value for lam assumes learning rate determined by optimizer
        Useful for implementing Reptile
        """
        # we can implement this with the regularization loss, but regularize around the target point
        # and with specific choice of lam=2.0 to preserve the learning rate of inner_opt
        self.outer_opt.zero_grad()
        loss = self.regularization_loss(target, lam=lam)
        loss.backward()
        self.outer_opt.step()

    def outer_step_with_grad(self, grad, flat_grad=False):
        """
        Given the gradient, step with the outer optimizer using the gradient.
        Assumed that the gradient is a tuple/list of size compatible with model.parameters()
        If flat_grad, then the gradient is a flattened vector
        """
        check = 0
        for p in self.model.parameters():
            check = check + 1 if type(p.grad) == type(None) else check
        if check > 0:
            # initialize the grad fields properly
            dummy_loss = self.regularization_loss(self.get_params())
            dummy_loss.backward()  # this would initialize required variables
        if flat_grad:
            offset = 0
            grad.to(self.device)
            for p in self.model.parameters():
                this_grad = grad[offset:offset + p.nelement()].view(p.size())
                p.grad.copy_(this_grad)
                offset += p.nelement()
        else:
            for i, p in enumerate(self.model.parameters()):
                p.grad = grad[i]
        self.outer_opt.step()

    def matrix_evaluator(self, x, edge_index, batch, lam, penalty_coefficient, device, regu_coef=1.0, lam_damping=10.0):
        """
        Constructor function that can be given to CG optimizer
        Works for both type(lam) == float and type(lam) == np.ndarray
        """
        if type(lam) == np.ndarray:
            lam.to(self.device)
        def evaluator(v):
            hvp = self.hessian_vector_product(x, edge_index, batch, v, penalty_coefficient, device)
            Av = (1.0 + regu_coef) * v + hvp / (lam + lam_damping)
            return Av
        return evaluator

    def hessian_vector_product(self, x, edge_index, batch, vector, penalty_coefficient, device, params=None):
        """
        Performs hessian vector product on the train set in task with the provided vector
        """
        if params is not None:
            self.set_params(params)
        tloss = self.get_loss(x, edge_index, batch, penalty_coefficient, device)
        grad_ft = torch.autograd.grad(tloss, self.model.parameters(), create_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_ft])
        vec = vector.to(self.device)
        h = torch.sum(flat_grad * vec)
        hvp = torch.autograd.grad(h, self.model.parameters())
        hvp_flat = torch.cat([g.contiguous().view(-1) for g in hvp])
        return hvp_flat
    
    def erdosloss_mcl(self, probs, edge_index, batch, penalty_coefficient, device):
        num_graphs = batch.max().item() + 1
        no_loop_index, _ = remove_self_loops(edge_index)  
        no_loop_row, no_loop_col = no_loop_index
        pairwise_prodsums = torch.zeros(num_graphs, device = str(device))
        for graph in range(num_graphs):
            batch_graph = (batch == graph)
            pairwise_prodsums[graph] = (torch.conv1d(probs[batch_graph].unsqueeze(-1), probs[batch_graph].unsqueeze(-1))).sum() / 2.
        self_sums = scatter_add((probs * probs), batch, 0, dim_size = num_graphs)
        expected_weight_G = scatter_add(probs[no_loop_row] * probs[no_loop_col], batch[no_loop_row], 0, dim_size = num_graphs) / 2.
        expected_clique_weight = (pairwise_prodsums.unsqueeze(-1) - self_sums) / 1.
        expected_distance = (expected_clique_weight - expected_weight_G)       
        expected_loss = (penalty_coefficient) * expected_distance * 0.5 - 0.5 * expected_weight_G  
        loss = expected_loss
        loss = loss.mean().squeeze()
        return loss
    
    def erdosloss_mvc(self, probs, edge_index, batch, penalty_coefficient, device):
        num_graphs = batch.max().item() + 1
        no_loop_index, _ = remove_self_loops(edge_index)  
        no_loop_row, no_loop_col = no_loop_index
        probs_sum = torch.zeros(num_graphs, device=device)
        for graph in range(num_graphs):
            batch_graph = (batch == graph)
            probs_sum[graph] = probs[batch_graph].unsqueeze(-1).sum()
        vertex_row = probs[no_loop_row]
        vertex_col = probs[no_loop_col]
        expected_distance = (1 - vertex_row) * (1 - vertex_col)
        expected_distance = expected_distance.sum() / num_graphs
        expected_weight = probs_sum.mean()
        loss = penalty_coefficient * expected_distance + expected_weight
        loss = loss.squeeze()
        return loss
    
    def erdosloss_mis(self, probs, edge_index, batch, penalty_coefficient, device):
        num_graphs = batch.max().item() + 1
        no_loop_index, _ = remove_self_loops(edge_index)  
        no_loop_row, no_loop_col = no_loop_index
        probs_sum = torch.zeros(num_graphs, device=device)
        for graph in range(num_graphs):
            batch_graph = (batch == graph)
            probs_sum[graph] = probs[batch_graph].unsqueeze(-1).sum()
        vertex_row = probs[no_loop_row]
        vertex_col = probs[no_loop_col]
        expected_distance = vertex_row * vertex_col
        expected_distance = expected_distance.sum() / num_graphs
        expected_weight = probs_sum.mean()
        loss = penalty_coefficient * expected_distance - expected_weight
        loss = loss.squeeze()
        return loss
    
    def erdosloss_mcut(self, probs, edge_index, batch, penalty_coefficient, device):
        num_graphs = batch.max().item() + 1
        no_loop_index, _ = remove_self_loops(edge_index)  
        no_loop_row, no_loop_col = no_loop_index

        vertex_row = probs[no_loop_row]
        vertex_col = probs[no_loop_col]

        expected_cut = (vertex_row * (1 - vertex_col) + (1 - vertex_row) * vertex_col).sum() / num_graphs
        
        loss = -penalty_coefficient * expected_cut
        
        loss = loss.squeeze()
        return loss