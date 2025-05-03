import torch
from torch import nn
import torch.nn.functional as F

def tsp_sample(adj, ze, mode='softmax', samples=1, epsilon=0.): # epsilon exploration
    device = adj.device
    assert mode in ['softmax', 'greedy']
    if mode == 'greedy':
        assert samples == 1
    batch_size, n_nodes, _ = adj.shape
    zex = ze.expand((samples, batch_size, n_nodes, n_nodes))
    adj_flat = adj.view(batch_size, n_nodes * n_nodes).expand((samples, batch_size, n_nodes * n_nodes))
    idx = torch.arange(n_nodes).expand((samples, batch_size, n_nodes)).to(device)
    mask = torch.ones((samples, batch_size, n_nodes), dtype = torch.bool).to(device)
    maskFalse = torch.zeros((samples, batch_size, 1), dtype = torch.bool).to(device)
    v0 = u = torch.zeros((samples, batch_size, 1), dtype = torch.long).to(device) # starts from v0:=0
    mask.scatter_(dim = -1, index = u, src = maskFalse)
    y = []
    if mode == 'softmax':
        logp, logq = [], []
    else:
        sol = [u]
    for i in range(1, n_nodes):
        zei = zex.gather(dim = -2, index = u.unsqueeze(dim = -1).expand((samples, batch_size, 1, n_nodes))).squeeze(dim = -2).masked_select(mask.clone()).view(samples, batch_size, n_nodes - i)
        if mode == 'softmax':
            pei = F.softmax(zei, dim = -1)
            qei = epsilon / (n_nodes - i) + (1. - epsilon) * pei
            vi = qei.view(samples * batch_size, n_nodes - i).multinomial(num_samples = 1, replacement = True).view(samples, batch_size, 1)
            logp.append(torch.log(pei.gather(dim = -1, index = vi)))
            logq.append(torch.log(qei.gather(dim = -1, index = vi)))
        elif mode == 'greedy':
            vi = zei.argmax(dim = -1, keepdim = True)
        v = idx.masked_select(mask).view(samples, batch_size, n_nodes - i).gather(dim = -1, index = vi)
        y.append(adj_flat.gather(dim = -1, index = u * n_nodes + v))
        u = v
        mask.scatter_(dim = -1, index = u, src = maskFalse)
        if mode == 'greedy':
            sol.append(u)
    y.append(adj_flat.gather(dim = -1, index = u * n_nodes + v0)) # ends at node v0
    y = torch.cat(y, dim = -1).sum(dim = -1).T # (batch_size, samples)
    if mode == 'softmax':
        logp = torch.cat(logp, dim = -1).sum(dim = -1).T
        logq = torch.cat(logq, dim = -1).sum(dim = -1).T
        return y, logp, logq # (batch_size, samples)
    elif mode == 'greedy':
        lens = y.squeeze(dim = 1), torch.cat(sol, dim = -1).squeeze(dim = 0) # (batch_size,)
        sols = sol
        return lens, sols

def tsp_greedy(adj, ze):
    return tsp_sample(adj, ze, mode='greedy') # y, sol

def tsp_optim(adj, ze0, opt_fn, steps, samples, epsilon=0.):
    device = adj.device
    batch_size, n_nodes, _ = adj.shape
    ze = nn.Parameter(ze0.to(device), requires_grad = True)
    opt = opt_fn([ze])
    y_means = []
    tbar = range(1, steps + 1)
    y_bl = torch.zeros((batch_size, 1)).to(device)
    for t in tbar:
        opt.zero_grad()
        y, logp, logq = tsp_sample(adj, ze, 'softmax', samples, epsilon)
        y_means.append(y.mean().item())
        y_bl = y.mean(dim = -1, keepdim = True)
        J = (((y - y_bl) * torch.exp(logp - logq)).detach() * logp).mean(dim = -1).sum()
        J.backward()
        opt.step()
    return ze
