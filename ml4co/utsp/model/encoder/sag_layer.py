import torch
import torch.nn as nn
import torch.nn.functional as F


def GCN_diffusion(
    W: torch.Tensor, 
    order: int, 
    feature: torch.Tensor, 
):
    """
    W: [batchsize,n,n]
    feature: [batchsize,n,n]
    """
    identity_matrices = torch.eye(W.size(1)).repeat(W.size(0), 1, 1)
    I_n = identity_matrices.to(W.device)
    A_gcn = W + I_n
    degrees = torch.sum(A_gcn, 2)
    degrees = degrees.unsqueeze(dim=2)
    D = degrees
    D = torch.pow(D, -0.5)
    gcn_diffusion_list = []
    A_gcn_feature = feature
    
    for _ in range(order):
        A_gcn_feature = D*A_gcn_feature
        A_gcn_feature = torch.matmul(A_gcn, A_gcn_feature)
        A_gcn_feature = torch.mul(A_gcn_feature, D)
        gcn_diffusion_list += [A_gcn_feature, ]

    return gcn_diffusion_list


def scattering_diffusionS4(sptensor: torch.Tensor, feature: torch.Tensor):
    '''
    A_tilte,adj_p,shape(N,N)
    feature:shape(N,3) :torch.FloatTensor
    all on cuda
    '''
    h_sct1, h_sct2, h_sct3, h_sct4 = SCT1stv2(sptensor, 4, feature)

    return h_sct1, h_sct2, h_sct3, h_sct4


def SCT1stv2(W,order: torch.Tensor, feature: torch.Tensor):
    '''
    W = [b,n,n]
    '''
    degrees = torch.sum(W,2)
    D = degrees
    D = torch.pow(D, -1)
    D = D.unsqueeze(dim=2)
    iteration = 2**order
    scale_list = list(2**i - 1 for i in range(order+1))
    feature_p = feature
    sct_diffusion_list = []
    for i in range(iteration):
        D_inv_x = D*feature_p
        W_D_inv_x = torch.matmul(W, D_inv_x)
        feature_p = 0.5*feature_p + 0.5*W_D_inv_x
        if i in scale_list:
            sct_diffusion_list += [feature_p, ]

    sct_feature1 = sct_diffusion_list[0] - sct_diffusion_list[1]
    sct_feature2 = sct_diffusion_list[1] - sct_diffusion_list[2]
    sct_feature3 = sct_diffusion_list[2] - sct_diffusion_list[3]
    sct_feature4 = sct_diffusion_list[3] - sct_diffusion_list[4]

    return sct_feature1, sct_feature2, sct_feature3, sct_feature4


class SCTConv(torch.nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hid = hidden_dim
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.a = nn.Parameter(torch.zeros(size=(2 * hidden_dim, 1)))

    def forward(
        self, 
        X: torch.Tensor, 
        adj: torch.Tensor, 
        moment: int = 1, 
    ) -> torch.Tensor:
        """
        Params
        ------
        adj [batch x nodes x nodes]: adjacency matrix
        X [batch x nodes x features]: node features matrix
        Returns
        -------
        X' [batch x nodes x features]: updated node features matrix
        """
        support0 = X
        B = support0.size(0)
        N = support0.size(1)
        h = support0
        
        gcn_diffusion_list = GCN_diffusion(adj, 3, support0)
        
        h_A = gcn_diffusion_list[0]
        h_A2 = gcn_diffusion_list[1]
        h_A3 = gcn_diffusion_list[2]
        
        h_A = nn.LeakyReLU()(h_A)
        h_A2 = nn.LeakyReLU()(h_A2)
        h_A3 = nn.LeakyReLU()(h_A3)
        
        h_sct1, h_sct2, h_sct3, h_sct4 = scattering_diffusionS4(adj, support0)
        h_sct1 = torch.abs(h_sct1) ** moment
        h_sct2 = torch.abs(h_sct2) ** moment
        h_sct3 = torch.abs(h_sct3) ** moment
        h_sct4 = torch.abs(h_sct4) ** moment
        
        a_input_A = torch.cat((h, h_A), dim=2).unsqueeze(1)
        a_input_A2 = torch.cat((h, h_A2), dim=2).unsqueeze(1)
        a_input_A3 = torch.cat((h, h_A3), dim=2).unsqueeze(1)
        
        a_input_sct1 = torch.cat((h, h_sct1), dim=2).unsqueeze(1)
        a_input_sct2 = torch.cat((h, h_sct2), dim=2).unsqueeze(1)
        a_input_sct3 = torch.cat((h, h_sct3), dim=2).unsqueeze(1)
        a_input_sct4 = torch.cat((h, h_sct4), dim=2).unsqueeze(1)
        
        a_input = torch.cat(
           (a_input_A, 
            a_input_A2,
            a_input_sct1, 
            a_input_sct2, 
            a_input_sct3,
            a_input_sct4
           ), 1
        ).view(B, 6, N, -1)
        e = torch.matmul(nn.functional.relu(a_input), self.a).squeeze(3)

        attention = F.softmax(e, dim=1).view(B,6, N, -1)

        h_all = torch.cat(
           (h_A.unsqueeze(dim=1), 
            h_A2.unsqueeze(dim=1),
            h_sct1.unsqueeze(dim=1),
            h_sct2.unsqueeze(dim=1),
            h_sct3.unsqueeze(dim=1),
            h_sct4.unsqueeze(dim=1))
            , dim=1
        ).view(B, 6,N,-1)

        h_prime = torch.mul(attention, h_all)
        h_prime = torch.mean(h_prime, 1) # (B,n,f)
        X = self.linear1(h_prime)
        X = F.leaky_relu(X)
        X = self.linear2(X)
        X = F.leaky_relu(X)
        return X