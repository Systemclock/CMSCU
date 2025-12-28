import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import itertools
class SpectralNetLoss(nn.Module):
    def __init__(self):
        super(SpectralNetLoss, self).__init__()

    def forward(self, W, Y, w, Y1, options, device, is_normalized: bool = True):
        alpha = options['alpha']
        beta = options['beta']
        gamma = options['gamma']
        tf = options['tf']
        loss1 = 0.0
        
        n = W[0].size(0)
        view_size = len(W)
        # normalized is better ?
        for v in range(view_size):
            if is_normalized:
                D_v = torch.sum(W[v], dim=1)
                Y[v] = Y[v] / D_v[:, None]
            Dy_v = torch.cdist(Y[v], Y[v]) ** 2
            loss_t = (W[v] * Dy_v).mean() * n
            loss1 = loss1 + loss_t
            # loss1 = loss1 + w[v] * loss_t

        # Attention loss for gpu
        H_v = Y
        K_v = []
        H_f = torch.zeros(Y[0].shape).to(device=device)
        K_c = torch.zeros(Y[0].shape[0], Y[0].shape[0]).to(device=device)
        for v in range(view_size):
            H_f = H_f + w[v] * Y[v]
            K_v.append(w[v] * self.gaussian_kernel(H_v[v], device=device))
            K_c = K_c + K_v[v]

        K_f = self.gaussian_kernel(H_f, device=device)
        loss3 = torch.norm((K_f - K_c), 'fro').pow(2)

        loss2 = 0.0
        # Constrastive Loss
        Affinity = W[0].repeat(2, 2)
        ls = itertools.combinations(range(len(W)), 2)
        for (view_i, view_j) in ls:
            loss2 = loss2 + self.constrastiveloss(Y1[view_i], Y1[view_j], n, tf, Affinity, device)

        loss = alpha * loss1 + beta * loss2

        loss = loss + gamma * loss3
        return loss

    def gaussian_kernel(self, H, device):

        D = torch.cdist(H, H)
        W = torch.exp(
            -1 * 21.5 * torch.pow(D, 2).to(device)

        )
        return W

    def constrastiveloss(self, emb_i, emb_j, n, tf, W, device):

        # print(emb_i.shape,emb_j.shape)
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        sim_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(sim_matrix, n)
        sim_ji = torch.diag(sim_matrix, -n)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        temperature = tf

        nominator = torch.exp(positives / temperature)
        rnegatives_mask = (~torch.eye(n*2, n*2, dtype=bool)).to(device)

        weighted_sim_matrix = sim_matrix * rnegatives_mask
        weighted_sim_matrix = weighted_sim_matrix + (~rnegatives_mask) * W
        denominator = torch.exp(weighted_sim_matrix / temperature)

        # denominator = rnegatives_mask * torch.exp((torch.ones(2*n, 2*n).to(device) - W) * sim_matrix / temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * n)

        return loss




