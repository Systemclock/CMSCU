import itertools
import numpy as np
import torch
import torch.nn as nn
import math

class MSCN(nn.Module):
    def __init__(self, input_list, data_list, n_clusters, view_size, device,
                 # arch: list = [256, 512, 1024, 5],
                 arch: list = [2048, 1024, 512],
                 epoches: int = 30,
                 lr: float = 1e-4,
                 lr_decay: float = 0.1,
                 min_lr: float = 1e-8,
                 patience: int = 10,
                 batch_size: int = 128):
        super(MSCN, self).__init__()
        self.input_list = input_list
        self.n_clusters = n_clusters
        self.view_size = view_size
        self.epoches = epoches
        self.lr = lr
        self.lr_decay = lr_decay
        self.min_lr = min_lr
        self.patience = patience
        self.batch_size = batch_size
        self.architecture = arch
        self.device = device
        self.layers = nn.ModuleList()
        self.orthonorm_weights = torch.rand(self.n_clusters, self.n_clusters)
        self.orthonorm_weights.requires_grad = False

        # input dim
        if isinstance(self.input_list, (list, tuple)):
            input_dim = self.input_list[0]
        else:
            input_dim = self.input_list
        current_dim = int(input_dim)


        for hidden_dim in self.architecture:
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            self.layers.append(nn.ReLU())  # ReLU activation after each hidden layer
            current_dim = hidden_dim  # Update current dimension for the next layer

        # output dim: n_clusters
        self.layers.append(nn.Linear(current_dim, self.n_clusters))
        self.layers.append(nn.Tanh())  # Tanh activation for the final layer


    def _make_orthonorm_weights(self, Y):

        m = Y.shape[0]
        x_2 = Y.T @ Y   # n_cluster * n_cluster
        x_2 = x_2 + (torch.eye(x_2.shape[0]) * 1e-4).to(self.device)
        L = torch.linalg.cholesky(x_2)
        L_inv = torch.linalg.inv(L)
        orthonorm_weights = np.sqrt(m) * L_inv.T

        return orthonorm_weights

    def forward(self, x, should_update_orth_weights: bool = True):

        for layer in self.layers:
            x = layer(x)
        # x = x.to(self.device)
        Y_tilde = x
        if should_update_orth_weights:  # default
            self.orthonorm_weights = self._make_orthonorm_weights(Y_tilde).to(self.device)

        Y = Y_tilde @ self.orthonorm_weights

        return Y


class Attention(nn.Module):
    def __init__(self, n_cluster, view_size, batch_size, options, device):
        super(Attention, self).__init__()
        self.n_cluster = n_cluster
        self.view_size = view_size
        self.batch_size = batch_size
        self.device = device
        self.options = options
        # input data Yn batch_size * c * v
        input_dim = self.n_cluster*self.view_size
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        out_dim = self.view_size
        self.w = torch.ones(self.view_size) / self.view_size
        # self.w = nn.Parameter(torch.full((self.view_size,), 1 / self.view_size), requires_grad=True)
        self.layers = nn.ModuleList()  # 4096 1000
        self.layers.append(nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU()))
        self.layers.append(nn.Sequential(nn.Linear(64, 256), nn.ReLU()))
        self.layers.append(nn.Sequential(nn.Linear(256, out_dim), nn.ReLU()))

    def forward(self, x):

        for layer in self.layers:
            x = layer(x).to(device=self.device)
        e_t = torch.sigmoid(x)
        t = self.options['t']
        e = torch.softmax(e_t/t, dim=1)
        self.w = torch.mean(e, dim=0)
        weight = self.w
        return weight

class ProjectionHead(nn.Module):
    def __init__(self, n_cluster):
        super(ProjectionHead, self).__init__()
        input_dim = n_cluster
        output_dim = input_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.layers(x)
























