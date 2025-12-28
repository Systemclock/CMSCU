import numpy as np
import scipy
import torch
import torch.optim as optim
from matplotlib.ticker import MultipleLocator
from tqdm import trange
import matplotlib.pyplot as plt
import Clustering
import data
import random
import network
import utils
import netloss
from utils import get_nearest_neighbors, compute_scale, get_gaussian_kernel

class SepctralTrainer:
    def __init__(self, input_list, data_list, n_clusters, view_size,  options, config: dict, device: torch.device):
        self.device = device
        self.input_list = input_list
        self.data_list = data_list
        self.n_clusters = n_clusters
        self.view_size = view_size
        self.spectral_config = config
        self.options = options
        self.lr = self.spectral_config["lr"]
        self.min_lr = self.spectral_config["min_lr"]
        self.epochs = self.spectral_config["epochs"]
        self.lr_decay = self.spectral_config["lr_decay"]
        self.patience = self.spectral_config["patience"]
        self.batch_size = self.spectral_config["batch_size"]
        self.n_nbg = self.spectral_config["n_nbg"]
        self.scale_k = self.spectral_config["scale_k"]
        self.is_local_scale = self.spectral_config["is_local_scale"]

    def train(self, X, y):
        self.X = X
        self.y = y
        self.criterion = netloss.SpectralNetLoss()


        self.spectralnet = []
        self.optimizer = []
        self.scheduler = []
        self.Projectmlp = []
        self.Projectopt = []
        self.Projectsch = []
        
        for v in range(self.view_size):
            # Create SpectralNet
            self.spectralnet.append(
                network.MSCN(input_list=self.input_list[v],
                             data_list=self.data_list[v],
                             n_clusters=self.n_clusters,
                             view_size=self.view_size,
                             device=self.device
                             ).to(self.device)
            )

            self.optimizer.append(
                optim.Adam(self.spectralnet[v].parameters(), lr=self.lr, weight_decay=0.01)
            )
            self.scheduler.append(
                optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer[v], mode="min", factor=self.lr_decay, patience=self.patience
                )
            )
         
        self.Attnet = network.Attention(
            n_cluster=self.n_clusters, 
            view_size=self.view_size, 
            batch_size=self.batch_size, 
            options=self.options, 
            device=self.device
        ).to(device=self.device)
        self.Attnetopt = optim.Adam(self.Attnet.parameters(), lr=1e-4, weight_decay=0.01)
        self.Attnetsche = optim.lr_scheduler.ReduceLROnPlateau(
            self.Attnetopt, mode="min", factor=self.lr_decay, patience=self.patience
        )


        print("Training SepctralNet:")
        t = trange(self.epochs, leave=True)

        Losses = []
        MACC = []
        MNMI = []
        MF = []
        MACC.append(0)
        MNMI.append(0)
        MF.append(0)
        train_data = self.X
        train_label = self.y
        batch_size = min(len(train_data[0]), self.batch_size)
        batches = utils.make_batches(len(train_data[0]), batch_size)

        for epoch in t:
            train_loss = 0.0
            embeddings = [[] for _ in range(self.view_size)]
            
            for j, (batch_start, batch_end) in enumerate(batches):
                # ========== pro ==========
                train_Y = []
                train_W = []
                train_X = []
                emb_t = []
                
                for v in range(self.view_size):
                    x_train = train_data[v][batch_start:batch_end].to(device=self.device)
                    train_X.append(x_train)
                    x_orth = x_train
                    
                    # orth step
                    self.spectralnet[v].eval()
                    self.spectralnet[v](x_orth, should_update_orth_weights=True)
                    
                    # pro step
                    self.spectralnet[v].train()
                    Y_t = self.spectralnet[v](x_train, should_update_orth_weights=False)
                    emb_t.append(Y_t.detach().cpu().numpy())
                    train_Y.append(Y_t)
                
                # compute similarity
                w_x_train = torch.concat(train_X, dim=1)
                train_w_tmp = utils._get_affinity_matrix(w_x_train, self.options['nbg']).to(device=self.device)
                for v in range(self.view_size):
                    train_W.append(train_w_tmp)
                

                y_concat = torch.cat(train_Y[:], dim=1).to(device=self.device)
                w = self.Attnet(y_concat).to(device=self.device)
                

                loss = self.criterion(train_W, train_Y, w, train_Y, self.options, self.device)
                

                for v in range(self.view_size):
                    self.optimizer[v].zero_grad()
                self.Attnetopt.zero_grad()

                loss.backward()

                for v in range(self.view_size):
                    self.optimizer[v].step()
                self.Attnetopt.step()
                
                train_loss += loss.item()

            train_loss /= train_data[0].size()[0]

            for v in range(self.view_size):
                self.scheduler[v].step(train_loss)
            self.Attnetsche.step(train_loss)

            current_lr = self.optimizer[0].param_groups[0]["lr"]
            if current_lr <= self.spectral_config["min_lr"]:
                break

            t.set_description(
                "Train Loss: {:.7f}, LR: {:.6f}".format(
                    train_loss, current_lr
                )
            )
            t.refresh()

        w = w.cpu().detach().numpy()

        return self.spectralnet, w



    def predict(self, X, y):
        #
        for v in range(self.view_size):
            X[v] = X[v].to(self.device)

        self.embeddings = []

        train_data = X
        batch_size = min(len(train_data[0]), self.batch_size)
        batches = utils.make_batches(len(train_data[0]), batch_size)

        with torch.no_grad():

            embedings = []
            for v in range(self.view_size):

                embeddings_ = self.spectralnet[v](train_data[v], should_update_orth_weights=False)

                embedings.append(embeddings_)
                embeddings_ = embeddings_.detach().cpu().numpy()

                self.embeddings.append(embeddings_)

            y_ = torch.cat(embedings[:], dim=1)
            w = self.Attnet(y_)
            w = w.cpu().detach().numpy()

        return self.embeddings, w

    def _get_affinity_matrix(self, X):
        is_local = self.is_local_scale
        n_neighbors = self.n_nbg
        scale_k = self.scale_k
        Dx = torch.cdist(X, X)
        Dis, indices = get_nearest_neighbors(X, k=n_neighbors + 1)
        scale = compute_scale(Dis, k=scale_k, is_local=is_local)
        W = get_gaussian_kernel(
            Dx, scale, indices, device=self.device, is_local=is_local
        )

        return W











