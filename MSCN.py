import numpy as np
import torch
from typing import List, Tuple, Dict

import Clustering
import trainer
from ae_trainer import AETrainer
from config import TrainingConfig


class MultiViewSpectralClustering:

    def __init__(self, config: TrainingConfig, device: torch.device):

        self.config = config
        self.device = device
        self.ae_config = config.ae_config
        self.spectral_config = config.spectral_config
        self.model_options = config.model_options
        

        self.ae_trainers: List[AETrainer] = []
        self.ae_nets: List[torch.nn.Module] = []
        self.spectral_trainer: trainer.SepctralTrainer = None
        
    def prepare_data(self, data_list: List[Tuple]) -> Tuple[List[torch.Tensor], torch.Tensor]:

        view_size = len(data_list)
        n_clusters = len(np.unique(data_list[0][1]))

        train_ae_list = []
        for v in range(view_size):
            x_train_v, _, x_test_v, _ = data_list[v]
            train_ae_list.append(torch.cat((x_train_v, x_test_v), axis=0))
        
        return train_ae_list, n_clusters
    
    def train_autoencoders(self, train_ae_list: List[torch.Tensor], n_clusters: int) -> List[torch.Tensor]:

        view_size = len(train_ae_list)
        self.ae_trainers = []
        self.ae_nets = []
        train_list = []

        for v in range(view_size):


            ae_config_dict = {
                "hiddens": self.ae_config.hiddens,
                "epochs": self.ae_config.epochs,
                "lr": self.ae_config.lr,
                "lr_decay": self.ae_config.lr_decay,
                "min_lr": self.ae_config.min_lr,
                "patience": self.ae_config.patience,
                "batch_size": self.ae_config.batch_size
            }
            ae_trainer = AETrainer(
                view=v,
                n_cluster=n_clusters,
                config=ae_config_dict,
                device=self.device
            )
            

            ae_net = ae_trainer.train(train_ae_list[v])

            embedded_features = ae_trainer.embed(train_ae_list[v])
            
            self.ae_trainers.append(ae_trainer)
            self.ae_nets.append(ae_net)
            train_list.append(embedded_features)

        return train_list
    
    def train_spectral_network(self, train_list: List[torch.Tensor], y_true: torch.Tensor, 
                              n_clusters: int, view_size: int) -> None:

        input_shape_list = [train_list[v].shape[1] for v in range(view_size)]
        

        spectral_config_dict = {
            "epochs": self.spectral_config.epochs,
            "lr": self.spectral_config.lr,
            "min_lr": self.spectral_config.min_lr,
            "lr_decay": self.spectral_config.lr_decay,
            "patience": self.spectral_config.patience,
            "batch_size": self.spectral_config.batch_size,
            "n_nbg": self.spectral_config.n_nbg,
            "scale_k": self.spectral_config.scale_k,
            "is_local_scale": self.spectral_config.is_local_scale
        }
        

        options_dict = {
            'nbg': self.model_options.nbg,
            't': self.model_options.t,
            'tf': self.model_options.tf,
            'alpha': self.model_options.alpha,
            'beta': self.model_options.beta,
            'gamma': self.model_options.gamma
        }
        

        self.spectral_trainer = trainer.SepctralTrainer(
            input_list=input_shape_list,
            data_list=train_list,
            n_clusters=n_clusters,
            view_size=view_size,
            options=options_dict,
            config=spectral_config_dict,
            device=self.device
        )
        
        # train
        self.spectral_trainer.train(train_list, y_true)

    
    def evaluate(self, train_list: List[torch.Tensor], y_true: torch.Tensor) -> Tuple[float, float, float, np.ndarray]:
        embeddings_list, view_weights = self.spectral_trainer.predict(train_list, y_true)

        acc, nmi, f_score = Clustering.Clustering(embeddings_list, y_true, view_weights)
        
        print(f"\n results:")
        print(f"  ACC (ACC): {acc:.4f}")
        print(f"  NMI (NMI): {nmi:.4f}")
        print(f"  F (F-score): {f_score:.4f}")
        
        return acc, nmi, f_score, view_weights


def run_net(data_list: List[Tuple], config: TrainingConfig) -> Tuple[float, float, float, np.ndarray]:

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    # model
    model = MultiViewSpectralClustering(config, device)
    
    # dataset
    train_ae_list, n_clusters = model.prepare_data(data_list)
    view_size = len(data_list)
    
    # label
    y_true = torch.cat((data_list[0][1], data_list[0][3]), axis=0)
    
    # train ae
    train_list = model.train_autoencoders(train_ae_list, n_clusters)
    
    # train sepctral
    model.train_spectral_network(train_list, y_true, n_clusters, view_size)
    
    # evaluate
    acc, nmi, f_score, w = model.evaluate(train_list, y_true)
    
    return acc, nmi, f_score, w
