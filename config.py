from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AutoEncoderConfig:

    hiddens: List[int] = None
    
    epochs: int = 500
    lr: float = 1e-3
    lr_decay: float = 0.1
    min_lr: float = 1e-7
    patience: int = 10
    batch_size: int = 256
    
    
    weights_path: str = "weights/ae_weights"
    
    def __post_init__(self):

        if self.hiddens is None:
            self.hiddens = [512, 512, 2048, 10]


@dataclass
class SpectralNetConfig:
    
    epochs: int = 200
    lr: float = 1e-4
    lr_decay: float = 0.1
    min_lr: float = 1e-8
    patience: int = 10
    batch_size: int = 512
    
    
    n_nbg: int = 5  
    scale_k: int = 15  
    is_local_scale: bool = True  
    

    architecture: List[int] = None
    
    def __post_init__(self):
        if self.architecture is None:
            self.architecture = [2048, 1024, 512]


@dataclass
class ModelOptions:
    nbg: int = 5  
    t: float = 0.3  
    tf: float = 0.7  
    alpha: float = 1.0  # sepctralclu loss weight
    beta: float = 0.1  # constrastive loss weight
    gamma: float = 0.1  # fusion loss weight


@dataclass
class TrainingConfig:
    
    # data config
    data_path: str = "data/handwritten_6viewsy.mat"
    
    num_runs: int = 1  # iteration num
    seed: int = 10  # seed
    
    ae_config: AutoEncoderConfig = None
    spectral_config: SpectralNetConfig = None
    model_options: ModelOptions = None
    
    def __post_init__(self):
        
        if self.ae_config is None:
            self.ae_config = AutoEncoderConfig()
        if self.spectral_config is None:
            self.spectral_config = SpectralNetConfig()
        if self.model_options is None:
            self.model_options = ModelOptions()

