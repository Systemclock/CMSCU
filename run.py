import os
import numpy as np
import pandas as pd
from itertools import product
from datetime import datetime

from data import get_data
import MSCN
from config import TrainingConfig, AutoEncoderConfig, SpectralNetConfig, ModelOptions
from utils import set_seed

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
DATA_PATH = 'data/handwritten_6viewsy.mat'
NUM_RUNS = 1
SEED = 10

MODEL_PARAMS = {
    'nbg': 6,
    't': 0.3,
    'tf': 0.7,
    'alpha': 1,
    'beta': 0.1,
    'gamma': 0.1
}
PARAM_GRID = {
    'nbg': [5],
    't': [0.3],
    'tf': [0.7],
    'alpha': [1],
    # 'beta': [1],
    # 'gamma': [0.1]
    'beta': [0.001, 0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
}

AE_CONFIG = {
    'hiddens': [512, 512, 2048, 10],
    'epochs': 500,
    'lr': 1e-3,
    'lr_decay': 0.1,
    'min_lr': 1e-7,
    'patience': 10,
    'batch_size': 256
}

SPECTRAL_CONFIG = {
    'epochs': 200,
    'lr': 1e-4,
    'lr_decay': 0.1,
    'min_lr': 1e-8,
    'patience': 10,
    'batch_size': 512,
    'n_nbg': 5,
    'scale_k': 15,
    'is_local_scale': True
}


EXPERIMENT_MODE = 'grid_search'


SAVE_PATH = 'results/parameter_grid_search_results2.xlsx'



def create_config(model_params):

    model_options = ModelOptions(**model_params)
    ae_config = AutoEncoderConfig(**AE_CONFIG)
    spectral_config = SpectralNetConfig(**SPECTRAL_CONFIG)

    return TrainingConfig(
        data_path=DATA_PATH,
        num_runs=NUM_RUNS,
        seed=SEED,
        ae_config=ae_config,
        spectral_config=spectral_config,
        model_options=model_options
    )


def run_experiment(config, data_list, exp_id=0, total=1):

    for key, value in config.model_options.__dict__.items():
        print(f"  {key}: {value}")

    set_seed(SEED)

    acc_results = []
    nmi_results = []
    f_results = []

    for run_idx in range(NUM_RUNS):
        acc, nmi, f, _ = MSCN.run_net(data_list, config)
        acc_results.append(acc)
        nmi_results.append(nmi)
        f_results.append(f)

    result = {
        'experiment_id': exp_id + 1,
        **{k: v for k, v in config.model_options.__dict__.items()},
        'acc_mean': np.mean(acc_results),
        'acc_std': np.std(acc_results),
        'nmi_mean': np.mean(nmi_results),
        'nmi_std': np.std(nmi_results),
        'f_mean': np.mean(f_results),
        'f_std': np.std(f_results),
        'num_runs': NUM_RUNS
    }

    print(f"\n  results: ACC={result['acc_mean']:.4f} std={result['acc_std']:.4f}, "
          f"NMI={result['nmi_mean']:.4f}¡À{result['nmi_std']:.4f}, "
          f"F={result['f_mean']:.4f}¡À{result['f_std']:.4f}")

    return result


def print_best_results(df):

    for metric in ['acc_mean', 'nmi_mean', 'f_mean']:
        best = df.loc[df[metric].idxmax()]
        metric_name = metric.replace('_mean', '').upper()
        print(f"\n{metric_name}:")
        print(f"nbg={best['nbg']}, t={best['t']}, tf={best['tf']}, "
              f"alpha={best['alpha']}, beta={best['beta']}, gamma={best['gamma']}")
        print(f"  ACC: {best['acc_mean']:.4f} and {best['acc_std']:.4f}")
        print(f"  NMI: {best['nmi_mean']:.4f} and {best['nmi_std']:.4f}")
        print(f"  F-score: {best['f_mean']:.4f} and {best['f_std']:.4f}")


def main():

    data_list = get_data(DATA_PATH)

    if EXPERIMENT_MODE == 'single':
        config = create_config(MODEL_PARAMS)
        result = run_experiment(config, data_list, 0, 1)
        print(f"ACC: {result['acc_mean']:.4f} ¡À {result['acc_std']:.4f}")
        print(f"NMI: {result['nmi_mean']:.4f} ¡À {result['nmi_std']:.4f}")
        print(f"F-score: {result['f_mean']:.4f} ¡À {result['f_std']:.4f}")

    elif EXPERIMENT_MODE == 'grid_search':

        param_names = list(PARAM_GRID.keys())
        param_values = list(PARAM_GRID.values())
        param_combinations = list(product(*param_values))
        total_experiments = len(param_combinations)

        all_results = []
        for exp_idx, param_combo in enumerate(param_combinations):
            params = dict(zip(param_names, param_combo))
            config = create_config(params)
            result = run_experiment(config, data_list, exp_idx, total_experiments)
            all_results.append(result)

        os.makedirs(os.path.dirname(SAVE_PATH) if os.path.dirname(SAVE_PATH) else '.', exist_ok=True)
        df = pd.DataFrame(all_results)
        df.to_excel(SAVE_PATH, index=False, engine='openpyxl')


        print_best_results(df)


if __name__ == "__main__":
    main()
