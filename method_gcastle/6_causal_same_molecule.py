from collections import defaultdict
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from castle.algorithms import PNL
from castle.algorithms import *
from copy import deepcopy
import os

# %% Constants.
dataset_name = 'default_15min'

# %% Load data.
ts_train, ts_test = pd.read_pickle(f'data/1_{dataset_name}_std.pkl')
ts = pd.concat([ts_train, ts_test], ignore_index=True)

# %% Split by height.
cols = ts.columns
mass_heights = map(lambda x: x.split('_') + [x], cols)
cols_grouped_mass = defaultdict(list)
all_possible_heights = []
for mass, height, mass_height in mass_heights:
    cols_grouped_mass[mass].append(mass_height)
    all_possible_heights.append(height)
all_possible_heights = np.unique(all_possible_heights)
heights_pair = list((fi, fj) for fi, fj in product(all_possible_heights, repeat=2) if fi != fj)

# %% Initialization.
os.makedirs(f'results/6_{dataset_name}_gc/', exist_ok=True)
os.environ['CASTLE_BACKEND'] = 'pytorch'
methods_dict = {
    'PC': PC(),
    # very slow, cannot use GPU, usage of CPU not stable
    # 'ANM': ANMNonlinear(),
    'DirectLiNGAM': DirectLiNGAM(),
    'ICALiNGAM': ICALiNGAM(random_state=2775, max_iter=5000),
    'GES': GES(method='r2'),
    # very slow
    # 'PNL': PNL(hidden_layers=2, hidden_units=32, batch_size=3500, epochs=2000, alpha=0.05, device_type='gpu',
    #            device_ids=0, activation=torch.nn.ReLU()),
    'NOTEARS': Notears(),
    'NOTEARS-MLP': NotearsNonlinear(hidden_layers=(32, 32, 1), model_type='mlp', device_type='gpu', device_ids=0),
    'NOTEARS-SOB': NotearsNonlinear(hidden_layers=(32, 32, 1), model_type='sob', device_type='gpu', device_ids=0),
    # Unknown true causal graph
    # 'NOTEARS-lOW-RANK': NotearsLowRank(),
    'DAG-GNN': DAG_GNN(device_type='gpu', batch_size=3500, device_ids=0),
    'GOLEM': GOLEM(seed=2775, device_type='gpu', device_ids=0, num_iter=10000),
    'GraNDAG': GraNDAG(hidden_dim=32, batch_size=3500, device_type='gpu', device_ids=0, random_seed=2775),
    'MCSL': MCSL(num_hidden_layers=2, hidden_dim=32, device_type='gpu', device_ids=0, random_seed=2775),
    'GAE': GAE(hidden_layers=2, hidden_dim=32, seed=2775, device_type='gpu', device_ids=0,
               early_stopping=True, early_stopping_thresh=0.995),
    # very slow, 1*NVIDIA GTX 4090, time = 90 s/iter * nb_epoch iter
    # 'RL': RL(batch_size=3500, seed=2775, nb_epoch=1000, device_type='gpu', device_ids=0),
    'CORL': CORL(batch_size=1500, random_seed=2775, iteration=1000, device_type='gpu', device_ids=0),
    # Unknown true causal graph
    # 'TTPM': TTPM()
}

# %% Infer causality.
for method, method_instance in methods_dict.items():
    if os.path.exists(f'raw/6_{dataset_name}_gc_{method}.pkl'):
        continue

    gc_val = pd.DataFrame(index=cols_grouped_mass.keys(), columns=pd.MultiIndex.from_tuples(heights_pair), dtype=float,
                          data=1)
    for mass, cols_this_mass in tqdm(cols_grouped_mass.items(), desc=method):
        if os.path.exists(f'results/6_{dataset_name}_gc/{method}_{mass}.eps'):
            continue

        x = ts[cols_this_mass].values
        model = deepcopy(method_instance)
        model.learn(x)

        m = len(cols_this_mass)
        cols_height = [col.split('_')[1] for col in cols_this_mass]
        for i, j in product(range(m), repeat=2):
            if i == j:
                continue
            gc_val.loc[mass, (cols_height[i], cols_height[j])] = model.causal_matrix[i, j]

        # Heatmap
        fig, ax = plt.subplots(figsize=(0.75 * gc_val.shape[1], 0.5 * gc_val.shape[0] + 1))
        cmap = ListedColormap(['#b40426', '#3b4cc0'])  # [warm, cool]
        heatmap = sns.heatmap(gc_val.values, square=True, linewidths=.5, cmap=cmap, vmin=0, vmax=None, cbar=False,
                              annot=gc_val.values, fmt='', ax=ax)
        ax.set_ylabel('Mass')
        ax.set_xlabel('(Cause, Effect)')
        ax.set_xticklabels(gc_val.columns, rotation=45)
        ax.set_yticklabels(gc_val.index, rotation=0)
        fig.subplots_adjust(bottom=0.2, top=0.95, left=0.05, right=0.95)
        sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
        fig.savefig(f'results/6_{dataset_name}_gc/{method}.eps')
        plt.close(fig)

    pd.to_pickle(gc_val, f'raw/6_{dataset_name}_gc_{method}.pkl')
