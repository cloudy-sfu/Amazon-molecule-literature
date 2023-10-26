import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from matplotlib.colors import ListedColormap
from castle.algorithms import PNL
from castle.algorithms import *
from copy import deepcopy

# %% Constants.
dataset_name = 'default_15min'

# %% Load data.
ts_train, ts_test = pd.read_pickle(f'data/1_{dataset_name}_std.pkl')
ts = pd.concat([ts_train, ts_test], ignore_index=True)

# %% Split by height.
cols = ts_train.columns
mass_heights = map(lambda x: x.split('_') + [x], cols)
cols_grouped_height = defaultdict(list)
for mass, height, mass_height in mass_heights:
    cols_grouped_height[height].append(mass_height)

# %% Initialization.
# https://github.com/huawei-noah/trustworthyAI/blob/master/gcastle/README.md
os.makedirs(f'results/5_{dataset_name}_gc/', exist_ok=True)
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
    if os.path.exists(f'raw/5_{dataset_name}_gc_{method}.pkl'):
        continue

    gc_val = {height: np.ones(shape=(len(x), len(x))) for height, x in cols_grouped_height.items()}
    for height, cols_this_height in tqdm(cols_grouped_height.items(), desc=method):
        if os.path.exists(f'results/5_{dataset_name}_gc/{method}_{height}.eps'):
            continue

        x = ts[cols_this_height].values
        model = deepcopy(method_instance)
        model.learn(x)
        # Direction confirmed: https://github.com/huawei-noah/trustworthyAI/issues/115
        gc_val[height] = model.causal_matrix

        # Heatmap
        fig, ax = plt.subplots(figsize=(7.5, 6))
        mask = np.zeros_like(gc_val[height], dtype=bool)
        mask[np.diag_indices_from(mask)] = True
        cmap = ListedColormap(['#b40426', '#3b4cc0'])  # [warm, cool]
        heatmap = sns.heatmap(gc_val[height], mask=mask, square=True, linewidths=.5, cmap=cmap, cbar=False,
                              vmin=0, vmax=None, annot=gc_val[height], fmt='', ax=ax)
        ax.set_ylabel('Cause')
        ax.set_xlabel('Effect')
        ax.set_xticklabels(cols_this_height, rotation=45)
        ax.set_yticklabels(cols_this_height, rotation=0)
        fig.subplots_adjust(bottom=0.15, top=0.95, left=0.10, right=1)
        sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
        fig.savefig(f'results/5_{dataset_name}_gc/{method}_{height}.eps')
        plt.close(fig)

    pd.to_pickle(gc_val, f'raw/5_{dataset_name}_gc_{method}.pkl')
