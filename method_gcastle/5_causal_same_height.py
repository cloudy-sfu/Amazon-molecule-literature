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
import torch
from copy import deepcopy

# %% Constants.
lag = 96
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
    'PNL': PNL(hidden_layers=2, hidden_units=32, batch_size=ts.shape[0], epochs=2000, alpha=0.05, device_type='gpu',
               device_ids=0, activation=torch.nn.ReLU()),
    'NOTEARS': Notears(),
}
pbar = tqdm(total=len(cols_grouped_height) * len(methods_dict))

# %% Infer causality.
for method, method_instance in methods_dict:
    if os.path.exists(f'raw/5_{dataset_name}_gc_{lag}_{method}.pkl'):
        pbar.update(len(cols_grouped_height))
        continue

    gc_val = {height: np.ones(shape=(len(x), len(x))) for height, x in cols_grouped_height.items()}
    for height, cols_this_height in tqdm(cols_grouped_height.items()):
        if os.path.exists(f'results/5_{dataset_name}_gc/{lag}_{method}_{height}.eps'):
            pbar.update(1)
            continue

        x = ts[cols_this_height].values
        model = deepcopy(method_instance)
        model.learn(x)
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
        fig.savefig(f'results/5_{dataset_name}_gc/{lag}_{method}_{height}.eps')
        plt.close(fig)

    pd.to_pickle(gc_val, f'raw/5_{dataset_name}_gc_{lag}_{method}.pkl')
