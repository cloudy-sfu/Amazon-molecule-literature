from collections import defaultdict
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from castle.algorithms import PNL
from castle.algorithms import *
from copy import deepcopy
import os

# %% Constants.
lag = 96
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
    'DirectLiNGAM': DirectLiNGAM(),
    'ICALiNGAM': ICALiNGAM(random_state=2775, max_iter=5000),
    'GES': GES(method='r2'),
    'PNL': PNL(hidden_layers=2, hidden_units=32, batch_size=ts.shape[0], epochs=2000, alpha=0.05, device_type='gpu',
               device_ids=0, activation=torch.nn.ReLU()),
    'ANM': ANMNonlinear(),
}
pbar = tqdm(total=len(cols_grouped_mass) * len(methods_dict))

# %% Infer causality.
for method, method_instance in methods_dict:
    if os.path.exists(f'raw/6_{dataset_name}_gc_{lag}_{method}.pkl'):
        pbar.update(len(cols_grouped_mass))
        continue

    gc_val = pd.DataFrame(index=cols_grouped_mass.keys(), columns=pd.MultiIndex.from_tuples(heights_pair), dtype=float,
                          data=1)
    for mass, cols_this_mass in cols_grouped_mass.items():
        if os.path.exists(f'results/6_{dataset_name}_gc/{lag}_{method}_{mass}.eps'):
            pbar.update(1)
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
        fig.savefig(f'results/6_{dataset_name}_gc/{lag}_{method}.eps')
        plt.close(fig)

    pd.to_pickle(gc_val, f'raw/6_{dataset_name}_gc_{lag}_{method}.pkl')
