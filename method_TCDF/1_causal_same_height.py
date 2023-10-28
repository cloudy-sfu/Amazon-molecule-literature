import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from TCDF import find_causes

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
gc_val = {height: np.zeros(shape=(len(x), len(x))) for height, x in cols_grouped_height.items()}
os.makedirs(f'results/1_{dataset_name}_gc/', exist_ok=True)
pbar = tqdm(total=ts_train.shape[1])

# %% Infer causality.
for height, cols_this_height in cols_grouped_height.items():
    x = ts[cols_this_height]
    m = len(cols_this_height)
    for j in range(m):
        causes, _, _ = find_causes(x=x, target_idx=j, cuda=True, epochs=1000, max_lag=lag, layers=2, seed=8596,
                                   dilation_c=lag)
        for i in causes:
            if i == j:
                continue
            gc_val[height][i, j] = 1
        pbar.update(1)

    # Heatmap
    fig, ax = plt.subplots(figsize=(7.5, 6))
    mask = np.zeros_like(gc_val[height], dtype=bool)
    mask[np.diag_indices_from(mask)] = True
    heatmap = sns.heatmap(gc_val[height], mask=mask, square=True, linewidths=.5, cmap='coolwarm',
                          vmin=0, vmax=None, annot=gc_val[height], fmt='', ax=ax)
    ax.set_ylabel('Cause')
    ax.set_xlabel('Effect')
    ax.set_xticklabels(cols_this_height, rotation=45)
    ax.set_yticklabels(cols_this_height, rotation=0)
    fig.subplots_adjust(bottom=0.15, top=0.95, left=0.10, right=1)
    sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
    fig.savefig(f'results/1_{dataset_name}_gc/{lag}_{height}.eps')
    plt.close(fig)

# %% Export.
pd.to_pickle(gc_val, f'raw/1_{dataset_name}_gc_{lag}.pkl')
