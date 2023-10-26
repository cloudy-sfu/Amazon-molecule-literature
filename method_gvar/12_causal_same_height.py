import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from training import training_procedure_trgc
from tqdm import tqdm

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
gc_val = {}
os.makedirs(f'results/12_{dataset_name}_gc/', exist_ok=True)

@np.vectorize
def decimal_non_zero(x_):
    return format(x_, '.2f').removeprefix('0')

# %% Infer causality.
for height, cols_this_height in tqdm(cols_grouped_height.items(), desc='Overall'):
    n = len(cols_this_height)
    x = ts[cols_this_height].values
    gc_val_height, _, _ = training_procedure_trgc(
        data=x, order=lag, hidden_layer_size=[32, 32], end_epoch=500, batch_size=4000,
        lmbd=0, gamma=0, seed=6615, use_cuda=True
    )
    gc_val[height] = gc_val_height

    # Heatmap
    fig, ax = plt.subplots(figsize=(7.5, 6))
    mask = np.zeros_like(gc_val[height], dtype=bool)
    mask[np.diag_indices_from(mask)] = True
    heatmap = sns.heatmap(gc_val[height], mask=mask, square=True, linewidths=.5, cmap='coolwarm',
                          vmin=0, vmax=None, annot=decimal_non_zero(gc_val[height]), fmt='', ax=ax)
    ax.set_ylabel('Cause')
    ax.set_xlabel('Effect')
    ax.set_xticklabels(cols_this_height, rotation=45)
    ax.set_yticklabels(cols_this_height, rotation=0)
    fig.subplots_adjust(bottom=0.15, top=0.95, left=0.10, right=1)
    sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
    fig.savefig(f'results/12_{dataset_name}_gc/{lag}_{height}.eps')
    plt.close(fig)

# %% Export.
pd.to_pickle(gc_val, f'raw/12_{dataset_name}_gc_{lag}.pkl')
