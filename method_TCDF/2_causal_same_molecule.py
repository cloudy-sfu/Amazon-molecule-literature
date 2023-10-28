import os
import pandas as pd
from collections import defaultdict
from TCDF import find_causes
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from itertools import product

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
gc_val = pd.DataFrame(index=cols_grouped_mass.keys(), columns=pd.MultiIndex.from_tuples(heights_pair), dtype=float,
                      data=0)
pbar = tqdm(total=ts_train.shape[1])

# %% Infer causality.
for mass, cols_this_mass in cols_grouped_mass.items():
    x = ts[cols_this_mass]
    m = len(cols_this_mass)
    cols_height = [col.split('_')[1] for col in cols_this_mass]
    for j in range(m):
        causes, _, _ = find_causes(x=x, target_idx=j, cuda=True, epochs=1000, max_lag=lag, layers=2, seed=8596,
                                   dilation_c=lag)
        for i in causes:
            if i == j:
                continue
            gc_val.loc[mass, (cols_height[i], cols_height[j])] = 1
        pbar.update(1)

# %% Heatmap
@np.vectorize
def decimal_non_zero(x_):
    return format(x_, '.2f').removeprefix('0')

fig, ax = plt.subplots(figsize=(0.75 * gc_val.shape[1], 0.5 * gc_val.shape[0] + 1))
heatmap = sns.heatmap(gc_val.values, square=True, linewidths=.5, cmap='coolwarm', vmin=0, vmax=0.1,
                      annot=gc_val.values, fmt='', ax=ax)
ax.set_ylabel('Mass')
ax.set_xlabel('(Cause, Effect)')
ax.set_xticklabels(gc_val.columns, rotation=45)
ax.set_yticklabels(gc_val.index, rotation=0)
fig.subplots_adjust(bottom=0.2, top=0.95, left=0.05, right=0.95)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
fig.savefig(f'results/2_{dataset_name}_gc_{lag}.eps')
plt.close(fig)

# %% Export.
pd.to_pickle(gc_val, f'raw/2_{dataset_name}_gc_{lag}.pkl')
