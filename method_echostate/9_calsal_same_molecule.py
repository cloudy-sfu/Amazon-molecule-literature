from scipy.io import loadmat
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns

# %% Constant.
dataset_name = 'default_15min'

# %% Load data.
data = loadmat(f"data/a2_{dataset_name}_gc.mat")
data_keys = data['gc_val'].dtype.fields.keys()
data_vals = data['gc_val'][0, 0]
gc_mat = {k: v for k, v in zip(data_keys, data_vals)}

all_heights = np.array([x[0][0] for x in data['all_heights']])
heights_pair = list((fi, fj) for fi, fj in product(all_heights, repeat=2) if fi != fj)
gc_val = pd.DataFrame(index=gc_mat.keys(), columns=pd.MultiIndex.from_tuples(heights_pair), dtype=float, data=1)
for mass, gc_est in gc_mat.items():
    m = gc_est.shape[0]
    for i, j in product(range(m), repeat=2):
        if i == j:
            continue
        gc_val.loc[mass, (all_heights[i], all_heights[j])] = gc_est[i, j]

# %% Heatmap.
@np.vectorize
def decimal_non_zero(x_):
    return format(x_, '.2f').removeprefix('0')

fig, ax = plt.subplots(figsize=(0.75 * gc_val.shape[1], 0.5 * gc_val.shape[0] + 1))
heatmap = sns.heatmap(gc_val.values, square=True, linewidths=.5, cmap='coolwarm_r', vmin=0, vmax=None, cbar=False,
                      annot=decimal_non_zero(gc_val.values), fmt='', ax=ax)
ax.set_ylabel('Mass')
ax.set_xlabel('(Cause, Effect)')
ax.set_xticklabels(gc_val.columns, rotation=45)
ax.set_yticklabels(gc_val.index, rotation=0)
fig.subplots_adjust(bottom=0.2, top=0.95, left=0.05, right=0.95)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
fig.savefig(f'results/9_{dataset_name}_gc.eps')
plt.close(fig)

# %% Export.
pd.to_pickle(gc_val, f'raw/9_{dataset_name}_gc.pkl')
