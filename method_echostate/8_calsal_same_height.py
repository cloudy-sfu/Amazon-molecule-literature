import os

from scipy.io import loadmat
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# %% Constant.
dataset_name = 'default_15min'

# %% Load data.
data = loadmat(f"data/a1_{dataset_name}_gc.mat")
data_keys = data['gc_val'].dtype.fields.keys()
data_vals = data['gc_val'][0, 0]
gc_val = {k: v for k, v in zip(data_keys, data_vals)}
data_vals = [h for h in data['gc_names'][0, 0]]
col_names = {
    k: [name[0][0] for name in v]
    for k, v in zip(data_keys, data_vals)
}
os.makedirs(f'results/8_{dataset_name}_gc/', exist_ok=True)

# %% Heatmap.
@np.vectorize
def decimal_non_zero(x_):
    return format(x_, '.2f').removeprefix('0')

for height, gc_mat in gc_val.items():
    fig, ax = plt.subplots(figsize=(7.5, 6))
    mask = np.zeros_like(gc_mat, dtype=bool)
    mask[np.diag_indices_from(mask)] = True
    heatmap = sns.heatmap(gc_mat, mask=mask, square=True, linewidths=.5, cmap='coolwarm_r',
                          vmin=0, vmax=None, annot=decimal_non_zero(gc_mat), fmt='', ax=ax)
    ax.set_ylabel('Cause')
    ax.set_xlabel('Effect')
    ax.set_xticklabels(col_names[height], rotation=45)
    ax.set_yticklabels(col_names[height], rotation=0)
    fig.subplots_adjust(bottom=0.15, top=0.95, left=0.10, right=1)
    sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
    fig.savefig(f'results/8_{dataset_name}_gc/{height}.eps')
    plt.close(fig)

# %% Export.
with open(f'raw/8_{dataset_name}_gc.pkl', 'wb') as f:
    pickle.dump(gc_val, f)
