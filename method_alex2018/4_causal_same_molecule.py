from collections import defaultdict
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from clstm import cLSTM, train_model_ista

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
                      data=1)
max_search_ = 8
pbar = tqdm(total=len(cols_grouped_mass.keys()) * max_search_)
device = torch.device('cuda')

# %% Bi-sectional search.
def bisearch(x, lambda_lb, lambda_ub, max_search, count=0):
    lambda_0 = (lambda_lb + lambda_ub) * 0.5
    clstm = cLSTM(x.shape[-1], hidden=32).cuda(device=device)
    train_loss_list = train_model_ista(clstm, x, context=10, lam=lambda_0, lam_ridge=0.1, lr=1e-3, max_iter=2000)
    gc_est = clstm.GC(threshold=False).cpu().data.numpy().T
    if count == max_search:
        return clstm, gc_est, lambda_0
    else:
        count += 1  # bi-sectional search times
        # basic idea: maximum regularization, but self-causality should be revealed.
        gc_est_diag = np.diag(gc_est)
        if np.sum(gc_est_diag < 1e-5) > 1:
            lambda_ub = lambda_0
        else:
            lambda_lb = lambda_0
        pbar.update(1)
        return bisearch(x, lambda_lb, lambda_ub, max_search, count)

# %% Infer causality.
for mass, cols_this_mass in cols_grouped_mass.items():
    m = len(cols_this_mass)
    x = ts[cols_this_mass].values
    x = torch.tensor(x[np.newaxis, :], dtype=torch.float32, device=device)
    clstm, gc_est, lambda_0 = bisearch(x, 0, 1, max_search_)
    torch.onnx.export(clstm, x, f"raw/4_{dataset_name}_clstm_{mass}_{round(lambda_0, 3)}.onnx", verbose=False)
    cols_height = [col.split('_')[1] for col in cols_this_mass]
    for i, j in product(range(m), repeat=2):
        if i == j:
            continue
        gc_val.loc[mass, (cols_height[i], cols_height[j])] = gc_est[i, j]

# %% Heatmap
@np.vectorize
def decimal_non_zero(x_):
    return format(x_, '.2f').removeprefix('0')

fig, ax = plt.subplots(figsize=(0.75 * gc_val.shape[1], 0.5 * gc_val.shape[0] + 1))
cmap = ListedColormap(['#b40426', '#3b4cc0'])
heatmap = sns.heatmap(gc_val.values > 0, square=True, linewidths=.5, cmap=cmap, vmin=0, vmax=None, cbar=False,
                      annot=decimal_non_zero(gc_val.values), fmt='', ax=ax)
ax.set_ylabel('Mass')
ax.set_xlabel('(Cause, Effect)')
ax.set_xticklabels(gc_val.columns, rotation=45)
ax.set_yticklabels(gc_val.index, rotation=0)
fig.subplots_adjust(bottom=0.2, top=0.95, left=0.05, right=0.95)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
fig.savefig(f'results/4_{dataset_name}_gc.eps')
plt.close(fig)

# %% Export.
pd.to_pickle(gc_val, f'raw/4_{dataset_name}_gc.pkl')
