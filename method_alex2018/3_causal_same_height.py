import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from clstm import cLSTM, train_model_ista
import torch
from matplotlib.colors import ListedColormap

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
gc_val = {height: np.ones(shape=(len(x), len(x))) for height, x in cols_grouped_height.items()}
os.makedirs(f'results/3_{dataset_name}_gc/', exist_ok=True)
max_search_ = 8
pbar = tqdm(total=len(cols_grouped_height.keys()) * max_search_)
device = torch.device('cuda')

@np.vectorize
def decimal_non_zero(x):
    return format(x, '.2f').removeprefix('0')

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
for height, cols_this_height in cols_grouped_height.items():
    x = ts[cols_this_height].values
    x = torch.tensor(x[np.newaxis, :], dtype=torch.float32, device=device)
    clstm, gc_est, lambda_0 = bisearch(x, 0, 1, max_search_)
    torch.onnx.export(clstm, x, f"raw/3_{dataset_name}_clstm_{lag}_{height}_{round(lambda_0, 3)}.onnx", verbose=False)
    gc_val[height] = gc_est

    # Heatmap
    fig, ax = plt.subplots(figsize=(7.5, 6))
    mask = np.zeros_like(gc_val[height], dtype=bool)
    mask[np.diag_indices_from(mask)] = True
    cmap = ListedColormap(['#b40426', '#3b4cc0'])
    heatmap = sns.heatmap(gc_val[height] > 0, mask=mask, square=True, linewidths=.5, cmap=cmap, cbar=False,
                          vmin=0, vmax=None, annot=decimal_non_zero(gc_val[height]), fmt='', ax=ax)
    ax.set_ylabel('Cause')
    ax.set_xlabel('Effect')
    ax.set_xticklabels(cols_this_height, rotation=45)
    ax.set_yticklabels(cols_this_height, rotation=0)
    fig.subplots_adjust(bottom=0.15, top=0.95, left=0.10, right=1)
    sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
    fig.savefig(f'results/3_{dataset_name}_gc/{lag}_{height}.eps')
    plt.close(fig)

# %% Export.
pd.to_pickle(gc_val, f'raw/3_{dataset_name}_gc_{lag}.pkl')
