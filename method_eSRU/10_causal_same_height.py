import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from models.esru_2LF import eSRU_2LF, train_eSRU_2LF
import torch
import joblib
import contextlib

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
gc_val = {}
os.makedirs(f'results/10_{dataset_name}_gc/', exist_ok=True)
device = torch.device('cuda')

@np.vectorize
def decimal_non_zero(x_):
    return format(x_, '.2f').removeprefix('0')

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument
    Reference: https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

# %% Infer causality.
def infer_node(x_, predicted_idx, n_nodes):
    model = eSRU_2LF(
        n_inp_channels=n_nodes, n_out_channels=1, dim_iid_stats=int(1.5 * n_nodes), dim_rec_stats=int(1.5 * n_nodes),
        dim_rec_stats_feedback=n_nodes,
        dim_final_stats=n_nodes, A=[0.0, 0.01, 0.1, 0.99], device=device
    )
    model.to(device)
    model, _ = train_eSRU_2LF(
        model=model, X=x_, device=device, batch_size=5000, predicted_idx=predicted_idx,
        max_iter=2000, lambda1=0.021544, lambda2=0.031623, lambda3=0.464159, lr=0.001,
        lr_gamma=0.99, lr_update_gap=4, stopping_thresh=1e-5
    )
    gc_est = torch.norm(model.lin_xr2phi.weight.data[:, :n_nodes], p=2, dim=0)
    return gc_est.cpu().numpy()

for height, cols_this_height in cols_grouped_height.items():
    n = len(cols_this_height)
    x = ts[cols_this_height].values.T
    x = torch.tensor(x, dtype=torch.float32, device=device)

    with tqdm_joblib(tqdm_object=tqdm(total=n, desc=height)):
        gc_val_height = joblib.Parallel(n_jobs=12)(
            joblib.delayed(infer_node)(x, j, n) for j in range(n)
        )
    gc_val[height] = np.array(gc_val_height).T

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
    fig.savefig(f'results/10_{dataset_name}_gc/{height}.eps')
    plt.close(fig)

# %% Export.
pd.to_pickle(gc_val, f'raw/10_{dataset_name}_gc.pkl')
