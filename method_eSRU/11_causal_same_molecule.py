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
from itertools import product

# %% Constants.
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
                      data=np.nan)
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

for mass, cols_this_mass in cols_grouped_mass.items():
    n = len(cols_this_mass)
    x = ts[cols_this_mass].values.T
    x = torch.tensor(x, dtype=torch.float32, device=device)

    cols_height = [col.split('_')[1] for col in cols_this_mass]
    with tqdm_joblib(tqdm_object=tqdm(total=n, desc=mass)):
        # Because there is only 3 height, use less instances than CPU cores.
        gc_val_mass = joblib.Parallel(n_jobs=3)(
            joblib.delayed(infer_node)(x, j, n) for j in range(n)
        )
    gc_val_mass = np.array(gc_val_mass).T  # transposed, raw entries (j, i)
    for i, j in product(range(n), repeat=2):
        if i == j:
            continue
        gc_val.loc[mass, (cols_height[i], cols_height[j])] = gc_val_mass[i, j]

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
fig.savefig(f'results/11_{dataset_name}_gc.eps')
plt.close(fig)

# %% Export.
pd.to_pickle(gc_val, f'raw/11_{dataset_name}_gc.pkl')
