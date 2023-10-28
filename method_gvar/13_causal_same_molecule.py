from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from training import training_procedure_trgc
from tqdm import tqdm
from itertools import product
import joblib
import contextlib

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
                      data=np.nan)
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
def infer_mass(dataset, mass, cols_this_mass):
    x = dataset[cols_this_mass].values
    # Direction confirmed https://github.com/i6092467/GVAR/issues/4
    gc_val_mass, _, _ = training_procedure_trgc(
        data=x, order=lag, hidden_layer_size=[32, 32], end_epoch=500, batch_size=4000,
        lmbd=0, gamma=0, seed=6615, use_cuda=True
    )
    return mass, gc_val_mass

n = len(cols_grouped_mass)
with tqdm_joblib(tqdm_object=tqdm(total=n, desc='Overall')):
    gc_vals = joblib.Parallel(n_jobs=12)(
        joblib.delayed(infer_mass)(ts, mass, cols_this_mass) for mass, cols_this_mass in cols_grouped_mass.items()
    )
gc_vals = {x[0]: x[1] for x in gc_vals}

for mass, cols_this_mass in tqdm(cols_grouped_mass.items(), desc='Overall'):
    n = len(cols_this_mass)
    cols_height = [col.split('_')[1] for col in cols_this_mass]
    gc_val_mass = gc_vals[mass]
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
fig.savefig(f'results/13_{dataset_name}_gc_{lag}.eps')
plt.close(fig)

# %% Export.
pd.to_pickle(gc_val, f'raw/13_{dataset_name}_gc_{lag}.pkl')
