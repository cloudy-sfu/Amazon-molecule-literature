import pandas as pd
from scipy.io import savemat

# %% Constants.
dataset_name = 'default_15min'

# %% Load data.
ts_train, ts_test = pd.read_pickle(f'data/1_{dataset_name}_std.pkl')
ts = pd.concat([ts_train, ts_test], ignore_index=True)

# %% Export.
savemat(f'results/7_{dataset_name}_std.mat',
    {'ts': ts.values, 'col_names': ts.columns.values},
    appendmat=False,
)
