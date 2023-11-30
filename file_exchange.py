"""
Copy the following files from Amazon molecule https://github.com/cloudy-sfu/Amazon-molecule to this program.
"""
import os
import shutil

# The path of Amazon molecule program in local machine.
base_dir = '/home/cld/amazon_molecular'

# pre-processed dataset
shutil.copy(os.path.join(base_dir, 'raw/1_default_15min_std.pkl'), 'data/')
# SSR results
shutil.copy(os.path.join(base_dir, 'raw/6_default_15min_p_96.pkl'), 'data/')
shutil.copy(os.path.join(base_dir, 'raw/12_default_15min_p_96.pkl'), 'data/')

dataset_name_lags = [
    ['default_15min', [48, 96, 192]],
    ['default_1h', [12, 24, 48]],
    ['default_4h', [3, 6, 12]],
]
for dataset_name, lags in dataset_name_lags:
    for lag in lags:
        # MLE results
        shutil.copy(os.path.join(base_dir, f'raw/8_{dataset_name}_p_{lag}.pkl'), 'data/')
        shutil.copy(os.path.join(base_dir, f'raw/17_{dataset_name}_p_{lag}.pkl'), 'data/')
        # Wilcoxon results
        shutil.copy(os.path.join(base_dir, f'raw/7_{dataset_name}_p_{lag}.pkl'), 'data/')
        shutil.copy(os.path.join(base_dir, f'raw/10_{dataset_name}_p_{lag}.pkl'), 'data/')
