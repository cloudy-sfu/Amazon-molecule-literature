"""
Copy the following files from Amazon molecule https://github.com/cloudy-sfu/Amazon-molecule to this program.
"""
import os
import shutil

# %% Constants.
dataset_name = 'default_15min'
lag = 96
# The path of Amazon molecule program in local machine.
base_dir = '/home/cld/amazon_molecular'

# %% Script.
# pre-processed dataset
shutil.copy(os.path.join(base_dir, f'raw/1_{dataset_name}_std.pkl'), 'data/')
# SSR results
shutil.copy(os.path.join(base_dir, f'raw/6_{dataset_name}_p_{lag}.pkl'), 'data/')
shutil.copy(os.path.join(base_dir, f'raw/12_{dataset_name}_p_{lag}.pkl'), 'data/')
# MLE results
shutil.copy(os.path.join(base_dir, f'raw/8_{dataset_name}_p_{lag}.pkl'), 'data/')
shutil.copy(os.path.join(base_dir, f'raw/17_{dataset_name}_p_{lag}.pkl'), 'data/')
