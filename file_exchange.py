"""
Copy the following files from Amazon molecule https://github.com/cloudy-sfu/Amazon-molecule to this program.
"""
import os
import shutil

# %% Constants.
dataset_name = 'default_15min'
lag = 96
# The path of Amazon molecule program in local machine.
base_dir = '~/amazon_molecular'

# %% Script.
# pre-processed dataset
shutil.copy(os.path.join(base_dir, f'raw/1_{dataset_name}_std.pkl'), 'data/')
