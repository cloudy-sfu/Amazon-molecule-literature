import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import re

# %% Constants.
lag = 96
dataset_name = 'default_15min'

# %% Load data.
distances = pd.read_pickle(f"raw/14_{dataset_name}_consistency_matrix_{lag}.pkl")
fishers = pd.read_pickle(f"raw/14_{dataset_name}_fisher_p_{lag}.pkl")

# %% Initialization.
os.makedirs(f'results/15_{dataset_name}_dendrogram_{lag}/', exist_ok=True)
method_family_colors = {
    'Ours': '#3b4cc0',
    'Other GC': '#96b7ff',
    'non-GC': '#c3543c'
}
default_color = '#dddcdc'

# %% Consistency per task by accuracy.
for task, dist in distances.items():
    dist_condensed = squareform(dist.values, checks=False)
    linkage_ = linkage(dist_condensed, method='average')
    fig, ax = plt.subplots(figsize=(6, 6))
    dendrogram(linkage_, labels=dist.columns, orientation='right', link_color_func=lambda x: 'k')
    ax.set_ylabel('Methods')
    ax.set_xlabel('Inconsistency')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    for label in ax.get_yticklabels():
        label_category = re.search(r'\((.*?)\)', label.get_text()).group(1)
        label.set_color(method_family_colors.get(label_category, default_color))
    fig.subplots_adjust(left=0.35, right=0.95, bottom=0.1, top=0.95)
    fig.savefig(f'results/15_{dataset_name}_dendrogram_{lag}/acc_{task}.eps')
    plt.close(fig)

# %% Consistency overall by accuracy.
dist_overall = np.mean([dist.values for dist in distances.values()], axis=0)
dist_condensed = squareform(dist_overall, checks=False)
linkage_ = linkage(dist_condensed, method='average')
fig, ax = plt.subplots(figsize=(6, 6))
dendrogram(linkage_, labels=dist.columns, orientation='right', link_color_func=lambda x: 'k')
ax.set_ylabel('Methods')
ax.set_xlabel('Inconsistency')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
for label in ax.get_yticklabels():
    label_category = re.search(r'\((.*?)\)', label.get_text()).group(1)
    label.set_color(method_family_colors.get(label_category, default_color))
fig.subplots_adjust(left=0.35, right=0.95, bottom=0.1, top=0.95)
fig.savefig(f'results/15_{dataset_name}_dendrogram_{lag}/acc_overall.eps')
plt.close(fig)

# %% Consistency per task by Fisher's exact test.
for task, dist in fishers.items():
    dist_condensed = squareform(dist.values, checks=False)
    linkage_ = linkage(dist_condensed, method='average')
    fig, ax = plt.subplots(figsize=(6, 6))
    dendrogram(linkage_, labels=dist.columns, orientation='right', link_color_func=lambda x: 'k')
    ax.set_ylabel('Methods')
    ax.set_xlabel('Independence')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    for label in ax.get_yticklabels():
        label_category = re.search(r'\((.*?)\)', label.get_text()).group(1)
        label.set_color(method_family_colors.get(label_category, default_color))
    fig.subplots_adjust(left=0.35, right=0.95, bottom=0.1, top=0.95)
    fig.savefig(f'results/15_{dataset_name}_dendrogram_{lag}/fisher_{task}.eps')
    plt.close(fig)

# %% Consistency overall by Fisher's exact test.
dist_overall = np.exp(np.mean([np.log(dist.values) for dist in fishers.values()], axis=0))
dist_condensed = squareform(dist_overall, checks=False)
linkage_ = linkage(dist_condensed, method='average')
fig, ax = plt.subplots(figsize=(6, 6))
dendrogram(linkage_, labels=dist.columns, orientation='right', link_color_func=lambda x: 'k')
ax.set_ylabel('Methods')
ax.set_xlabel('Independence')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
for label in ax.get_yticklabels():
    label_category = re.search(r'\((.*?)\)', label.get_text()).group(1)
    label.set_color(method_family_colors.get(label_category, default_color))
fig.subplots_adjust(left=0.35, right=0.95, bottom=0.1, top=0.95)
fig.savefig(f'results/15_{dataset_name}_dendrogram_{lag}/fisher_overall.eps')
plt.close(fig)
