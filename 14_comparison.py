import logging
import os
import re
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from scipy.stats import fisher_exact
from sklearn.metrics import confusion_matrix

# %% Constants.
lag = 96
dataset_name = 'default_15min'
# type:
#     p: p-value, float [0, 1], p < CV (critical value) represents causality
#     gc: GC statistics, float [0, +∞], GC_est > CV represents causality
#     b: boolean, {0, 1}, 1 means causality
methods = {
    '(Ours) SSR': {
        'same_height': f'data/6_{dataset_name}_p_{lag}.pkl',
        'same_molecule': f'data/12_{dataset_name}_p_{lag}.pkl',
        'type': 'p',
    },
    '(Ours) MLE': {
        'same_height': f'data/8_{dataset_name}_p_{lag}.pkl',
        'same_molecule': f'data/17_{dataset_name}_p_{lag}.pkl',
        'type': 'p'
    },
    '(Ours) Wilcoxon': {
        'same_height': f'data/7_{dataset_name}_p_{lag}.pkl',
        'same_molecule': f'data/10_{dataset_name}_p_{lag}.pkl',
        'type': 'p'
    },
    '(Other GC) Echo': {
        'same_height': f'raw/8_{dataset_name}_gc.pkl',
        'same_molecule': f'raw/9_{dataset_name}_gc.pkl',
        'type': 'gc'
    },
    '(Other GC) TCDF': {
        'same_height': f'raw/1_{dataset_name}_gc_{lag}.pkl',
        'same_molecule': f'raw/2_{dataset_name}_gc_{lag}.pkl',
        'type': 'b'
    },
    '(Other GC) cLSTM': {
        'same_height': f'raw/3_{dataset_name}_gc.pkl',
        'same_molecule': f'raw/4_{dataset_name}_gc.pkl',
        'type': 'gc'
    },
    '(Other GC) eSRU': {
        'same_height': f'raw/10_{dataset_name}_gc.pkl',
        'same_molecule': f'raw/11_{dataset_name}_gc.pkl',
        'type': 'gc'
    },
    '(Other GC) GVAR': {
        'same_height': f'raw/12_{dataset_name}_gc_{lag}.pkl',
        'same_molecule': f'raw/13_{dataset_name}_gc_{lag}.pkl',
        'type': 'gc'
    }
} | {
    '(non-GC) ' + method: {
        'same_height': f'raw/5_{dataset_name}_gc_{method}.pkl',
        'same_molecule': f'raw/6_{dataset_name}_gc_{method}.pkl',
        'type': 'b'
    }
    for method in ['PC', 'DirectLiNGAM', 'ICALiNGAM', 'GES', 'NOTEARS', 'NOTEARS-MLP', 'NOTEARS-SOB', 'DAG-GNN',
                   'GOLEM', 'GraNDAG', 'MCSL', 'GAE', 'CORL']
}

# %% Initialization.
heights = list(pd.read_pickle(list(methods.values())[0]['same_height']).keys())
dist_mat = {
    task: pd.DataFrame(data=np.nan, index=methods.keys(), columns=methods.keys(), dtype=float)
    for task in heights + ['same_molecule']
}
# Odds ratio: https://en.wikipedia.org/wiki/Odds_ratio
# H_0: odds_ratio=1 -> independent -> overlap by random chance
# H_1: odds_ratio!=1 -> significant agreement/consistency -> overlap by dependency
fisher_mat = {
    task: pd.DataFrame(data=np.nan, index=methods.keys(), columns=methods.keys(), dtype=float)
    for task in heights + ['same_molecule']
}
logging.basicConfig(
    filename=f'raw/14_{dataset_name}_log_{lag}.txt',
    filemode='w',
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
cvs = [0.01, 0.05, 0.1]  # critical values if type is 'p'

# %% Maximize consistency.
def inconsistency(threshold, continuous_array, boolean_array, gt=False):
    if gt:
        bool_conti = continuous_array > threshold
    else:
        bool_conti = continuous_array < threshold
    return np.mean(bool_conti != boolean_array)

# %% Convert to list of boolean matrices
#          bool             p                   GC
# bool   accuracy      traverse p           traverse GC
# p                 compare p=1/5/10%   p=1/5/10%, traverse GC
# GC                                         1 - |corr|
# Same height
for height in heights:
    for (name1, cfg1), (name2, cfg2) in combinations(methods.items(), r=2):
        gc_dict_1 = pd.read_pickle(cfg1['same_height'])
        gc_dict_2 = pd.read_pickle(cfg2['same_height'])
        gc_val_1 = gc_dict_1[height].astype(float)
        gc_val_2 = gc_dict_2[height].astype(float)
        # So far the gc_val/p_val matrices in "same height" experiments all follow the database column orders.
        # The order of time series are consistent.
        mask = np.eye(gc_val_1.shape[0], dtype=bool)
        gc_val_1 = gc_val_1[~mask]
        gc_val_2 = gc_val_2[~mask]
        assert gc_val_1.shape == gc_val_2.shape
        if cfg1['type'] == cfg2['type'] == 'b':
            dist_mat[height].loc[name1, name2] = np.mean(gc_val_1 != gc_val_2)
            cmat = confusion_matrix(gc_val_1, gc_val_2, labels=[True, False])
            odd, p = fisher_exact(cmat)
            fisher_mat[height].loc[name1, name2] = p
        elif cfg1['type'] == 'b' and cfg2['type'] == 'p':
            result = minimize(
                fun=inconsistency, x0=np.array([0.05]),
                args=(gc_val_2, gc_val_1, False),
                method='L-BFGS-B', bounds=[(0, 1)]
            )
            logging.info(f'Compare {name1} and {name2} in task {height}, {name1} is boolean, traverse p-value of '
                         f'{name2}, optimum critical value is {result.x}.')
            dist_mat[height].loc[name1, name2] = result.fun
            bool_2 = gc_val_2 < result.x
            cmat = confusion_matrix(gc_val_1, bool_2, labels=[True, False])
            odd, p = fisher_exact(cmat)
            fisher_mat[height].loc[name1, name2] = p
        elif cfg1['type'] == 'p' and cfg2['type'] == 'b':
            result = minimize(
                fun=inconsistency, x0=np.array([0.05]),
                args=(gc_val_1, gc_val_2, False),
                method='L-BFGS-B', bounds=[(0, 1)]
            )
            logging.info(f'Compare {name1} and {name2} in task {height}, {name2} is boolean, traverse p-value of '
                         f'{name1}, optimum critical value is {result.x}.')
            dist_mat[height].loc[name1, name2] = result.fun
            bool_1 = gc_val_1 < result.x
            cmat = confusion_matrix(gc_val_2, bool_1, labels=[True, False])
            odd, p = fisher_exact(cmat)
            fisher_mat[height].loc[name1, name2] = p
        elif cfg1['type'] == 'b' and cfg2['type'] == 'gc':
            result = minimize(
                fun=inconsistency, x0=np.array([0]),
                args=(gc_val_2, gc_val_1, True),
                method='L-BFGS-B', bounds=[(0, None)]
            )
            logging.info(f'Compare {name1} and {name2} in task {height}, {name1} is boolean, traverse GC statistics '
                         f'of {name2}, optimum threshold is {result.x}.')
            dist_mat[height].loc[name1, name2] = result.fun
            bool_2 = gc_val_2 > result.x
            cmat = confusion_matrix(gc_val_1, bool_2, labels=[True, False])
            odd, p = fisher_exact(cmat)
            fisher_mat[height].loc[name1, name2] = p
        elif cfg1['type'] == 'gc' and cfg2['type'] == 'b':
            result = minimize(
                fun=inconsistency, x0=np.array([0]),
                args=(gc_val_1, gc_val_2, True),
                method='L-BFGS-B', bounds=[(0, None)]
            )
            logging.info(f'Compare {name1} and {name2} in task {height}, {name2} is boolean, traverse GC statistics '
                         f'of {name1}, optimum threshold is {result.x}.')
            dist_mat[height].loc[name1, name2] = result.fun
            bool_1 = gc_val_1 > result.x
            cmat = confusion_matrix(gc_val_2, bool_1, labels=[True, False])
            odd, p = fisher_exact(cmat)
            fisher_mat[height].loc[name1, name2] = p
        elif cfg1['type'] == cfg2['type'] == 'p':
            dist_ = []
            for cv in cvs:
                bool_1 = gc_val_1 < cv
                bool_2 = gc_val_2 < cv
                logging.info(f'Compare {name1} and {name2} in task {height}, critical value is {cv}. Ratio of causal '
                             f'relationships for {name1} is {np.mean(bool_1)}, for {name2} is {np.mean(bool_2)}.')
                dist_.append(np.mean(bool_1 != bool_2))
            b = np.argmin(dist_)
            logging.info(f'Compare {name1} and {name2} in task {height}, p-value leading to maximum consistency is '
                         f'{cvs[b]}.')
            dist_mat[height].loc[name1, name2] = dist_[b]

            bool_1 = gc_val_1 < cvs[b]
            bool_2 = gc_val_2 < cvs[b]
            cmat = confusion_matrix(bool_1, bool_2, labels=[True, False])
            odd, p = fisher_exact(cmat)
            fisher_mat[height].loc[name1, name2] = p
        elif cfg1['type'] == 'p' and cfg2['type'] == 'gc':
            dist_ = []
            optim_threshold = []
            for cv in cvs:
                bool_1 = gc_val_1 < cv
                result = minimize(
                    fun=inconsistency, x0=np.array([0]),
                    args=(gc_val_2, bool_1, True),
                    method='L-BFGS-B', bounds=[(0, None)]
                )
                logging.info(f'Compare {name1} and {name2} in task {height}, critical value for p-value array {name1} '
                             f'is {cv}, ratio of causal relationships is {np.mean(bool_1)}. Traverse GC statistics '
                             f'{name2}, optimum threshold {result.x}.')
                dist_.append(result.fun)
                optim_threshold.append(result.x)
            b = np.argmin(dist_)
            logging.info(f'Compare {name1} and {name2} in task {height}, p-value leading to maximum consistency is '
                         f'{cvs[b]}.')
            dist_mat[height].loc[name1, name2] = dist_[b]
            bool_1 = gc_val_1 < cvs[b]
            bool_2 = gc_val_2 > optim_threshold[b]
            cmat = confusion_matrix(bool_1, bool_2, labels=[True, False])
            odd, p = fisher_exact(cmat)
            fisher_mat[height].loc[name1, name2] = p
        elif cfg1['type'] == 'gc' and cfg2['type'] == 'p':
            dist_ = []
            optim_threshold = []
            for cv in cvs:
                bool_2 = gc_val_2 < cv
                result = minimize(
                    fun=inconsistency, x0=np.array([0]),
                    args=(gc_val_1, bool_2, True),
                    method='L-BFGS-B', bounds=[(0, None)]
                )
                logging.info(f'Compare {name1} and {name2} in task {height}, critical value for p-value array {name2} '
                             f'is {cv}, ratio of causal relationships is {np.mean(bool_2)}. Traverse GC statistics '
                             f'{name1}, optimum threshold {result.x}.')
                dist_.append(result.fun)
                optim_threshold.append(result.x)
            b = np.argmin(dist_)
            logging.info(f'Compare {name1} and {name2} in task {height}, p-value leading to maximum consistency is '
                         f'{cvs[b]}.')
            dist_mat[height].loc[name1, name2] = dist_[b]
            bool_1 = gc_val_1 > optim_threshold[b]
            bool_2 = gc_val_2 < cvs[b]
            cmat = confusion_matrix(bool_1, bool_2, labels=[True, False])
            odd, p = fisher_exact(cmat)
            fisher_mat[height].loc[name1, name2] = p
        elif cfg1['type'] == cfg2['type'] == 'gc':
            corr = np.corrcoef(gc_val_1, gc_val_2)[0, 1]
            dist_mat[height].loc[name1, name2] = 1 - np.abs(corr)
            fisher_mat[height].loc[name1, name2] = 1 - np.abs(corr)
        else:
            raise Exception(f"Types not supported, type of method 1 is {cfg1['type']}, that of method 2 is "
                            f"{cfg2['type']}.")
# Same molecule
for (name1, cfg1), (name2, cfg2) in combinations(methods.items(), r=2):
    if name1 == name2:
        continue
    gc_df_1 = pd.read_pickle(cfg1['same_molecule'])
    gc_df_2 = pd.read_pickle(cfg2['same_molecule'])
    gc_df_2.reindex(index=gc_df_1.index, columns=gc_df_1.columns, copy=False)
    gc_val_1 = gc_df_1.values.flatten()
    gc_val_2 = gc_df_2.values.flatten()
    if cfg1['type'] == cfg2['type'] == 'b':
        dist_mat['same_molecule'].loc[name1, name2] = np.mean(gc_val_1 != gc_val_2)
        cmat = confusion_matrix(gc_val_1, gc_val_2, labels=[True, False])
        odd, p = fisher_exact(cmat)
        fisher_mat['same_molecule'].loc[name1, name2] = p
    elif cfg1['type'] == 'b' and cfg2['type'] == 'p':
        result = minimize(
            fun=inconsistency, x0=np.array([0.05]),
            args=(gc_val_2, gc_val_1, False),
            method='L-BFGS-B', bounds=[(0, 1)]
        )
        logging.info(f'Compare {name1} and {name2} in task same_molecule, {name1} is boolean, traverse p-value of '
                     f'{name2}, optimum critical value is {result.x}.')
        dist_mat['same_molecule'].loc[name1, name2] = result.fun
        bool_2 = gc_val_2 < result.x
        cmat = confusion_matrix(gc_val_1, bool_2, labels=[True, False])
        odd, p = fisher_exact(cmat)
        fisher_mat['same_molecule'].loc[name1, name2] = p
    elif cfg1['type'] == 'p' and cfg2['type'] == 'b':
        result = minimize(
            fun=inconsistency, x0=np.array([0.05]),
            args=(gc_val_1, gc_val_2, False),
            method='L-BFGS-B', bounds=[(0, 1)]
        )
        logging.info(f'Compare {name1} and {name2} in task same_molecule, {name2} is boolean, traverse p-value of '
                     f'{name1}, optimum critical value is {result.x}.')
        dist_mat['same_molecule'].loc[name1, name2] = result.fun
        bool_1 = gc_val_1 < result.x
        cmat = confusion_matrix(gc_val_2, bool_1, labels=[True, False])
        odd, p = fisher_exact(cmat)
        fisher_mat['same_molecule'].loc[name1, name2] = p
    elif cfg1['type'] == 'b' and cfg2['type'] == 'gc':
        result = minimize(
            fun=inconsistency, x0=np.array([0]),
            args=(gc_val_2, gc_val_1, True),
            method='L-BFGS-B', bounds=[(0, None)]
        )
        logging.info(f'Compare {name1} and {name2} in task same_molecule, {name1} is boolean, traverse GC statistics '
                     f'of {name2}, optimum threshold is {result.x}.')
        dist_mat['same_molecule'].loc[name1, name2] = result.fun
        bool_2 = gc_val_2 > result.x
        cmat = confusion_matrix(gc_val_1, bool_2, labels=[True, False])
        odd, p = fisher_exact(cmat)
        fisher_mat['same_molecule'].loc[name1, name2] = p
    elif cfg1['type'] == 'gc' and cfg2['type'] == 'b':
        result = minimize(
            fun=inconsistency, x0=np.array([0]),
            args=(gc_val_1, gc_val_2, True),
            method='L-BFGS-B', bounds=[(0, None)]
        )
        logging.info(f'Compare {name1} and {name2} in task same_molecule, {name2} is boolean, traverse GC statistics '
                     f'of {name1}, optimum threshold is {result.x}.')
        dist_mat['same_molecule'].loc[name1, name2] = result.fun
        bool_1 = gc_val_1 > result.x
        cmat = confusion_matrix(gc_val_2, bool_1, labels=[True, False])
        odd, p = fisher_exact(cmat)
        fisher_mat['same_molecule'].loc[name1, name2] = p
    elif cfg1['type'] == cfg2['type'] == 'p':
        dist_ = []
        for cv in cvs:
            bool_1 = gc_val_1 < cv
            bool_2 = gc_val_2 < cv
            logging.info(f'Compare {name1} and {name2} in task same_molecule, critical value is {cv}. Ratio of causal '
                         f'relationships for {name1} is {np.mean(bool_1)}, for {name2} is {np.mean(bool_2)}.')
            dist_.append(np.mean(bool_1 != bool_2))
        b = np.argmin(dist_)
        logging.info(f'Compare {name1} and {name2} in task same_molecule, p-value leading to maximum consistency is '
                     f'{cvs[b]}.')
        dist_mat['same_molecule'].loc[name1, name2] = dist_[b]
        bool_1 = gc_val_1 < cvs[b]
        bool_2 = gc_val_2 < cvs[b]
        cmat = confusion_matrix(bool_1, bool_2, labels=[True, False])
        odd, p = fisher_exact(cmat)
        fisher_mat['same_molecule'].loc[name1, name2] = p
    elif cfg1['type'] == 'p' and cfg2['type'] == 'gc':
        dist_ = []
        optim_threshold = []
        for cv in cvs:
            bool_1 = gc_val_1 < cv
            result = minimize(
                fun=inconsistency, x0=np.array([0]),
                args=(gc_val_2, bool_1, True),
                method='L-BFGS-B', bounds=[(0, None)]
            )
            logging.info(f'Compare {name1} and {name2} in task same_molecule, critical value for p-value array {name1} '
                         f'is {cv}, ratio of causal relationships is {np.mean(bool_1)}. Traverse GC statistics '
                         f'{name2}, optimum threshold {result.x}.')
            dist_.append(result.fun)
            optim_threshold.append(result.x)
        b = np.argmin(dist_)
        logging.info(f'Compare {name1} and {name2} in task same_molecule, p-value leading to maximum consistency is '
                     f'{cvs[b]}.')
        dist_mat['same_molecule'].loc[name1, name2] = dist_[b]
        bool_1 = gc_val_1 < cvs[b]
        bool_2 = gc_val_2 > optim_threshold[b]
        cmat = confusion_matrix(bool_1, bool_2, labels=[True, False])
        odd, p = fisher_exact(cmat)
        fisher_mat['same_molecule'].loc[name1, name2] = p
    elif cfg1['type'] == 'gc' and cfg2['type'] == 'p':
        dist_ = []
        optim_threshold = []
        for cv in cvs:
            bool_2 = gc_val_2 < cv
            result = minimize(
                fun=inconsistency, x0=np.array([0]),
                args=(gc_val_1, bool_2, True),
                method='L-BFGS-B', bounds=[(0, None)]
            )
            logging.info(f'Compare {name1} and {name2} in task same_molecule, critical value for p-value array {name2} '
                         f'is {cv}, ratio of causal relationships is {np.mean(bool_2)}. Traverse GC statistics '
                         f'{name1}, optimum threshold {result.x}.')
            dist_.append(result.fun)
            optim_threshold.append(result.x)
        b = np.argmin(dist_)
        logging.info(f'Compare {name1} and {name2} in task same_molecule, p-value leading to maximum consistency is '
                     f'{cvs[b]}.')
        dist_mat['same_molecule'].loc[name1, name2] = dist_[b]
        bool_1 = gc_val_1 > optim_threshold[b]
        bool_2 = gc_val_2 < cvs[b]
        cmat = confusion_matrix(bool_1, bool_2, labels=[True, False])
        odd, p = fisher_exact(cmat)
        fisher_mat['same_molecule'].loc[name1, name2] = p
    elif cfg1['type'] == cfg2['type'] == 'gc':
        corr = np.corrcoef(gc_val_1, gc_val_2)[0, 1]
        dist_mat['same_molecule'].loc[name1, name2] = 1 - np.abs(corr)
        fisher_mat['same_molecule'].loc[name1, name2] = 1 - np.abs(corr)
    else:
        raise Exception(f"Types not supported, type of method 1 is {cfg1['type']}, that of method 2 is "
                        f"{cfg2['type']}.")

# %% Export.
pd.to_pickle(dist_mat, f"raw/14_{dataset_name}_consistency_matrix_{lag}.pkl")
pd.to_pickle(fisher_mat, f"raw/14_{dataset_name}_fisher_p_{lag}.pkl")

# %% Initialization.
os.makedirs(f'results/14_{dataset_name}_dendrogram_{lag}/', exist_ok=True)
method_family_colors = {
    'Ours': '#3b4cc0',
    'Other GC': '#96b7ff',
    'non-GC': '#c3543c'
}
default_color = '#dddcdc'

# %% Consistency per task by accuracy.
for task, dist in dist_mat.items():
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
    fig.savefig(f'results/14_{dataset_name}_dendrogram_{lag}/acc_{task}.eps')
    plt.close(fig)

# %% Consistency overall by accuracy.
dist_overall = np.mean([dist.values for dist in dist_mat.values()], axis=0)
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
fig.savefig(f'results/14_{dataset_name}_dendrogram_{lag}/acc_overall.eps')
plt.close(fig)

# %% Consistency per task by Fisher's exact test.
for task, dist in fisher_mat.items():
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
    fig.savefig(f'results/14_{dataset_name}_dendrogram_{lag}/fisher_{task}.eps')
    plt.close(fig)

# %% Consistency overall by Fisher's exact test.
dist_overall = np.exp(np.mean([np.log(dist.values) for dist in fisher_mat.values()], axis=0))
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
fig.savefig(f'results/14_{dataset_name}_dendrogram_{lag}/fisher_overall.eps')
plt.close(fig)
