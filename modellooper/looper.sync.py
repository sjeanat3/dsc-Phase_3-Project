# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os
import pickle
from glob import iglob
import json
from joblib import dump
from time import perf_counter
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.errors import ParserError
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, plot_confusion_matrix,
                             plot_roc_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from multimodel import MultiModel

# %%
run "../graph_tool.py"

# %%
# %reload_ext autoreload
# %autoreload 1 
# %aimport modeling 

# %%
with open('../features/cond_encoding.pkl', 'rb') as encodings:
    target = pickle.load(encodings)
    target = [col_str.strip('/% ').rstrip() for col_str in target.keys()]
    target = [itm.split() for itm in target]
    for ind, itm in enumerate(target):
        while "/" in itm:
            itm.remove("/")
        target[ind] = "_".join(itm)

# %%
y_data = pd.read_csv('../data/profile.txt', sep="\t", header=None, names=target)
for col in y_data.columns:
    y_data[col].to_pickle(f"./target_variables/{col}.pkl")

# %%
features = [] 
for itm in iglob('../features/*.pkl'):
    filename = os.path.basename(itm)[:-4]
    if filename in ["cond_encoding", "sensor_info"]:
        continue
    features.append((filename, itm)) 
target_vars = [] 
for f_path in iglob("./target_variables/*.pkl"):
    filename = os.path.basename(f_path)[:-4]
    target_vars.append((filename, f_path))

# %%
args_list = []
for name, path in target_vars:
    for feature, pkl in features:
        args_list.append((pkl, path, feature, name))

# %%
def pool_func(args):
    start_t = perf_counter()
    _, _, feature, target = args
    model_inst = MultiModel(*args)
    model_inst.fit_model()
    dump(model_inst, f"../models/{target}_{feature}.pkl")
    with open(f"../models/{target}_{feature}.txt", "w") as log:
        log.writelines(model_inst.print_m())
    end_t = perf_counter()
    return (f"Target variable {target} with feature set {feature}", end_t - start_t)
    
# %%
for args in args_list:
    msg, duration = pool_func(args)
    print(f"{msg} took {duration} to compute.")

# %%
test_summary = []
for text in iglob("../models/*.txt"):
    with open(text, 'r') as log:
        output = log.readlines()
        index = 0
        test_summary.append("-" * 60 + "\n")
        for line in output:
            if not index:
                test_summary.append(line.split("-")[-1] + "\n")
            if index == 1:
                test_summary.append(line.split("L")[0] + "\n")
            if "{" in line:
                 continue
            if line[0] in ["K", "X"]:
                test_summary.append(f"{line}\n")
            index += 1

# %%
with open('../models/test_summary.txt', 'w') as sum:
    sum.writelines(test_summary)
            

