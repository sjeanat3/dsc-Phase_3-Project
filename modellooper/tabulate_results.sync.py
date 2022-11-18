# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (learn-env)
#     language: python
#     name: python3
# ---

# %%
import os
from tabulate import tabulate
from glob import glob, iglob
import pandas as pd
from joblib import dump, load
import pickle
from typing import Union, BinaryIO

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
std_colors = mcolors.TABLEAU_COLORS
import seaborn as sns

# %%
with open("../models/test_summary.txt", "r") as file:
    summary = file.readlines()
for line in summary:
    if line.startswith("\n"):
        summary.remove(line)


# %%
table = [["Feature", "Target", "KNN Accuracry", "XGBoost Accuracy"]]
output = []
for line in range(len(summary)):
    if summary[line].startswith("F"):
        feat = summary[line].split()[-1]
        targ = summary[line+1].split()[-1]
        knn = summary[line+2].split()[-1]
        xgb = summary[line+3].split()[-1]
        table.append([feat, targ, knn, xgb])
print(tabulate(table, headers='firstrow', tablefmt='github'))

# %% [markdown]
# | Feature    | Target                    |   KNN Accuracry |   XGBoost Accuracy |
# |------------|---------------------------|-----------------|--------------------|
# | avg_3rds   | Cooler_Condition          |        1        |           1        |
# | avg_change | Cooler_Condition          |        0.924471 |           0.987915 |
# | cycle_mean | Cooler_Condition          |        1        |           1        |
# | dx_3rds    | Cooler_Condition          |        0.927492 |           0.996979 |
# | std_3rds   | Cooler_Condition          |        0.996979 |           0.990937 |
# | std_dev    | Cooler_Condition          |        0.990937 |           0.990937 |
# | avg_3rds   | Hydraulic_accumulator_bar |        0.963746 |           0.987915 |
# | avg_change | Hydraulic_accumulator_bar |        0.818731 |           0.963746 |
# | cycle_mean | Hydraulic_accumulator_bar |        0.963746 |           0.969789 |
# | dx_3rds    | Hydraulic_accumulator_bar |        0.670695 |           0.942598 |
# | std_3rds   | Hydraulic_accumulator_bar |        0.930514 |           0.975831 |
# | std_dev    | Hydraulic_accumulator_bar |        0.8429   |           0.915408 |
# | avg_3rds   | Internal_pump_leakage     |        0.990937 |           0.996979 |
# | avg_change | Internal_pump_leakage     |        0.697885 |           0.854985 |
# | cycle_mean | Internal_pump_leakage     |        0.990937 |           0.987915 |
# | dx_3rds    | Internal_pump_leakage     |        0.646526 |           0.794562 |
# | std_3rds   | Internal_pump_leakage     |        0.92145  |           0.969789 |
# | std_dev    | Internal_pump_leakage     |        0.963746 |           0.984894 |
# | avg_3rds   | stable_flag               |        0.966767 |           0.963746 |
# | avg_change | stable_flag               |        0.827795 |           0.942598 |
# | cycle_mean | stable_flag               |        0.963746 |           0.966767 |
# | dx_3rds    | stable_flag               |        0.800604 |           0.912387 |
# | std_3rds   | stable_flag               |        0.957704 |           0.969789 |
# | std_dev    | stable_flag               |        0.954683 |           0.969789 |
# | avg_3rds   | Valve_Condition           |        0.984894 |           0.987915 |
# | avg_change | Valve_Condition           |        0.465257 |           0.504532 |
# | cycle_mean | Valve_Condition           |        0.918429 |           0.97281  |
# | dx_3rds    | Valve_Condition           |        0.44713  |           0.483384 |
# | std_3rds   | Valve_Condition           |        0.694864 |           0.933535 |
# | std_dev    | Valve_Condition           |        0.752266 |           0.960725 |

# %% [markdown]
# | Feature    |   Target                  |   KNN Accuracry |   XGBoost Accuracy |
# |   :---:    |          :---:            |      :---:      |        :---:       |
# | std_3rds   | Cooler_Condition          |        0.996979 |           0.996979 |
# | avg_3rds   | Hydraulic_accumulator_bar |        0.963746 |           0.987915 |
# | avg_3rds   | Internal_pump_leakage     |        0.990937 |           0.996979 |
# | avg_3rds   | stable_flag               |        0.966767 |           0.963746 |
# | avg_3rds   | Valve_Condition           |        0.984894 |           0.987915 |

# %%
run "../src/RegularModel.py"

# %%
model = load("../models/Internal_pump_leakage_avg_3rds.pkl")


# %%
def feature_importance(file_path:str):
    model = load(file_path)
    feature_importances = model.xgb_grid_search.best_estimator_.steps[1][1].feature_importances_
    feature_columns = model._X_train.columns
    fi_list = list(filter(lambda x : x[1] > 0, zip(feature_columns, feature_importances)))
    fi_list.sort(reverse=True, key = lambda x : x[1])
    return fi_list


# %%
final_pairs = ['Cooler_Condition_std_3rds', 'Hydraulic_accumulator_bar_avg_3rds', 'Internal_pump_leakage_avg_3rds', 'stable_flag_avg_3rds', 'Valve_Condition_avg_3rds']


# %%
# files = [file for file in iglob("../models/*.pkl") if os.path.basename(file)[:-4] in final_pairs]

def get_feature_counts(sensor_info:str="./**/sensor_info.pkl", model_search:str="", top:int=5):
    with open(sensor_info, "rb") as file:
        sensors = pickle.load(file)
        col_names = sensors.keys() 
    files = [file for file in iglob(model_search)]
    feature_counts= {}
    for file in files:
        pair = os.path.basename(file)[:-4]
        fi = feature_importance(file)[:top]
    #     print(f"\n{pair}:")
        for name, itm in fi:
            name = name.split("_")
            feature = [word for word in name if word in col_names]
    #         print(f"{next(feature)}: {itm:.5f}")
            if feature[0] in feature_counts.keys():
                feature_counts[feature[0]] += 1
                continue
            feature_counts.update({feature[0]: 1})

    features = list(feature_counts.items())
    features.sort(reverse=True, key=lambda x : x[1])
    feature_ids, feature_count = [[k for k, v in features],
                 [v for k, v in features]]
    return (feature_ids, feature_count)


# %%
targets = ['Cooler_Condition*', 'Hydraulic_accumulator_bar*', 'Internal_pump_leakage*', 'stable_flag*', 'Valve_Condition*']

# %%
def plot_feature_counts(feature_search:Union[list, str], plot_columns:int=2, figsize=(10,10)):
    targets = [*feature_search]
    num_items = len(targets)
    print(num_items)
    if plot_columns > num_items:
        plot_columns = num_items
    if len(targets) < 2:
        fig, ax = plt.subplots(figsize=figsize, tight_layout="tight")
    else:
        rows, rem = divmod(num_items, plot_columns)
        shape = [rows + bool(rem), plot_columns]
        print(shape)
        fig, ax = plt.subplots(*shape, figsize=figsize, tight_layout="tight")

    for item, model in enumerate(targets):
        ax_i = ax
        if num_items > 2:
            print(divmod(item, plot_columns))
            ax_i = ax[divmod(item, plot_columns)]
        elif num_items == 2:
            ax_i = ax[item]
            print(type(ax_i))
        x, y = get_feature_counts("../features/sensor_info.pkl", model, top=5) # TODO - Generalize
        ax_i.set_title(model[10:-5])
        sns.barplot(x=x, y=y, ax=ax_i, palette=color_dict)

    return fig, ax

# %%
plot_feature_counts(feature_search=['../models/Cooler_Condition*.pkl', '../models/Valve_Condition*.pkl', '../models/Hydraulic_accumulator_bar*.pkl', '../models/stable_flag*.pkl', '../models/Internal_pump_leakage*.pkl'])
plt.show()
# divmod(6, 2)

# %%
def get_feature_avg(sensor_info:str="./**/sensor_info.pkl", model_search:str="", top:int=5):
    with open(sensor_info, "rb") as file:
        sensors = pickle.load(file)
        col_names = sensors.keys() 
    files = [file for file in iglob(model_search)]
    feature_counts= {}
    for file in files:
        pair = os.path.basename(file)[:-4]
        fi = feature_importance(file)[:top]
    #     print(f"\n{pair}:")
        for name, itm in fi:
            name = name.split("_")
#             feature = [word for word in name if word in col_names]
            feature = [name[ind] for ind in range(len(name)) if name[ind].isupper()]
            if not name[-1] == feature[0]:
                feature[0] = (feature[0], name[-1])
#             feature = [name[ind+1] for ind in range(len(name)) if name[ind].isupper()]
#             print(f"{next(feature)}: {itm:.5f}")
            if feature[0] in feature_counts.keys():
                feature_counts[feature[0]].append(itm)
                continue
            feature_counts.update({feature[0]: [itm]})

    features = list(feature_counts.items())
    features.sort(reverse=True, key=lambda x : x[1])
    feature_ids, feature_arr = [[k for k, v in features],
                 [v for k, v in features]]
    feature_avg = []
    for arr in feature_arr:
        feature_avg.append(np.mean(arr))
    return feature_ids, feature_avg


# %%
name, avg = get_feature_avg("../features/sensor_info.pkl", "../models/*.pkl", top=5)

# %%
period = name
x = list(zip(period, avg))
x.sort(reverse=True, key=lambda x : x[1])
zero, one, two, three = [[(zero, avg) for zero, avg in x if len(zero[0]) < 2],
                         [(one, avg) for one, avg in x if one[1] == '1'],
                         [(two, avg) for two, avg in x if two[1] == '2'],
                         [(three, avg) for three, avg in x if three[1] == '3']]

# %%
zero_x, zero_y = [[name for name, avg in zero],
                  [avg for name, avg in zero]]
one_x, one_y = [[name[0] for name, avg in one],
                  [avg for name, avg in one]]
two_x, two_y = [[name[0] for name, avg in two],
                  [avg for name, avg in two]]
three_x, three_y = [[name[0] for name, avg in three],
                  [avg for name, avg in three]]


# %%
with open("../features/sensor_info.pkl", "rb") as binary:
    sensors = pickle.load(binary)
columns = list(sensors.keys())
colors = sns.cubehelix_palette(n_colors=17, start=2, gamma=.5, dark=.7, light=.6, hue=2, rot=9)
color_dict = {col: color for col, color in zip(columns, colors)}
colors

# %%
fig, ax = plt.subplots(2, 2, figsize=(12, 10), tight_layout='tight')
fig.suptitle("Feature Effect Averages", fontsize=16)
ax[0,0].set_title("Full Test Cycle")
ax[0,1].set_title("First 20 Seconds")
ax[1,0].set_title("Middle 20 Seconds")
ax[1,1].set_title("Last 20 Seconds")
sns.barplot(x=zero_x, y=zero_y, palette=color_dict, ax=ax[0,0])
sns.barplot(x=one_x, y=one_y,  palette=color_dict, ax=ax[0,1])
sns.barplot(x=two_x, y=two_y, palette=color_dict, ax=ax[1,0])
sns.barplot(x=three_x, y=three_y, palette=color_dict, ax=ax[1,1])
plt.show()

# %%
col = 2
itms = 12
[divmod(num, col) for num in range(itms)]
row, rem = divmod(itms, col)
print(row, rem)
shape = (row + bool(rem), col)
shape
