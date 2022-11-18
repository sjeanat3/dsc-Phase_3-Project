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

FINAL_PAIRS = ['Cooler_Condition_std_3rds', 'Hydraulic_accumulator_bar_avg_3rds', 'Internal_pump_leakage_avg_3rds', 'stable_flag_avg_3rds', 'Valve_Condition_avg_3rds']


def feature_importance(file_path:str):
    model = load(file_path)
    feature_importances = model.xgb_grid_search.best_estimator_.steps[1][1].feature_importances_
    feature_columns = model._X_train.columns
    fi_list = list(filter(lambda x : x[1] > 0, zip(feature_columns, feature_importances)))
    fi_list.sort(reverse=True, key = lambda x : x[1])
    return fi_list

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


def plot_feature_counts(feature_search:Union[list, str], plot_columns:int=2, figsize=(10,10)):
    targets = [*feature_search]
    num_items = len(targets)
    # print(num_items)
    if plot_columns > num_items:
        plot_columns = num_items
    if len(targets) < 2:
        fig, ax = plt.subplots(figsize=figsize, tight_layout="tight")
    else:
        rows, rem = divmod(num_items, plot_columns)
        shape = [rows + bool(rem), plot_columns]
        # print(shape)
        fig, ax = plt.subplots(*shape, figsize=figsize, tight_layout="tight")

    for item, model in enumerate(targets):
        ax_i = ax
        if num_items > 2:
            ax_i = ax[divmod(item, plot_columns)]
            # print(divmod(item, plot_columns))
        elif num_items == 2:
            ax_i = ax[item]
            # print(type(ax_i))
        x, y = get_feature_counts(os.path.abspath("./features/sensor_info.pkl"), model, top=5) # TODO - Generalize
        ax_i.set_title(model[9:-5])
        sns.barplot(x=x, y=y, ax=ax_i, palette=color_dict)

    return fig, ax


def get_feature_counts(sensor_info:str="./features/sensor_info.pkl", model_search:str="", top:int=5):
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


def get_f1_weighted_score(name, grid_search, X_test, y_test):
    test_preds = grid_search.predict(X_test)
    text = "F-1 Weighted Score: "
    result = f1_score(y_test, test_preds, average = 'weighted')
    return text, result


def get_roc_auc_score(name, gird_search, X_test, y_test):
    test_probas = gird_search.predict_proba(X_test)
    score = roc_auc_score(y_test, 
                          test_probas,
                          multi_class = 'ovo',
                          average = 'weighted')
    text = "ROC-AUC Score: "
    return text, score

def get_accuracy(name, grid_search, X_test, y_test): 
    test_preds = grid_search.predict(X_test)
    text = 'Accuracy Score: '
    result = accuracy_score(y_test, test_preds)
    return text, result

def get_metrics(name, estimator:int, model_path:str):
    model = load(model_path)
    X_test = model._X_test
    y_test = model._y_test

    result = []
     
    estimator = model.grids[estimator]

    result.append(get_accuracy(name, estimator, X_test, y_test))
    result.append(get_roc_auc_score(name, estimator, X_test, y_test))
    result.append(get_f1_weighted_score(name, estimator, X_test, y_test))
    return result


def true_pos_rate(grid_search, X_test, y_test):
    test_preds = grid_search.predict(X_test)
    true_pos_rate = recall_score(y_test, test_preds)
    return true_pos_rate
