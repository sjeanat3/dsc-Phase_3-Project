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
from glob import iglob
import pandas as pd

# %%
with open("../models/test_summary.txt", "r") as file:
    summary = file.readlines()
for line in summary:
    if line.startswith("\n"):
        summary.remove(line)
print(*summary)

# %%
table = [["Target", "Feature", "KNN Accuracry", "XGBoost Accuracy"]]
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
# | Target     | Feature                   |   KNN Accuracry |   XGBoost Accuracy |
# |   :---:    |          :---:            |      :---:      |        :---:       |
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
# | Target     | Feature                   |   KNN Accuracry |   XGBoost Accuracy |
# |   :---:    |          :---:            |      :---:      |        :---:       |
# | dx_3rds    | Cooler_Condition          |        0.927492 |           0.996979 |
# | avg_3rds   | Hydraulic_accumulator_bar |        0.963746 |           0.987915 |
# | avg_3rds   | Internal_pump_leakage     |        0.990937 |           0.996979 |
# | avg_3rds   | stable_flag               |        0.966767 |           0.963746 |
# | avg_3rds   | Valve_Condition           |        0.984894 |           0.987915 |
