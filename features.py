"""
Feature list:
    * Average Rate of Change
    * Stddev
    * Min/Max
    * Averages in 3rds of total time
"""
import pandas as pd
import numpy as np

def avg_change(row):
    dx_sum = sum([row[i+1] - row[i] for i in range(len(row)) if i < 59])
    dx_avg = dx_sum/(len(row)- 1)
    return dx_avg


def table_apply(data, func, suffix=None, **kwargs):
    tbl_list = list(data.keys())
    output = pd.DataFrame()
    for key in tbl_list:
        output[f"{suffix}_{key}"] = data[key].apply(func, **kwargs)
    return output

