"""
Feature list:
    * Average Rate of Change
    * Stddev
    * Min/Max
    * Averages in 3rds of total time
"""
from typing import Callable
import pandas as pd
import numpy as np

def avg_change(row):
    dx_sum = sum([row[i+1] - row[i] for i in range(len(row)) if i < 59])
    dx_avg = dx_sum/(len(row)- 1)
    return dx_avg

def divide_row_into_thrids():
    """Divides a row into thrids."""

    pass


def table_apply(data:dict, func:Callable, suffix:str="", **kwargs):
    """Function eliminates some boiler-plate code when extracting new features. Loops through a dictionary of DataFrame objects and applies a function to each.

        Args:
            data (dict): A dictionary of Pandas DataFrame objects.
            func (function): A function to be applied to each DataFrame.
            suffix (str): A string to add to the front of the column names.
            **kwargs: Keyword arguments to pass to the apply() function. Pass 'axis=1' for row-wise aggregation.

        Returns:
            The aggregated data from each table as columns in a new DataFrame object."""

    tbl_list = list(data.keys())
    output = pd.DataFrame()
    for key in tbl_list:
        output[f"{suffix}_{key}"] = data[key].apply(func, **kwargs)
    return output

