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

def feature_wrap(data, *, wrap_func:Callable, name:str, suffix:str):
    return pd.Series(wrap_func(data), index=[f"{suffix}{name}"])

def avg_change(row, *, name:str="", suffix:str=""):
    dx_sum = sum([row[i+1] - row[i] for i in range(len(row)) if i < len(row)-1])
    dx_avg = dx_sum/(len(row)- 1)
    return dx_avg

def thirds_apply(row, *, thirds_func:Callable, suffix:str="", name:str):
    """Divides a row into thrids."""
    cursor = 0
    data = []
    row_size = int(len(row)/3)
    for third in range(3):
        val = []
        for itm in range(row_size):
            val.append(row[cursor])
            cursor += 1
        data.append(thirds_func(val, name=name, suffix=name))
    output = pd.Series(data=data, index=[f"{suffix}{name}_1", f"{suffix}{name}_2", f"{suffix}{name}_3"]) 
    return output 
    # return type(row_size)


def table_apply(data:dict, func:Callable, suffix:str="", **kwargs):
    """Function eliminates some boiler-plate code when extracting new features. Loops through a dictionary of DataFrame objects and applies a function to each.

        Args:
            data (dict): A dictionary of Pandas DataFrame objects.
            func (function): A function to be applied to each DataFrame.
            suffix (str): A string to add to the front of the column names.
            **kwargs: Keyword arguments to pass to the apply() function. Pass 'axis=1' for row-wise aggregation.

        Returns:
            The aggregated data from each table as columns in a new DataFrame object."""
    
    if suffix:
        suffix = f"{suffix}_"
        kwargs.update({'suffix':suffix})
    tbl_list = list(data.keys())
    output = pd.DataFrame(index=np.linspace(0,2204,2205, dtype=int)) # TODO Generalize this intialization
    for key in tbl_list:
        kwargs.update({'name':key})
        print(f"Calculating {kwargs['suffix']} for {kwargs['name']}...")
        result = data[key].apply(func, **kwargs) # TODO add loading bar to functions that prints a '.' for some division of rows processed
        output = output.join(result)
    return output

 
