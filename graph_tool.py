from functools import lru_cache
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from tabulate import tabulate

def make_array(num, div, opt=None, verbose=False):
    def v_print(msg_string):
        if verbose is True:
            return print(msg_string)

    rows, rem = divmod(num, div)
    if rem == 0:
        result = []
        for row in range(rows):
            row_out = []
            for col in range(div):
                # print(row_out)
                row_out.append(col)
            result.append(row_out)
        return result

    v_print(f"{div} does not go into {num} and equal amount of times:")
    v_print(f"{num}/{div} returns {rows} with {rem} remaining.")
    div_up, div_down = (div, div)
    while div_up < 1000:
        if opt in ("+", None):
            div_up += 1
        if opt in ("-", None) and div_down > 0:
            div_down -= 1
        v_print(f"trying {num}/{div_up}")
        if divmod(num, div_up)[1] == 0:
            v_print(f"success! recursing with {num} and {div_up}")
            return make_array(num, div_up, verbose=verbose)
        v_print(f"trying {num}/{div_down}")
        if divmod(num, div_down)[1] == 0:
            v_print(f"success! recursing with {num} and {div_down}")
            return make_array(num, div_down, verbose=verbose)

# This is all code from a jupyter notebook that needs to be refactored into a useable tool
def hist_grid (numeric_data, figure_size):
    numeric_col_names = list(numeric_data.columns)
    get_col_name = iter(numeric_col_names)
    color = iter(colors)
    num_of_columns = len(numeric_col_names)
    base_array = np.shape(make_array(num_of_columns, 4))
    fig, ax = plt.subplots(base_array[0], base_array[1], figsize=(figure_size))
    fig.set_tight_layout(tight=True)
    for row, obj in enumerate(base_array):
        for col in obj:
            name = next(get_col_name)
            ax[row, col].hist(x=numeric_data[name], bins=16, color=next(color), alpha=.4)
            ax[row, col].set_title(name)
        # print(row, col, next(column_gen), next(color))
    return (fig, ax)


class Data_Grid():

    """
    A Class to divide and group a dataset into a grid. Defaults to a Hexogonal 
    grid which is offset by 1/2 units every other row.
    """
    def __init__ (self,
                  data,
                  x_axis,
                  y_axis,
                  grid_type='h',
                  grid_base=100) -> None:

        self.data = data
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.grid_type = grid_type
        self.grid_base = grid_base


        if not isinstance(x_axis, str):
            raise TypeError("Both column name arguments must be passed as strings.")
        if not isinstance(y_axis, str):
            raise TypeError("Both column name arguments must be passed as strings.")


    @lru_cache(1)
    def xy_limits(self):
        """
        returns a tuple of arrays representing the points along each axis.
        """

        # To Do:
        # I want to be able to cache this somehow so that once it's run for a particular obj
        # the values are stored and just returned when it's called. Seems like this is something
        # that is pretty straightforward to do if I knew what I was doing.
        # I was right! Just add the functools @cache decorator

        x_col = self.data[self.x_axis]
        y_col = self.data[self.y_axis]

        # column limits and data paramters
        x_max, x_min = (x_col.max(), x_col.min())
        x_span = x_max - x_min
        y_max, y_min = (y_col.max(), y_col.min())
        y_span = y_max - y_min

        # Grid Parameters
        grid_height = round(self.grid_base * y_span / x_span)

        x_line = np.linspace(0, self.grid_base, self.grid_base + 1)
        y_line = np.linspace(0, grid_height, grid_height + 1)

        return (x_line, y_line)


    def grid_points(self, xy_limits):
        """
        returns a list of each point in the grid.
        """
        # grid points
        x_line, y_line = xy_limits()
        x_coord = []
        y_coord = []
        x_coord_a = []
        xx, yy = np.meshgrid(x_line, y_line)

        for row in xx:
            x_coord.extend(row)
        for col in yy:
            y_coord.extend(col)
        for ind, row in enumerate(xx):
            if ind % 2 == 0:
                x_coord_a.extend(row + 0.5)
                continue
            x_coord_a.extend(row)
        self.grid = {"grid":list(zip(x_coord, y_coord)), "offset_grid":list(zip(x_coord_a, y_coord))}

