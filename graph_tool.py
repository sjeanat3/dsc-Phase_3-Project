from functools import lru_cache
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from tabulate import tabulate

def make_array(num:int, div:int, opt:str=None, force_col:bool=False, verbose:bool=False):
    """Function that takes a length and then finds the 'best-fit' array for that number.

        Args:
            num (int): A length or number of elements to match the 2d array to.
            div (int): A number of divisions or columns to try to match.
            opt (str): An optional argument to pass to the guesser. This will bias the output towards returning more or fewer columns.
                    None : Default option. The guesser will move in both directions decreasing
                           and increasing the column count until it finds a match this will bias towards the larger column count.
                    '+'  : will tell the guesser to only increase divs, or column count until a match is found.
                    '-'  : will tell the guesser to only decrease divs, or column count until a match is found.
                    TODO:
                    '+|-': tell the guesser to move in both directions but if it finds two matches at one step to bias towards the larger column count.
                    '-|+': tell the guesser to move in both directions but if it finds two matches at one step to bias towards the lower column count.

            force_col (bool): When true the div parameter is forced and a grid with 'div' columns is returned.
            verbose (bool): Turn debugging messages on and off. Deafult off. Turn on if you want it to print what it's doing at each step.
    """

    def v_print(msg_string):
        if verbose is True:
            return print(msg_string)

    v_print(f"Passed arguments:\nnum = {num}, div = {div}, opt = {opt}, forced = {force_col}, verbose = {verbose}")
    if force_col is True:
        v_print("Working from forced branch.")
        rows, rem =  divmod(num, div) 
        if rem:
            rows += 1
        result = []
        count = 1
        for row in range(rows): 
            row_out = []
            for col in range(div):
                v_print(f"count: {count}")
                if count > num:
                    col = 'x'
                row_out.append((row, col))
                count += 1
            result.append(row_out)
        return result

    rows, rem = divmod(num, div)
    if rem == 0:
        result = []
        for row in range(rows):
            if div == 1:
                result.append((row,))
                continue
            item = []
            for col in range(div):
                # print(item)
                item.append((row, col))
            result.append(item)
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

def hist_grid (numeric_data:pd.DataFrame, *, size:float=2.5, grid_cols:int=4, colors: list=None, **kwargs) -> tuple[Figure, Axes]:
    """This function takes a dataframe of numeric values and returns a plt.subplots() object with an array of histogram plots

           Args:
                numeric_data (pandas.DataFrame): Works with a DataFrame object. In order to get a grid back it needs to be a non-prime number.
                size (float): Determines the size of each plot. Only accepts a single float value. Default is 2.5.
                grid_cols (int): Designates the desired number of columns. Use keyword arg 'force_col = True' to force desired column count.
                colors (list): Option to provide a list of colors that fit the Matplotlib color requirements. Defualts to the TABLEAU color dicitonary.
            Returns:
                It Returns a tuple of the 'figure' and 'axes' objects unpacked from 'subplots'. To
                render the output you need to write plt.show() or plt.savefig().
        """
    get_col_name = iter(numeric_data.columns)
    n_columns = len(numeric_data.columns)
    if not colors:
        colors = list(mcolors.TABLEAU_COLORS.keys())
    ind = 0

    while len(colors) < n_columns:
        colors.append(colors[ind % len(colors)])
        ind += 1
    color = iter(colors)

    base_array = make_array(n_columns, grid_cols, **kwargs)
    shape = [*np.shape(base_array)]
    pos_list = []
    if len(shape) > 2:
        shape.pop(-1)
        [pos_list.extend(row) for row in base_array]
        # print(f"shape is {shape}\nshape length = {len(shape)}")
        # print(pos_list)
    else:
        pos_list = base_array
        # print(f"shape is {shape}\nshape length = {len(shape)}")
        # print(pos_list)

    fig_height, fig_width = [shape[i] * size if(len(shape) > 1 or i < 1) else size * i  for i in range(2)]
    # print(f"figure width: {fig_width}\nfigure height: {fig_height}")
    fig, ax = plt.subplots(*shape, figsize=(fig_width, fig_height))
    fig.set_tight_layout(tight=True)
    for pos in pos_list:
        # print(pos)
        if 'x' in pos:
            continue
        name = next(get_col_name)
        ax[pos].hist(x=numeric_data[name], bins=16, color=next(color), alpha=.4)
        ax[pos].set_title(name)
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

