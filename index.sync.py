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

# %% [markdown]
# # The Beginning
# These first few rows are my very first attempts at EDA on this data. It took a little while for me
# to wrap my head around how to structure and use it all.

# %%
import numpy as np
import pandas as pd
from pandas.errors import ParserError
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# %%
df = pd.read_csv('./data/CE.txt', sep='\t')
df.describe() # type: ignore

# %% [markdown]
# ## Pulling in the Predictor Values
# Below is just a quick loop that stores each `txt` file as its own *DataFrame* object and sticks them
# all in a dictionary where their **key** is the 'basename' of each file path.

# %%
cwd = os.getcwd()
print(cwd)
tables = {}
for itm in glob.iglob("./**/*.txt"):
    id = os.path.basename(itm)
    id = id[:-4]
    if id in ["documentation", "description", "profile"]:
        continue
    print(id)
    try:
        tables.update({id: pd.read_csv(itm, header=None,  sep='\t')})
    except ParserError as err:
        print(err)
        continue

# %% [markdown]
# **The cell below is a nice visual of the data structure**. Each table in the dictionary has equal
# numbers of rows, and a column count that cooresponds to the frequency of samples taken during each
# pump cycle.

# %%
count = 0
for table in list(tables.keys()):
    print(table)
    print(tables[table].shape)

# %% [markdown]
# ## Structure of the Data
# **Okay, so the structure of the data is this:**
# 1. The rows represent 1 cycle of the hydraulic test rig.
# 2. The individual txt files are sensor readings, rows represent a cycle, each column is a reading
#    from that specific sensor.
# 3. Readings from each table are given in hz, and each cycle lasted 60 seconds. So, a 1hz sensor
#    provides a 60 column by 2205 row table.
# 4. "Profile.txt" contains a 5 column by 2205 row table with system states encoded in each column.

# %% [markdown]
# ## The Condition Encodings
# The `profile.txt` table contains the recorded pump conditions for each cycle (row). Below I am
# intitializing a dictionary to hold basic information about each table. This along with the dictionary
# of sensor information that follows will help establish a base on which to build out our model
# functionality.
#
# Practically, anytime we need to use this information in the future we can refer to it here, and 
# if we decide we need to change it later we can do it once here and those changes will propagate
# through the rest of our program automatically.

# %%
# Create a dictionary that will translate the values in "Profile.txt" into their text categories 
# describing the state of the test rig for each cycle.
encoding = {
        "Cooler Condition / %": {
             3:"close to failure",
             20:"reduced efficiency",
             100:"full eficiency",
            },
        "Valve Condition / %": {
             100:"optimal switching behavior",
             90:"small lag",
             80:"severe lag",
             73:"close to total failure"
            },
        "Internal pump leakage": {
             0:"no leakage",
             1:"weak leakage",
             2:"severe leakage"
            },
        "Hydraulic accumulator / bar": {
             130:"optimal pressure",
             115:"slightly reduced pressure",
             100:"severely reduced pressure",
             90:"close to total failure"
            },
        "stable flag": {
             0:"conditions were stable",
             1:"static conditions might not have been reached yet",
            }
        }

# %% [markdown]
# ### Predictor Variables
# I am going to create another dictionary below that maps the sensor table short hand names to something
# more readable. As I add features it will be more and more important to keep everything straight. To
# keep things straight it will be important to only refer to common information from a single point.
#
# This will help keep naming conventions and data meanings clear and will come in handy later if I 
# want to scale the rows in each column proportionally in some way that requires using their frequency
# or respecting their unit of measurement.
#
# #### Sensor Frequencies and Other ID Info:

# %%
sensor_dict = {
        "PS1": {
            "name": None,
            "type": "pressure",
            "unit": "bar",
            "samp_rate": 100,
            }, 
        "PS2": {
            "name": None,
            "type": "pressure",
            "unit": "bar",
            "samp_rate": 100,
            }, 
        "PS3": {
            "name": None,
            "type": "pressure",
            "unit": "bar",
            "samp_rate": 100,
            }, 
        "PS4": {
            "name": None,
            "type": "pressure",
            "unit": "bar",
            "samp_rate": 100,
            }, 
        "PS5": {
            "name": None,
            "type": "pressure",
            "unit": "bar",
            "samp_rate": 100,
            }, 
        "PS6": {
            "name": None,
            "type": "pressure",
            "unit": "bar",
            "samp_rate": 100,
            }, 
        "EPS1": {
            "name": None,
            "type": "motor_power",
            "unit": "W",
            "samp_rate": 100,
            }, 
        "FS1": {
            "name": None,
            "type": "volume_flow",
            "unit": "l/min",
            "samp_rate": 10,
            }, 
        "FS2": {
            "name": None,
            "type": "volume_flow",
            "unit": "l/min",
            "samp_rate": 10,
            }, 
        "TS1": {
            "name": None,
            "type": "temperature",
            "unit": "°C",
            "samp_rate": 1,
            }, 
        "TS2": {
            "name": None,
            "type": "temperature",
            "unit": "°C",
            "samp_rate": 1,
            }, 
        "TS3": {
            "name": None,
            "type": "temperature",
            "unit": "°C",
            "samp_rate": 1,
            }, 
        "TS4": {
            "name": None,
            "type": "temperature",
            "unit": "°C",
            "samp_rate": 1,
            }, 
        "VS1": {
            "name": None,
            "type": "vibration",
            "unit": "mm/s",
            "samp_rate": 1,
            }, 
        "CE": {
            "name": None,
            "type": "cooling_efficiency",
            "unit": "percent",
            "samp_rate": 1,
            }, 
        "CP": {
            "name": None,
            "type": "cooling_power",
            "unit": "kW",
            "samp_rate": 1,
            }, 
        "SE": {
            "name": None,
            "type": "efficiency_factor",
            "unit": "percent",
            "samp_rate": 1,
            }
        }

# %% [markdown]
# ### Adding Names Programmatically
# Here I am going to add names to the dictionary using a function so that if later I need or want
# to change them I already have quick and easy way to do that.

# %%
def rename_cols(data_dict=None, input_str=None, numbers=True,  *args, **kwargs):
    """A function to rename my columns from a dictionary containing information about
       each sensor.
            Args:
                data_dict ('dict'): The dictionary that will be processed
                    by the function. This dictionary should be built into the
                    feature extraction process.
                numbers (bool, 'defualt' = True): Tells the function whether or not to use the
                    numbers from the sensor dict keys in the output names.
                input_str ('str', optional): An input string to be proccessed
                    by the function.

            Returns:
                A dictionary with <sensor>['name'] field reprocessed
    """
    if data_dict is None:
        return None
    if input_str is None:
        input_str = ""

    for key in data_dict.keys():
        name = data_dict[key]['type'][:4]
        base = key
        num = ""
        num = ""
        if key[-1].isdigit() and numbers is True:
            num = f"_{key[-1]}"
            base = key[0:-1]
        if len(input_str) > 0:
            base = f"{input_str}"
        data_dict[key]['name'] = f"{base}_{name}{num}"

    return None

# %%
rename_cols(sensor_dict, "", True)

# %% [markdown]
# # Target Variables
# **Now that we can see the structure** of our target variables a little more clearly lets take a
# look at the `profile.txt` file in our dataset. 
#
# I will pull it inot a primary DataFrame object, so that we can continue to work with it; adding 
# predictor variables and iterating over a test pipeline to find the best combinations for prediction.
#
# Setting this up just requires pulling in the five columns and assigning column names based on our
# encoding keys from the above dictionary.

# %%
target_cols = list(encoding.keys())
df = pd.read_csv('./data/profile.txt', sep='\t', header=None, names=target_cols)
print(df.shape)
df.head()

# %% [markdown]
# ### Graphing the Cycles
# The columns of our DataFrame give us the encoded status of the test rig, and the rows represent
# each 60 second test cycle. That means we can graph these cycles to get a sense of the testing setup
# we will be trying to predict.
#
# Below I quickly iterate through the columns. Plotting each test components status for each cycle.

# %%
import matplotlib.colors as mcolors
colors = list(mcolors.TABLEAU_COLORS.values())

# %%
fig, ax = plt.subplots(5, 1, figsize=(12,14), tight_layout='tight')
x_axis = df.index
for col in range(len(target_cols)):
    ax[col].plot(x_axis, df[target_cols[col]], c=colors[col])
    ax[col].set_title(target_cols[col])
plt.show()

# %% [markdown]
# ## Next Steps
# Because the there is so much test data I need to setup a workflow that also me to iterate over it
# in many different ways. I think I am currently well positioned to do that, because I have my target
# variables all in a single DataFrame object and each of the sensor tables in its own separate df.
#
# ### Tables as Columns, Rows as Cells
# With the data structured the way it is I want to test out different ways of aggregating it into the
# test frame. There are essentially several thousand columns of data for each cycle. However, my
# plan now is to iterate through each table of sensor data and decide how to aggregate it for each
# each row. So, the simplest way to do this would be to collapse each table into a single column that
# captures the average value of each row.
#
# I can then expand the functionality of my iterating process to output standard-deviations, averages,
# and min/max values for each row. Then maybe test the effect of binning rows down to some proportion
# of their original size.
#
# The first thing to do that may come in handy is to make a dicitonary that holds the hertz values 
# for each sensor table. I could do this by hand but it will probably be easier to just write a loop.

# %% [markdown]
# ## Feature Extraction
# I want to establish my intial goals for the feature extraction and testing:
# 1. A function to return simple averages for each sensor table.
# 2. A function to return summary statistics for each cycle from the sensor tables.
# 3. A function to return summary statistics for each cycle as thirds of the cycle (20 seconds of
#    sensor data).
# 4. A function that returns the average change in sensor values for each cycle of sensor data.
#
# Iteratively test each of these sets of columns together and separately against each of the target
# variables to see which features produce the most accurate predicitons.
#
# ### Additional Features to Try
# * average rate of change during each cycle.
#
# ### Things to make sure to check for:
# * distributions of sensor datas
# * NaNs
# * outliers

# %%
fig, ax = plt.subplots()
for num in range(10):
    rand_row = np.random.randint(0, 2204)
    x_val = tables['TS1'].iloc[[rand_row]].columns
    y_val = tables['TS1'].iloc[[rand_row]].values.flatten()
    ax.set_title('Random Sample of Cycles')
    ax.plot(x_val, y_val, label=f"Cycle: {rand_row}")
ax.set(xlabel="Seconds", ylabel="Temperature")
ax.legend()
plt.show()

# %% [markdown]
# ### First Feature
# So this will be the basic pattern for feature extraction. Not too bad, I will need to write functions
# to capture features beyond basic aggregations.
#
# ### Feature: 60sec Mean Readings

# %%
feature_avg = pd.DataFrame()
for table in tables.keys():
    col_name = "avg_"
   feature_avg[table] = tables[table].apply(np.mean, axis=1) 
feature_avg.head()

# %%
from graph_tool import make_array

# %%
colors = list(mcolors.TABLEAU_COLORS.keys())
while len(colors) <  len(feature_avg.columns):
    rand_index = np.random.randint(0, len(colors))
    colors.append(colors[rand_index])

# %%
def hist_grid (numeric_data, figure_size):
    numeric_col_names = list(numeric_data.columns)
    get_col_name = iter(numeric_col_names)
    color = iter(colors)
    num_of_columns = len(numeric_col_names)
    base_array = make_array(num_of_columns, 4)
    fig, ax = plt.subplots(np.shape(base_array)[0], np.shape(base_array)[1], figsize=(figure_size))
    fig.set_tight_layout(tight=True)
    for row, obj in enumerate(base_array):
        for col in obj:
            name = next(get_col_name)
            ax[row, col].hist(x=numeric_data[name], bins=16, color=next(color), alpha=.4)
            ax[row, col].set_title(name)
        # print(row, col, next(column_gen), next(color))
    return (fig, ax)

# %%
test = feature_avg.drop(['TS4'], axis=1)
test.head()

# %%
hist_grid(test,(10, 10))
plt.show()