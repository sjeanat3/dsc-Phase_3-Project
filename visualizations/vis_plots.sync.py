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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# %%
df = pd.read_pickle('../features/std_3rds.pkl')
df.head()

# %%
def bounce(time:float, amp:float, freq:float, decay:float):
    return 1 + (-1 * amp * np.cos(time*freq*np.pi)/np.e**(time*decay))
    

# %%
fig, ax = plt.subplots()
plt.xlim(0, 3)
plt.ylim(-.5, 2)
x_axis = np.linspace(0, 3, 200)
y_axis = []
amp = 1
freq = 4.0
decay = 2
for y_t in x_axis:
    y_axis.append(bounce(y_t,amp, freq, decay))
ax.plot(x_axis, y_axis)
plt.show()

# %%

