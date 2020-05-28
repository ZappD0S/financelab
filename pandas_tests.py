import numpy as np
import pandas as pd
from numba import njit


df = pd.read_csv("stock_data/IBM.csv", index_col=0, usecols=[0, 1, 2, 3, 4], parse_dates=True)

print(df.index.dayofyear.to_numpy())