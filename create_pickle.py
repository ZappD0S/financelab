import numpy as np
import pandas as pd
import datetime

PAIR = "eurusd"
YEAR = 2019


# arr = np.loadtxt(f"tick_data/{PAIR}_{YEAR}.csv", delimiter=',', usecols=(1, 2))

df = pd.read_csv(f"tick_data/{PAIR}_{YEAR}.csv", header=None, index_col=0,
                 names=['timestamp', 'sell', 'buy'], usecols=[0, 1, 2])

df.index = pd.to_datetime(df.index, format="%Y%m%d %H%M%S%f")
# df = df[~df.index.duplicated()]

df.to_pickle(f"tick_data/{PAIR}_{YEAR}.pkl")
