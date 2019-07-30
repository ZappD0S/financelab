import numpy as np
import pandas as pd
import datetime


arr = np.loadtxt('tick_data/eurusd_2018.csv', delimiter=',', usecols=(1, 2))

# df = pd.read_csv('tick_data/eurusd_2018.csv', sep=',', header=None, index_col=0,
#                  names=['timestamp', 'sell', 'buy'], usecols=[0, 1, 2])
#
# df.index = pd.to_datetime(df.index, format='%Y%m%d %H%M%S%f')
#
# df = df[~df.index.duplicated()]
#
# pd.to_pickle(df, 'tick_data/eurusd_2018.pkl')
