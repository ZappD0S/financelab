import numpy as np
import pandas as pd
from datetime import datetime
import os
import re

SAVE_DIR = "./tick_data"
PATTERN = r"\w{6}_201801-202001.csv"

# https://stackoverflow.com/questions/15799162/resampling-within-a-pandas-multiindex
# https://stackoverflow.com/questions/19798229/how-to-do-group-by-on-a-multiindex-in-pandas

# df = pd.read_csv(
#     f"tick_data/{PAIR}_{YEAR}.csv", header=None, index_col=0, names=["timestamp", "sell", "buy"], usecols=[0, 1, 2]
# )

paths = [os.path.join(SAVE_DIR, path) for path in os.listdir(SAVE_DIR) if re.match(PATTERN, path)]

df = pd.read_csv(paths[0], index_col=0, names=["timestamp", "sell", "buy"], usecols=[0, 1, 2])
df.index = pd.to_datetime(df.index, format="%Y%m%d %H%M%S%f")

# df = df.stack()


# df = df[~df.index.duplicated()]
# df.to_pickle(f"tick_data/{PAIR}_{YEAR}.pkl")
