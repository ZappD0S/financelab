import numpy as np
import pandas as pd
from numba import njit, prange

@njit(parallel=True)
def compute_stuff(p, lower, upper):
    res = np.zeros((lower.size, upper.size, 2))

    for t0 in prange(p.size):
        for l in prange(lower.size):
            for u in prange(upper.size):
                for t in range(t0 + 1, p.size):
                    if p[t] - p[t0] < lower[l]:
                        res[l, u, 0] += 1
                        break
                    if p[t] - p[t0] > upper[u, 1]:
                        res[l, u, 1] += 1
                        break
    return res

df = pd.read_parquet("tick_data/eurusd_2019.parquet.gzip")
p = df["buy"].values

lower = -np.linspace(1e-5, 1e-1, 200)
upper = np.linspace(1e-5, 1e-1, 200)

thresholds = np.stack((lower, upper), axis=-1)
# res = compute_stuff(p, thresholds)