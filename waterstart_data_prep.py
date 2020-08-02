import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit, prange, int32, float32


@njit(parallel=True, fastmath=True)
# def compute_deltas(p, N, spread):
def compute_stop_inds(p, N, spread):
    res = np.zeros(N, dtype=int32)

    for i in prange(N):
        for j in range(i + 1, p.size):
            if abs(p[j] - p[i]) > spread:
                res[i] = j
                break

    return res


@njit(fastmath=True)
def compute_avg_trend(p, lengths, n_samples):
    means = np.empty(lengths.size, dtype=float32)
    stds = np.empty(lengths.size, dtype=float32)

    for i, length in enumerate(lengths):
        print(i, length)
        max_samples = p.size - length + 1
        inds = np.random.choice(max_samples, size=min(n_samples, max_samples), replace=False)
        A = np.arange(length, dtype=float32).reshape(-1, 1) ** np.arange(2)[::-1]
        slopes = np.empty(inds.size, dtype=float32)
        for j, ind in enumerate(inds):
            coeffs, res, rnk, s = np.linalg.lstsq(A, p[ind : ind + length])
            assert coeffs.size == 2
            slopes[j] = coeffs[0]

        means[i] = slopes.mean()
        stds[i] = slopes.std() / np.sqrt(slopes.size)

    return means, stds


df = pd.read_parquet("tick_data/eurusd_2019.parquet.gzip")
p = df["buy"].values
t = df.index.values

N = int(5e6)

# stop_inds = compute_stop_inds(p, N, 1.5e-4)
# deltas = t[stop_inds] - t[:N]

lengths = np.linspace(p.size, p.size / 100, 30, dtype="int64")
means, stds = compute_avg_trend(p.astype("float32"), lengths, 100)
