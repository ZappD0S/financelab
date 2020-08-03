import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit, prange, int64, float32


@njit(parallel=True, fastmath=True)
# def compute_deltas(p, N, spread):
def compute_stop_inds(p, N, spread):
    res = np.zeros(N, dtype=int64)

    for i in prange(N):
        for j in range(i + 1, p.size):
            if abs(p[j] - p[i]) > spread:
                res[i] = j
                break

    return res


@njit(parallel=True, fastmath=True)
def compute_avg_trend(p, lengths, n_samples):
    means = np.empty(lengths.size, dtype=float32)
    stds = np.empty(lengths.size, dtype=float32)

    for i, length in enumerate(lengths):
        print(i, length)
        max_samples = p.size - length + 1
        inds = np.random.choice(max_samples, size=min(n_samples, max_samples), replace=False)
        A = np.arange(length).astype(float32).reshape(-1, 1) ** np.arange(2)[::-1]
        slopes = np.empty(inds.size, dtype=float32)

        for j in prange(inds.size):
            ind = inds[j]
            coeffs, _, _, _ = np.linalg.lstsq(A, p[ind : ind + length])
            slopes[j] = coeffs[0]

        means[i] = slopes.mean()
        stds[i] = slopes.std() / np.sqrt(slopes.size)

    return means, stds


@njit(fastmath=True)
def remove_small_deltas(p, times, threshold):
    assert p.ndim == times.ndim == 1
    assert p.size == times.size

    inds = np.empty(p.size, dtype=int64)
    last_t = 0
    count = 0
    for t in range(1, p.size):
        if times[t] - times[last_t] > threshold:
            gap1, gap2 = abs(times[t - 1] - threshold), abs(times[t] - threshold)
            inds[count] = t + (min(gap1, gap2) == gap2)
            last_t = inds[count]
            count += 1

    return inds[:count]


df = pd.read_parquet("tick_data/eurusd_2019.parquet.gzip")
p = df["buy"].values.astype("float32")
t = df.index.values

N = int(5e6)

stop_inds = compute_stop_inds(p, N, 1.5e-4)
deltas = t[stop_inds] - t[:N]
deltas.sort()

# threshold = np.int64(deltas.mean() / 100)
# threshold = np.int64(deltas[int(0.1 * (deltas.size - 1))] / 100)
threshold = np.timedelta64(1, "s").astype(deltas.dtype).astype("int64")

inds = remove_small_deltas(p, (t - t[0]).astype("int64"), threshold)


# lengths = np.linspace(p.size, p.size / 100, 30, dtype="int64")
# lengths = np.linspace(1.52e7, 1.42e7, 10, dtype="int64")
# means, stds = compute_avg_trend(p, lengths, 100)
