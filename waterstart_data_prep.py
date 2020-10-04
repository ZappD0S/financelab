import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit, prange, int64, float32


@njit(parallel=True, fastmath=True)
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
def remove_small_deltas(times, threshold):
    assert times.ndim == 1

    inds = np.empty(times.size, dtype=int64)
    t0 = 0
    count = 0
    for t in range(1, times.size):
        if times[t] - times[t0] > threshold:
            gap1, gap2 = abs(times[t - 1] - threshold), abs(times[t] - threshold)
            inds[count] = t + (min(gap1, gap2) == gap2)
            t0 = inds[count]
            count += 1

    return inds[:count]


df = pd.read_parquet("tick_data/eurusd_2019.parquet.gzip")
sell_price = df["sell"].values.astype("float64")
t = df.index.values

N = int(5e6)

stop_inds = compute_stop_inds(sell_price, N, 1.5e-4)
deltas = t[stop_inds] - t[:N]
deltas.sort()

# threshold = np.int64(deltas.mean() / 100)
# threshold = np.int64(deltas[int(0.1 * (deltas.size - 1))] / 100)
threshold = np.timedelta64(10, "s").astype(deltas.dtype).astype("int64")

inds = remove_small_deltas(t.astype("int64"), threshold)

sell_price = sell_price[inds]
t = t[inds]

dt = np.diff(t).astype("float64")
sell_price = sell_price[1:]

# order: buy, sell
prices = np.stack((sell_price + 1.5e-4, sell_price), axis=-1)

logp = np.log(prices[:, 0])
logp -= logp.mean()
logp /= logp.std()

# dt = np.diff(t).astype("float64")
dt /= dt.std()

input = np.stack((logp, dt), axis=-1)

input = input.astype("float32")
prices = prices.astype("float32")

np.savez_compressed("train_data/train_data_10s.npz", input=input, prices=prices)
# input = (input - input.mean(axis=0, keepdims=True)) / input.std(axis=0, keepdims=True)

# lengths = np.linspace(p.size, p.size / 100, 30, dtype="int64")
# lengths = np.linspace(1.52e7, 1.42e7, 10, dtype="int64")
# means, stds = compute_avg_trend(p, lengths, 100)
