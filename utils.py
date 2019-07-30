import numpy as np
import pandas as pd
from numba import njit


def remove_time_gaps(df, threshold):
    deltas = np.diff(df.index.values.astype(np.int64))

    mask = deltas > threshold
    mu = deltas[~mask].mean()
    sigma = deltas[~mask].std()

    new_values = np.random.randn(mask.sum()) * sigma + mu
    new_values = np.rint(new_values)

    deltas[mask] = new_values

    new_index = np.full_like(df.index.values, df.index.values[0])
    new_index[1:] += np.cumsum(deltas).astype(np.timedelta64)

    df.index = pd.DatetimeIndex(new_index)



# def remove_time_gaps(df, threshold):


def valid_windowing_parameters(x_size, width, step_size):
    return ((x_size - width) % step_size) == 0


def window_indexer(x_size, width, step_size=1):
    if not valid_windowing_parameters(x_size, width, step_size):
        raise Exception("riprova..")

    return (np.arange(width, dtype=int)[np.newaxis, :] + step_size *
            np.arange((x_size - width) / step_size + 1, dtype=int)[:, np.newaxis])


def rolling_window(x, window_size):
    stride = x.strides[0]
    return np.lib.stride_tricks.as_strided(x, (x.size - window_size + 1, window_size), (stride, stride))


def compute_mean(f, x):
    f /= np.sum(f)

    mu = np.sum(f * x)
    std = np.sqrt(np.sum(f * (x - mu) ** 2))

    count = 0
    while True:

        a = mu - 3*std
        b = mu + 3*std

        # interval = np.s_[np.abs(x - a).argmin():np.abs(x - b).argmin()]
        interval = slice(np.abs(x - a).argmin(),
                         np.abs(x - b).argmin())

        new_mu = np.sum(f[interval]/f[interval].sum() * x[interval])

        if new_mu == mu:
            print(count)
            return mu

        mu = new_mu
        std = np.sqrt(np.sum(f[interval]/f[interval].sum() * (x[interval] - mu)**2))
        count += 1


@njit
def filtered_mean_std(y, lag, threshold, influence):
    mean = np.zeros(y.size - lag)
    # std = np.zeros(y.size - lag)
    # mask = np.zeros
    std = np.zeros(y.size - 2*lag)

    for i in range(lag):
        mean[i] = np.mean(y[i:i+lag])

    filtered_y = y.copy()
    # for i in range(y.size - lag):
    for i in range(lag, y.size - lag):
        # delta = min(lag, i)

        # mean[i] = np.mean(filtered_y[i:i+lag])
        mean[i] = np.mean(y[i:i+lag])
        # std[i] = np.std(filtered_y[i:i+lag] - mean[i-lag:i])
        # std[i-lag] = np.std(filtered_y[i:i+lag] - mean[i-lag:i])
        std[i-lag] = np.std(y[i:i+lag] - mean[i-lag:i])
        # std[i-lag] = np.std(filtered_y[i:i+lag])

        # if np.abs(y[i+lag+1] - mean[i]) > threshold*std[i]:
        #     filtered_y[i+lag+1] = influence*y[i+lag+1] + (1 - influence)*filtered_y[i+lag]
        # else:
        #     filtered_y[i+lag+1] = y[i+lag+1]
        # if np.abs(y[i+lag] - mean[i]) > threshold*std[i]:
        if np.abs(y[i+lag] - mean[i]) > threshold*std[i-lag]:
            filtered_y[i+lag] = influence*y[i+lag] + (1 - influence)*filtered_y[i+lag-1]
        else:
            filtered_y[i+lag] = y[i+lag]

    return mean[lag:], std

