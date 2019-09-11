import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import numba
from numba import njit, prange
# from utils import rolling_window
# from scipy import stats
from scipy.signal import correlate
from fracdiff import get_weights


def autocorr(x):
    mu = x.mean()
    x_std = (x - mu) / x.std() + mu
    result = correlate(x_std, x_std, mode='full')
    return result[result.size // 2:] / x.size


@njit(parallel=True)
def build_output_arr(y):
    for i in prange(len(buy_close) - lookahead):
        sell_threshold_reached = False
        buy_threshold_reached = False
        profitable = False

        for j in range(lookahead):
            if sell_low[i + j] / buy_close[i] < loss_threshold:
                buy_threshold_reached = True

            if sell_close[i] / buy_high[i + j] < loss_threshold:
                sell_threshold_reached = True

            if not buy_threshold_reached and sell_close[i + j] > buy_close[i]:
                y[i] = 1
                profitable = True
                break
            if not sell_threshold_reached and sell_close[i] > buy_close[i + j]:
                y[i] = -1
                profitable = True
                break

        if not profitable:
            y[i] = 0


@njit(parallel=True)
def compute_good(log_gmin):
    shift = 0
    buy_counts = []
    sell_counts = []
    buy_mask = np.zeros(len(logp) - 1, dtype=numba.bool_)
    sell_mask = np.zeros(len(logp) - 1, dtype=numba.bool_)
    while True:
        shift += 1
        buy_count = 0
        sell_count = 0
        for i in prange(len(logp) - shift):
            # buy_mask[i] |= logp[i + shift] - logp[i] > log_gmin
            # sell_mask[i] |= logp[i] - logp[i + shift] > log_gmin
            if not buy_mask[i] and logp[i + shift] - logp[i] > log_gmin:
                buy_mask[i] = True
                buy_count += 1

            if not sell_mask[i] and logp[i] - logp[i + shift] > log_gmin:
                sell_mask[i] = True
                sell_count += 1

        buy_counts.append(buy_count)
        sell_counts.append(sell_count)
        # if np.sum(~(sell_mask | buy_mask)) / buy_mask.size < 1e-5:
        if np.all(sell_mask | buy_mask) or shift > 300:
        # if (sum(buy_counts) + sum(sell_counts)) / len(logp) >= 0.99:
        # if (cum_buy_count + cum_sell_count) / len(logp) >= 0.99:
            print(shift)
            break
    # return buy_mask, sell_mask
    # return buy_counts, sell_counts
    buy_counts = np.asarray(buy_counts)
    sell_counts = np.asarray(sell_counts)
    return (buy_mask, sell_mask), (buy_counts, sell_counts)


d = 0.6
min_steps = 10
# lookbehind = 100
lookbehind = 6000
lookahead = 100
loss_threshold = 0.9998
# timeframe = '1s'
timeframe = '30s'


df = pd.read_pickle('tick_data/eurusd_2018.pkl')
df = df.resample(timeframe).ohlc().dropna()
# df.to_pickle('tick_data/eurusd_2018_1min.pkl')

# buy_close, buy_high = df['buy', 'close'].values, df['buy', 'high'].values
# sell_close, sell_low = df['sell', 'close'].values, df['sell', 'low'].values

buy, sell = np.log(df['buy', 'close'].values), np.log(df['sell', 'close'].values)
logp = np.log(df['buy', 'close'].values)
# logp = np.log(df['buy'].values)
# buy_mask, sell_mask = compute_good(1.7e-5)
(buy_mask, sell_mask), (buy_counts, sell_counts) = compute_good(1.2e-4)
# raise Exception
to_skip = len(buy_counts)
# buy_mask, sell_mask = compute_good(6.1e-5)

y = np.empty(len(logp) - 1, dtype='uint8')
y[buy_mask & sell_mask] = 0
y[buy_mask & ~sell_mask] = 1
y[~buy_mask & sell_mask] = 2
if np.any(~(buy_mask | sell_mask)):
    y[~(buy_mask | sell_mask)] = 3
# build_output_arr(y)
y = y[:-to_skip + 1]
logp = logp[:-to_skip]

assert y.size == logp.size
# y = y[lookbehind - 1:]

# weights = get_weights(d, lookbehind).flatten()
# dlogp = np.convolve(np.log(df['buy', 'close'].values), weights, mode='valid')[:-lookahead]
# assert dlogp.size == y.size

with h5py.File(f'train_data/train_data_tf{timeframe}_lfz.h5', 'w') as f:
    # f.create_dataset('dlogp', data=dlogp, compression='lzf')
    f.create_dataset('logp', data=logp, compression='lzf')
    f.create_dataset('y', data=y, compression='lzf')
    f.flush()
