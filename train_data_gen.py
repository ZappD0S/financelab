import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from numba import njit, prange
# from utils import rolling_window
# from scipy import stats
from scipy.signal import correlate
# import h5py
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


min_steps = 10
# lookbehind = 100
lookbehind = 2500
lookahead = 100
loss_threshold = 0.9998
timeframe = '1min'


df = pd.read_pickle('tick_data/eurusd_2018.pkl')
df = df.resample(timeframe).ohlc().dropna()
df.to_pickle('tick_data/eurusd_2018_1min.pkl')
raise Exception

buy_close, buy_high = df['buy', 'close'].values, df['buy', 'high'].values
sell_close, sell_low = df['sell', 'close'].values, df['sell', 'low'].values

y = np.empty(len(buy_close) - lookahead, dtype='int16')
# build_output_arr(y)
# np.save('y.npy', y)

y = y[lookbehind - 1:]

weights = get_weights(0.6, lookbehind).flatten()
dlogp = np.convolve(np.log(df['buy', 'close'].values), weights, mode='valid')[:-lookahead]

np.savez(f'train_data/train_data_tf{timeframe}.npz', dlogp=dlogp, y=y)
