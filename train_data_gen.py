import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit, njit
from utils import rolling_window
from scipy import stats
from scipy.signal import correlate
import h5py
from fracdiff import get_weights

def autocorr(x):
    mu = x.mean()
    x_std = (x - mu)/x.std() + mu
    result = correlate(x_std, x_std, mode='full')
    return result[result.size//2:]/x.size

def build_mu_sigma(mu_sigma):
    # for i in range(mu_sigma.shape[0]):
    #     for j in range(mu_sigma.shape[1]):
    #         mu_sigma[i, j, 0] = np.mean(windowed_logrets[i, :min_steps + j])
    #         mu_sigma[i, j, 1] = np.std(windowed_logrets[i, :min_steps + i])
    #         mu_sigma[i, j, 2] = stats.skew(windowed_logrets[i, :min_steps + j])
    #         mu_sigma[i, j, 3] = stats.kurtosis(windowed_logrets[i, :min_steps + j])

    for i in range(lookbehind - min_steps):
        # TODO: bisogna invertire la direzione!
        mu_sigma[:, i, 0] = np.mean(windowed_logrets[:, :min_steps+i], axis=1)
        mu_sigma[:, i, 1] = np.std(windowed_logrets[:, :min_steps+i], axis=1)
        mu_sigma[:, i, 2] = stats.skew(windowed_logrets[:, :min_steps+i], axis=1)
        mu_sigma[:, i, 3] = stats.kurtosis(windowed_logrets[:, :min_steps+i], axis=1)


def build_input_array(x):
    # x[..., 0] = windowed_logrets

    for i in range(lookbehind):
        # x[:, i, 0] = logrets[i::lookbehind+1]
        x[:, i, 0] = logrets[i:-lookbehind+i+1]

    # x.write_direct(logrets, np.s_[...], np.s_[..., 0])
    # x.write_direct(logrets, np.s_[...], np.s_[..., 0])
    # x[..., 1] = np.mean(windowed_logrets, axis=1)
    # x[..., 2] = np.std(windowed_logrets, axis=1)
    # x[..., 3] = stats.skew(windowed_logrets, axis=1)
    # x[..., 4] = stats.kurtosis(windowed_logrets, axis=1)


@njit
def build_output_arr(y):
    for i in range(len(buy_close) - lookahead):
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

# minutely = df.resample('1min').last().dropna()['buy'].apply(np.log).values
# hourly = df.resample('1h').last().dropna()['buy'].apply(np.log).values
# daily = df.resample('1d').last().dropna()['buy'].apply(np.log).values


# logrets = np.diff(np.log(df['buy', 'close'].values))

# logrets = np.diff(np.log(df['buy'].values))

# windowed_logrets = rolling_window(logrets, lookbehind)

# fname = f'train_data/mu_sigma_w{lookbehind}.dat'
# fname = f'train_data/X_w{lookbehind}.dat'
# fname = f'train_data/X_w{lookbehind}.hdf5'

# np.save(fname)
# mu_sigma = np.memmap(fname, dtype='float64', mode='r+', shape=(*windowed_logrets.shape, 2))
# X = np.memmap(fname, dtype='float64', mode='r+',
#                      shape=(windowed_logrets.shape[0], windowed_logrets.shape[1] - min_steps, 4))



# X = np.memmap(fname, dtype='float32', mode='w+',
#                      shape=(windowed_logrets.shape[0], windowed_logrets.shape[1], 5))

buy_close, buy_high = df['buy', 'close'].values, df['buy', 'high'].values
sell_close, sell_low = df['sell', 'close'].values, df['sell', 'low'].values

y = np.empty(len(buy_close) - lookahead, dtype='int8')
# build_output_arr(y)

# y = y[:-lookahead-1]
y = y[lookbehind-1:]

weights = get_weights(0.6, lookbehind).flatten()
dlogp = np.convolve(np.log(df['buy', 'close'].values), weights, mode='valid')[:-lookahead]

np.savez(f'train_data/train_data_tf{timeframe}.npz', dlogp=dlogp, y=y)

# build_mu_sigma(mu_sigma, windowed_logrets)
#d = dict(zip(df.columns.values, df.to_numpy().T))
# y = np.empty(len(df.index) - lookahead, dtype='int8')
# build_output_arr(y, df)
# build_output_arr(y)
