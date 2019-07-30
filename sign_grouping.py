import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import remove_time_gaps, rolling_window


def remove_false_positives(mask1, mask2):
    res = np.zeros_like(mask1)
    types = np.zeros_like(mask1, dtype=int)
    types[mask1] = 1
    types[mask2] = 2
    mask = mask1 | mask2

    nonzero_types = types[mask]
    # inds = np.argwhere(mask)
    inds = np.nonzero(mask)[0]

    false_positives = np.zeros_like(nonzero_types, dtype=bool)
    false_positives[1:] |= (nonzero_types[:-1] == 1) & (nonzero_types[1:] == 1)
    false_positives[1:] |= (nonzero_types[:-1] == 2) & (nonzero_types[1:] == 2)

    res[inds[~false_positives]] = True
    return res


def build_sign_change_mask(arr):
    mask1 = (arr[:-1] <= 0) & (arr[1:] > 0)
    mask2 = (arr[:-1] >= 0) & (arr[1:] < 0)
    return remove_false_positives(mask1, mask2)


df = pd.read_pickle('tick_data/eurusd_2018.pkl')
# remove_time_gaps(df, 6e10)
df = df.resample('1min').pad()
# df = df.resample('2h').pad()
df = df[1:]

p = df['buy'].values


window_size = 40

window = np.hanning(window_size)
window /= window.sum()
sma = np.convolve(p, window, 'valid')
# sma = rolling_window(p, lookahead_size).mean(axis=1)

p_deriv = np.diff(p)
# p_deriv = np.diff(sma)

mu = p_deriv.mean()
sigma = p_deriv.std()

# p_deriv = np.random.randn(p_deriv.size)*sigma + mu

# mask1 = (p_deriv[:-1] <= 0) & (p_deriv[1:] > 0)
# mask2 = (p_deriv[:-1] >= 0) & (p_deriv[1:] < 0)

# sign_change = remove_false_positives(mask1, mask2)
sign_change = build_sign_change_mask(p_deriv)

# inds = np.cumsum(np.append(False, sign_change))

# samesign_counts = np.bincount(inds)
# samesign_sums = np.bincount(inds, p_deriv)


# samesign_counts = samesign_counts[:samesign_counts.size - samesign_counts.size % 2]
# samesign_sums = samesign_sums[:samesign_sums.size - samesign_sums.size % 2]


inds = np.cumsum(samesign_counts) - 1

sma1 = rolling_window(samesign_sums[::2], window_size).sum(axis=1)
# sma1 = np.convolve(samesign_sums[::2], window, 'valid')
sma2 = rolling_window(samesign_sums[1::2], window_size).sum(axis=1)
# sma2 = np.convolve(samesign_sums[1::2], window, 'valid')

#ratio = -samesign_sums[1::2]/samesign_sums[::2]

# autocorr = np.correlate(samesign_sums[1::2], -samesign_sums[::2], 'same')
# autocorr = np.correlate(sma1, -sma2, 'same')
# autocorr = autocorr[autocorr.size//2:]
# plt.plot(np.abs(np.fft.rfft(autocorr)))

buy_price = df['buy'].values
sell_price = df['sell'].values


ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0), sharex=ax1)


ax1.plot(buy_price, color='lightgreen')
ax1.plot(sell_price, color='darkgreen')
ax1.plot(sma)

# ax2.plot(inds[1::2][lookahead_size-1:], -sma2/sma1)
ax2.plot(inds[1::2], -samesign_sums[1::2]/samesign_sums[::2])
# plt.plot(autocorr)
