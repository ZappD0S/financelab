import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit, njit, bool_, int_, float_
from utils import rolling_window, remove_time_gaps
from scipy import signal
from scipy.stats import spearmanr


def simplify_buy_sell_masks(buy_mask, sell_mask):
    buy_inds = np.argwhere(buy_mask).squeeze()
    sell_inds = np.argwhere(sell_mask).squeeze()

    inds = np.concatenate((buy_inds, sell_inds))
    types = np.concatenate((np.full_like(buy_inds, 1), np.full_like(sell_inds, 2)))

    sort_indexer = np.argsort(inds)

    types = types[sort_indexer]
    sorted_inds = inds[sort_indexer]

    buy_after_sell = (types[:-1] == 2) & (types[1:] == 1)
    sell_after_buy = (types[:-1] == 1) & (types[1:] == 2)

    buy_mask[0] = types[0] == 1
    buy_mask[1:] = buy_after_sell

    sell_mask[0] = False
    sell_mask[1:] = sell_after_buy

    # sell_mask[:buy_mask.argmax()] = False
    sell_mask[:buy_mask.argmax()+1] = False
    buy_mask[-sell_mask[::-1].argmax()-1:] = False

    return sorted_inds


def roots_mask(x, mode='both'):
    roots = np.zeros_like(x, dtype=np.bool)
    if mode == 'both':
        roots[1:] |= x[1:] * x[:-1] < 0
    if mode == 'positive':
        roots[1:] |= (x[:-1] < 0) & (x[1:] > 0)
    if mode == 'negative':
        roots[1:] |= (x[:-1] > 0) & (x[1:] < 0)

    # roots |= x == 0
    return roots


@njit
def compute_strategy():
    buy_mask = np.zeros_like(p, dtype=bool_)
    sell_mask = np.zeros_like(p, dtype=bool_)

    position_open = False
    last_buy_ind = 0
    for i in range(1, p.size):
        if not position_open:
            # a = buy_price[i] > long_sma[i] and buy_price[i - 1] < long_sma[i - 1]
            # a = buy_price[i] > long_sma[i] + 2e-4
            a = p_deriv_mean_deriv[i] > 0 and p_deriv_mean_deriv[i - 1] < 0 and p_deriv_mean[i] < 0
            # b = p_deriv_mean[i] < -5.27e-5
            # b = (p_deriv_mean[i] > 0 and p_deriv_mean[i - 1] < 0) and p_deriv_mean_deriv2[i] > 0
            if a:
                buy_mask[i] = True
                position_open = True
                last_buy_ind = i
        else:

            #b = p_deriv_mean[i] < 0 and p_deriv_mean[i - 1] > 0
            a = p_deriv_mean_deriv[i] < 0 and p_deriv_mean_deriv[i - 1] > 0 # and p_deriv_mean[i] > 0
            b = sell_price[i] > buy_price[last_buy_ind]
            # b = p_deriv_mean[i] > 2e-5
            # b = sell_price[i] < long_sma[i] and sell_price[i - 1] > long_sma[i - 1]
            c = sell_price[i] <= (1 - k/(leverage*100))*buy_price[last_buy_ind]
            if (a and b) or c:
                sell_mask[i] = True
                position_open = False

    if position_open:
        buy_mask[last_buy_ind] = False

    return buy_mask, sell_mask


leverage = 50
k = 3
short_window_size = 51
long_window_size = 201
llong_window_size = 501
# influence = 0.5
threshold = 1.5
# threshold = 2

df = pd.read_pickle('tick_data/eurusd_2018.pkl')
remove_time_gaps(df, 6e10)
df = df.resample('1min').pad()
df = df[1:]

p = df['buy'].values

# short_window_inds = window_indexer(p.size, short_window_size)
# short_sma = p[short_window_inds].mean(axis=-1)




short_window = np.hanning(short_window_size)
# short_window = tukey(short_window_size)
short_window /= short_window.sum()
short_sma = np.convolve(p, short_window, 'valid')
# p = p[-short_sma.size:]
# short_std = np.sqrt(np.convolve((p-short_sma)**2, short_window, 'valid') -
#                     np.convolve(p-short_sma, short_window, 'valid')**2)


long_window = np.hanning(long_window_size)
long_window /= long_window.sum()
long_sma = np.convolve(p, long_window, 'valid')


# p = p[-long_sma.size:]

# std_window_size = 100
std_window_size = 51
# std_window_size = 100
std_window = np.hanning(std_window_size)
std_window /= std_window.sum()


# def compute_spearmanr():
#     t = np.arange(std_window_size)
#     windowed_p = rolling_window(p, std_window_size)
#     spr = np.empty(windowed_p.shape[0], dtype=float)
#     # i = 0
#     # for window in rw:
#     for i in range(spr.size):
#         spr[i] = spearmanr(windowed_p[i], t)[0]
#
#     return spr


# spr = compute_spearmanr()

std = np.sqrt(np.convolve(p**2, std_window, 'valid') -
              np.convolve(p, std_window, 'valid')**2)


long_sma_deriv1 = np.diff(long_sma)
long_sma_deriv2 = np.diff(long_sma, 2)

p_deriv = np.diff(p)

p_deriv_mean = np.convolve(p_deriv, std_window, 'valid')
p_deriv_mean_deriv = np.diff(p_deriv_mean)
p_deriv_mean_deriv2 = np.diff(p_deriv_mean, 2)

p_deriv_std = np.sqrt(np.convolve(p_deriv**2, std_window, 'valid') -
                      np.convolve(p_deriv, std_window, 'valid')**2)


# delay = (std_window_size - 1) // 2
# z = signal.hilbert(p_deriv_mean)

# delay = (long_window_size - 1) // 2
# z = signal.hilbert(long_sma)

# phase_shift = np.angle(z[delay:]/z[:-delay])

# inst_amp = np.abs(z)

# phase_shift = np.unwrap(np.angle(z[delay:]/z[:-delay]))

# z2 = signal.hilbert(inst_freq)
# cosa = np.abs(z2)


# p_deriv_std2 = np.convolve(p_deriv**2, std_window, 'valid')/np.convolve(p_deriv, std_window, 'valid')**2 - 1


# min_size = np.min((long_sma_deriv1.size, long_sma.size, short_sma.size))

# min_size = long_sma_deriv1.size
# min_size = long_sma_deriv2.size
# min_size = long_std.size
# min_size = llong_sma.size
min_size = p.size - 1000

p = p[-min_size:]
buy_price = df['buy'][-min_size:].values
sell_price = df['sell'][-min_size:].values

long_sma = long_sma[-min_size:]
std = std[-min_size:]

short_sma = short_sma[-min_size:]
# short_std = short_std[-min_size:]

# short_sma_deriv1 = short_sma_deriv1[-min_size:]
# short_sma_deriv2 = short_sma_deriv2[-min_size:]
#
long_sma_deriv1 = long_sma_deriv1[-min_size:]
long_sma_deriv2 = long_sma_deriv2[-min_size:]
# p_deriv_std = p_deriv_std[-min_size:]
# p_deriv_std2 = p_deriv_std2[-min_size:]
p_deriv = p_deriv[-min_size:]
p_deriv_std = p_deriv_std[-min_size:]
p_deriv_mean = p_deriv_mean[-min_size:]
# phase_shift = phase_shift[-min_size:]
p_deriv_mean_deriv = p_deriv_mean_deriv[-min_size:]
p_deriv_mean_deriv2 = p_deriv_mean_deriv2[-min_size:]

#
# ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
# ax2 = plt.subplot2grid((3, 1), (2, 0), sharex=ax1)
#
# ax1.plot(buy_price, color='lightgreen')
# ax1.plot(sell_price, color='darkgreen')
#
# long_delay = 0.5*(long_window_size - 1)
#
# ax1.plot(long_sma)
# # ax1.plot(long_sma + np.log(p_deriv_std3/p_deriv_std3.mean()) - np.log(p_deriv_mean3/p_deriv_mean3.mean()))
# # ax1.plot(long_sma + np.log(p_deriv_std4/p_deriv_std4.mean()) - np.log(p_deriv_mean4/p_deriv_mean4.mean()))
#
# extrema = roots_mask(long_sma_deriv1)
#
# # ax1.plot(np.arange(long_sma.size)[long_sma_deriv2 > 0], long_sma[long_sma_deriv2 > 0], 'r^', markevery=8)
# # ax1.plot(np.arange(long_sma.size)[long_sma_deriv2 < 0], long_sma[long_sma_deriv2 < 0], 'bv', markevery=8)
# ax1.plot(np.arange(long_sma.size)[extrema], long_sma[extrema], '.', color='purple')
#
# ax2.plot(buy_price - long_sma)
#
# norm_std = (p_deriv_std - p_deriv_std.mean())/p_deriv_std.std()
# norm_mean = (p_deriv_mean - p_deriv_mean.mean())/p_deriv_mean.std()
# ax2.plot(norm_std)
# ax2.plot(norm_mean)
# ax2.plot(np.log10(p_deriv_std**2/p_deriv_mean**2))
# plt.tight_layout(pad=0)
#
# raise Exception

# plt.plot(llong_sma)
# plt.plot(np.arange(long_sma.size) - long_delay, long_sma)
# plt.plot(long_sma - long_std)
# plt.plot(long_sma + long_std)


# plt.plot(long_sma - 2e-4*np.sqrt(1+long_sma_deriv1**2))
# plt.plot(long_sma + 2e-4*np.sqrt(1+long_sma_deriv1**2))
#
# plt.plot(np.arange(long_sma.size)[long_sma_deriv2 > 0], long_sma[long_sma_deriv2 > 0], 'r^', markevery=8)
# plt.plot(np.arange(long_sma.size)[long_sma_deriv2 < 0], long_sma[long_sma_deriv2 < 0], 'bv', markevery=8)


# raise Exception

# buy_cond1 = (long_sma_deriv1 > 0) & (long_sma_deriv2 > 0)
# buy_cond2 = short_sma > long_sma
# buy_cond3 = p - short_sma < -threshold * short_std
# buy_cond3 = p - short_sma > threshold * short_std

# buy_mask = buy_cond1 & buy_cond2 & buy_cond3
# buy_mask = roots_mask(p - long_sma, 'positive')

# sell_mask = p - short_sma > threshold * short_std
# sell_cond = p - short_sma < -threshold * short_std

# sell_mask = roots_mask(p - long_sma, 'negative')

# sorted_inds = simplify_buy_sell_masks(buy_mask, sell_mask)

# buy_inds = np.argwhere(buy_cond).reshape(-1)
# sell_inds = np.argwhere(sell_cond).reshape(-1)
#
# inds = np.concatenate((buy_inds, sell_inds))
# types = np.concatenate((np.full_like(buy_inds, 1), np.full_like(sell_inds, 2)))
#
# sort_indexer = np.argsort(inds)
#
# types = types[sort_indexer]
# sorted_inds = inds[sort_indexer]
#
# buy_after_sell = (types[:-1] == 2) & (types[1:] == 1)
# sell_after_buy = (types[:-1] == 1) & (types[1:] == 2)
#
#
# buy_mask = np.empty_like(types, dtype=np.bool)
# buy_mask[0] = types[0] == 1
# buy_mask[1:] = buy_after_sell
#
# sell_mask = np.empty_like(types, dtype=np.bool)
# sell_mask[0] = False
# sell_mask[1:] = sell_after_buy
#
#
# sell_mask[:buy_mask.argmax()] = 0
# buy_mask[-sell_mask[::-1].argmax()-1:] = 0

print('computing strategy..')
buy_mask, sell_mask = compute_strategy()
print('stategy computed..')

ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0), sharex=ax1)


ax1.plot(buy_price, color='lightgreen')
ax1.plot(sell_price, color='darkgreen')
ax1.plot(short_sma)
ax1.plot(long_sma)
# plt.plot(short_sma - threshold*short_std)
# plt.plot(short_sma + threshold*short_std)

# ax1.plot(np.arange(p.size)[buy_mask], buy_price[buy_mask], '.', color='red')
# ax1.plot(np.arange(p.size)[sell_mask], sell_price[sell_mask], '.', color='blue')


ax2.plot(p_deriv_mean)
# ax2.plot(phase_shift)



# ax2.plot(p_deriv_mean_deriv2/p_deriv_mean_deriv2.std() * p_deriv_mean.std())
# ax2.plot(p_deriv_std)
# ax2.plot(inst_phase)
# ax2.plot(inst_amp)
# ax2.plot(inst_freq)
# ax2.plot(cosa)
# ax2.plot(p_deriv_mean_deriv2)
# ax2.plot(p_deriv_std)

# raise Exception
# b = buy_price[sorted_inds[buy_mask]]
# s = sell_price[sorted_inds[sell_mask]]

b = buy_price[buy_mask]
s = sell_price[sell_mask]

returns = s/b

mask = returns < 1
gain_change = ~mask[:-1] & mask[1:]
inds = np.cumsum(np.append(False, gain_change))
ss_counts = np.bincount(inds[mask])


# buy_inds = np.argwhere(buy_mask).reshape(-1)
buy_inds = np.nonzero(buy_mask)[0]

print(buy_inds[np.argmin(returns)])

# print('index min return: ', sorted_inds[buy_mask][np.argmin(returns)])

cum_returns = np.cumprod(returns)

lev_returns = (returns - 1)*leverage + 1

# gain = np.prod(returns)
# lev_gain = np.prod(lev_returns)
# print(gain)
# print(lev_gain)
# b = buy_price[sorted_inds[sell_mask]]
# s = sell_price[sorted_inds[buy_mask]]
#
# inv_gain = np.prod(s[1:]/b[:-1])
# print(inv_gain)

