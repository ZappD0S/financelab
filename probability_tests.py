import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bisect import bisect, insort
import numba
from numba import njit, prange
from utils import rolling_window, remove_time_gaps
from scipy import signal
import scipy.stats as st
from arch import arch_model


def roots_mask(x, mode='both'):
    roots = np.zeros_like(x, dtype=np.bool)
    if mode == 'both':
        roots[1:] |= x[1:] * x[:-1] < 0
    elif mode == 'positive':
        roots[1:] |= (x[:-1] < 0) & (x[1:] > 0)
    elif mode == 'negative':
        roots[1:] |= (x[:-1] > 0) & (x[1:] < 0)
    else:
        raise
    # roots |= x == 0
    return roots


@numba.njit
def profitable_delays_and_mins(buy_p, sell_p, lookahead=100):
    delays = []
    mins = []
    # delays = np.zeros_like(buy_p, dtype=numba.int64)
    # mins = np.zeros_like(buy_p, dtype=numba.float64)
    # gains = np.zeros_like(buy_p, dtype=numba.float64)
    # mask = np.zeros_like(buy_p, dtype=numba.bool_)

    # profitable = np.zeros(buy_p.size - lookahead, dtype=numba.bool_)
    # delays = np.zeros(buy_p.size - lookahead, dtype=numba.int64)
    # mins = np.zeros(buy_p.size - lookahead, dtype=numba.float64)
    # profitable = np.zeros(buy_p.size - lookahead, dtype='bool')
    # delays = np.zeros(buy_p.size - lookahead, dtype='int64')
    # mins = np.zeros(buy_p.size - lookahead, dtype='float64')

    # mins[:] = sell_p[-lookahead] / buy_p[lookahead:]
    # mins[:] = sell_p[-lookahead]

    # for i in range(1, lookahead):
    #     print(i/lookahead)
    #     for j in range(gains.size):
    #         if not profitable[j]:
    #             if gains[j] > 1:
    #                 profitable[j] = True
    #                 delays[j] = i
    #             else:
    #                 mins[j] = min(mins[j], gains[j])
    #
    #     gains = sell_p[i:-lookahead+i]/buy_p[lookahead-i:-i]
    #     mask = gains > 1
    #     mask = sell_p[i:-lookahead+i] > buy_p[lookahead-i:-i]
    #
    #     mins[~profitable] = np.minimum(mins[~profitable], sell_p[~profitable])
    #
    #     profitable |= mask
    #     delays[mask] = i

    for i in range(len(buy_p) - lookahead):
        # min_gain = sell_p[i]/buy_p[i]
        min_p = sell_p[i]
        for j in range(1, lookahead):
            #gain = sell_p[i+j]/buy_p[i]
            # sell_p[i+j] > buy_p[i]
            # sell_p[i+j] < min_p

            if gain < min_gain:
                min_gain = gain
            if gain > 1:
                delays.append(j)
                mins.append(min_gain)
                break

        mask = sell_p[i:i+lookahead] > buy_p[i]
        if np.any(mask):
            first_profitable = np.argmax(mask)
            delays.append(first_profitable)
            mins.append(np.min(sell_p[:first_profitable]))

    # return delays[profitable], mins[profitable]
    return np.asarray(delays), np.asarray(mins)
    # return delays[mask], mins[mask]


@njit
def dynprog(inds, types):
    profitable = []
    # waiting = []
    # profitable = set()
    # waiting = set()

    # profitable = np.zeros_like(buy_p, dtype='bool')
    #
    # not_profitable_inds = np.arange(profitable.size)
    #
    # counts = []
    #
    # for i in range(1, int(1e5)):
    #     not_profitable_inds = not_profitable_inds[not_profitable_inds < buy_p.size - i]
    #     mask = sell_p[i:][not_profitable_inds] > buy_p[:-i][not_profitable_inds]
    #     count = np.sum(mask)
    #     print(i, count)
    #     counts.append(count)
    #     # profitable[:-i] |= mask
    #     not_profitable_inds = not_profitable_inds[~mask]

    start = 0

    while types[start] != 1:
        start += 1

    waiting = [types[start]]

    for i in range(start + 1, ps.size):
        index = np.searchsorted(waiting, inds[i])
        if types[i] == 1:
            waiting.insert(index, inds[i])
            # if waiting:
            #     index = np.searchsorted(waiting, inds[i])
            #     waiting.insert(index, inds[i])
            # else:
            #     waiting.append(inds[i])

            # waiting.append(inds[i])
            # waiting.add(inds[i])
        else:
            # index = np.searchsorted(waiting, inds[i])
            profitable.extend(waiting[:index])
            del waiting[:index]
            # became_proft = set([ind for ind in waiting if ind < inds[i]])
            # waiting.difference_update(became_proft)
            # profitable.update(became_proft)

    return profitable


@njit
def segment_geomean(arr, segments, out):
    # segment_vals, segments_counts = np.unique(segments, return_counts=True)
    #segment_vals = tmp[0]
    # segments_counts = tmp[1]
    segment_vals = np.unique(segments)
    segments_counts = np.bincount(segments)
    segments_counts = segments_counts[segments_counts != 0]

    if segment_vals.size != segments_counts.size:
        return 1
    # counts = np.zeros_like(segment_vals)
    # ret = np.ones(segment_vals.size)
    # masks = np.equal.outer(arr, segment_vals)
    out[...] = 1
    for i in range(len(arr)):
        for j, val in enumerate(segment_vals):
            if segments[i] == val:
                ind = j
                break
        else:
            return 1

        # ret[ind] *= arr[ind]
        out[ind] *= arr[i]**(1/segments_counts[ind])
        # counts[ind] += 1

    # out = out**(1/counts)
    # for i, val in enumerate(segment_vals):
    #     prod = 1
    #         if segments[j] == val:
    #             prod *=
    #
    #     mask = segments == val
    #     n = np.sum(mask)
    #     ret[i] = np.prod(arr[mask])**(1/n)

    return 0
    #return ret


df = pd.read_pickle('tick_data/eurusd_2018.pkl')
# remove_time_gaps(df, 6e10)
# df = df.resample('1min').pad().dropna()

# df = df.resample('1h').pad().dropna()
#df = df[1:]

#dp = df['buy'].diff().dropna()
# dp = df.apply(np.log).reset_index().diff() #.dropna()
# p = df.reset_index()

#p['timestamp'] = p['timestamp'].astype('int64')

# dp = p.diff().dropna()
# p = p.set_index('timestamp')

# dp = dp[dp['timestamp'] < int(1e13)]

# p = dp.cumsum() + p.iloc[0]

# p['timestamp'] = p['timestamp'].astype('timedelta64[ns]')
# p = p.set_index('timestamp')

# raise Exception
# p = p.resample('1h').pad().dropna()

# raise Exception

# dp = dp.cumsum().set_index('timestamp').resample('1h').pad().dropna()

# dp = [dp['timestamp'].diff() < np.timedelta64(int(1e12))]
# dp = dp[dp['timestamp'].astype('int64') < 1e12]
# dp = dp.resample('1h').pad().dropna()
returns = np.diff(np.log(df['buy'].values))
# sell_p = df['sell'].values

raise Exception


# logrets = np.diff(np.log(np.asarray(buy_p*1e5, dtype='int64')))
# returns = np.diff(np.log(buy_p))

# returns = 100 * df['buy'].pct_change().dropna()

# returns = returns[~np.isclose(returns, 0)]
# returns = returns[returns != 0]

# returns = returns[2000:2000+9000]

# logrets = logrets[logrets != 0]

#logrets = (logrets - logrets.mean())/logrets.std() * 0.5
#am = arch_model(returns, mean='HARX', lags=10, dist='StudentsT')
#am = arch_model(returns, mean='HARX', lags=8)
am = arch_model(returns**2, vol='FIGARCH', power=1)
res1 = am.fit(update_freq=5)
# am = arch_model(returns**2, vol='FIGARCH', power=1)
# res2 = am.fit(update_freq=5)

# ps = np.concatenate((buy_p, sell_p))
# inds = np.tile(np.arange(buy_p.size), 2)
# types = np.concatenate((np.full_like(buy_p, 1, dtype='int64'), np.full_like(sell_p, 2, dtype='int64')))
#
# indexer = np.argsort(ps)

# ps = ps[indexer]
#inds = inds[indexer]
# types = types[indexer]

#res = dynprog(inds, types)

#delays, mins = profitable_delays_and_mins(buy_p, sell_p, int(1e5))

raise Exception

# dp = np.diff(np.asarray(buy_p*1e5, dtype='int64'))
# dp = np.diff(np.asarray(buy_p*1e5, dtype='int64'))
# dt = np.diff(df.index.values)


# vals, counts = np.unique(dt, return_counts=True)


# dp_selected = dp[dt == vals[counts.argmax()]]

# sample_size = 10000

# sample_means = np.random.permutation(dp_selected)[:-(dp_selected.size%sample_size)].reshape(-1, sample_size).sum(axis=1)
# sample_means = np.random.permutation(dp_selected)
# plt.plot(*np.unique(sample_means, return_counts=True))
# plt.show()
# raise Exception

# lookbehind = 51  # 200
lookbehind = 50  # 200
lookahead = 100
min_steps = 10

# window = np.hanning(lookbehind)
window = np.hanning(lookbehind + 1)
# short_window = tukey(short_window_size)
window /= window.sum()
sma = np.convolve(buy_p, window, 'valid')

# buy_p = buy_p[-sma.size:]
# sell_p = sell_p[-sma.size:]

# crossabove_mask = roots_mask(buy_p - sma, 'positive')
# crossabove_mask = crossabove_mask[:-lookahead + 1]

# crossabove_mask = roots_mask(buy_p[lookbehind:-lookahead + 1] - sma[:-lookahead], 'positive')
crossabove_mask = roots_mask(buy_p[lookbehind:-lookahead + 1] - sma[:-lookahead + 1], 'positive')
# crossabove_mask = crossabove_mask[:-lookahead + 1]

windowed_buy_p = rolling_window(buy_p, lookahead + lookbehind + 1)
windowed_sell_p = rolling_window(sell_p, lookahead + lookbehind + 1)

#dobbiamo fare tutto sulla derivata!
mean = np.add.accumulate(windowed_buy_p[:, lookbehind::-1], axis=1)[:, min_steps-1:]/np.arange(min_steps, lookbehind + 2)

mean2 = np.add.accumulate(windowed_buy_p[:, lookbehind::-1]**2, axis=1)[:, min_steps-1:]/np.arange(min_steps, lookbehind + 2)

std = np.sqrt(mean2 - mean**2)
print(windowed_buy_p.shape)
# crossabove_mask = roots_mask(windowed_buy_p[:, 0] - sma, 'positive')

N = windowed_buy_p.shape[0]

# gains = windowed_sell_p / windowed_buy_p[:, 0, np.newaxis]
gains = windowed_sell_p[:, lookbehind:] / windowed_buy_p[:, lookbehind, np.newaxis]
# gains = windowed_sell_p[:, 0, np.newaxis]/windowed_buy_p

# profitable = np.any(gains > 1, axis=1)
#
# print(np.sum(profitable)/profitable.size)
#
# gains = gains[crossabove_mask]

profitable = np.any(gains > 1, axis=1)

print(np.sum(profitable)/profitable.size)

gains = gains[profitable]


inds_first_profitable = np.argmax(gains > 1, axis=1).astype('int64')

# ranges = np.dstack((np.zeros_like(inds_first_profitable, dtype='int64'), inds_first_profitable))
ranges = np.column_stack((np.zeros_like(inds_first_profitable, dtype='int64'), inds_first_profitable))

# multi_ind = np.ix_(np.arange(inds_first_profitable.size).repeat(2), ranges.flatten())

# flat_inds = np.ravel_multi_index((np.arange(inds_first_profitable.size).repeat(2), ranges.flatten()), gains.shape)
# flat_inds = np.ravel_multi_index((np.arange(inds_first_profitable.size)[np.newaxis].repeat(2, axis=0), ranges),
#                                  gains.shape).flatten()

flat_inds = np.ravel_multi_index((np.arange(inds_first_profitable.size)[:, np.newaxis].repeat(2, axis=1), ranges),
                                 gains.shape).flatten()

min_before_profitable = np.minimum.reduceat(gains.flatten(), flat_inds)[::2]

loss = 1 - min_before_profitable

# maxs = np.max(gains, axis=1)
maxs = np.maximum.reduceat(gains.flatten(), flat_inds)[1::2]

# ranges = np.stack((, inds_first_profitable))

# per le transazioni che diventanto vangaggiose
# qual è il valore più basso raggiungo prima di diventare vantaggiose

# dopo 5-6 passi si può già chiudre le transazioni svantaggiose, wow..

#print(profitable.sum()/N)

# probs = np.bincount(inds_first_profitable)/N
# probs = np.bincount(inds_first_profitable)/gains.shape[0]
# plt.plot(probs[1:])

# price = np.load('tick_data/eurusd_2018.npy')
dp = np.diff(np.asarray(buy_p*1e5, dtype='int64'))

dt = np.diff(df.index.values.astype('int64'))


# dt = np.floor_divide(dt, np.gcd.reduce(dt))

lcm_ = np.lcm.reduce(dt)

k_dt = lcm_/dt
# k, pmf = np.unique(diffs, return_counts=True)

# pmf = pmf/pmf.sum()
# mu = np.sum(pmf*k)


