import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numba
from numba import njit
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.signal import resample


@njit
def get_delays(p, spread):
    N = p.size
    mask = np.zeros(N, dtype=numba.bool_)
    delays = np.zeros(N, dtype=numba.int64)
    for i in range(N):
        p0 = p[i]
        j = 1
        while i + j < N:
            if abs(p[i + j] - p0) > spread:
                mask[i] = True
                delays[i] = j
                break
            j += 1

    return mask, delays


spread = 1.3e-4
df = pd.read_pickle('tick_data/eurusd_2018.pkl')

timestamps = df.index.values
timestamps = timestamps - timestamps[0]
p = df['buy'].values

deltas = np.diff(timestamps)

deltas_sec = deltas / np.timedelta64(1, 's')

mask = deltas_sec < 500

plt.plot(deltas_sec[mask], np.diff(p)[mask], ',')
plt.show()
raise Exception

# mask = deltas > np.timedelta64(2, 'm')
mask = deltas > np.timedelta64(20, 's')
inds = mask.nonzero()[0]

deltas = deltas[~mask]
timestamps = np.cumsum(np.append(np.timedelta64(0), deltas))

p1 = p[inds]
p2 = p[inds + 1]

p[inds] = (p[inds] + p[inds + 1]) / 2
p = np.delete(p, inds + 1)


mask, delays = get_delays(p, spread)

starts = np.nonzero(mask)[0]
stops = starts + delays[mask]
deltas = timestamps[stops] - timestamps[starts]


# deltas = deltas[deltas < np.timedelta64(2, 'm')]
deltas = deltas / np.timedelta64(1, 's')

x = np.linspace(0, 500, 500)
ecdf = ECDF(deltas)


# n_sec = (timestamps[-1] - timestamps[0]) // np.timedelta64(1, 's')
n_sec = 2**24
# p_1s = resample(p, n_sec, t=timestamps)
# plt.plot(x, ecdf(x))
# plt.show()
