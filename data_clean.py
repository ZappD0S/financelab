import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numba
from numba import njit
from statsmodels.distributions.empirical_distribution import ECDF


@njit
def get_delays(p, spread):
    N = p.size
    mask = np.zeros(N, dtype=numba.bool_)
    delays = np.zeros(N, dtype=numba.int64)
    for i in range(N):
        start_p = p[i]
        j = 1
        while i + j < N:
            if abs(p[i + j] - start_p) > spread:
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

mask, delays = get_delays(p, spread)

starts = np.nonzero(mask)[0]
stops = starts + delays[mask]
deltas = timestamps[stops] - timestamps[starts]


# deltas = deltas[deltas < np.timedelta64(2, 'm')]
deltas = deltas / np.timedelta64(1, 's')

x = np.linspace(0, 500, 500)
ecdf = ECDF(deltas)

plt.plot(x, ecdf(x))
plt.show()
