import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numba
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def compute_returns(diffs, prices):
    N = len(diffs)
    returns = np.full_like(prices, np.nan)
    deltas = np.full_like(prices, np.nan)
    for i in prange(N):
        cumdelta = 0
        for j in range(i, N):
            delta1, delta2 = cumdelta, cumdelta + diffs[j]

            if delta1 <= 365 <= delta2:
                options = np.array([delta1, delta2])
                choice = np.argmin(np.abs(options - 365))
                returns[i] = np.log(prices[j + choice]) - np.log(prices[i])
                deltas[i] = options[choice]
                break

            cumdelta += diffs[j]

    return returns, deltas

df = pd.read_csv("stock_data/IBM.csv", index_col=0, usecols=[0, 1, 2, 3, 4], parse_dates=True)
df.sort_index(inplace=True)

ind = df.index
diffs = ind[1:] - ind[:-1]
diffs = diffs.days.to_numpy()

dates = np.stack((diffs), axis=-1)

prices = df['close'].to_numpy()

returns, deltas = compute_returns(dates, prices)

mean_delta = deltas[~np.isnan(deltas)].mean()

print(np.mean(returns[~np.isnan(returns)]) / mean_delta)
print(np.mean((np.log(prices[1:]) - np.log(prices[:-1]))) / diffs.mean())

# plt.plot(ind.to_numpy(), res)
# plt.show()
