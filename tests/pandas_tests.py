import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numba
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def compute_returns(diffs, prices, period):
    assert len(prices) - len(diffs) == 1
    N = len(diffs)
    returns = np.empty_like(prices)
    deltas = np.empty_like(prices)
    mask = np.zeros_like(prices)
    for i in prange(N):
        cumdelta = diffs[i]
        for j in range(i + 1, N):
            delta1 = cumdelta
            delta2 = cumdelta + diffs[j]

            if delta1 <= period <= delta2:
                options = np.array([delta1, delta2])
                choice = np.argmin(np.abs(options - period))
                # returns[i] = np.log(prices[j + choice]) - np.log(prices[i])
                returns[i] = prices[j + choice] / prices[i]
                deltas[i] = options[choice]
                mask[i] = True
                break

            cumdelta = delta2

    return returns, deltas, mask


df = pd.read_csv("stock_data/CVX.csv", index_col=0, usecols=[0, 1, 2, 3, 4], parse_dates=True)
df.sort_index(inplace=True)

diffs = df.index[1:] - df.index[:-1]
diffs = diffs.days.to_numpy()
prices = df["close"].to_numpy()


def compute_mean_returns(periods):
    for period in periods:
        returns, deltas, mask = compute_returns(diffs, prices, period)
        mask = mask.astype(bool)
        # returns, deltas = returns[mask], deltas[mask]
        norm_returns = returns[mask] ** (1 / deltas[mask])
        yield norm_returns.mean(), norm_returns.std() / np.sqrt(norm_returns.size)


# periods = np.arange(1, 20)
# periods = np.linspace(10, 500, 200, dtype=int)
periods = np.linspace(800, 3000, 500, dtype=int)
# periods = np.linspace(7000, 7300, 500, dtype=int)
means, stds = np.array(list(compute_mean_returns(periods))).T
plt.plot(periods, means)
plt.plot(periods, means + stds)
plt.plot(periods, means - stds)
plt.show()
