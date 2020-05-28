import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numba
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def compute_returns(dates, prices):
    assert(len(prices) == len(dates))
    N = len(prices)
    returns = np.full_like(prices, np.nan)
    for i in prange(N):
        start_date = dates[i]
        for j in range(i, N):
            # year_diff, month_diff, day_diff = dates[i + j] - start_date
            year_diff, dayofyear_diff = dates[i + j] - start_date

            if year_diff < 1:
                continue
            elif year_diff > 1:
                break

            # if month_diff < 0:
            #     continue
            # elif month_diff > 0:
            #     break

            if dayofyear_diff < 0:
                continue
            else:
                if dayofyear_diff == 0:
                    returns[i] = prices[i + j] / prices[i]
                break

    return returns

df = pd.read_csv("stock_data/IBM.csv", index_col=0, usecols=[0, 1, 2, 3, 4], parse_dates=True)
df.sort_index(inplace=True)

ind = df.index

dates = np.stack((
    ind.year.to_numpy(),
    ind.dayofyear.to_numpy()
    # ind.month.to_numpy(),
    # ind.day.to_numpy()
), axis=-1)

prices = df['close'].to_numpy()

returns = compute_returns(dates, prices)

# plt.plot(ind.to_numpy(), res)
# plt.show()
