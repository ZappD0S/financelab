import numpy as np
import pandas as pd
from numba import njit
from enum import IntEnum
import matplotlib.pyplot as plt
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize, gbrt_minimize
from utils import rolling_window, remove_time_gaps


PositionType = IntEnum('PositionType', [('Buy', 0), ('Sell', -1)])


def regularize_gain(returns, k, lev=1):
    # g_mean = np.mean(returns[returns > 1])
    N = np.sum(returns > 1)
    g_mean = np.prod(returns[returns > 1])**(1/N)
    # r_mean = np.mean(returns[returns > 1] - 1)*lev
    r_mean = (g_mean - 1)*lev
    # g_mean -= np.std(returns[returns > 1])
    # r_mean = g_mean - 1
    tot = 1
    loss = 0
    cum_tot = np.empty_like(returns)
    # k = k0
    for i, g in enumerate(returns):
        r = g - 1
        if not loss:
            if r < 0:
                # loss += -k*tot*r
                loss += -k*tot*r*lev

            # tot += k*tot*r
            tot += k*tot*r*lev
        else:
            # x dovrebbe avere un minimo?
            x = loss/r_mean
            if x > tot:
                print('warning!')
                print(loss)
                print(x, tot)
            x = min(tot, x)
            # x = np.clip(x, k*tot, tot)

            # if x > tot:
            #     print(x)
            #     print(i)
            #     raise Exception

            # if r < 0:
            # loss += -x*r
            loss += -x*r*lev
            loss = max(0, loss)

            # tot += x*r
            tot += x*r*lev
        cum_tot[i] = tot

    return cum_tot


def get_timer(x):
    cum_max = np.maximum.accumulate(x)
    new_max_inds = np.nonzero(cum_max == x)[0]
    last_max_inds = np.zeros_like(x, dtype='int64')
    last_max_inds[new_max_inds] = new_max_inds
    last_max_inds = np.maximum.accumulate(last_max_inds)
    return np.arange(x.size) - last_max_inds

@njit
def strategy1(buy_p, sell_p, params, lev=1):
    # k, q, m = params
    q, m = params
    # q1, m1, q2, m2 = params
    n = buy_p.size

    # open_mask = np.zeros((2, n), dtype=bool)
    # buy_mask = np.zeros(n, dtype=bool)
    # sell_mask = np.zeros(n, dtype=bool)
    # close_mask = np.zeros(n, dtype=bool)

    returns = []
    tot_gain = 1
    pos_type = PositionType.Buy
    pos_open = True
    count = 0
    i0 = 410
    i = i0 + 1
    # last_peak = windowed_sell_p[i0]
    gmax = 1
    # was_profitable = False

    while i < n:
        if pos_type == PositionType.Buy:
            gnow = sell_p[i] / buy_p[i0]

            if gnow > gmax:
                gmax = gnow
                count = 0

            # a = windowed_buy_p[i0]
            # b = windowed_sell_p[i]
            # if not last_peak or b > last_peak:
            #     last_peak = b
            #
            # gnow = b / a
            # gmax = last_peak / a
        elif pos_type == PositionType.Sell:
            gnow = sell_p[i0] / buy_p[i]

            if gnow > gmax:
                gmax = gnow
                count = 0

            # a = windowed_buy_p[i]
            # b = windowed_sell_p[i0]
            # if not last_peak or a < last_peak:
            #     last_peak = a
            #
            # gnow = b / a
            # gmax = b / last_peak
        else:
            raise

        # profitable = gnow > 1
        # if not profitable and was_profitable:
        #     # print('was profitable, but not any more')
        #     pos_open = False
        #     continue
        #
        # was_profitable = profitable

        # if gnow <= (1 - k/lev):
        #     pos_open = False
        # elif gnow > 1:

        max_dd = -q + m * count
        # rel_dd = (gnow - gmax) / (gmax - 1)
        rel_dd = (gnow - gmax) / gmax
        assert(rel_dd <= 0)

        if rel_dd < max_dd:
            pos_open = False
        # else:
        #     count += 1

        count += 1

        if not pos_open:
            # tot_gain *= gnow
            tot_gain *= (gnow - 1) * lev + 1
            returns.append(gnow)
            pos_open = True
            pos_type = ~pos_type
            i0 = i
            count = 0
            # last_peak = 0
            gmax = 1

        i += 1

    return -tot_gain, np.asarray(returns)


df = pd.read_pickle('tick_data/eurusd_2018.pkl')
remove_time_gaps(df, 6e10)
df = df.resample('1min').pad()
df = df[1:]

buy_p = df['buy'].values
sell_p = df['sell'].values

lev = 50
# lev = 10


space = [(0.3, 0.7),
         (1e-4, 1e-2, 'log-uniform')]


def objective(params):
    # return -np.prod(strategy1(windowed_buy_p, windowed_sell_p, params, opt=True))
    return strategy1(buy_p, sell_p, params)[0]


#res_gp = gp_minimize(objective, space, n_calls=100, verbose=True)
# k, q, m = res_gp.x

# res_gp = gbrt_minimize(objective, space, n_calls=400, verbose=True)
# q, m = res_gp.x


# q, m = [0.18781724359773516, 0.0001540076947000537]
# q, m = [0.9646670635272306, 0.00092110672626639]
# [0.9409831044250364, 4.343856964383541e-05]

# [0.5010334487978061, 0.00046692531732295706]
q, m = [0.5010334487978061, 0.00046692531732295706]


# k, q, m = [0.14883648944572664, 1.0, 6.348019164048938e-06]
# k, q, m = [0.14883648944572664, 1.0, 4e-06]
# k, q, m = [0.3, 1.3457875238175223, 0.0001282612327957939]
# k, q, m = [0.16957805779262938, 4.408096024096525, 0.001]
# [0.15797978339770632, 4.002057008006755, 0.0007380629511342397]

# k, q, m = [0.17378804012499427, 3.906171659483189, 0.0007288880372574512]


# k, q, m = [0.05112778761843332, 1.8546207297324926, 1.011935229638011e-05] # da studiare..

# k, q, m = [0.08524266553920269, 1.0126159669760681, 4.103712630612201e-06] # 5.3...


# k, q, m = [0.06756375652099142, 1.972602395430446, 0.0003587127257023572]

# k, q, m = [0.0996075711247225, 0.7470410982900649, 4.563979806943556e-07]


# k, q, m = [0.16552451312523317, 1.3244034444221113, 0.0003348927459384232]


# gain = strategy1(windowed_buy_p, windowed_sell_p, (k, q, m))

# _, returns = strategy1(windowed_buy_p, windowed_sell_p, (k, q, m))
_, returns = strategy1(buy_p, sell_p, (q, m))
lev_returns = (returns - 1)*lev + 1

print(np.prod(returns))
print(np.prod(lev_returns))

print(np.sum(returns > 1), np.sum(returns < 1))
print(returns[returns > 1].prod(), returns[returns < 1].prod())
mask = returns < 1
gain_change = ~mask[:-1] & mask[1:]


# gain_change = mask[:-1] != mask[1:]
# inds = np.nonzero(np.append(False, ))


# gain_change_inds = np.nonzero(gain_change)[0]
# inds = np.zeros_like(returns, dtype='int64')
# inds[gain_change_inds] = gain_change_inds
# np.maximum.accumulate(inds, out=inds)

inds = np.cumsum(np.append(False, gain_change))
ss_counts = np.bincount(inds[mask])

