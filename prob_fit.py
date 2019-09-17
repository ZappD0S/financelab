import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.distributions import levy_stable, t
from statsmodels.distributions.empirical_distribution import ECDF
# from levy import neglog_levy, par_bounds, convert_from_par0, convert_to_par0
from arch.univariate.distribution import SkewStudent
import tensorflow as tf
import tensorflow_probability as tfp
from scipy import optimize
from concurrent.futures import ProcessPoolExecutor
import sys


def fit_t(x, initial_guess):
    def neglogp(param):
        args, (loc, scale) = param[:-2], param[-2:]
        logpdf = t._logpdf((x - loc) / scale, *args)
        n_log_scale = len(x) * np.log(scale)
        return -np.sum(logpdf) + n_log_scale

    bounds = ((0., 1e10), (None, None), (1e-6, 1e10))
    res = optimize.minimize(neglogp, initial_guess, method='L-BFGS-B', bounds=bounds)
    return res.x


# def fit_levy(x, initial_guess):
#     def neglog_density(param):
#         return np.sum(neglog_levy(x, *param))

#     res = optimize.minimize(neglog_density, initial_guess, method='L-BFGS-B', bounds=par_bounds)
#     return res.x


def fit_skewt(x, initial_guess):

    def objective(params):
        args = params[:-2]
        loc, scale = params[-2:]
        logpdf = skewt.loglikelihood(args, (x - loc) / scale, 1, individual=True)
        n_log_scale = len(x) * np.log(scale)
        return -np.sum(logpdf) + n_log_scale

    bounds = ((2.05, 300.), (-1., 1.), (None, None), (1e-6, 1e10))
    skewt = SkewStudent()
    res = optimize.minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)
    return res.x


def get_gain_loss_probs(logp, stop_loss, take_gain, lookahead):
    mask = np.zeros(logp.size - lookahead, dtype='bool')
    gain_prob, loss_prob = 0, 0
    N = logp.size

    for i in range(lookahead):
        shift = i + 1
        # diffs = logp[shift:N - lookahead + shift] - logp[lookahead - shift:N - shift]
        diffs = logp[shift:] - logp[:-shift]
        diffs = diffs[:mask.size]
        diffs = diffs[~mask]
        mask1 = diffs >= take_gain
        mask2 = diffs <= stop_loss
        gain_prob += np.sum(mask1) / mask.size
        loss_prob += np.sum(mask2) / mask.size
        mask[~mask] |= mask1 | mask2

    return gain_prob, loss_prob

# def compute_params(shift):
#     try:
#         gains = sell[shift:] - buy[:-shift]
#         gains = (gains - gains.mean()) / gains.std()
#         params_arr[shift - 1] = t.fit(gains)
#         print('finished fit for shift', shift)
#     except Exception as ex:
#         print(ex)
#     sys.stdout.flush()


# df = pd.read_pickle('tick_data/eurusd_2018.pkl')
# df = df.resample('1s').ohlc().dropna()
# buy = np.log(df['buy', 'close'].values)
# sell = np.log(df['sell', 'close'].values)

# tf.enable_eager_execution()


# @tf.function
# def compute_nll(nll, params_arr):
#     for i in range(params_arr.shape[0]):
#         shift = i + 1
#         gains = logp[shift:] - logp[:-shift]
#         gains = (gains - tf.reduce_mean(gains)) / tf.math.reduce_std(gains)
#         nll[i].assign(-tf.reduce_sum(tfp.distributions.StudentT(
#             df=params_arr[i, 0],
#             loc=params_arr[i, 1],
#             scale=params_arr[i, 2]).log_prob(gains)))
#     return nll


def objective(params):
    return -tf.reduce_sum(tfp.distributions.StudentT(
        df=params[0],
        loc=params[1],
        scale=params[2]**2).log_prob(gains))


data = np.load('buy_sell_comp.npz')

# buy, sell = data['buy'], data['sell']
logp = data['buy']
logp = tf.constant(logp, dtype='float32')
gains = logp[1:] - logp[:-1]
gains = (gains - tf.reduce_mean(gains)) / tf.math.reduce_std(gains)

res = tfp.optimizer.lbfgs_minimize(lambda xs: tfp.math.value_and_gradient(objective, xs), tf.constant([2.3, 0, 0.7]))

best_params = res.position

x = tf.linspace(-4., 4., 1000)
t = tfp.distributions.StudentT(df=best_params[0], loc=best_params[1], scale=best_params[2]**2)

# buy, sell = data['buy'], data['sell']
logp = data['buy']
logp = tf.constant(logp, dtype='float32')
max_shift = 1000
params_arr = tf.random.uniform((max_shift, 3), dtype='float32')
nll = tf.Variable(np.zeros((params_arr.shape[0],), dtype='float32'))
# params_arr = np.empty((max_shift, 3), dtype='float32')
raise Exception
# gains = sell[1:] - buy[:-1]
# gains = logp[1:] - logp[:-1]
# gains = (gains - gains.mean()) / gains.std()

# df, loc, scale = 1, 2, 3
# x = 0.3
# tfp.distributions.StudentT(df=df, loc=loc, scale=scale).log_prob(x)

# tfp.math.value_and_gradient()
# tfp.optimizer.lbfgs_minimize()

# params_arr[0] = t.fit(gains)
#
# for i in range(1, max_shift):
#     shift = i + 1
#     gains = sell[shift:] - buy[:-shift]
#     gains = (gains - gains.mean()) / gains.std()
#     shape, loc, scale = params_arr[i - 1]
#     params_arr[i] = t.fit(gains, shape, loc=loc, scale=scale)
#     print('finished fit for shift', shift)

# if __name__ == '__main__':
#     with ProcessPoolExecutor() as executor:
#         executor.map(compute_params, range(1, max_shift), chunksize=10)

# logp = np.load('logp.npy', allow_pickle=True)
# gains = data[10:] - data[:-10]
# gains = logp[100:] - logp[:-100]

# gains = (gains - gains.mean()) / gains.std()

# np.random.shuffle(gains)
# sample_size = 2**12
# n_samples = gains.size // sample_size
# max_samples = 500
# n_samples = min(n_samples, max_samples)
# batched_gains = np.resize(gains, (n_samples, sample_size))
# batched_gains = batched_gains[:max_samples]
# params_arr = np.empty((batched_gains.shape[0], 4), dtype='float32')
# initial_guess = levy_stable._fitstart(gains)
# params_arr[0] = fit_levy(batched_gains[0], convert_to_par0['1'](initial_guess))
#
# for i in range(1, batched_gains.shape[0]):
#     print(i)
#     initial_guess = params_arr[:i].mean(axis=0)
#     params_arr[i] = fit_levy(batched_gains[i], initial_guess)
#
# params1 = convert_from_par0['1'](params_arr.mean(axis=0))
t_params = t.fit(gains)
# skewt_params = fit_skewt(gains, (5.7, 0, 0, 1))
ecdf = ECDF(gains)

x = np.linspace(-8, 8, 2000)
plt.plot(x, ecdf(x), '--', color='grey')
plt.plot(x, t.cdf(x, *t_params))
# plt.plot(x, SkewStudent().cdf((x - skewt_params[-2])/skewt_params[-1], skewt_params[:-2]))
# plt.plot(x, levy(x, alpha=alpha, beta=beta, mu=mu, sigma=sigma, cdf=True))
# plt.plot(x, levy_stable.cdf(x, *params1))
plt.show()
