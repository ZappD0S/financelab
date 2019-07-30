import os
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Concatenate
from keras.layers.recurrent import LSTM
from utils import rolling_window

# from scipy.stats import skew, kurtosis
import scipy.stats as spt


class DataGenerator(keras.utils.Sequence):
    def __init__(self, dlogp, y, lookbehind, batch_size=32, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lookbehind = lookbehind
        self.dlogp = (dlogp - dlogp.mean) / dlogp.std()
        self.windowed_dlogp = rolling_window(self.dlogp, lookbehind)
        self.y = y[lookbehind-1:]
        self.y -= self.y.min()
        self.size = self.y.size

        assert(self.size == self.windowed_dlogp.shape[0])
        self.indexes = np.arange(self.size)

        stats_fname = 'statistics.npy'

        try:
            self.statistics = np.load(stats_fname, allow_pickle=True)
        except IOError:
            self.statistics = self.compute_statistics()
            np.save(stats_fname, self.statistics)

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.size / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def compute_statistics(self):
        statistics = np.empty((self.size, 4), dtype='float64')

        statistics[:, 0] = np.mean(self.windowed_dlogp, axis=1)
        statistics[:, 1] = np.std(self.windowed_dlogp, axis=1)
        statistics[:, 2] = spt.skew(self.windowed_dlogp, axis=1)
        statistics[:, 3] = spt.kurtosis(self.windowed_dlogp, axis=1)

        statistics = (statistics - np.mean(statistics, axis=0)) / np.std(statistics, axis=0)
        return statistics

        # means = np.append(np.mean(self.dlogp), np.mean(statistics, axis=0))
        # stds = np.append(np.std(self.dlogp), np.std(statistics, axis=0))
        # return means, stds

        # X = self[0][0]
        # X2 = X**2
        # N = len(self)
        # for i in range(N):
        #     Xi = self[i][0]
        #     X += Xi
        #     X2 += Xi**2
        #
        # mus = np.mean(X / N, axis=(0, 1))
        # sigmas = np.sqrt(np.mean(X2 / N, axis=(0, 1)) - self.mu**2)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X = self.windowed_dlogp[indexes]
        Stats = self.statistics[indexes]

        return X, Stats, keras.utils.to_categorical(self.y[indexes], num_classes=3)

        # X[..., 1:] = self.statistics[]
        #
        # X[..., 0] = self.dlogp[indexer]
        # X[..., 1] = np.mean(self.dlogp[indexer], axis=1)[:, np.newaxis]
        # X[..., 2] = np.std(self.dlogp[indexer], axis=1)[:, np.newaxis]
        # X[..., 3] = spt.skew(self.dlogp[indexer], axis=1)[:, np.newaxis]
        # X[..., 4] = spt.kurtosis(self.dlogp[indexer], axis=1)[:, np.newaxis]

        # return X, keras.utils.to_categorical(self.y[indexes], num_classes=3)
        # return (X - self.mu)/self.sigma, keras.utils.to_categorical(self.y[indexes], num_classes=3)


data = np.load('train_data/train_data_tf1min.npz')

y = data['y']
dlogp = data['dlogp']

dgen = DataGenerator(dlogp, y, 2000)


# x = Input(batch_shape=)