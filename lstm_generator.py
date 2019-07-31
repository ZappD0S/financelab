import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from keras.layers.recurrent import LSTM
# from utils import rolling_window
import scipy.stats as spt


def rolling_window(x, window_size):
    stride = x.strides[0]
    return np.lib.stride_tricks.as_strided(x, (x.size - window_size + 1, window_size), (stride, stride))


class DataGenerator(keras.utils.Sequence):
    def __init__(self, dlogp, y, lookbehind, batch_size=32, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lookbehind = lookbehind
        self.dlogp = (dlogp - dlogp.mean()) / dlogp.std()
        self.windowed_dlogp = rolling_window(self.dlogp, lookbehind)
        y = y[lookbehind - 1:]
        y -= y.min()
        self.size = y.size
        self.y = keras.utils.to_categorical(y, num_classes=3)
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

        statistics = (statistics - statistics.mean(axis=0)) / statistics.std(axis=0)
        return statistics

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        x_batch = self.windowed_dlogp[indexes, :, np.newaxis]
        stats_batch = self.statistics[indexes]
        y_batch = self.y[indexes]

        # return [X, Stats], keras.utils.to_categorical(self.y[indexes], num_classes=3)
        return [x_batch, stats_batch], y_batch


# file = io.BytesIO(uploaded['train_data_tf1min.npz'])
# data = np.load(file, allow_pickle=True)
data = np.load('train_data/train_data_tf1min.npz')

y = data['y']
dlogp = data['dlogp']

dgen = DataGenerator(dlogp, y, 2000, batch_size=500)


x = Input(batch_shape=(dgen.batch_size, dgen.lookbehind, 1))

lstm1 = LSTM(100, dropout=0.2, recurrent_dropout=0.15, return_sequences=True)(x)
lstm2 = LSTM(100, dropout=0.2, recurrent_dropout=0.15)(lstm1)

stats = Input(batch_shape=(dgen.batch_size, 4))
dense_input = Concatenate(axis=1)([lstm2, stats])


output = Dense(3, activation='softmax')(dense_input)

model = Model(inputs=[x, stats], outputs=output)

opt = keras.optimizers.RMSprop()

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit_generator(dgen, epochs=100)
