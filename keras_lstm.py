import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from keras.layers.recurrent import LSTM
# from utils import rolling_window
import scipy.stats as spt

def compute_statistics(windowed_dlogp):
    statistics = np.empty((dlogp.shape[0], 4), dtype='float64')

    statistics[:, 0] = np.mean(windowed_dlogp, axis=1)
    statistics[:, 1] = np.std(windowed_dlogp, axis=1)
    statistics[:, 2] = spt.skew(windowed_dlogp, axis=1)
    statistics[:, 3] = spt.kurtosis(windowed_dlogp, axis=1)

    statistics = (statistics - statistics.mean(axis=0)) / statistics.std(axis=0)
    return statistics

def rolling_window(x, window_size):
    stride = x.strides[0]
    return np.lib.stride_tricks.as_strided(x, (x.size - window_size + 1, window_size), (stride, stride))


batch_size = 200
lookbehind = 2000

data = np.load('train_data/train_data_tf1min.npz')
y = data['y']
dlogp = data['dlogp']

dlogp = (dlogp - dlogp.mean()) / dlogp.std()
windowed_dlogp = rolling_window(dlogp, lookbehind)
y = y[lookbehind - 1:]
y -= y.min()
y = keras.utils.to_categorical(y, num_classes=3)

stats_fname = 'statistics.npy'

try:
    statistics = np.load(stats_fname, allow_pickle=True)
except IOError:
    statistics = compute_statistics(windowed_dlogp)
    np.save(stats_fname, statistics)


# def __getitem__(self, index):
#     indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

#     x_batch = self.windowed_dlogp[indexes, :, np.newaxis]
#     stats_batch = self.statistics[indexes]
#     y_batch = self.y[indexes]

#     # return [X, Stats], keras.utils.to_categorical(self.y[indexes], num_classes=3)
#     return [x_batch, stats_batch], y_batch


# file = io.BytesIO(uploaded['train_data_tf1min.npz'])
# data = np.load(file, allow_pickle=True)

x = Input(batch_shape=(batch_size, lookbehind, 1))

lstm1 = LSTM(100, dropout=0.2, recurrent_dropout=0.15, return_sequences=True)(x)
lstm2 = LSTM(100, dropout=0.2, recurrent_dropout=0.15)(lstm1)

stats = Input(batch_shape=(batch_size, 4))
dense_input = Concatenate(axis=1)([lstm2, stats])


output = Dense(3, activation='softmax')(dense_input)

model = Model(inputs=[x, stats], outputs=output)

opt = keras.optimizers.RMSprop()

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x=[windowed_dlogp, statistics], y=y, batch_size=batch_size, epochs=100, validation_split=0.2)
# model.fit_generator(dgen, epochs=100)
