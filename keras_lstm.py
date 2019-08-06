import os
import re
import glob
import numpy as np
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Permute, Lambda, RepeatVector, Flatten, multiply, concatenate
from keras.layers.recurrent import LSTM
from keras.layers import CuDNNLSTM
from keras.layers.wrappers import Bidirectional
# from utils import rolling_window
import scipy.stats as spt
from google.colab import drive


def rolling_window(x, window_size):
    stride = x.strides[0]
    return np.lib.stride_tricks.as_strided(x, (x.size - window_size + 1, window_size), (stride, stride))


def attention_block(inputs, single_attention_vector=False):
    input_dim, time_steps = inputs.shape[-2:]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output = multiply([inputs, a_probs], name='attention_mul')
    # return output
    return K.sum(output, axis=1)


def compute_statistics(windowed_dlogp):
    statistics = np.empty((windowed_dlogp.shape[0], 4), dtype='float64')

    statistics[:, 0] = np.mean(windowed_dlogp, axis=1)
    statistics[:, 1] = np.std(windowed_dlogp, axis=1)
    statistics[:, 2] = spt.skew(windowed_dlogp, axis=1)
    statistics[:, 3] = spt.kurtosis(windowed_dlogp, axis=1)

    statistics = (statistics - statistics.mean(axis=0)) / statistics.std(axis=0)
    return statistics

# def create_model():
#     x = Input(batch_shape=(None, lookbehind, 1))
#     lstm1 =  LSTM(100, dropout=0.2, recurrent_dropout=0.15, return_sequences=True)(x)
#     lstm2 = LSTM(100, dropout=0.2, recurrent_dropout=0.15)(lstm1)
#     stats = Input(batch_shape=(None, 4))
#     dense_input = keras.layers.concatenate([lstm2, stats], axis=1)
#     output = Dense(3, activation='softmax')(dense_input)
#     return Model(inputs=[x, stats], outputs=output)


def create_model():
    inputs = Input(batch_shape=(None, lookbehind, 1))
    out = Bidirectional(LSTM(128, dropout=0.4, recurrent_dropout=0.4, activation='relu', return_sequences=True))(inputs)
    out = Bidirectional(CuDNNLSTM(64, return_sequences=True))(out)
    stats = Input(batcch_shape=(None, 4))
    stats = RepeatVector(out.shape[1])(stats)
    out = concatenate([out, stats], axis=-1)
    out = attention_block(out)

    # dense_input = keras.layers.concatenate([lstm2, stats], axis=1)
    out = Flatten()(out)
    out = Dense(3, activation='softmax')(out)
    return Model(inputs=[inputs], outputs=out)


batch_size = 200
lookbehind = 2000

data = np.load('train_data_tf1min.npz')
y = data['y']
dlogp = data['dlogp']

dlogp = (dlogp - dlogp.mean()) / dlogp.std()
windowed_dlogp = rolling_window(dlogp, lookbehind)
y = y[lookbehind - 1:]
y -= y.min()
y = keras.utils.to_categorical(y, num_classes=3)

stats_filepath = 'statistics.npy'

try:
    statistics = np.load(stats_filepath, allow_pickle=True)
except IOError:
    statistics = compute_statistics(windowed_dlogp)
    np.save(stats_filepath, statistics)

drive.mount('/content/drive')
weights_directory = "/content/drive/My Drive/nn_weights/keras_lstm/"

files = glob.glob(weights_directory + "weights.*.hdf5")
regex = re.compile(r'weights\.(\d+)-(\d+\.\d+)\.hdf5')

most_recent_file = None
last_epoch = 0

if files:
    for file in files:
        match = regex.search(os.path.basename(file))
        if match:
            current_epoch = int(match.group(1))
            if current_epoch > last_epoch:
                last_epoch = current_epoch
                most_recent_file = file

if most_recent_file:
    print(f'file trovato: last_epoch = {last_epoch}')
    model = keras.models.load_model(most_recent_file)
else:
    model = create_model()

weights_filepath = weights_directory + "weights.{epoch:03d}-{val_acc:.3f}.hdf5"
checkpointer = keras.callbacks.ModelCheckpoint(weights_filepath, verbose=1, save_best_only=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(patience=3, verbose=1)

opt = keras.optimizers.RMSprop(lr=0.01)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

try:
    model.fit(x=[windowed_dlogp[..., np.newaxis], statistics], y=y,
              batch_size=batch_size, epochs=100 - last_epoch, validation_split=0.3, callbacks=[checkpointer, reduce_lr])
except KeyboardInterrupt:
    # checkpointer.set_model(model)
    # checkpointer.on_epoch_end(model.history.epoch[-1])
    print("stopped correctly..")
