import os
import re
import glob
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout
# from keras.layers.recurrent import LSTM
from keras.layers import CuDNNLSTM
from keras.regularizers import l2
from keras.layers.wrappers import Bidirectional
import scipy.stats as spt
from attention import Attention


def rolling_window(x, window_size):
    stride = x.strides[0]
    return np.lib.stride_tricks.as_strided(x, (x.size - window_size + 1, window_size), (stride, stride))


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


def create_model(lookbehind):
    inputs = Input(batch_shape=(None, lookbehind, 1))
    # out = Bidirectional(LSTM(8, dropout=0.4, recurrent_dropout=0.4, activation='relu', return_sequences=True))(inputs)
    out = Bidirectional(CuDNNLSTM(2, return_sequences=True, kernel_regularizer=l2(), recurrent_regularizer=l2(0.001)))(inputs)
    out = Bidirectional(CuDNNLSTM(4, return_sequences=True, kernel_regularizer=l2(), recurrent_regularizer=l2(0.001)))(inputs)
    out = Bidirectional(CuDNNLSTM(8, return_sequences=True, kernel_regularizer=l2(), recurrent_regularizer=l2(0.001)))(out)
    out = Bidirectional(CuDNNLSTM(16, return_sequences=True, kernel_regularizer=l2(), recurrent_regularizer=l2(0.001)))(out)
    # out = Dropout(0.1)(out)
    out = Attention(lookbehind)(out)
    out = Dense(15, activation='relu')(out)
    out = Dropout(0.2)(out)
    # stats = Input(batch_shape=(None, 4))
    # out = concatenate([out, stats], axis=1)
    out = Dense(3, activation='softmax')(out)
    return Model(inputs=[inputs], outputs=out)


if __name__ == "__main__":

    batch_size = 16
    lookbehind = 2000

    from google.colab import drive
    drive.mount('/content/drive')

    data = np.load('/content/drive/My Drive/train_data/train_data_tf1min.npz')
    y = data['y']
    dlogp = data['dlogp']

    dlogp = (dlogp - dlogp.mean()) / dlogp.std()
    windowed_dlogp = rolling_window(dlogp, lookbehind)
    y = y[lookbehind - 1:]
    y -= y.min()
    y = keras.utils.to_categorical(y, num_classes=3)

    # stats_filepath = '/content/statistics.npy'

    # try:
    #     statistics = np.load(stats_filepath, allow_pickle=True)
    # except IOError:
    #     statistics = compute_statistics(windowed_dlogp)
    #     np.save(stats_filepath, statistics)

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
        model = create_model(lookbehind)

    weights_filepath = os.path.join(weights_directory, 'weights.{epoch:03d}-{val_acc:.3f}.hdf5')
    checkpointer = keras.callbacks.ModelCheckpoint(weights_filepath, verbose=1, save_best_only=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(verbose=1)

    # opt = keras.optimizers.RMSprop(lr=0.01)
    opt = keras.optimizers.Adam()

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    try:
        model.fit(x=[windowed_dlogp[..., np.newaxis]], y=y,
                  batch_size=batch_size, epochs=100 - last_epoch, validation_split=0.3,
                  callbacks=[checkpointer, reduce_lr])
    except KeyboardInterrupt:
        # checkpointer.set_model(model)
        # checkpointer.on_epoch_end(model.history.epoch[-1])
        print("stopped correctly..")
