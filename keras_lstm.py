import os
import sys
import re
import glob
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input, Dense, Lambda, Bidirectional, Conv1D, BatchNormalization
from tensorflow.python.keras.layers import CuDNNLSTM
from tensorflow.python.keras.callbacks import ModelCheckpoint
from attention import Attention
from cyclical_learning_rate import CyclicLR
import h5py
import matplotlib.pyplot as plt


def rolling_window(x, window_size):
    stride = x.strides[0]
    return np.lib.stride_tricks.as_strided(x, (x.size - window_size + 1, window_size), (stride, stride))


def create_model():
    inputs = Input(batch_shape=(None, lookbehind, 1))
    # inputs = Input(batch_shape=(None, n_windows, window_size))
    out = Lambda(lambda x: K.conv1d(x, K.expand_dims(K.eye(window_size), 1), strides=window_size // 2))(inputs)
    shortcut = Conv1D(2 * 128, 1)(out)
    out = Bidirectional(CuDNNLSTM(128, return_sequences=True))(out)
    out = tf.keras.layers.add([out, shortcut])
    out = BatchNormalization()(out)
    shortcut = Conv1D(2 * 64, 1)(out)
    # out = Bidirectional(CuDNNLSTM(64, return_sequences=True))(out)
    out = Bidirectional(CuDNNLSTM(64))(out)
    out = tf.keras.layers.add([out, shortcut])
    out = BatchNormalization()(out)
    out = Lambda(lambda x: x[:, -1, :])(out)
    # out = Attention(n_windows)(out)
    out = Dense(64, activation='relu')(out)
    out = Dense(3, activation='softmax')(out)
    return Model(inputs=[inputs], outputs=out)


def find_lr(model, data, steps, init_value=1e-8, final_value=10., beta=0.98):
    mult = (final_value / init_value) ** (1 / steps)
    lr = init_value
    K.set_value(model.optimizer.lr, init_value)
    avg_loss = 0.
    best_loss = 0.
    losses = []
    log_lrs = []
    x, y = data.make_one_shot_iterator().get_next()
    for i in range(steps):
        sys.stdout.write(f'\r{i+1}/{steps}')
        sys.stdout.flush()
        # As before, get the loss for this mini-batch of inputs/outputs
        outs = model.train_on_batch(x, y)
        loss = outs[model.metrics_names.index('loss')]
        # loss = dict(zip(model.metrics_names, outs))['loss']

        avg_loss = beta * avg_loss + (1 - beta) * loss
        smoothed_loss = avg_loss / (1 - beta**(i + 1))
        # Stop if the loss is exploding
        if i > 0 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        # Record the best loss
        if smoothed_loss < best_loss or i == 0:
            best_loss = smoothed_loss
        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(np.log10(lr))
        # Update the lr for the next step
        lr *= mult
        K.set_value(model.optimizer.lr, lr)

    return log_lrs, losses


if __name__ == "__main__":
    K.clear_session()
    batch_size = 16
    n_windows = 300
    window_size = 100
    lookbehind = (n_windows + 1) * window_size // 2
    steps_per_epoch = 1500
    val_batch_size = 512
    val_steps = 2**4

    # from google.colab import drive
    # drive.mount('/content/drive')

    # with h5py.File('/content/drive/My Drive/train_data/train_data_tf1s_d0.6_lfz.h5', 'r') as f:
    with h5py.File('train_data/train_data_tf1s_d0.6_lfz.h5', 'r') as f:
        y = f['y'][()]
        dlogp = f['dlogp'][()]

    dlogp = (dlogp - dlogp.mean()) / dlogp.std()
    windowed_dlogp = rolling_window(dlogp, lookbehind)
    # y = y[lookbehind - 1:]
    y = np.asarray(y - y.min(), dtype='uint8')

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
        model = tf.keras.models.load_model(most_recent_file)
    else:
        model = create_model()

    weights_filepath = os.path.join(weights_directory, 'weights.{epoch:03d}-{val_acc:.3f}.hdf5')
    callbacks = [ModelCheckpoint(weights_filepath, verbose=1, save_best_only=True),
                 CyclicLR(3e-5, 5e-4, step_size=4 * steps_per_epoch)]

    # model.compile(loss='categorical_crossentropy',
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    ds_x = tf.data.Dataset.from_tensor_slices(dlogp[..., np.newaxis])
    # ds_y = tf.data.Dataset.from_tensor_slices(tf.one_hot(y, 3))
    ds_y = tf.data.Dataset.from_tensor_slices(y)
    ds_x = ds_x.window(lookbehind, shift=1).flat_map(lambda x: x.batch(lookbehind))
    ds_y = ds_y.skip(lookbehind - 1)
    # def map_fn(x):
    #     return x.window(window_size, shift=window_size // 2).flat_map(lambda z: z.batch(window_size, drop_remainder=True)).batch(n_windows)

    # ds_x = ds_x.map(map_fn, 4).flat_map(lambda x: x)
    # ds = tf.data.Dataset.zip((ds_x, ds_y))
    ds = tf.data.Dataset.zip((ds_x, ds_y)).filter(lambda x, y: tf.not_equal(y, 3))
    train_ds = ds.skip(val_batch_size * val_steps).shuffle(2000).batch(batch_size).prefetch(100)
    test_ds = ds.take(val_batch_size * val_steps).repeat().batch(val_batch_size)
    model.fit(train_ds, epochs=100, steps_per_epoch=steps_per_epoch, validation_data=test_ds, validation_steps=val_steps, callbacks=callbacks)
    # ds = ds.apply(tf.data.experimental.prefetch_to_device('/device:GPU:0', 10))

    # lrs = tf.constant(np.logspace(np.log10(from_lr), np.log10(to_lr), n_lrs, dtype='float32'))

    # index = tf.placeholder(tf.int32, shape=())
    # set_lr_op = model.optimizer.lr.assign(lrs[index])
    # k = np.exp((np.log(to_lr) - np.log(from_lr)) / n_lrs)
    # increase_lr_op = model.optimizer.lr.assign(k * model.optimizer.lr)

    log_lrs, losses = find_lr(model, ds, 1000)
