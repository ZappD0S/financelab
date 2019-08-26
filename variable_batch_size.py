import copy
import numpy as np
import tensorflow as tf
# import tensorflow.python
from tensorflow.python.keras.callbacks import CallbackList
import tensorflow.keras.backend as K
# import keras


class IncreaseBatchSizeOnPlateau(object):
    # def __init__(self, initial_batch_size, train_samples, patience, cooldown):
    def __init__(self, initial_batch_size, patience, cooldown):
        self.batch_size = initial_batch_size
        # self.train_samples = train_samples
        self.patience = patience
        self.cooldown = cooldown
        self.wait = 0
        self.cooldown_counter = 0
        self.min_loss = 0

    def get_batch_size(self, epoch, epoch_logs):
        if epoch_logs['val_loss'] < self.min_loss:
            self.min_loss = epoch_logs['val_loss']
            self.wait = 0
        elif self.cooldown_counter == 0:
            self.wait += 1
            if self.wait >= self.patience:
                # if self.batch_size * 2 < self.train_samples:
                self.batch_size *= 2
                print('\nEpoch %05d: increasing batch_size to %s.' % (epoch + 1, self.batch_size))
                self.cooldown_counter = self.cooldown
                self.wait = 0
        return self.batch_size


def train(model, dset, val_samples, epochs, initial_batch_size, filepath, patience, cooldown, initial_epoch=0):
    # num_train_samples = x.shape[0]
    batch_size_increaser = IncreaseBatchSizeOnPlateau(initial_batch_size, patience=patience, cooldown=cooldown)

    # callbacks = [tf.keras.callbacks.BaseLogger(stateful_metrics=model.stateful_metric_names),
    #              tf.keras.callbacks.ProgbarLogger(stateful_metrics=model.stateful_metric_names),
    #              tf.keras.callbacks.ModelCheckpoint(filepath, verbose=1, save_best_only=True)]
    callbacks = [tf.keras.callbacks.BaseLogger(stateful_metrics=model.metrics_names[1:]),
                 tf.keras.callbacks.ProgbarLogger(stateful_metrics=model.metrics_names[1:]),
                 tf.keras.callbacks.ModelCheckpoint(filepath, verbose=1, save_best_only=True)]

    callbacks = CallbackList(callbacks)
    # callbacks = tf.python.keras.callbacks.configure_callbacks(callbacks, do_validation=True, batch_size=initial_batch_size, epochs=epochs)

    # it's possible to callback a different model than itself
    # (used by Sequential models)
    if hasattr(model, 'callback_model') and model.callback_model:
        callback_model = model.callback_model
    else:
        callback_model = model

    callback_metrics = copy.copy(model.metrics_names) + [
        'val_' + n for n in model.metrics_names]

    callbacks.set_model(callback_model)
    callbacks.set_params({
        # 'batch_size': batch_size,
        'epochs': epochs,
        # 'steps': steps_per_epoch,
        'steps': None,
        # 'samples': num_train_samples,
        'samples': None,
        'verbose': 1,
        'do_validation': 1,
        'metrics': callback_metrics or [],
    })

    train_dset = dset.skip(val_samples)
    val_dset = dset.take(val_samples).batch(val_samples)
    val_iter = val_dset.make_one_shot_iterator()
    val_x, val_y = val_iter.get_next()

    for cbk in callbacks:
        cbk.validation_data = (val_x, val_y)

    batch_size = initial_batch_size

    callbacks.on_train_begin()
    callback_model.stop_training = False

    for epoch in range(initial_epoch, epochs):
        # for m in model.stateful_metric_functions:
        #     m.reset_states()
        model.reset_metrics()
        callbacks.on_epoch_begin(epoch)
        epoch_logs = {}

        # train_iter = train_dset.batch(batch_size).make_initializable_iterator()
        # K.get_session().run(train_iter.initializer)
        train_iter = train_dset.batch(batch_size).make_one_shot_iterator()
        x, y = train_iter.get_next()
        batch_index = 0
        while True:
            batch_logs = {'batch': batch_index, 'size': K.eval(tf.shape(x)[0])}
            callbacks.on_batch_begin(batch_index, batch_logs)

            try:
                outs = model.train_on_batch(x, y, reset_metrics=False)
            except tf.errors.OutOfRangeError:
                break

            for l, o in zip(model.metrics_names, outs):
                batch_logs[l] = o

            callbacks.on_batch_end(batch_index, batch_logs)
            if callback_model.stop_training:
                break

            batch_index += 1

        val_outs = model.test_on_batch(val_x, val_y)
        for l, o in zip(model.metrics_names, val_outs):
            epoch_logs['val_' + l] = o
        callbacks.on_epoch_end(epoch, epoch_logs)
        batch_size = batch_size_increaser.get_batch_size(epoch, epoch_logs)

        if callback_model.stop_training:
            break

    callbacks.on_train_end()
    return model.history


# def train(model, x, y, val_x, val_y, epochs, initial_batch_size, filepath, patience, cooldown, initial_epoch=0):
#     num_train_samples = x.shape[0]
#     batch_size_increaser = IncreaseBatchSizeOnPlateau(initial_batch_size, num_train_samples, patience=patience, cooldown=cooldown)
#
#     callbacks = [keras.callbacks.BaseLogger(stateful_metrics=model.stateful_metric_names),
#                  keras.callbacks.ProgbarLogger(stateful_metrics=model.stateful_metric_names),
#                  keras.callbacks.ModelCheckpoint(filepath, verbose=1, save_best_only=True)]
#
#     callbacks = keras.callbacks.CallbackList(callbacks)
#
#     # it's possible to callback a different model than itself
#     # (used by Sequential models)
#     if hasattr(model, 'callback_model') and model.callback_model:
#         callback_model = model.callback_model
#     else:
#         callback_model = model
#
#     callback_metrics = copy.copy(model.metrics_names) + [
#         'val_' + n for n in model.metrics_names]
#
#     callbacks.set_model(callback_model)
#     callbacks.set_params({
#         # 'batch_size': batch_size,
#         'epochs': epochs,
#         # 'steps': steps_per_epoch,
#         'steps': None,
#         'samples': num_train_samples,
#         'verbose': 1,
#         'do_validation': 1,
#         'metrics': callback_metrics or [],
#     })
#
#     callbacks.on_train_begin()
#     callback_model.stop_training = False
#     for cbk in callbacks:
#         cbk.validation_data = (val_x, val_y)
#
#     index_array = np.arange(num_train_samples)
#     batch_size = initial_batch_size
#
#     for epoch in range(initial_epoch, epochs):
#         for m in model.stateful_metric_functions:
#             m.reset_states()
#         callbacks.on_epoch_begin(epoch)
#         epoch_logs = {}
#
#         np.random.shuffle(index_array)
#
#         remainder = num_train_samples % batch_size
#         num_batches = num_train_samples // batch_size + min(remainder, 1)
#
#         for batch_index in range(num_batches):
#             batch_start, batch_end = batch_size * batch_index, batch_size * (batch_index + 1)
#             batch_inds = index_array[batch_start:batch_end]
#
#             x_batch, y_batch = x[batch_inds], y[batch_inds]
#
#             # we need to use len(batch_inds) because
#             # for the last batch it could be different from batch_size
#             batch_logs = {'batch': batch_index, 'size': len(batch_inds)}
#             callbacks.on_batch_begin(batch_index, batch_logs)
#
#             outs = model.train_on_batch(x_batch, y_batch)
#             # outs = to_list(outs)
#
#             for l, o in zip(model.metrics_names, outs):
#                 batch_logs[l] = o
#
#             callbacks.on_batch_end(batch_index, batch_logs)
#             if callback_model.stop_training:
#                 break
#
#             if batch_index == num_batches - 1:  # Last batch.
#                 val_outs = model.test_on_batch(val_x, val_y)
#                 # val_outs = to_list(val_outs)
#                 # out_labels
#                 for l, o in zip(model.metrics_names, val_outs):
#                     epoch_logs['val_' + l] = o
#         callbacks.on_epoch_end(epoch, epoch_logs)
#         batch_size = batch_size_increaser.get_batch_size(epoch, epoch_logs)
#
#         if callback_model.stop_training:
#             break
#     callbacks.on_train_end()
