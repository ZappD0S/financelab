import numpy as np
from tensorflow.python.keras.utils import Sequence
from tensorflow.python.keras.utils import to_categorical


def rolling_window(x, window_size):
    stride = x.strides[0]
    return np.lib.stride_tricks.as_strided(x, (x.size - window_size + 1, window_size), (stride, stride))


class WindowsGenerator(Sequence):
    def __init__(self, x, y, batch_size, lookbehind, skip_last):
        self.batch_size = batch_size
        min_size = min(x.size, y.size)
        x = x[:min_size - skip_last]
        y = y[:min_size - skip_last]

        self.windowed_x = rolling_window(x, lookbehind)
        y = y[lookbehind - 1:]
        assert self.windowed_x.shape[0] == y.shape[0]

        mask = y != 3
        self.y = to_categorical(np.where(mask, y, -1), 3)
        self.valid_inds = mask.nonzero()[0]

        self.n_batches = int(np.ceil(self.valid_inds.size / batch_size))

        np.random.shuffle(self.valid_inds)

    def __len__(self):
        return self.n_batches

    def __getitem__(self, index):
        batch_inds = self.valid_inds[index:index + self.batch_size]
        x = self.windowed_x[batch_inds]
        y = self.y[batch_inds]
        return x, y

    def on_epoch_end(self):
        np.random.shuffle(self.valid_inds)

