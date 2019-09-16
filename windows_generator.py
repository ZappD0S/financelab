import numpy as np
from tensorflow.python.keras.utils import Sequence


def rolling_window(x, window_size):
    stride = x.strides[0]
    return np.lib.stride_tricks.as_strided(x, (x.size - window_size + 1, window_size), (stride, stride))


def categorical_to_binary_list(y):
    classes = np.unique(y)

    done_mask = np.zeros(y.size, dtype='bool')
    inds_outputs = []

    for cls_ in classes[:-1]:
        cur_inds = np.nonzero(~done_mask)[0]
        mask = y[~done_mask] == cls_
        cur_y = np.empty(mask.size, dtype='uint8')
        cur_y[mask] = 0
        cur_y[~mask] = 1
        inds_outputs.append((cur_inds, cur_y))
        done_mask[~done_mask] = mask

    return inds_outputs


def get_stratified_inds(y, train_length, inds=None, freqs=None, seed=None):
    assert train_length < y.size
    if freqs is None:
        classes, counts = np.unique(y, return_counts=True)
        freqs = counts / counts.sum()
    else:
        classes = np.unique(y)
        assert np.isclose(np.sum(freqs), 1)
        assert len(classes) == len(freqs)

    if inds is None:
        inds = np.arange(y.size)
    else:
        assert y.size == inds.size

    train_inds = []
    test_inds = []

    if seed is not None:
        np.random.seed(seed)

    for i, cls_ in enumerate(classes):
        inds_per_cls = inds[y == cls_]
        np.random.shuffle(inds_per_cls)
        split_ind = int(freqs[i] * train_length)
        train_inds.append(inds_per_cls[:split_ind])
        test_inds.append(inds_per_cls[split_ind:])

    train_inds = np.concatenate(train_inds)
    np.random.shuffle(train_inds)
    test_inds = np.concatenate(test_inds)
    np.random.shuffle(test_inds)

    return train_inds, test_inds


class WindowsGenerator(Sequence):
    def __init__(self, windowed_x, y, batch_size, inds):
        self.windowed_x = windowed_x
        self.y = y
        self.batch_size = batch_size
        self.inds = inds
        self.n_batches = int(np.ceil(self.inds.size / batch_size))

    def __len__(self):
        return self.n_batches

    def __getitem__(self, index):
        batch_inds = self.inds[index:index + self.batch_size]
        x = self.windowed_x[batch_inds]
        y = self.y[batch_inds]
        return x, y


# def get_stratified_inds(y, fold_length, n_folds, seed):
#     assert n_folds * fold_length <= y.size
#     classes = np.unique(y)
#     inds = np.arange(y.size)
#     train_inds = [[] for _ in range(n_folds)]
#     test_inds = []

#     np.random.seed(seed)

#     for cls_ in classes:
#         inds_per_cls = inds[y == cls_]
#         frac = int(fold_length * inds_per_cls.size / y.size)
#         np.random.shuffle(inds_per_cls)
#         for i in range(n_folds):
#             train_inds[i].append(inds_per_cls[i * frac:(i + 1) * frac])
#         test_inds.append(inds_per_cls[(n_folds + 1) * frac:])

#     for i in range(n_folds):
#         train_inds[i] = np.concatenate(train_inds[i])
#         np.random.shuffle(train_inds[i])

#     test_inds = np.concatenate(test_inds)
#     np.random.shuffle(test_inds)

#     return train_inds, test_inds

# class WindowsGenerator(Sequence):
#     def __init__(self, x, y, batch_size, lookbehind, skip_last):
#         self.batch_size = batch_size
#         min_size = min(x.size, y.size)
#         x = x[:min_size - skip_last]
#         y = y[:min_size - skip_last]

#         self.windowed_x = rolling_window(x, lookbehind)
#         y = y[lookbehind - 1:]
#         assert self.windowed_x.shape[0] == y.shape[0]

#         mask = y != 3
#         self.y = to_categorical(np.where(mask, y, -1), 3)
#         self.valid_inds = mask.nonzero()[0]

#         self.n_batches = int(np.ceil(self.valid_inds.size / batch_size))

#         np.random.shuffle(self.valid_inds)

#     def __len__(self):
#         return self.n_batches

#     def __getitem__(self, index):
#         batch_inds = self.valid_inds[index:index + self.batch_size]
#         x = self.windowed_x[batch_inds]
#         y = self.y[batch_inds]
#         return x, y

#     def on_epoch_end(self):
#         np.random.shuffle(self.valid_inds)
