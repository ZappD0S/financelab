import torch
import torch.nn as nn
from torch.utils import data
from multi_scale_resnet import MSResNet
from tqdm import tqdm


class EndOfSamples(Exception):
    pass


# class ParametersCalculator:
#     def __init__(self, data_size, max_windows_per_sample, window_size, sample_stride, window_stride):
#         self.data_size = data_size
#         self.sample_size = max_windows_per_sample
#         self.window_size = window_size
#         self.window_stride = window_stride
#         self.sample_stride = sample_stride
#         self.n_windows = (data_size - window_size) // window_stride
#         self.n_samples = (self.n_windows - max_windows_per_sample) * window_stride // sample_stride
#
#     def get_window_slice(self, n_window):
#         return slice(self.window_stride * n_window,
#                      self.window_stride * n_window + self.window_size)


class MyDataset(data.Dataset):
    def __init__(self, p, window_size):
        self.window_size = window_size
        self.p_size = p.size
        self.p = torch.from_numpy((p - p.mean()) / p.std())

    def __getitem__(self, index):
        window = self.p[index:index + self.window_size]
        window -= window[0]
        return window

    def __len__(self):
        return self.p_size - self.window_size


class MySampler(data.Sampler):
    def __init__(self, dataset, sample_stride, window_stride, max_windows_per_sample, batch_size):
        self.batch_size = batch_size
        n_windows = len(dataset)
        self.n_samples = (n_windows - max_windows_per_sample * window_stride) // sample_stride
        self.window_stride = window_stride
        self.max_windows_per_sample = max(sample_stride // window_stride, max_windows_per_sample)
        self.sample_inds = torch.randperm(self.n_samples) * sample_stride
        self.counter = 0

    def __iter__(self):
        if self.counter * self.batch_size > self.n_samples:
            raise EndOfSamples()

        start_batch_inds = self.sample_inds[self.counter * self.batch_size:(self.counter + 1) * self.batch_size]
        self.counter += 1

        for i in range(self.max_windows_per_sample):
            yield start_batch_inds + i * self.window_stride


class Model(nn.Module):
    def __init__(self, hidden_size, input_channels):
        super().__init__()
        self.resnet = MSResNet(input_channels)
        self.lstm = nn.LSTM(256 * 3, hidden_size)
        self.fc = nn.Linear(hidden_size, 2, bias=True)

    def forward(self, input, hidden=None):
        out = self.resnet(input)
        out, hidden = self.lstm(out.unsqueeze(1), hidden)
        out = self.fc(out.squeeze())
        fs, ks = out.unbind(dim=1)
        fs = torch.tanh(fs)
        ks = torch.sigmoid(ks)
        return (fs, ks), hidden


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_pickle("~/source/repos/financelab/tick_data/eurusd_2018.pkl")
    df = df.resample('1s').ohlc().dropna()
    p = df['sell', 'close'].values[:10000]

    ds = MyDataset(p, 100)
    batch_size = 16
    spread = 1.3e-4
    sampler = MySampler(ds, sample_stride=300, window_stride=4, max_windows_per_sample=100, batch_size=batch_size)
    dloader = data.DataLoader(ds, batch_sampler=sampler)

    while True:
        try:
            for i, x in enumerate(dloader):
                pass
            print('fine della serie temporale')
        except EndOfSamples:
            print("fine dell'epoch")
            break

    # out = torch.rand(batch_size, 2)
    # fs, ks = out.unbind(dim=1)
    # fs = fs*2 - 1
    # fs = fs.view(batch_size, 1, 1)
    # e0 = torch.rand(batch_size, 1, 1) + 1
    # d0 = torch.rand(batch_size, 1, 1) + 1
    # p = torch.rand(batch_size) + 1

    def get_new_state(fs, ks, e0, d0, p, ignore_threshold=False):
        p = torch.stack([p, p + spread], dim=1)
        p1 = p.unsqueeze(dim=2)
        p2 = p.unsqueeze(dim=1)

        e1 = (fs - 1) * (d0 + e0 * p1) / ((fs - 1) * p1 - fs * p2)
        d1 = (fs * p2 * (d0 + e0 * p1)) / (-fs * p1 + fs * p2 + p1)
        y = -(d0 * (fs - 1) + e0 * fs * p2) / ((fs - 1) * p1 - fs * p2)

        cond1 = torch.stack([y[:, 0, :] > 0, y[:, 1, :] < 0], dim=1)
        cond2 = torch.stack([d1[:, :, 0] < 0, d1[:, :, 1] > 0], dim=2)
        cond = cond1 & cond2
        y = y[cond]
        assert y.shape == (batch_size,)
        e1 = e1[cond]
        d1 = d1[cond]

        if not ignore_threshold:
            mask = y.abs() > ks
            e1 = torch.where(mask, e1, e0)
            d1 = torch.where(mask, d1, d0)
        return e1, d1

    def train(model, dataset, optimizer):
        while True:
            e0 = torch.ones(batch_size)
            d0 = torch.zeros(batch_size)
            iterator = iter(dataset)
            batch = next(iterator)
            (fs, ks), hidden = model(batch)
            e0, d0 = get_new_state(fs, ks, e0, d0, batch[:, -1])

            try:
                for batch in iterator:
                    (fs, ks), hidden = model(batch, hidden)
                    e0, d0 = get_new_state(fs, ks, e0, d0, batch[:, -1])
            # except EndOfTimeSeries:
                e0, _ = get_new_state(torch.zeros(batch_size), None, e0, d0, batch[:, -1])
                loss = e0.mean().neg()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print("fine della serie temporale")
            except EndOfSamples:
                # print("fine dell'epoch")
                break

