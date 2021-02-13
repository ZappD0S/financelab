from math import log
from typing import Tuple

import torch
import torch.nn as nn


def clamp_preserve_gradients(x: torch.Tensor, min: float, max: float):
    return x + torch.detach_(x.clamp(min, max) - x)


class GatedTrasition(nn.Module):
    def __init__(self, input_dim: int, z_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.log_sigma_min = log(torch.finfo(torch.get_default_dtype()).eps)
        self.log_sigma_max = -self.log_sigma_min

        self.lin_xr = nn.Linear(input_dim, hidden_dim)
        self.lin_hr = nn.Linear(z_dim, hidden_dim)

        self.lin_xm_ = nn.Linear(input_dim, z_dim)
        self.lin_rm_ = nn.Linear(hidden_dim, z_dim)

        self.lin_xg = nn.Linear(input_dim, z_dim)
        self.lin_hg = nn.Linear(z_dim, z_dim)

        self.lin_hm = nn.Linear(z_dim, z_dim)

        self.lin_m_s = nn.Linear(z_dim, z_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r = torch.relu_(self.lin_xr(x) + self.lin_hr(h))
        mean_ = self.lin_xm_(x) + self.lin_rm_(r)

        g = torch.sigmoid_(self.lin_xg(x) + self.lin_hg(h))

        mean = (1 - g) * self.lin_hm(h) + g * mean_

        log_sigma = self.lin_m_s(mean_.relu())
        log_sigma = clamp_preserve_gradients(log_sigma, self.log_sigma_min, self.log_sigma_max)
        sigma = log_sigma.exp_()

        return mean, sigma


class Emitter(nn.Module):
    def __init__(self, z_dim: int, n_cur: int, hidden_dim: int):
        super().__init__()
        self.n_cur = n_cur
        self.lin1 = nn.Linear(z_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, 2 * n_cur)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: batch_dims..., z_dim

        out = self.lin1(z).relu_()
        out = self.lin2(out).relu_()

        return self.lin3(out)


class CNN(nn.Module):
    def __init__(
        self,
        seq_len: int,
        n_samples: int,
        batch_size: int,
        window_size: int,
        in_features: int,
        out_features: int,
        n_cur: int,
        max_trades: int,
    ):
        super().__init__()
        # TODO: we might add another conv layer for prev_step_data to go through before we cat it with the rest
        # and pass it to the last conv.
        self.seq_len = seq_len
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.window_size = window_size
        # self.in_features = in_features
        self.kernel_size = 3
        self.n_cur = n_cur
        self.conv1 = nn.Conv1d(in_features, in_features, kernel_size=(1, self.kernel_size))
        l_in = window_size + 1 - self.kernel_size
        # TODO: is this a good value?
        l_out = 32
        self.conv2 = nn.Conv1d(in_features, l_out, kernel_size=(1, l_in))
        self.conv3 = nn.Conv1d(1, out_features, kernel_size=n_cur * (l_out + 2 * max_trades) + 1)

    def forward(self, x: torch.Tensor, prev_step_data: torch.Tensor):
        # x: (seq_len * batch_size, n_features, n_cur, window_size)
        # prev_step_data: (n_samples * seq_len * batch_size, 2 * n_cur * max_trades + 1)

        out = self.conv1(x).relu_()
        out = self.conv2(out).squeeze(3).relu_()

        out = (
            out.transpose_(1, 2)
            .expand(self.n_samples, -1, -1, -1)
            .contiguous()
            .view(-1, self.n_cur * self.conv2.out_channels)
        )

        out = torch.cat([out, prev_step_data], dim=1).unsqueeze_(1)
        out = self.conv3(out).squeeze_(2).relu_()

        return out


class NeuralBaseline(nn.Module):
    def __init__(self, z_dim: int, n_cur: int, max_trades: int, hidden_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(2 * n_cur * (max_trades + 1) + z_dim + 1, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, n_cur)

    def forward(self, x):
        # x: (batch_dims..., 2 * n_cur * (max_trades + 1) + z_dim + 1)

        out = self.lin1(x).relu_()
        out = self.lin2(out)

        return out
