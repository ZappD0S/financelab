from math import log
from typing import Optional, Tuple

import torch
import torch.nn as nn


def clamp_preserve_gradients(x: torch.Tensor, min: float, max: float):
    return x + torch.detach_(x.clamp(min, max) - x)


class GatedTrasition(nn.Module):
    def __init__(self, input_dim: int, z_dim: int, hidden_dim: int):
        super().__init__()
        # self.h0 = nn.Parameter(torch.zeros(z_dim))
        # self.softplus = nn.Softplus()
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
        # log_sigma = clamp_preserve_gradients(log_sigma, -5.0, 3.0)
        log_sigma = clamp_preserve_gradients(log_sigma, self.log_sigma_min, self.log_sigma_max)
        sigma = log_sigma.exp_()

        return mean, sigma


class Emitter(nn.Module):
    def __init__(self, z_dim: int, n_cur: int, hidden_dim: int):
        super().__init__()
        self.n_cur = n_cur
        # TODO: compute the right in_features value
        self.lin1 = nn.Linear(z_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        # self.lin3 = nn.Linear(hidden_dim, 4 * n_cur)
        self.lin3 = nn.Linear(hidden_dim, 2 * n_cur)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: batch_size * n_samples, z_dim

        out = self.lin1(z).relu_()
        out = self.lin2(out).relu_()

        return self.lin3(out)


class CNN(nn.Module):
    def __init__(
        self, window_size: int, in_features: int, out_features: int, n_cur: int, n_samples: int, max_trades: int
    ):
        super().__init__()
        # TODO: we might add another conv layer for prev_step_data to go through before we cat it with the rest
        # and pass it to the last conv.
        self.window_size = window_size
        self.n_cur = n_cur
        self.n_samples = n_samples
        self.in_features = in_features
        self.kernel_size = 3
        self.conv1 = nn.Conv1d(in_features, in_features, kernel_size=(1, self.kernel_size))
        l_in = window_size + 1 - self.kernel_size
        l_out = 32
        self.conv2 = nn.Conv1d(in_features, l_out, kernel_size=(1, l_in))
        self.conv3 = nn.Conv1d(l_out, out_features, kernel_size=n_cur * (l_out + 2 * max_trades) + 1)

    def forward(self, x: torch.Tensor, prev_step_data: torch.Tensor):
        # x: (batch_size * seq_len, n_features, n_cur, window_size)
        # prev_step_data: (batch_size * seq_len * n_samples, 2 * n_cur * max_trades + 1)

        out = self.conv1(x).relu_()
        out = self.conv2(out).squeeze(3).relu_()

        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(-1, self.n_cur * self.conv2.out_channels)
            .unsqueeze(1)
            .expand(-1, self.n_samples, -1)
            .contiguous()
            .view(-1, self.n_cur * self.conv2.out_channels)
        )
        out = torch.cat([out, prev_step_data], dim=1)

        out = self.conv3(out).relu_()
        return out
