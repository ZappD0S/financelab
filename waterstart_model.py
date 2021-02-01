from typing import Optional, Tuple

import torch
import torch.nn as nn


def clamp_preserve_gradients(x: torch.Tensor, min: float, max: float):
    return x + (x.clamp(min, max) - x).detach()


class GatedTrasition(nn.Module):
    def __init__(self, input_dim: int, z_dim: int, hidden_dim: int):
        super().__init__()
        # self.h0 = nn.Parameter(torch.zeros(z_dim))
        # self.softplus = nn.Softplus()
        self.input_dim = input_dim
        self.z_dim = z_dim

        self.lin_xr = nn.Linear(input_dim, hidden_dim)
        self.lin_hr = nn.Linear(z_dim, hidden_dim)

        self.lin_xm_ = nn.Linear(input_dim, z_dim)
        self.lin_rm_ = nn.Linear(hidden_dim, z_dim)

        self.lin_xg = nn.Linear(input_dim, z_dim)
        self.lin_hg = nn.Linear(z_dim, z_dim)

        self.lin_hm = nn.Linear(z_dim, z_dim)

        self.lin_m_s = nn.Linear(z_dim, z_dim)

    # TODO: we need to recheck the structure of this model
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r = torch.relu_(self.lin_xr(x) + self.lin_hr(h))
        mean_ = self.lin_xm_(x) + self.lin_rm_(r)

        g = torch.sigmoid_(self.lin_xg(x) + self.lin_hg(h))

        mean = (1 - g) * mean_ + g * self.lin_hm(h)

        log_sigma = self.lin_m_s(mean_.relu())
        # TODO: make the min lower
        log_sigma = clamp_preserve_gradients(log_sigma, -5.0, 3.0)
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
        # out[..., -1].sigmoid_()

        # return self.lin3(out).sigmoid_()
        return self.lin3(out)


class CNN(nn.Module):
    def __init__(self, window_size: int, in_features: int, out_features: int, n_cur: int):
        super().__init__()

        self.window_size = window_size
        self.n_cur = n_cur
        self.in_features = in_features
        self.kernel_size = 3
        self.conv1 = nn.Conv1d(in_features, in_features, kernel_size=(1, self.kernel_size))
        l_in = window_size + 1 - self.kernel_size
        l_out = 32
        self.conv2 = nn.Conv1d(in_features, l_out, kernel_size=(1, l_in - l_out + 1))
        self.conv3 = nn.Conv1d(l_out, out_features, kernel_size=(n_cur, l_out))

    def forward(self, x: torch.Tensor):
        # x: (batch_size, n_features, n_cur, window_size)

        # batch_shape = x.shape[:-3]
        # x = x.flatten(end_dim=-4)

        out = self.conv1(x).relu_()
        out = self.conv2(out).relu_()
        out = self.conv3(out).relu_()

        # return out.unflatten(-4, batch_shape)
        # return out.view(*batch_shape, *out.shape[-4:])
        return out
