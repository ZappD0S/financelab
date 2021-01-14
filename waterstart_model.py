from typing import Optional, Tuple

import torch
import torch.nn as nn

# the model input data will be normalized dividing by the
# last prices.

# the shape of the input is (batch_size, 3, seq_len, n_cur)


class GatedTrasition(nn.Module):
    def __init__(self, input_dim: int, z_dim: int, hidden_dim: int):
        super().__init__()
        self.h0 = nn.Parameter(torch.zeros(z_dim))
        self.softplus = nn.Softplus()
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

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if h is None:
            batch_size = x.size(0)
            h = self.h0.expand(batch_size, -1).contiguous()

        r = torch.relu_(self.lin_xr(x) + self.lin_hr(h))
        mean_ = self.lin_xm_(x) + self.lin_rm_(r)

        g = torch.sigmoid_(self.lin_xg(x) + self.lin_hg(h))

        mean = (1 - g) * mean_ + g * self.lin_hm(h)

        # TODO: instead of using softplus we might interpred
        # the output of the linear as the log of sigma and
        # then apply torch.exp
        sigma = self.softplus(self.lin_m_s(mean_.relu()))

        return mean, sigma


class Emitter(nn.Module):
    def __init__(self, z_dim: int, n_cur: int, hidden_dim: int):
        super().__init__()
        self.n_cur = n_cur
        self.lin1 = nn.Linear(z_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, 4 * n_cur)

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
        # n_features might be (sell, buy) x (open, high, low, close) + delta = 9 features
        # x: (batch_size, n_features, n_cur, window_size)
        assert x.size(1) == self.in_features
        assert x.size(2) == self.n_cur
        assert x.size(3) == self.window_size
        # batch_shape = x.shape[:-3]
        # x = x.flatten(end_dim=-4)

        out = self.conv1(x).relu_()
        out = self.conv2(out).relu_()
        out = self.conv3(out).relu_()

        # return out.unflatten(-4, batch_shape)
        # return out.view(*batch_shape, *out.shape[-4:])
        return out
