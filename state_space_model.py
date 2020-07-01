import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.distribution import Distribution

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import affine_autoregressive
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam


class RoundedDistribution(Distribution):
    def __init__(self, base_distr, loc, scale, batch_shape=torch.Size(), event_shape=torch.Size(), validate_args=None):
        super(RoundedDistribution, self).__init__(
            batch_shape=batch_shape, event_shape=event_shape, validate_args=validate_args
        )

        self.base_distr = base_distr
        self.loc = loc
        self.scale = scale
        self.has_rsample = base_distr.has_rsample

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            base_sample = self.base_distr.sample(sample_shape)
            return (torch.round(base_sample * self.scale + self.loc) - self.loc) / self.scale

    def rsample(self, sample_shape=torch.Size()):
        if not self.has_rsample:
            raise NotImplementedError

        base_sample = self.base_distr.rsample(sample_shape)
        return (torch.round(base_sample * self.scale + self.loc) - self.loc) / self.scale

    def log_prob(self, value):
        high = (torch.round(value * self.scale + self.loc) + 0.5 - self.loc) / self.scale
        low = (torch.round(value * self.scale + self.loc) - 0.5 - self.loc) / self.scale
        return torch.log(self.base_distr.cdf(high) - self.base_distr.cdf(low))


# class Observation(nn.Module):
#     def __init__(self, z_dim, x_dim, hidden_dim=20):
#         super(Observation, self).__init__()
#         self.lin_loc_z_to_hidden = nn.Linear(z_dim, hidden_dim)
#         self.lin_loc_hidden_to_x = nn.Linear(hidden_dim, x_dim)

#         self.lin_scale_z_to_hidden = nn.Linear(z_dim, hidden_dim)
#         self.lin_scale_hidden_to_x = nn.Linear(hidden_dim, x_dim)

#     def forward(self, z_t):
#         _loc = F.relu(self.lin_loc_z_to_hidden(z_t))
#         loc = F.relu(self.lin_loc_hidden_to_x(_loc))

#         _scale = F.relu(self.lin_scale_z_to_hidden(z_t))
#         scale = F.relu(self.lin_scale_hidden_to_x(_scale))

#         return loc, scale


class LocScaleTransformer(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(LocScaleTransformer, self).__init__()

        self.lin_gate_input_to_hidden = nn.Linear(in_dim, hidden_dim)
        self.lin_gate_hidden_to_z = nn.Linear(hidden_dim, out_dim)

        self.lin_proposed_loc_z_to_hidden = nn.Linear(in_dim, hidden_dim)
        self.lin_proposed_loc_hidden_to_z = nn.Linear(hidden_dim, out_dim)

        self.lin_z_to_loc = nn.Linear(in_dim, out_dim)
        # self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        # self.lin_z_to_loc.bias.data = torch.zeros(z_dim)

        self.lin_z_to_scale = nn.Linear(out_dim, out_dim)

    def forward(self, input):
        _gate = F.relu(self.lin_gate_input_to_hidden(input))
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(_gate))

        _proposed_mean = F.relu(self.lin_proposed_loc_z_to_hidden(input))
        proposed_mean = self.lin_proposed_loc_hidden_to_z(_proposed_mean)

        loc = (1 - gate) * self.lin_z_to_loc(input) + gate * proposed_mean
        scale = F.softplus(self.lin_z_to_scale(F.relu(proposed_mean)))

        return loc, scale


class StateSpaceModel(nn.Module):
    def __init__(self, z_dim, x_dim, rnn_dim, trans_hidden_dim, obs_hidden_dim, use_cuda=False):
        super(StateSpaceModel, self).__init__()

        self.rnn_cell = nn.GRUCell(z_dim, rnn_dim)
        self.trans = LocScaleTransformer(rnn_dim, z_dim, trans_hidden_dim)
        # self.obs = Observation(z_dim, x_dim, obs_hidden_dim)
        self.obs = LocScaleTransformer(z_dim, x_dim, obs_hidden_dim)

        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.h_0 = nn.Parameter(torch.zeros(rnn_dim))

        if use_cuda:
            self.cuda()

    def model(self, mini_batch, annealing_factor=1.0):
        pyro.module("ssm", self)
        T = mini_batch.size(1)

        z_prev = self.z_0.expand(mini_batch.size(0), -1)
        h_prev = self.h_0.expand(mini_batch.size(0), -1)

        with pyro.plate("z_minibatch", len(mini_batch)):
            for t in range(T):

                rnn_out = self.rnn_cell(z_prev, h_prev)
                z_loc, z_scale = self.trans(rnn_out)

                with poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample(f"z_{t + 1}", dist.Normal(z_loc, z_scale).to_event(1))

                x_loc, x_scale = self.obs(z_t)

                pyro.sample(
                    f"x_{t + 1}", dist.Normal(x_loc, x_scale).to_event(1), obs=mini_batch[:, t, :],
                )

                z_prev = z_t
                h_prev = rnn_out


class Combiner(nn.Module):
    def __init__(self, z_dim, rnn_dim):
        super(Combiner, self).__init__()
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)

    def forward(self, z_t_1, h_rnn):
        h_combined = 0.5 * (torch.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)

        loc = self.lin_hidden_to_loc(h_combined)
        scale = F.softplus(self.lin_hidden_to_scale(h_combined))

        return loc, scale


class VariationalDistribution(nn.Module):
    def __init__(self, z_dim, x_dim, rnn_dim, num_iafs=0, iaf_dim=15, use_cuda=False):
        super(VariationalDistribution, self).__init__()
        self.rnn = nn.GRU(input_size=x_dim, hidden_size=rnn_dim, batch_first=True)
        self.comb = Combiner(z_dim, rnn_dim)

        self.iafs = [affine_autoregressive(z_dim, hidden_dims=[iaf_dim]) for _ in range(num_iafs)]
        self.iafs_modules = nn.ModuleList(self.iafs)

        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.h_0 = nn.Parameter(torch.zeros(rnn_dim))

        if use_cuda:
            self.cuda()

    def guide(self, mini_batch, annealing_factor=1.0):
        pyro.module("vardistr", self)
        T = mini_batch.size(1)

        z_prev = self.z_0.expand(mini_batch.size(0), -1)
        # the first dimension is num_layers * num_directions
        h_0_contig = self.h_0.expand(1, mini_batch.size(0), -1).contiguous()

        rnn_output, _ = self.rnn(mini_batch, h_0_contig)

        with pyro.plate("z_minibatch", len(mini_batch)):
            for t in range(T):
                z_loc, z_scale = self.comb(z_prev, rnn_output[:, t, :])
                z_dist = TransformedDistribution(dist.Normal(z_loc, z_scale), self.iafs)
                assert z_dist.event_shape == (self.z_0.size(0),)
                assert z_dist.batch_shape[-1:] == (len(mini_batch),)

                with pyro.poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample(f"z_{t + 1}", z_dist)

                z_prev = z_t


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    torch.set_default_dtype(torch.float64)

    window = 300
    batch_size = 32
    # split = 0.75
    epochs = 5000
    train_frac = 0.7
    n_val_batches = 500
    use_cuda = False

    # df = pd.read_parquet("drive/My Drive/train_data/eurusd_2019.parquet.gzip")
    df = pd.read_parquet("tick_data/eurusd_2019.parquet.gzip")

    delta_p = np.diff(np.round(df["buy"].values * 1e5))

    delta_t = np.diff(df.index.values)
    delta_t[delta_t < np.timedelta64(0)] += np.timedelta64(1, "h")
    delta_t = delta_t.astype(float)
    assert np.all(delta_t > 0)

    t_scaler = RobustScaler(with_centering=False, quantile_range=(0, 100 - 1e-2))
    delta_t = t_scaler.fit_transform(delta_t.reshape(-1, 1))

    p_scaler = StandardScaler()
    delta_p = delta_p.reshape(-1, 1)
    # delta_p is almost standardized from the start. For now
    # we leave it unaltered
    # delta_p = p_scaler.fit_transform(delta_p.reshape(-1, 1))

    X = torch.from_numpy(np.hstack((delta_p, delta_t)))

    sizes = (X.size(0) - window + 1, window) + X.size()[1:]
    strides = (X.stride(0),) + X.stride()

    X = torch.as_strided(X, sizes, strides)
    inds = torch.randperm(X.size(0) - X.size(0) % batch_size).view(-1, batch_size)

    split1 = int(train_frac * inds.size(0))
    split2 = split1 + n_val_batches
    train_inds, val_inds, test_inds = inds[:split1], inds[split1:split2], inds[split2:]

    N_mini_batches = train_inds.size(0)

    ssm = StateSpaceModel(z_dim=15, x_dim=2, rnn_dim=30, trans_hidden_dim=30, obs_hidden_dim=20, use_cuda=use_cuda)
    vardistr = VariationalDistribution(z_dim=15, x_dim=2, rnn_dim=30, num_iafs=1, use_cuda=use_cuda)

    adam_args = {"lr": 0.001}
    adam = ClippedAdam(adam_args)

    elbo = Trace_ELBO()
    svi = SVI(ssm.model, vardistr.guide, adam, loss=elbo)

    annealing_epochs = 1000
    min_annealing_factor = 0.1

    def process_minibatch(epoch, which_minibatch, shuffled_inds):
        if epoch < annealing_epochs:
            annealing_factor = min_annealing_factor + (1 - min_annealing_factor) * (
                which_minibatch + epoch * N_mini_batches + 1
            ) / (annealing_epochs * N_mini_batches)
        else:
            annealing_factor = 1.0

        minibatch_ind = shuffled_inds[which_minibatch]
        minibatch = X[minibatch_ind]

        if use_cuda:
            minibatch = minibatch.cuda()

        return svi.step(minibatch, annealing_factor)

    def do_evaluation():
        vardistr.rnn.eval()
        val_nll = svi.evaluate_loss(X[val_inds])
        vardistr.rnn.train()

        return val_nll

    # quit()

    times = [time.time()]
    for epoch in range(epochs):
        epoch_nll = 0.0
        train_inds = train_inds[torch.randperm(train_inds.size(0))]

        for which_mini_batch in range(N_mini_batches):
            epoch_nll += process_minibatch(epoch, which_mini_batch, train_inds)
            quit()

        times.append(time.time())
        epoch_time = times[-1] - times[-2]
        print("[training epoch %04d]  %.4f \t\t\t\t(dt = %.3f sec)" % (epoch, epoch_nll, epoch_time))

        # if val_test_frequency > 0 and epoch > 0 and epoch % val_test_frequency == 0:
        val_nll = do_evaluation()
        print("[validation epoch %04d]  %.4f" % (epoch, val_nll))
