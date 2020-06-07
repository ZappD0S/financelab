import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import affine_autoregressive
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam


class Observation(nn.Module):
    def __init__(self, z_dim, x_dim, hidden_dim=20):
        super(Observation, self).__init__()
        self.lin_loc_z_to_hidden = nn.Linear(z_dim, hidden_dim)
        self.lin_loc_hidden_to_x = nn.Linear(hidden_dim, x_dim)

        self.lin_scale_z_to_hidden = nn.Linear(z_dim, hidden_dim)
        self.lin_scale_hidden_to_x = nn.Linear(hidden_dim, x_dim)

    def forward(self, z_t):
        _loc = F.relu(self.lin_loc_z_to_hidden(z_t))
        loc = F.relu(self.lin_loc_hidden_to_x(_loc))

        _scale = F.relu(self.lin_scale_z_to_hidden(z_t))
        scale = F.relu(self.lin_scale_hidden_to_x(_scale))

        return loc, scale


class GatedTransition(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(GatedTransition, self).__init__()

        self.lin_gate_z_to_hidden = nn.Linear(z_dim, hidden_dim)
        self.lin_gate_hidden_to_z = nn.Linear(hidden_dim, z_dim)

        self.lin_proposed_loc_z_to_hidden = nn.Linear(z_dim, hidden_dim)
        self.lin_proposed_loc_hidden_to_z = nn.Linear(hidden_dim, z_dim)

        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)

        self.lin_z_to_scale = nn.Linear(z_dim, z_dim)

    def forward(self, z_t_1):
        _gate = F.relu(self.lin_gate_z_to_hidden(z_t_1))
        gate = F.sigmoid(self.lin_gate_hidden_to_z(_gate))

        _proposed_mean = F.relu(self.lin_proposed_loc_z_to_hidden(z_t_1))
        proposed_mean = self.lin_proposed_loc_hidden_to_z(_proposed_mean)

        loc = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean
        scale = self.softplus(self.lin_z_to_scale(self.relu(proposed_mean)))

        return loc, scale


class StateSpaceModel(nn.Module):
    def __init__(self, z_dim, x_dim, trans_hidden_dim, obs_hidden_dim, use_cuda=False):
        super(StateSpaceModel, self).__init__()

        self.trans = GatedTransition(z_dim, trans_hidden_dim)
        self.obs = Observation(z_dim, x_dim, obs_hidden_dim)
        self.rnn_cell = nn.GRUCell(z_dim, z_dim)

        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.h_0 = nn.Parameter(torch.zeros(z_dim))

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
                pyro.sample(f"x_{t + 1}", dist.Normal(x_loc, x_scale).to_event(1), obs=mini_batch[:, t, :])

                z_prev = z_t
                h_prev = rnn_out


class Combiner(nn.Module):
    def __init__(self, z_dim, rnn_dim):
        super(Combiner, self).__init__()
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)

    def forward(self, z_t_1, h_rnn):
        h_combined = 0.5 * (F.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)

        loc = self.lin_hidden_to_loc(h_combined)
        scale = F.softplus(self.lin_hidden_to_scale(h_combined))

        return loc, scale


class VariationalDistribution(nn.Module):
    def __init__(self, input_dim, z_dim, rnn_dim, num_iafs=0, iaf_dim=50, use_cuda=False):
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=rnn_dim, nonlinearity='relu',
                          batch_first=True, bidirectional=False)
        self.comb = Combiner(z_dim, rnn_dim)

        self.iafs = [affine_autoregressive(z_dim, hidden_dims=[iaf_dim]) for _ in range(num_iafs)]
        self.iafs_modules = nn.ModuleList(self.iafs)

        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.h_0 = nn.Parameter(torch.zeros(rnn_dim))

    def guide(self, mini_batch, annealing_factor=1.0):
        T = mini_batch.size(1)
        pyro.module("vardistr", self)

        # la prima dimensione è num_layers * num_directions
        h_0_contig = self.h_0.expand(1, mini_batch.size(0), -1).contiguous()
        rnn_output, _ = self.rnn(mini_batch, h_0_contig)
        z_prev = self.z_0.expand(mini_batch.size(0), -1)

        with pyro.plate("z_minibatch", len(mini_batch)):
            for t in range(T):
                z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t, :])
                z_dist = TransformedDistribution(dist.Normal(z_loc, z_scale), self.iafs)
                assert z_dist.event_shape == (self.z_q_0.size(0),)
                assert z_dist.batch_shape[-1:] == (len(mini_batch),)

                with pyro.poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample(f"z_{t + 1}", z_dist)

                z_prev = z_t
