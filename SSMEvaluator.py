from typing import List, Optional

import torch
import torch.distributions as dist
import torch.jit
import torch.nn as nn
from pyro.distributions import TransformModule
from pyro.distributions.transforms.affine_autoregressive import affine_autoregressive

from waterstart_model import Emitter, GatedTrasition


# class SSMEvaluator(nn.Module):
#     def __init__(
#         self,
#         trans: GatedTrasition,
#         emitter: Emitter,
#         iafs: List[TransformModule],
#         seq_len: int,
#         batch_size: int,
#         n_samples: int,
#         n_cur: int,
#         device: Optional[torch.device] = None,
#     ):
#         super().__init__()
#         self.trans = trans
#         self.emitter = emitter
#         self.iafs = iafs
#         self.iafs_modules = nn.ModuleList(iafs)
#         self.seq_len = seq_len
#         self.batch_size = batch_size
#         self.n_samples = n_samples
#         self.n_cur = n_cur
#         self.device = device
#         self.to(device)

#     def forward(self, input: torch.Tensor):
#         # input: (seq_len, batch_size, n_features)

#         z_logprobs = input.new_empty(self.seq_len, self.n_samples, self.batch_size)
#         x_logprobs = input.new_empty(self.seq_len, 3, self.n_cur, self.n_samples, self.batch_size)
#         samples = input.new_empty(self.seq_len, 3, self.n_cur, self.n_samples, self.batch_size)
#         fractions = input.new_empty(self.seq_len, 1, self.n_cur, self.n_samples, self.batch_size)

#         last_z: Optional[torch.Tensor] = None

#         for i in range(self.seq_len):
#             z_loc, z_scale = self.trans(input[i].repeat(self.n_samples, 1), last_z)
#             z_dist = dist.TransformedDistribution(dist.Normal(z_loc, z_scale), self.iafs)
#             last_z = z_sample = z_dist.rsample()

#             z_logprobs[i] = z_dist.log_prob(z_sample).view(self.n_samples, self.batch_size)

#             probs, fractions[i] = (
#                 self.emitter(z_sample).t().view(4, self.n_cur, self.n_samples, self.batch_size).split([3, 1])
#             )

#             # NOTE: if probs is not between 0 and 1 this fails!
#             x_dist = dist.Bernoulli(probs=probs)
#             samples[i] = x_dist.sample()
#             x_logprobs[i] = x_dist.log_prob(samples[i])

#         return samples, fractions, x_logprobs, z_logprobs


class SSMEvaluator(nn.Module):
    def __init__(
        self,
        trans: GatedTrasition,
        emitter: Emitter,
        iafs: List[TransformModule],
        seq_len: int,
        batch_size: int,
        n_samples: int,
        n_cur: int,
    ):
        super().__init__()
        self.trans = trans
        self.emitter = emitter
        self.iafs = iafs
        self._iafs_modules = nn.ModuleList(iafs)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.n_cur = n_cur

    def forward(self, input: torch.Tensor, z0: torch.Tensor):
        # input: (seq_len, batch_size, n_features)
        # z0: (batch_size, n_samples, z_dim)

        z_logprobs = input.new_empty(self.seq_len, self.batch_size, self.n_samples)
        x_logprobs = input.new_empty(self.seq_len, self.batch_size, self.n_samples, self.n_cur, 3)

        z_samples = input.new_empty(self.seq_len, self.batch_size, self.n_samples, self.trans.z_dim)
        x_samples = input.new_empty(self.seq_len, self.batch_size, self.n_samples, self.n_cur, 3)

        raw_fractions = input.new_empty(self.seq_len, self.batch_size, self.n_samples, self.n_cur, 1)

        last_z_sample = z0.view(self.batch_size * self.n_samples, self.trans.z_dim)

        for i in range(self.seq_len):
            z_loc, z_scale = self.trans(input[i].repeat(self.n_samples, 1), last_z_sample)
            z_dist = dist.TransformedDistribution(dist.Normal(z_loc, z_scale), self.iafs)
            last_z_sample = z_dist.rsample()

            z_samples[i] = last_z_sample.view(self.batch_size, self.n_samples, -1).detach()

            z_logprobs[i] = z_dist.log_prob(last_z_sample).view(self.batch_size, self.n_samples)

            logits, raw_fractions[i] = (
                self.emitter(last_z_sample).view(self.batch_size, self.n_samples, self.n_cur, 4).split([3, 1], dim=-1)
            )

            x_dist = dist.Bernoulli(logits=logits)
            last_x_sample = x_dist.sample()
            x_samples[i] = last_x_sample
            x_logprobs[i] = x_dist.log_prob(last_x_sample)

        fractions = raw_fractions.squeeze(-1).sigmoid_()
        return x_samples, z_samples, x_logprobs, z_logprobs, fractions


if __name__ == "__main__":
    n_cur = 10
    n_samples = 100
    leverage = 50
    z_dim = 128

    n_features = 256
    # n_features = 50
    seq_len = 109
    batch_size = 2
    iafs = [affine_autoregressive(z_dim, [200]) for _ in range(2)]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    trans = torch.jit.script(GatedTrasition(n_features, z_dim, 20))
    emitter = torch.jit.script(Emitter(z_dim, n_cur, 20))

    ssm_eval = SSMEvaluator(trans, emitter, iafs, seq_len, batch_size, n_samples, n_cur).to(device)

    input = torch.randn(seq_len, batch_size, n_features, device=device)
    z0 = torch.randn(batch_size, n_samples, z_dim, device=device)

    # prices = torch.randn(seq_len, 1, n_cur, batch_size, device=device).div_(100).cumsum(0).expand(-1, 2, -1, -1)
    # prices[:, 1] += 1.5e-4

    dummy_input = (input, z0)

    ssm_eval = torch.jit.trace(ssm_eval, dummy_input, check_trace=False)
    with torch.autograd.set_detect_anomaly(True):
        res = ssm_eval(*dummy_input)
        # res.sum().backward()
