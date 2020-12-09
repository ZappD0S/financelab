from typing import List, Optional

import torch
import torch.jit
import torch.distributions as dist
import torch.nn as nn
from pyro.distributions import TransformModule
from pyro.distributions.transforms.affine_autoregressive import affine_autoregressive
from waterstart_model import GatedTrasition, Emitter


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
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.trans = trans
        self.emitter = emitter
        self.iafs = iafs
        self.iafs_modules = nn.ModuleList(iafs)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.n_cur = n_cur
        self.device = device
        self.to(device)

    def forward(self, input: torch.Tensor):
        # input: (seq_len, batch_size, n_features)

        z_logprobs = input.new_empty(self.seq_len, self.n_samples, self.batch_size)
        x_logprobs = input.new_empty(self.seq_len, 3, self.n_cur, self.n_samples, self.batch_size)
        samples = input.new_empty(self.seq_len, 3, self.n_cur, self.n_samples, self.batch_size)
        fractions = input.new_empty(self.seq_len, 1, self.n_cur, self.n_samples, self.batch_size)

        last_z: Optional[torch.Tensor] = None

        for i in range(self.seq_len):
            z_loc, z_scale = self.trans(input[i].repeat(self.n_samples, 1), last_z)
            z_dist = dist.TransformedDistribution(dist.Normal(z_loc, z_scale), self.iafs)
            last_z = z_sample = z_dist.rsample()

            z_logprobs[i] = z_dist.log_prob(z_sample).view(self.n_samples, self.batch_size)

            probs, fractions[i] = (
                self.emitter(z_sample).t().view(4, self.n_cur, self.n_samples, self.batch_size).split([3, 1])
            )

            # NOTE: if probs is not between 0 and 1 this fails!
            x_dist = dist.Bernoulli(probs=probs)
            samples[i] = x_dist.sample()
            x_logprobs[i] = x_dist.log_prob(samples[i])

        return samples, fractions, x_logprobs, z_logprobs


class SSMEvaluator2(nn.Module):
    def __init__(
        self,
        trans: GatedTrasition,
        emitter: Emitter,
        iafs: List[TransformModule],
        seq_len: int,
        batch_size: int,
        n_samples: int,
        n_cur: int,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.trans = trans
        self.emitter = emitter
        self.iafs = iafs
        self.iafs_modules = nn.ModuleList(iafs)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.n_cur = n_cur
        self.device = device
        self.to(device)

    def forward(self, input: torch.Tensor):
        # input: (batch_size, n_features)

        z_logprobs = input.new_empty(self.seq_len, self.n_samples, self.batch_size)
        x_logprobs = input.new_empty(self.seq_len, self.n_samples, self.batch_size, self.n_cur, 3)
        samples = input.new_empty(self.seq_len, self.n_samples, self.batch_size, self.n_cur, 3)
        raw_fractions = input.new_empty(self.seq_len, self.n_samples, self.batch_size, self.n_cur, 1)

        last_z: Optional[torch.Tensor] = None

        for i in range(self.seq_len):
            z_loc, z_scale = self.trans(input[i].repeat(self.n_samples, 1), last_z)
            z_dist = dist.TransformedDistribution(dist.Normal(z_loc, z_scale), self.iafs)
            last_z = z_sample = z_dist.rsample()

            z_logprobs[i] = z_dist.log_prob(z_sample).view(self.n_samples, self.batch_size)

            logits, raw_fractions[i] = (
                self.emitter(z_sample).view(self.n_samples, self.batch_size, self.n_cur, 4).split([3, 1], dim=-1)
            )

            x_dist = dist.Bernoulli(logits=logits)
            samples[i] = x_dist.sample()
            x_logprobs[i] = x_dist.log_prob(samples[i])

        fractions = raw_fractions.squeeze(-1).sigmoid_()
        return samples, fractions, x_logprobs, z_logprobs


if __name__ == "__main__":
    n_cur = 10
    n_samples = 100
    leverage = 50
    z_dim = 128

    n_features = 50
    seq_len = 100
    batch_size = 2
    iafs = [affine_autoregressive(z_dim, [200]) for _ in range(2)]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    trans = torch.jit.script(GatedTrasition(n_features, z_dim, 20))
    emitter = torch.jit.script(Emitter(z_dim, n_cur, 20))

    ssm_eval = SSMEvaluator(trans, emitter, iafs, seq_len, batch_size, n_samples, n_cur, device)

    input = torch.randn(seq_len, batch_size, n_features, device=device)

    # prices = torch.randn(seq_len, 1, n_cur, batch_size, device=device).div_(100).cumsum(0).expand(-1, 2, -1, -1)
    # prices[:, 1] += 1.5e-4

    dummy_input = (input,)

    ssm_eval = torch.jit.trace(ssm_eval, dummy_input, check_trace=False)
    with torch.autograd.set_detect_anomaly(True):
        res = ssm_eval(*dummy_input)
        # res.sum().backward()
