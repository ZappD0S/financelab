from typing import List, Optional

import torch
import torch.jit
import torch.distributions as dist
import torch.nn as nn
from pyro.distributions import TransformModule
from pyro.distributions.transforms.affine_autoregressive import affine_autoregressive
from torch.distributions import Distribution
from torch.distributions.transformed_distribution import TransformedDistribution
from waterstart_model import GatedTrasition, Emitter


class LossEvaluator(nn.Module):
    def __init__(
        self,
        trans: GatedTrasition,
        emitter: Emitter,
        iafs: List[TransformModule],
        seq_len: int,
        batch_size: int,
        n_samples: int,
        n_cur: int,
        leverage: float,
        force_probs: bool = False,
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
        self.leverage = leverage
        self.force_probs = force_probs
        self.device = device
        self.to(device)

    def forward(self, input: torch.Tensor, prices: torch.Tensor):
        # input: (seq_len, batch_size, n_features)
        # prices: (seq_len, batch_size, n_cur, 2)

        # assert input.size(0) == prices.size(0) == self.seq_len
        # assert input.size(1) == prices.size(1) == self.batch_size
        # assert input.size(2) == self.trans.input_dim and prices.size(2) == self.n_cur

        last_z: Optional[torch.Tensor] = None

        all_inds = (
            torch.arange(self.n_samples)[:, None, None],
            torch.arange(self.batch_size)[:, None],
            torch.arange(self.n_cur),
        )

        pos_states = input.new_zeros(self.n_samples, self.batch_size, self.n_cur, dtype=torch.long)
        pos_types = input.new_empty(self.n_samples, self.batch_size, self.n_cur, dtype=torch.long)

        cum_z_logprob = input.new_zeros(self.n_samples, self.batch_size)

        cash_value = input.new_ones(self.n_samples, self.batch_size)
        initial_pos_values = input.new_empty(self.n_samples, self.batch_size, self.n_cur)
        total_pos_values = input.new_zeros(self.n_samples, self.batch_size, self.n_cur)

        pos_pl = input.new_zeros(self.n_samples, self.batch_size, self.n_cur)
        pos_pl_terms = input.new_empty(self.n_samples, self.batch_size, self.n_cur, 2, 2)

        # cash_logprobs keeps track of all the logprobs that affect the cash
        # amount, so both opened positions and closed position
        cash_logprob = input.new_zeros(self.n_samples, self.batch_size)
        pos_cum_exec_logprobs = input.new_zeros(self.n_samples, self.batch_size, self.n_cur)
        initial_pos_value_logprobs = input.new_empty(self.n_samples, self.batch_size, self.n_cur)
        bankrupt_mask = input.new_zeros(self.n_samples, self.batch_size, dtype=torch.bool)

        costs = input.new_zeros(self.n_samples, self.batch_size, self.n_cur)

        loss = input.new_zeros(self.n_samples, self.batch_size, device=self.device)

        # NOTE: if a mask is used more than once for indexing it's faster
        # to use nonzero() to compute the indices first and then use them for indexing.

        for i in range(self.seq_len):
            # NOTE: we do not need to take into account the bankrupt_mask because
            # where it is set the pos_states can't be 1
            open_pos_mask = pos_states == 1
            closed_pos_mask = pos_states == 0

            open_pos_inds = open_pos_mask.nonzero().unbind(1)

            # TODO: fix pos_pl_terms
            # compute the profit/loss terms of open positions, whether we are closing or not
            pos_pl_terms[(*open_pos_inds, 0, 1)] = 0

            open_pos_types = pos_types[open_pos_inds]
            open_pos_prices = prices[(i, *open_pos_inds[1:], open_pos_types)]
            coeffs = open_pos_prices.new([1.0, -1.0])[open_pos_types]
            pos_pl_terms[(*open_pos_inds, 1, 1)] = -coeffs / open_pos_prices

            # pos_pl = torch.zeros(self.n_samples, self.batch_size, self.n_cur, device=self.device)
            pos_pl.zero_()
            pos_pl[open_pos_inds] = initial_pos_values[open_pos_inds] * pos_pl_terms[open_pos_inds].sum(-1).prod(-1)

            # total_pos_values = torch.zeros(self.n_samples, self.batch_size, self.n_cur, device=self.device)
            total_pos_values.zero_()
            total_pos_values[open_pos_inds] = initial_pos_values[open_pos_inds] + pos_pl[open_pos_inds]
            portfolio_value = cash_value + total_pos_values.sum(-1)

            any_open_pos_inds = open_pos_mask.any(-1).nonzero().unbind(1)
            bankrupt_mask[any_open_pos_inds] = portfolio_value[any_open_pos_inds] <= 0.0

            # TODO: a possible optimization here is to compute this only where not bankrupt
            # z_loc, z_scale = self.trans(input[i].expand(n_samples, -1, -1), last_z)
            z_loc, z_scale = self.trans(input[i].repeat(self.n_samples, 1), last_z)
            z_dist = TransformedDistribution(dist.Normal(z_loc, z_scale), self.iafs)

            z_sample = z_dist.rsample()
            cum_z_logprob += z_dist.log_prob(z_sample).view(self.n_samples, self.batch_size)
            last_z = z_sample

            open_close_probs, short_probs, fractions = (
                self.emitter(z_sample).view(self.n_samples, self.batch_size, self.n_cur, 4).split([2, 1, 1], dim=-1)
            )
            short_probs = short_probs.squeeze(-1)
            fractions = fractions.squeeze(-1)

            # 0 -> open prob, 1 -> close_prob
            # pos_state == 0 -> the position is closed, so we consider the open prob
            # pos_state == 1 -> the position is open, so we consider the close prob
            exec_probs = open_close_probs[(*all_inds, pos_states.clone())]

            if self.force_probs and i == seq_len - 1:
                force_probs_mask = torch.ones_like(bankrupt_mask)
            else:
                force_probs_mask = bankrupt_mask

            force_probs_inds = force_probs_mask.nonzero().unbind(1)
            # set the prob to 0 where the position is closed ad to 1 where it's open
            exec_probs[force_probs_inds] = exec_probs.new([0.0, 1.0])[pos_states[force_probs_inds]]

            exec_dist: Distribution = dist.Bernoulli(probs=exec_probs)
            exec_samples = exec_dist.sample()

            pos_cum_exec_logprobs += exec_dist.log_prob(exec_samples)

            event_mask = exec_samples == 1
            event_inds = event_mask.nonzero().unbind(1)

            open_mask = closed_pos_mask & event_mask
            open_inds = open_mask.nonzero().unbind(1)

            close_mask = open_pos_mask & event_mask
            close_inds = close_mask.nonzero().unbind(1)

            # pos_states[event_inds] = pos_states[event_inds].add_(1).remainder_(2)
            pos_states[event_inds] = (pos_states[event_inds] + 1) % 2

            pos_type_dist: Distribution = dist.Bernoulli(probs=short_probs[open_inds])
            # 0 -> long, 1 -> short
            opened_pos_types = pos_type_dist.sample()
            pos_type_logprobs = pos_type_dist.log_prob(opened_pos_types)
            opened_pos_types = opened_pos_types.type(torch.long)
            pos_types[open_inds] = opened_pos_types

            pos_cum_exec_logprobs[open_inds] += pos_type_logprobs

            # compute the profit/loss terms due to opening
            opened_pos_prices = prices[(i, *open_inds[1:], opened_pos_types)]
            pos_pl_terms[(*open_inds, 0, 0)] = self.leverage * opened_pos_prices

            coeffs = 1 / self.leverage + opened_pos_prices.new([1.0, -1.0])[opened_pos_types]
            pos_pl_terms[(*open_inds, 1, 0)] = coeffs / opened_pos_prices

            # costs = torch.zeros(self.n_samples, self.batch_size, self.n_cur, device=self.device)
            costs.zero_()
            costs[close_inds] = pos_pl[close_inds]

            for j in range(self.n_cur):
                mask = open_inds[2] == j
                cur_open_inds = tuple(open_inds[i][mask] for i in range(2))

                new_pos_initial_value = fractions[(*cur_open_inds, j)] * cash_value[cur_open_inds]
                initial_pos_values[(*cur_open_inds, j)] = new_pos_initial_value
                cash_value[cur_open_inds] -= new_pos_initial_value

                cash_logprob[cur_open_inds] += pos_cum_exec_logprobs[(*cur_open_inds, j)]
                initial_pos_value_logprobs[(*cur_open_inds, j)] = cash_logprob[cur_open_inds]
                pos_cum_exec_logprobs[(*cur_open_inds, j)] = 0.0

                mask = close_inds[2] == j
                cur_close_inds = tuple(close_inds[i][mask] for i in range(2))

                cost = costs[(*cur_close_inds, j)]
                # TODO: what if we used as baseline the pos_pl of the most
                # recently closed pos? in that case we'd have to keep track
                # of a tensor called last_costs
                baseline = costs[:, cur_close_inds[1], j].mean(0)

                cost_logprob = (
                    cum_z_logprob[cur_close_inds]
                    + initial_pos_value_logprobs[(*cur_close_inds, j)]
                    + pos_cum_exec_logprobs[(*cur_close_inds, j)]
                )

                loss[cur_close_inds] += cost_logprob * torch.detach(cost - baseline) + cost
                cash_value[cur_close_inds] += initial_pos_values[(*cur_close_inds, j)] + cost
                cash_logprob[cur_close_inds] += pos_cum_exec_logprobs[(*cur_close_inds, j)]
                pos_cum_exec_logprobs[(*cur_close_inds, j)] = 0.0

        return loss


if __name__ == "__main__":
    n_cur = 10
    n_samples = 100
    leverage = 50
    z_dim = 128

    n_features = 50
    seq_len = 100
    batch_size = 2
    iafs = [affine_autoregressive(z_dim, [200]) for _ in range(2)]

    trans = torch.jit.script(GatedTrasition(n_features, z_dim, 20))
    emitter = torch.jit.script(Emitter(z_dim, n_cur, 20))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    loss_eval = LossEvaluator(
        trans, emitter, iafs, seq_len, batch_size, n_samples, n_cur, leverage, force_probs=False, device=device
    )

    input = torch.randn(seq_len, batch_size, n_features, device=device)

    prices = torch.randn(seq_len, batch_size, n_cur, 1, device=device).div_(100).cumsum(0).expand(-1, -1, -1, 2)
    prices[..., 1] += 1.5e-4

    dummy_input = (input, prices)

    # loss_eval = torch.jit.trace(loss_eval, dummy_input, check_trace=False)
    with torch.autograd.set_detect_anomaly(True):
        res = loss_eval(*dummy_input)
        res.sum().backward()
