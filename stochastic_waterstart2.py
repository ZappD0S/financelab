from typing import List, Optional

import torch
from torch.distributions import bernoulli
import torch.jit
import torch.distributions as dist
import torch.nn as nn
from pyro.distributions import TransformModule
from pyro.distributions.transforms.affine_autoregressive import affine_autoregressive
from torch.distributions import Distribution
from torch.distributions.transformed_distribution import TransformedDistribution


class LossEvaluator(nn.Module):
    def __init__(
        self,
        seq_len: int,
        batch_size: int,
        n_samples: int,
        n_cur: int,
        leverage: float,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.n_cur = n_cur
        self.leverage = leverage
        self.device = device
        self.to(device)

    def forward(self, probs: torch.Tensor, prices: torch.Tensor, force_probs: bool = False):
        # probs: (seq_len, batch_size, n_cur, 4)
        # prices: (seq_len, batch_size, n_cur, 2)

        last_z: Optional[torch.Tensor] = None

        all_inds = (
            torch.arange(self.n_samples)[:, None, None],
            torch.arange(self.batch_size)[:, None],
            torch.arange(self.n_cur),
        )

        pos_states = torch.zeros(self.n_samples, self.batch_size, self.n_cur, dtype=torch.long, device=self.device)
        pos_types = torch.empty(self.n_samples, self.batch_size, self.n_cur, dtype=torch.long, device=self.device)

        cash_value = torch.ones(self.n_samples, self.batch_size, device=self.device)
        initial_pos_values = torch.empty(self.n_samples, self.batch_size, self.n_cur, device=self.device)
        total_pos_values = torch.zeros(self.n_samples, self.batch_size, self.n_cur, device=self.device)

        pos_pl = torch.zeros(self.n_samples, self.batch_size, self.n_cur, device=self.device)
        pos_pl_terms = torch.empty(self.n_samples, self.batch_size, self.n_cur, 2, 2, device=self.device)

        # cash_logprobs keeps track of all the logprobs that affect the cash
        # amount, so both opened positions and closed position
        cash_logprob = torch.zeros(self.n_samples, self.batch_size, device=self.device)
        pos_cum_exec_logprobs = torch.zeros(self.n_samples, self.batch_size, self.n_cur, device=self.device)
        initial_pos_value_logprobs = torch.empty(self.n_samples, self.batch_size, self.n_cur, device=self.device)
        bankrupt_mask = torch.zeros(self.n_samples, self.batch_size, dtype=torch.bool, device=self.device)

        costs = torch.zeros(self.n_samples, self.batch_size, self.n_cur, device=self.device)

        loss = torch.zeros(self.n_samples, self.batch_size, device=self.device)

        # NOTE: if a mask is used more than once for indexing it's faster
        # to use nonzero() to compute the indices first and then use them for indexing.

        for i in range(self.seq_len):
            # NOTE: we do not need to take into account the bankrupt_mask because
            # where it is set the pos_states can't be 1
            open_pos_mask = pos_states == 1
            closed_pos_mask = pos_states == 0

            open_pos_inds = open_pos_mask.nonzero().unbind(1)

            # compute the profit/loss terms of open positions, whether we are closing or not
            pos_pl_terms[(*open_pos_inds, 0, 1)] = 0

            open_pos_types = pos_types[open_pos_inds]
            open_pos_prices = prices[(i, *open_pos_inds[1:], open_pos_types)]
            coeffs = torch.tensor([1.0, -1.0], device=self.device)[open_pos_types]
            pos_pl_terms[(*open_pos_inds, 1, 1)] = -coeffs / open_pos_prices

            # pos_pl = torch.zeros(self.n_samples, self.batch_size, self.n_cur, device=self.device)
            pos_pl.zero_()
            pos_pl[open_pos_inds] = initial_pos_values[open_pos_inds] * pos_pl_terms[open_pos_inds].sum(-1).prod(-1)

            # total_pos_values = torch.zeros(self.n_samples, self.batch_size, self.n_cur, device=self.device)
            total_pos_values.zero_()
            total_pos_values[open_pos_inds] = initial_pos_values[open_pos_inds] + pos_pl[open_pos_inds]
            portfolio_value = cash_value.clone() + total_pos_values.sum(-1)

            any_open_pos_inds = open_pos_mask.any(-1).nonzero().unbind(1)
            bankrupt_mask[any_open_pos_inds] = portfolio_value[any_open_pos_inds] <= 0

            exec_probs, short_probs, fractions = (
                probs[i, None].expand(self.n_samples, -1, -1, -1).split([2, 1, 1], dim=-1)
            )
            short_probs = short_probs.squeeze(-1)
            fractions = fractions.squeeze(-1)

            # 0 -> open prob, 1 -> close_prob
            # pos_state == 0 -> the position is closed, so we consider the open prob
            # pos_state == 1 -> the position is open, so we consider the close prob

            # TODO: can we improve this part a bit?
            if force_probs and i == seq_len - 1:
                exec_probs = torch.tensor([0.0, 1.0], device=self.device)[pos_states.clone()]
            else:
                exec_probs = exec_probs.clone()
                inds = torch.nonzero(open_pos_mask & bankrupt_mask.unsqueeze(-1)).unbind(1)
                exec_probs[(*inds, 1)] = 1.0

                inds = torch.nonzero(closed_pos_mask & bankrupt_mask.unsqueeze(-1)).unbind(1)
                exec_probs[(*inds, 0)] = 0.0

                exec_probs = exec_probs[(*all_inds, pos_states.clone())]

            exec_dist: Distribution = dist.Bernoulli(probs=exec_probs)
            exec_samples = exec_dist.sample()

            pos_cum_exec_logprobs += exec_dist.log_prob(exec_samples)

            event_mask = exec_samples == 1
            event_inds = event_mask.nonzero().unbind(1)

            open_mask = closed_pos_mask & event_mask
            open_inds = open_mask.nonzero().unbind(1)

            close_mask = open_pos_mask & event_mask
            close_inds = close_mask.nonzero().unbind(1)

            pos_states[event_inds] = pos_states[event_inds].add_(1).remainder_(2)

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

            coeffs = 1 / self.leverage + torch.tensor([1.0, -1.0], device=self.device)[opened_pos_types]
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
                    # cum_z_logprob[cur_close_inds]
                    initial_pos_value_logprobs[(*cur_close_inds, j)]
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

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    loss_eval = LossEvaluator(seq_len, batch_size, n_samples, n_cur, leverage, device)

    open_close_probs = torch.bernoulli(torch.full((seq_len, batch_size, n_cur, 2), 1e-2, device=device))
    pos_type_probs = torch.full((seq_len, batch_size, n_cur, 1), 0.5, device=device)
    fractions = torch.rand(seq_len, batch_size, n_cur, 1, device=device)

    net_output = torch.cat((open_close_probs, pos_type_probs, fractions), dim=-1)

    prices = torch.randn(seq_len, batch_size, n_cur, 1, device=device).div_(100).cumsum(0).expand(-1, -1, -1, 2)
    prices[..., 1] += 1.5e-4

    dummy_input = (net_output, prices)

    # loss_eval = torch.jit.trace(loss_eval, dummy_input, check_trace=False)
    with torch.autograd.set_detect_anomaly(True):
        res = loss_eval(*dummy_input)
        # res.sum().backward()
