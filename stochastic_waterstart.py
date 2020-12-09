from typing import List, Optional

import torch
import torch.jit
import torch.distributions as dist
import torch.nn as nn
from pyro.distributions import TransformModule
from pyro.distributions.transforms.affine_autoregressive import affine_autoregressive
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
        # prices: (seq_len, 2, n_cur, batch_size)

        prices = prices.unsqueeze(3)

        last_z: Optional[torch.Tensor] = None

        pos_states = input.new_zeros(self.n_cur, self.n_samples, self.batch_size, dtype=torch.long)
        pos_types = input.new_zeros(self.n_cur, self.n_samples, self.batch_size, dtype=torch.long)

        # this are only necessary to avoid computing these masks twice in the loop
        long_pos_type_mask = input.new_zeros(self.n_cur, self.n_samples, self.batch_size, dtype=torch.bool)
        short_pos_type_mask = input.new_zeros(self.n_cur, self.n_samples, self.batch_size, dtype=torch.bool)

        cum_z_logprob = input.new_zeros(self.n_samples, self.batch_size)

        cash_value = input.new_ones(self.n_samples, self.batch_size)
        initial_pos_values = input.new_empty(self.n_cur, self.n_samples, self.batch_size)

        pos_pl_terms = input.new_empty(2, 2, self.n_cur, self.n_samples, self.batch_size)

        # cash_logprobs keeps track of all the logprobs that affect the cash
        # amount, so both opened positions and closed position
        cash_logprob = input.new_zeros(self.n_samples, self.batch_size)
        pos_cum_exec_logprobs = input.new_zeros(self.n_cur, self.n_samples, self.batch_size)
        initial_pos_value_logprobs = input.new_empty(self.n_cur, self.n_samples, self.batch_size)
        bankrupt_mask = input.new_zeros(self.n_samples, self.batch_size, dtype=torch.bool)

        costs = input.new_zeros(self.n_cur, self.n_samples, self.batch_size)
        loss = input.new_zeros(self.n_samples, self.batch_size)

        # TODO: we obtained mixed results regarding sampling on GPU. We have to investigate better.

        # TODO: in replay buffer we need to save only the pl_terms due to opening so we might
        # have to split pos_pl_terms in pos_open_pl_terms and pos_close_pl_terms and save only the former

        # TODO: we might try to replace the += and -= with _ = _ + _ and _ = _ - _ and remove some clone(),
        # to se if it's faster
        for i in range(self.seq_len):
            # NOTE: we do not need to take into account the bankrupt_mask because
            # where it is set the pos_states can't be 1
            open_pos_mask = pos_states == 1
            closed_pos_mask = pos_states == 0

            # compute the profit/loss terms of open positions, whether we are closing or not
            pos_pl_terms[1, 0] = pos_pl_terms.new_zeros([]).where(open_pos_mask, pos_pl_terms[1, 0])

            # in this term buy and sell prices are inverted!
            new_terms = 1 / torch.where(
                long_pos_type_mask,
                prices[i, 1],
                torch.where(short_pos_type_mask, prices[i, 0], prices.new_tensor(float("nan"))),
            )
            # assert not new_terms.isnan().any()
            pos_pl_terms[1, 1] = new_terms.where(open_pos_mask, pos_pl_terms[1, 1])

            pos_pl = torch.where(
                open_pos_mask, initial_pos_values.clone() * pos_pl_terms.sum(0).prod(0), pos_pl_terms.new_zeros([])
            )

            total_pos_values = torch.where(open_pos_mask, initial_pos_values.clone() + pos_pl, pos_pl.new_zeros([]))
            portfolio_value = cash_value.clone() + total_pos_values.sum(0)

            bankrupt_mask = torch.where(open_pos_mask.any(0), portfolio_value <= 0.0, bankrupt_mask)

            # z_loc, z_scale = self.trans(input[i].expand(n_samples, -1, -1), last_z)
            z_loc, z_scale = self.trans(input[i].repeat(self.n_samples, 1), last_z)
            z_dist = TransformedDistribution(dist.Normal(z_loc, z_scale), self.iafs)

            z_sample = z_dist.rsample()
            cum_z_logprob += z_dist.log_prob(z_sample).view(self.n_samples, self.batch_size)
            last_z = z_sample

            open_probs, close_probs, short_probs, fractions = (
                self.emitter(z_sample).t().view(4, self.n_cur, self.n_samples, self.batch_size).unbind()
            )

            # 0 -> open prob, 1 -> close_prob
            # pos_state == 0 -> the position is closed, so we consider the open prob
            # pos_state == 1 -> the position is open, so we consider the close prob

            if self.force_probs and i == seq_len - 1:
                force_probs_mask = torch.ones_like(bankrupt_mask)
            else:
                force_probs_mask = bankrupt_mask

            # set the prob to 0 where the position is closed ad to 1 where it's open
            open_probs = torch.where(force_probs_mask, open_probs.new_zeros([]), open_probs)
            close_probs = torch.where(force_probs_mask, close_probs.new_ones([]), close_probs)

            exec_probs = torch.where(
                closed_pos_mask,
                open_probs,
                torch.where(open_pos_mask, close_probs, close_probs.new_tensor(float("nan"))),
            )
            # assert not exec_probs.isnan().any()

            exec_dist = dist.Bernoulli(probs=exec_probs)
            exec_samples = exec_dist.sample()
            pos_cum_exec_logprobs += exec_dist.log_prob(exec_samples)

            event_mask = exec_samples == 1
            open_mask = closed_pos_mask & event_mask
            close_mask = open_pos_mask & event_mask

            # NOTE: when using the % operator torch.jit.trace causes trouble. Also, this
            # way the operation becomes inplace
            pos_states = torch.where(event_mask, (pos_states + 1).remainder_(2), pos_states)

            # NOTE: 0 -> long, 1 -> short
            pos_type_dist = dist.Bernoulli(probs=short_probs)
            opened_pos_types = pos_type_dist.sample()
            pos_cum_exec_logprobs += pos_type_dist.log_prob(opened_pos_types).where(
                open_mask, pos_cum_exec_logprobs.new_zeros([])
            )

            pos_types = opened_pos_types.type(torch.long).where(open_mask, pos_types)
            long_pos_type_mask = pos_types == 0
            short_pos_type_mask = pos_types == 1

            # compute the profit/loss terms due to opening
            new_terms = self.leverage * torch.where(
                long_pos_type_mask,
                prices[i, 0],
                torch.where(short_pos_type_mask, -prices[i, 1], pos_pl_terms.new_tensor(float("nan"))),
            )
            # assert not new_terms.isnan().any()
            pos_pl_terms[0, 0] = new_terms.where(open_mask, pos_pl_terms[0, 0])

            new_terms = 1 / torch.where(
                long_pos_type_mask,
                prices[i, 0],
                torch.where(short_pos_type_mask, prices[i, 1], prices.new_tensor(float("nan"))),
            )
            # assert not new_terms.isnan().any()
            pos_pl_terms[0, 1] = new_terms.where(open_pos_mask, pos_pl_terms[0, 1])

            costs = pos_pl.where(close_mask, pos_pl.new_zeros([]))

            for j in range(self.n_cur):
                cur_open_mask = open_mask[j]

                new_pos_initial_value = fractions[j] * cash_value.clone()
                initial_pos_values[j] = new_pos_initial_value.where(cur_open_mask, initial_pos_values[j].clone())
                cash_value -= new_pos_initial_value.where(cur_open_mask, cash_value.new_zeros([]))

                cash_logprob += pos_cum_exec_logprobs[j].clone().where(cur_open_mask, cash_logprob.new_zeros([]))
                initial_pos_value_logprobs[j] = cash_logprob.clone().where(cur_open_mask, initial_pos_value_logprobs[j])

                # TODO: we can put the reset of the pos_cum_exec_logprobs after the loop, right?
                # we should reset using the event mask
                pos_cum_exec_logprobs[j] = pos_cum_exec_logprobs.new_zeros([]).where(
                    cur_open_mask, pos_cum_exec_logprobs[j].clone()
                )

                cur_close_mask = close_mask[j]

                cost = costs[j]
                # TODO: what if we used as baseline the pos_pl of the most
                # recently closed pos? in that case we'd have to keep track
                # of a tensor called last_costs
                baseline = costs[j].mean(0, keepdim=True)
                cost_logprob = (
                    cum_z_logprob.clone() + initial_pos_value_logprobs[j].clone() + pos_cum_exec_logprobs[j].clone()
                )

                loss += torch.where(
                    cur_close_mask, cost_logprob * torch.detach(cost - baseline) + cost, loss.new_zeros([])
                )

                cash_value += torch.where(
                    cur_close_mask, initial_pos_values[j].clone() + cost, cash_value.new_zeros([])
                )
                cash_logprob += pos_cum_exec_logprobs[j].clone().where(cur_close_mask, cash_logprob.new_zeros([]))
                pos_cum_exec_logprobs[j] = pos_cum_exec_logprobs.new_zeros([]).where(
                    cur_close_mask, pos_cum_exec_logprobs[j].clone()
                )

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

    prices = torch.randn(seq_len, 1, n_cur, batch_size, device=device).div_(100).cumsum(0).expand(-1, 2, -1, -1)
    prices[:, 1] += 1.5e-4

    dummy_input = (input, prices)

    loss_eval = torch.jit.trace(loss_eval, dummy_input, check_trace=False)
    print("jit done")

    with torch.autograd.set_detect_anomaly(True):
        res = loss_eval(*dummy_input)
        res.sum().backward()
