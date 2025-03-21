import torch
import torch.jit
import torch.distributions as dist
import torch.nn as nn
from waterstart_model import CNN, GatedTransition, Emitter, NeuralBaseline
from pyro.distributions import TransformModule
from typing import List, Tuple, Optional


class LossEvaluator(nn.Module):
    def __init__(
        self,
        cnn: CNN,
        trans: GatedTransition,
        iafs: List[TransformModule],
        emitter: Emitter,
        nn_baseline: NeuralBaseline,
        batch_size: int,
        seq_len: int,
        n_samples: int,
        n_cur: int,
        max_trades: int,
        z_dim: int,
        leverage: float = 1.0,
    ):
        super().__init__()
        self.cnn = cnn
        self.trans = trans
        self.iafs = iafs
        self._iafs_modules = nn.ModuleList(iafs)
        self.emitter = emitter
        self.nn_baseline = nn_baseline
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_samples = n_samples
        self.n_cur = n_cur
        self.max_trades = max_trades
        self.z_dim = z_dim
        self.leverage = leverage

    @staticmethod
    def compute_used_and_unused_margin(
        total_margin: torch.Tensor, open_pos_margins: torch.Tensor,
    ):
        # total_margin: (..., n_samples, seq_len, batch_size)
        # open_pos_margins: (..., n_cur, n_samples, seq_len, batch_size)
        # account_cur_rates: (..., n_cur, 1, seq_len, batch_size)

        total_used_margin = open_pos_margins.abs().sum(-4)
        total_unused_margin = total_margin - total_used_margin

        total_unused_margin = total_unused_margin.maximum(total_unused_margin.new_zeros([]))
        return total_used_margin, total_unused_margin

    @staticmethod
    def compute_close_rates(open_pos_sizes, rates):
        # open_pos_sizes: (..., n_cur, n_samples, seq_len, batch_size)
        # rates: (..., 2, n_cur, 1, seq_len, batch_size)

        return torch.where(
            open_pos_sizes > 0,
            rates.select(-5, 0),
            torch.where(open_pos_sizes < 0, rates.select(-5, 1), rates.new_ones([])),
        )

    @staticmethod
    def compute_open_rates(fractions: torch.Tensor, rates: torch.Tensor):
        # fractions: (..., n_cur, n_samples, seq_len, batch_size)
        # rates: (..., 2, n_cur, 1, seq_len, batch_size)

        return torch.where(
            fractions > 0, rates.select(-5, 1), torch.where(fractions < 0, rates.select(-5, 0), rates.new_zeros([]))
        )

    @staticmethod
    def compute_pl(exec_sizes, open_pos_rates, close_rates):
        # return pos_sizes.abs() / account_cur_rates * (1 - open_pos_rates / close_rates)
        return exec_sizes * (1 - open_pos_rates / close_rates)

    @staticmethod
    def compute_closeout_mask(total_margin, total_used_margin, open_pos_pl):
        # total_margin: (..., n_samples, seq_len, batch_size)
        # total_used_margin: (..., n_samples, seq_len, batch_size)
        # open_pos_pl: (..., n_cur, n_samples, seq_len, batch_size)

        account_value = total_margin + open_pos_pl.sum(-4)
        return account_value < 0.5 * total_used_margin

    @staticmethod
    def compute_open_pos_data(total_margin, open_pos_margins, total_unused_margin, open_pos_rates, close_rates):
        # total_margin: (..., n_samples, seq_len, batch_size)
        # open_pos_margins: (..., n_cur, n_samples, seq_len, batch_size)
        # total_unused_margin: (..., n_samples, seq_len, batch_size)
        # open_pos_rates: (..., n_cur, n_samples, seq_len, batch_size)
        # close_rates: (..., n_cur, n_samples, seq_len, batch_size)

        rel_margins = torch.cat([open_pos_margins, total_unused_margin.unsqueeze(-4)]) / total_margin.unsqueeze(-4)
        rel_open_rates = open_pos_rates / close_rates

        return torch.cat([rel_margins, rel_open_rates], dim=-4).movedim(-4, -1)

    @staticmethod
    def apply_mask_map(x: torch.Tensor, src_mask: torch.Tensor, dest_mask: torch.Tensor):
        out = torch.zeros_like(dest_mask, dtype=x.dtype)
        out[dest_mask] = x[src_mask]
        return out

    def compute_trans_input(self, market_data: torch.Tensor, open_pos_data: torch.Tensor):
        # market_data: (..., seq_len, batch_size, n_features, n_cur, window_size)
        # open_pos_data:  (..., n_samples, seq_len, batch_size, 2 * n_cur + 1)

        open_pos_data = open_pos_data.movedim(-4, 0)

        out: torch.Tensor = self.cnn(market_data.flatten(0, -4), open_pos_data.flatten(0, -2))
        batch_shape = open_pos_data[..., 0].size()
        out = out.unflatten(0, batch_shape).movedim(0, -4)
        return out

    def compute_baseline(self, trans_input: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
        # trans_input: (..., n_samples, seq_len, batch_size, in_features)
        # z0: (..., n_samples, seq_len, batch_size, z_dim)

        input = torch.cat([trans_input.detach(), z0.detach()], dim=-1)
        return self.nn_baseline(input)

    def sample_hidden_state_dist(self, trans_input: torch.Tensor, z0: torch.Tensor):
        # trans_input: (..., n_samples, seq_len, batch_size, n_features)
        # z0: (..., n_samples, seq_len, batch_size, z_dim)

        z_loc, z_scale = self.trans(trans_input, z0)

        z_dist = dist.TransformedDistribution(dist.Normal(z_loc, z_scale), self.iafs)
        z_sample = z_dist.rsample()
        z_logprob = z_dist.log_prob(z_sample)

        return z_sample, z_logprob

    def sample_exec_dist(self, z_samples) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z_samples: (..., n_samples, seq_len, batch_size, z_dim)

        exec_logits, raw_fractions = self.emitter(z_samples).movedim(-1, -4).unflatten(-4, (2, self.n_cur)).unbind(-5)

        fractions = raw_fractions.tanh()
        exec_dist = dist.Bernoulli(logits=exec_logits)

        exec_samples = exec_dist.sample()
        exec_logprobs = exec_dist.log_prob(exec_samples)
        fractions = fractions.where(exec_samples == 1, fractions.new_zeros([]))

        return exec_logprobs, fractions

    def update(
        self,
        total_margin: torch.Tensor,
        pos_sizes: torch.Tensor,
        pos_rates: torch.Tensor,
        rates: torch.Tensor,
        account_cur_rates: torch.Tensor,
        market_data: torch.Tensor,
        z0: torch.Tensor,
        prev_logprobs: Optional[torch.Tensor] = None,
        compute_loss: bool = False,
    ):
        account_cur_pos_sizes = pos_sizes / account_cur_rates
        pos_margins = account_cur_pos_sizes / self.leverage

        total_used_margin, total_unused_margin = self.compute_used_and_unused_margin(total_margin, pos_margins)

        close_rates = self.compute_close_rates(pos_sizes, rates)

        pos_pl = self.compute_pl(account_cur_pos_sizes.abs(), close_rates)
        closeout_mask = self.compute_closeout_mask(total_margin, total_used_margin, pos_pl)

        pos_data = self.compute_open_pos_data(total_margin, pos_margins, total_unused_margin, pos_rates, close_rates)

        trans_input = self.compute_trans_input(market_data, pos_data)

        z_samples, z_logprobs = self.sample_hidden_state_dist(trans_input, z0)
        exec_logprobs, fractions = self.sample_exec_dist(z_samples)

        open_rates = self.compute_open_rates(fractions, rates)

        open_mask = torch.zeros_like(pos_sizes, dtype=torch.bool)
        close_mask = torch.zeros_like(pos_sizes, dtype=torch.bool)

        if compute_loss:
            surrogate_loss = torch.zeros_like(total_margin)
            loss = torch.zeros_like(total_margin)

            cum_exec_logprobs = torch.zeros_like(total_margin)
            baseline = self.compute_baseline(trans_input, z0).movedim(-1, -4)

            if prev_logprobs is None:
                prev_logprobs = torch.zeros_like(exec_logprobs)

        for i in range(self.n_cur):
            _, total_unused_margin = self.compute_used_and_unused_margin(total_margin, pos_margins)

            available_margin = total_unused_margin.unsqueeze(-4) + pos_margins.select(-4, i)
            exec_size = fractions.select(-4, i) * available_margin * self.leverage * account_cur_rates.select(-4, i)

            exec_size = pos_sizes.select(-4, i).neg().where(closeout_mask.unsqueeze(-4), exec_size)

            new_pos_size = pos_sizes.select(-4, i) + exec_size
            new_pos_size = new_pos_size.new_zeros([]).where(
                new_pos_size.isclose(new_pos_size.new_zeros([])), new_pos_size
            )

            close_pos_mask = (pos_sizes.select(-4, i) != 0) & (new_pos_size == 0)
            remove_reduce_pos_mask = pos_sizes.select(-4, i) * exec_size < 0

            flip_pos_mask = pos_sizes.select(-4, i) * new_pos_size < 0
            new_pos_mask = flip_pos_mask | ((pos_sizes.select(-4, i) == 0) & (new_pos_size != 0))

            open_mask.select(-4, i)[...] = new_pos_mask
            close_mask.select(-4, i)[...] = remove_reduce_pos_mask

            closed_size = torch.minimum(exec_size.abs(), pos_sizes.select(-4, i).abs()).where(
                remove_reduce_pos_mask, exec_size.new_zeros([])
            )
            pl = self.compute_pl(closed_size, pos_rates.select(-4, i), close_rates.select(-4, i))

            if compute_loss:
                cum_exec_logprobs = cum_exec_logprobs + exec_logprobs.select(-4, i).where(
                    close_mask, exec_logprobs.new_zeros([])
                )

                logprob = prev_logprobs.select(-4, i) + z_logprobs + cum_exec_logprobs.select(-4, i)
                cost = torch.log1p_(pl / total_margin)

                surrogate_loss = (
                    surrogate_loss
                    - (logprob * torch.detach_(cost - baseline.select(-4, i)) + cost)
                    + (cost.detach() - baseline.select(-4, i)) ** 2
                )

                loss = loss - cost.detach()

            total_margin = total_margin + pl

            pos_sizes.select(-4, i)[...] = new_pos_size.detach().where(new_pos_mask, pos_sizes.select(-4, i))
            # TODO: in the future we'll update only when we open a new position so that here we only need one where.
            # This means that we the rates might be non-zero where the sizes are 0, but it doesn't matter.
            pos_rates.select(-4, i)[...] = open_rates.select(-4, i).where(
                new_pos_mask, pos_rates.new_zeros([]).where(close_pos_mask, pos_rates.select(-4, i))
            )

            pos_margins.select(-4, i)[...] = pos_sizes.select(-4, i) / (self.leverage * account_cur_rates.select(-4, i))

        if compute_loss:
            return total_margin, pos_sizes, pos_rates, open_mask, close_mask, surrogate_loss, loss
        else:
            return total_margin, pos_sizes, pos_rates, open_mask, close_mask, z_logprobs, exec_logprobs

    def forward(
        self,
        market_data: torch.Tensor,
        z0: torch.Tensor,
        rates: torch.Tensor,
        account_cur_rates: torch.Tensor,
        total_margin: torch.Tensor,
        pos_sizes: torch.Tensor,
        pos_rates: torch.Tensor,
        left_pos_mask: torch.Tensor,
        # left_open_pos_inds: torch.Tensor,
        # left_cur_inds: torch.Tensor,
        # left_batch_inds: torch.Tensor,
    ):
        # market_data: (n_cur + 1, seq_len, batch_size, n_features, n_cur, window_size)
        # z0: (n_cur + 1, n_samples, seq_len, batch_size, z_dim)

        # rates: (n_cur + 1, 2, n_cur, seq_len, batch_size)
        # this is the midpoint for the rate between
        # the base currency of the pair and the account currency (euro)
        # account_cur_rates: (n_cur + 1, n_cur, seq_len, batch_size)

        # total_margin: (n_cur + 1, n_samples, seq_len, batch_size)
        # pos_sizes: (n_cur + 1, n_cur, n_samples, seq_len, batch_size)
        # pos_rates: (n_cur + 1, n_cur, n_samples, seq_len, batch_size)
        # this mask tells us for which currencies there is an open position
        # left_pos_mask: (n_cur, n_samples, seq_len, batch_size)

        # NOTE: to broadcast with the n_samples dim
        rates = rates.unsqueeze(2)
        account_cur_rates = account_cur_rates.unsqueeze(1)

        left_market_data = market_data[:-1]
        left_z0 = z0[:-1]
        left_rates = rates[:-1]
        left_account_cur_rates = account_cur_rates[:-1]
        left_total_margin = total_margin[:-1]
        left_pos_sizes = pos_sizes[:-1]
        left_pos_rates = pos_rates[:-1]

        (
            left_total_margin,
            left_pos_sizes,
            left_pos_rates,
            left_open_mask,
            left_close_mask,
            left_z_logprobs,
            left_exec_logprobs,
        ) = self.update(
            left_total_margin,
            left_pos_sizes,
            left_pos_rates,
            left_rates,
            left_account_cur_rates,
            left_market_data,
            left_z0,
        )

        left_open_logprobs = left_exec_logprobs.where(left_open_mask, left_exec_logprobs.new_zeros())
        left_close_logprobs = left_exec_logprobs.where(left_close_mask, left_exec_logprobs.new_zeros())

        cum_left_exec_logprobs = left_open_logprobs + torch.cat(
            torch.zeros_like(left_close_logprobs[:, -1:]), left_close_logprobs[:, :-1].cumsum(1), dim=1
        )

        splatted_left_pos_mask = left_pos_mask & torch.eye(
            self.n_cur, dtype=torch.bool, device=left_pos_mask.device
        ).view(self.n_cur, self.n_cur, 1, 1, 1)

        if torch.jit.is_scripting():
            self.apply_mask_map = torch.jit.script(self.apply_mask_map)

        open_pos_sizes = self.apply_mask_map(left_pos_sizes, splatted_left_pos_mask, left_pos_mask)
        open_pos_rates = self.apply_mask_map(left_pos_rates, splatted_left_pos_mask, left_pos_mask)
        prev_z_logprobs = self.apply_mask_map(left_z_logprobs, splatted_left_pos_mask, left_pos_mask)
        prev_cum_exec_logprobs = self.apply_mask_map(cum_left_exec_logprobs, splatted_left_pos_mask, left_pos_mask)

        right_market_data = market_data[-1]
        right_z0 = z0[-1]
        right_rates = rates[-1]
        right_account_cur_rates = account_cur_rates[-1]
        right_total_margin = total_margin[-1]
        right_pos_sizes = pos_sizes[-1]
        right_pos_rates = pos_rates[-1]

        same_pos_mask = open_pos_rates == right_pos_rates
        right_pos_sizes = right_pos_sizes + torch.where(
            same_pos_mask, open_pos_sizes - open_pos_sizes.detach(), open_pos_sizes.new_zeros([])
        )

        return self.update(
            right_total_margin,
            right_pos_sizes,
            right_pos_rates,
            right_rates,
            right_account_cur_rates,
            right_market_data,
            right_z0,
            prev_logprobs=prev_z_logprobs + prev_cum_exec_logprobs,
            compute_loss=True,
        )

        # account_cur_pos_sizes = left_pos_sizes / left_account_cur_rates
        # left_pos_margins = account_cur_pos_sizes / self.leverage

        # total_used_margin, total_unused_margin = self.compute_used_and_unused_margin(
        #     left_total_margin, left_pos_margins
        # )

        # close_rates = self.compute_close_rates(left_pos_sizes, left_rates)

        # pos_pl = self.compute_pl(account_cur_pos_sizes.abs(), close_rates)
        # closeout_mask = self.compute_closeout_mask(left_total_margin, total_used_margin, pos_pl)

        # pos_data = self.compute_open_pos_data(
        #     left_total_margin, left_pos_margins, total_unused_margin, left_pos_rates, close_rates
        # )

        # left_z_samples, left_z_logprobs = self.sample_hidden_state_dist(left_batch_data, left_z0, pos_data)
        # left_exec_samples, left_exec_logprobs, fractions = self.sample_exec_dist(left_z_samples)

        # open_rates = self.compute_open_rates(fractions, left_rates)

        # close_mask = torch.zeros_like(left_pos_sizes, dtype=torch.bool)
        # open_mask = torch.zeros_like(left_pos_sizes, dtype=torch.bool)

        # for i in range(self.n_cur):
        #     _, total_unused_margin = self.compute_used_and_unused_margin(left_total_margin, left_pos_margins)

        #     available_margin = total_unused_margin.unsqueeze(1) + left_pos_margins[:, i]
        #     exec_size = fractions[:, i] * available_margin * self.leverage * left_account_cur_rates[:, i]

        #     new_pos_size = left_pos_sizes[:, i] + exec_size
        #     new_pos_size = new_pos_size.new_zeros([]).where(
        #         closeout_mask.unsqueeze(1) | new_pos_size.isclose(new_pos_size.new_zeros([])), new_pos_size
        #     )

        #     remove_reduce_pos_mask = left_pos_sizes[:, i] * exec_size < 0
        #     flip_pos_mask = left_pos_sizes[:, i] * new_pos_size < 0
        #     new_pos_mask = flip_pos_mask | ((left_pos_sizes[:, i] == 0) & (new_pos_size != 0))
        #     close_pos_mask = (left_pos_sizes[:, i] != 0) & (new_pos_size == 0)

        #     close_mask[:, i] = remove_reduce_pos_mask
        #     open_mask[:, i] = new_pos_mask

        #     closed_size = torch.minimum(exec_size.abs(), left_pos_sizes[:, i].abs()).where(
        #         remove_reduce_pos_mask, exec_size.new_zeros([])
        #     )

        #     left_total_margin = left_total_margin + self.compute_pl(
        #         closed_size, left_pos_rates[:, i], close_rates[:, i]
        #     )

        #     left_pos_sizes[:, i] = new_pos_size.detach().where(new_pos_mask, left_pos_sizes[:, i])
        #     # TODO: in the future we'll update only when we open a new position so that here we only need one where.
        #     # This meaans that we the rates might be non-zero where the sizes are 0, but it doesn't matter.
        #     left_pos_rates[:, i] = open_rates[:, i].where(
        #         new_pos_mask, left_pos_rates.new_zeros([]).where(close_pos_mask, left_pos_rates[:, i])
        #     )

        #     left_pos_margins[:, i] = left_pos_sizes[:, i] / (self.leverage * left_account_cur_rates[:, i])

        # open_logprobs = left_exec_logprobs.where(open_mask, left_exec_logprobs.new_zeros())
        # close_logprobs = left_exec_logprobs.where(close_mask, left_exec_logprobs.new_zeros())

        # cum_left_exec_logprobs = open_logprobs + torch.cat(
        #     torch.zeros_like(close_logprobs[:, -1:]), close_logprobs[:, :-1].cumsum(1), dim=1
        # )

        # splatted_left_pos_mask = left_pos_mask & torch.eye(
        #     self.n_cur, dtype=torch.bool, device=left_pos_mask.device
        # ).view(self.n_cur, self.n_cur, 1, 1, 1)

        # # TODO: create a function where we create the following tensors and jit.script it
        # open_pos_sizes = left_pos_sizes.new_zeros([self.n_cur, self.n_samples, self.seq_len, self.batch_size])
        # open_pos_rates = left_pos_rates.new_zeros([self.n_cur, self.n_samples, self.seq_len, self.batch_size])

        # open_pos_sizes[left_pos_mask] = left_pos_sizes[splatted_left_pos_mask]
        # open_pos_rates[left_pos_mask] = left_pos_rates[splatted_left_pos_mask]

        # right_batch_data = batch_data[-1]
        # right_z0 = z0[-1]
        # right_rates = rates[-1]
        # right_account_cur_rates = account_cur_rates[-1]
        # right_total_margin = total_margin[-1]
        # right_pos_sizes = pos_sizes[-1]
        # right_pos_rates = pos_rates[-1]

        # same_pos_mask = open_pos_rates == right_pos_rates

        # # it can happen that the initial size of the position has been reduced at some point
        # # between the left and right timesteps. But it's also possible that the action
        # # taken by the model doesn't match the old one that was taken before.
        # # in the the fist case we want to add (or subtract )
        # # right_pos_sizes = open_pos_sizes - open_pos_sizes.detach() + right_pos_sizes
        # right_pos_sizes = right_pos_sizes + torch.where(
        #     same_pos_mask, open_pos_sizes - open_pos_sizes.detach(), open_pos_sizes.new_zeros([])
        # )

        # # right_pos_sizes = torch.where(
        # #     same_pos_mask, open_pos_sizes + right_pos_sizes - open_pos_sizes.detach(), right_pos_sizes
        # # )

        # account_cur_pos_sizes = right_pos_sizes / right_account_cur_rates
        # right_pos_margins = account_cur_pos_sizes / self.leverage

        # total_used_margin, total_unused_margin = self.compute_used_and_unused_margin(
        #     right_total_margin, right_pos_margins
        # )

        # close_rates = self.compute_close_rates(right_pos_sizes, right_rates)

        # pos_pl = self.compute_pl(account_cur_pos_sizes.abs(), close_rates)
        # closeout_mask = self.compute_closeout_mask(right_total_margin, total_used_margin, pos_pl)

        # pos_data = self.compute_open_pos_data(
        #     right_total_margin, right_pos_margins, total_unused_margin, right_pos_rates, close_rates
        # )

        # right_z_samples, right_z_logprobs = self.sample_hidden_state_dist(right_batch_data, right_z0, pos_data)
        # right_exec_samples, right_exec_logprobs, fractions = self.sample_exec_dist(right_z_samples)

        # open_rates = self.compute_open_rates(fractions, right_rates)

        # close_mask = torch.zeros_like(right_pos_sizes, dtype=torch.bool)
        # open_mask = torch.zeros_like(right_pos_sizes, dtype=torch.bool)

        # for i in range(self.n_cur):
        #     _, total_unused_margin = self.compute_used_and_unused_margin(right_total_margin, right_pos_margins)

        #     available_margin = total_unused_margin.unsqueeze(1) + right_pos_margins[:, i]
        #     exec_size = fractions[:, i] * available_margin * self.leverage * left_account_cur_rates[:, i]

        #     new_pos_size = left_pos_sizes[:, i] + exec_size
        #     new_pos_size = new_pos_size.new_zeros([]).where(
        #         closeout_mask.unsqueeze(1) | new_pos_size.isclose(new_pos_size.new_zeros([])), new_pos_size
        #     )

        #     remove_reduce_pos_mask = left_pos_sizes[:, i] * exec_size < 0
        #     flip_pos_mask = left_pos_sizes[:, i] * new_pos_size < 0
        #     new_pos_mask = flip_pos_mask | ((left_pos_sizes[:, i] == 0) & (new_pos_size != 0))
        #     close_pos_mask = (left_pos_sizes[:, i] != 0) & (new_pos_size == 0)

        #     close_mask[:, i] = remove_reduce_pos_mask
        #     open_mask[:, i] = new_pos_mask

        #     closed_size = torch.minimum(exec_size.abs(), left_pos_sizes[:, i].abs()).where(
        #         remove_reduce_pos_mask, exec_size.new_zeros([])
        #     )

        #     left_total_margin = left_total_margin + self.compute_pl(
        #         closed_size, left_pos_rates[:, i], close_rates[:, i]
        #     )

        #     left_pos_sizes[:, i] = new_pos_size.detach().where(new_pos_mask, left_pos_sizes[:, i])
        #     # TODO: in the future we'll update only when we open a new position so that here we only need one where.
        #     # This meaans that we the rates might be non-zero where the sizes are 0, but it doesn't matter.
        #     left_pos_rates[:, i] = open_rates[:, i].where(
        #         new_pos_mask, left_pos_rates.new_zeros([]).where(close_pos_mask, left_pos_rates[:, i])
        #     )

        #     left_pos_margins[:, i] = left_pos_sizes[:, i] / (self.leverage * left_account_cur_rates[:, i])
