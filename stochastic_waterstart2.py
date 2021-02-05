import torch
import torch.jit
import torch.distributions as dist
import torch.nn as nn
from waterstart_model import CNN, GatedTrasition, Emitter
from pyro.distributions import TransformModule
from typing import List


class LossEvaluator(nn.Module):
    def __init__(
        self,
        cnn: CNN,
        trans: GatedTrasition,
        emitter: Emitter,
        iafs: List[TransformModule],
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
        self.emitter = emitter
        self.iafs = iafs
        self._iafs_modules = nn.ModuleList(iafs)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_samples = n_samples
        self.n_cur = n_cur
        self.max_trades = max_trades
        self.z_dim = z_dim
        self.leverage = leverage

    def forward(
        self,
        batch_data: torch.Tensor,
        z0: torch.Tensor,
        rates: torch.Tensor,
        account_cur_rates: torch.Tensor,
        total_margin: torch.Tensor,
        open_trades_sizes: torch.Tensor,
        open_trades_rates: torch.Tensor,
    ):
        # batch_data: (batch_size * seq_len, n_features, n_cur, window_size)
        # z0: (batch_size, seq_len, n_samples, z_dim)

        # rates: (batch_size, seq_len, n_cur, 2)
        # this is the midpoint for the rate between
        # the base currency of the pair and the account currency (euro)
        # account_cur_rates: (batch_size, seq_len, n_cur)

        # total_margin: (batch_size, seq_len, n_samples)
        # open_trades_sizes: (batch_size, seq_len, n_samples, n_cur, max_trades)
        # open_trades_rates: (batch_size, seq_len, n_samples, n_cur, max_trades)

        # NOTE: to broadcast with the n_samples dim
        rates = rates.unsqueeze(2)
        account_cur_rates = account_cur_rates.unsqueeze(2)

        # NOTE: the trades go from the oldest to the newest and ar aligned to the right
        # TODO: maybe switch the names of these two?
        first_open_trades_sizes_view = open_trades_sizes[..., -1]
        last_open_trades_sizes_view = open_trades_sizes[..., 0]

        account_cur_open_trades_sizes = open_trades_sizes / account_cur_rates.unsqueeze(4)
        open_trades_margins = account_cur_open_trades_sizes / self.leverage

        # TODO: set the right dims. they should be max_trades and n_cur
        total_used_margin = open_trades_margins.abs().sum([3, 4])
        total_unused_margin = total_margin - total_used_margin

        # NOTE: the sign of the first trade determines the type of the position (long, short)
        close_rates = torch.where(
            first_open_trades_sizes_view > 0,
            rates[..., 0],
            torch.where(first_open_trades_sizes_view < 0, rates[..., 1], rates.new_ones([])),
        )

        no_zeros_open_trades_rates = open_trades_rates.where(open_trades_rates != 0, open_trades_rates.new_ones([]))
        open_trades_pl = account_cur_open_trades_sizes * (1 - close_rates.unsqueeze(4) / no_zeros_open_trades_rates)

        account_value = total_margin + open_trades_pl.sum([3, 4])
        closeout_mask = account_value < 0.5 * total_used_margin

        rel_margins = torch.cat(
            [open_trades_margins.flatten(3, 4), total_unused_margin.unsqueeze(3)], dim=3
        ) / total_margin.unsqueeze(3)
        rel_open_rates = open_trades_rates / rates[:, -1, None].mean(4, keepdim=True)

        prev_step_data = torch.cat([rel_margins, rel_open_rates.flatten(3, 4)], dim=3).view(
            self.batch_size * self.seq_len * self.n_samples, -1
        )
        out = self.cnn(batch_data, prev_step_data).view(self.batch_size, self.seq_len, self.n_samples, -1)
        z_loc, z_scale = self.trans(out, z0)

        z_dist = dist.TransformedDistribution(dist.Normal(z_loc, z_scale), self.iafs)
        z_sample = z_dist.rsample()
        z_logprobs = z_dist.log_prob(z_sample)

        exec_logits, raw_fractions = (
            self.emitter(z_sample).view(self.batch_size, self.seq_len, self.n_samples, self.n_cur, 2).unbind(-1)
        )

        fractions = raw_fractions.tanh_()
        exec_dist = dist.Bernoulli(logits=exec_logits)

        exec_samples = exec_dist.sample()
        exec_logprobs = exec_dist.log_prob(exec_samples)

        opposite_types_mask = fractions * first_open_trades_sizes_view <= 0
        exec_mask = exec_samples == 1

        ignore_add_trade_mask = ~opposite_types_mask & (last_open_trades_sizes_view != 0)
        add_trade_mask = ~opposite_types_mask & (last_open_trades_sizes_view == 0)

        open_rates = torch.where(
            fractions > 0, rates[..., 1], torch.where(fractions < 0, rates[..., 0], rates.new_zeros([])),
        )

        for i in range(self.n_cur):
            pos_sizes = open_trades_margins.sum(4)
            used_margins = pos_sizes.abs() / (self.leverage * account_cur_rates)

            total_unused_margin = total_margin - used_margins.sum(3)

            available_margins = total_unused_margin + used_margins[..., i].where(
                opposite_types_mask[..., i], used_margins.new_zeros([])
            )

            exec_sizes = fractions[..., i] * available_margins * self.leverage * account_cur_rates[..., i]
            exec_sizes = exec_sizes.where(exec_mask[..., i], exec_sizes.new_zeros([])).where(
                ~closeout_mask.unsqueeze(3), -pos_sizes
            )

            cur_open_trades_sizes_view = open_trades_sizes[..., i, :]
            cur_open_trades_rates_view = open_trades_rates[..., i, :]

            shifted_cur_open_trades_sizes = cur_open_trades_sizes_view.roll(shifts=-1, dims=3)
            shifted_cur_open_trades_rates = cur_open_trades_rates_view.roll(shifts=-1, dims=3)

            shifted_cur_open_trades_sizes[..., -1] = exec_sizes
            shifted_cur_open_trades_rates[..., -1] = open_rates[..., i]

            cur_open_trades_sizes_view[...] = shifted_cur_open_trades_sizes.where(
                add_trade_mask[..., i, None], cur_open_trades_sizes_view
            )
            cur_open_trades_rates_view[...] = shifted_cur_open_trades_rates.where(
                add_trade_mask[..., i, None], cur_open_trades_rates_view
            )

            # TODO: here left is meant as remaining, while variables below he have left and right
            # we we should find another name
            left_exec_sizes = exec_sizes.where(opposite_types_mask[..., i], exec_sizes.new_zeros([]))

            right_cum_size_diffs = left_exec_sizes.unsqueeze(3) + cur_open_trades_sizes_view.cumsum(3)
            left_cum_size_diffs = torch.empty_like(right_cum_size_diffs)
            left_cum_size_diffs[..., 0] = left_exec_sizes
            left_cum_size_diffs[..., 1:] = right_cum_size_diffs[..., :-1]

            reduce_trade_size_mask = left_cum_size_diffs * right_cum_size_diffs < 0
            close_trades_mask = right_cum_size_diffs * left_exec_sizes.unsqueeze(3) >= 0

            closed_trades_sizes = torch.zeros_like(cur_open_trades_sizes_view)

            closed_trades_sizes[close_trades_mask] = cur_open_trades_sizes_view[close_trades_mask]
            cur_open_trades_sizes_view[close_trades_mask] = 0
            cur_open_trades_rates_view[close_trades_mask] = 0

            closed_trades_sizes[reduce_trade_size_mask] = left_cum_size_diffs[reduce_trade_size_mask]
            cur_open_trades_sizes_view[reduce_trade_size_mask] = right_cum_size_diffs[reduce_trade_size_mask]

            closed_trades_pl = (
                closed_trades_sizes.abs_()
                / account_cur_rates[..., i, None]
                # NOTE: here we can use no_zeros_open_trades_rates without updating it because the values
                # that we use are only those of the trades that were there at the beginning
                * (1 - close_rates[..., i, None] / no_zeros_open_trades_rates[..., i, :])
            )

            total_margin = total_margin + closed_trades_pl.sum(3)

            add_opposite_type_trade_mask = close_trades_mask[..., -1]
            cur_open_trades_sizes_view[..., -1] = (
                right_cum_size_diffs[..., -1]
                .detach()
                .where(add_opposite_type_trade_mask, cur_open_trades_sizes_view[..., -1])
            )
            cur_open_trades_rates_view[..., -1] = open_rates[..., i].where(
                add_opposite_type_trade_mask, cur_open_trades_rates_view[..., -1]
            )

        logprobs = z_logprobs + exec_logprobs.where(
            ~(closeout_mask.unsqueeze(3) | ignore_add_trade_mask), exec_logprobs.new_zeros([])
        ).sum_(3)

        costs = total_margin.log().neg_()
        baselines = costs.mean(2, keepdim=True)
        # loss = torch.mean(logprobs * torch.detach_(costs - baselines) + costs, dim=2).sum()
        losses = logprobs * torch.detach_(costs - baselines) + costs

        total_margin = total_margin.detach_()
        z_sample = z_sample.detach_()
        assert not open_trades_sizes.requires_grad
        assert not open_trades_rates.requires_grad

        return losses, z_sample, total_margin, open_trades_sizes, open_trades_rates


if __name__ == "__main__":
    n_cur = 13
    n_samples = 100
    leverage = 1
    z_dim = 128

    seq_len = 100
    batch_size = 2

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    loss_eval = LossEvaluator(seq_len, batch_size, n_samples, n_cur, leverage=1).to(device)

    dummy_pos_state = torch.randint(2, size=(n_cur, n_samples, batch_size), dtype=torch.int8, device=device)
    dummy_rates = torch.randn(seq_len, 1, n_cur, batch_size, device=device).div_(100).log1p_().cumsum(
        0
    ).exp_() + torch.tensor([0.0, 1e-4], device=device).view(2, 1, 1)

    dummy_input = (
        torch.randint(2, size=(seq_len, 3, n_cur, n_samples, batch_size), dtype=torch.float32, device=device),
        torch.rand(seq_len, n_cur, n_samples, batch_size, device=device),
        torch.rand(seq_len, 3, n_cur, n_samples, batch_size, device=device).log_(),
        torch.rand(seq_len, n_samples, batch_size, device=device).log_(),
        dummy_rates,
        torch.randn(seq_len, n_cur, batch_size, device=device).div_(100).log1p_().cumsum(0).exp_(),
        dummy_pos_state,
        torch.randint(2, size=(n_cur, n_samples, batch_size), dtype=torch.int8, device=device),
        torch.ones(n_samples, batch_size, device=device),
        torch.where(
            dummy_pos_state == 1,
            torch.rand(n_cur, n_samples, batch_size, device=device),
            torch.zeros(n_cur, n_samples, batch_size, device=device),
        ),
        torch.where(
            dummy_pos_state == 1,
            dummy_rates[torch.randint(seq_len, size=(n_samples,)), 0].movedim(0, 1),
            torch.zeros(n_cur, n_samples, batch_size, device=device),
        ),
    )

    loss_eval = torch.jit.trace(loss_eval, dummy_input)
    print("jit done")

    # with torch.autograd.set_detect_anomaly(True):
    with torch.jit.optimized_execution(False):
        loss, all_pos_states, all_pos_types, all_total_margins, all_open_pos_sizes, all_open_rates = loss_eval(
            *dummy_input
        )
        # res.sum().backward()
