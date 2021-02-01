import torch
import torch.jit
import torch.distributions as dist
import torch.nn as nn
from waterstart_model import GatedTrasition, Emitter
from pyro.distributions import TransformModule
from typing import List


class LossEvaluator(nn.Module):
    def __init__(
        self,
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
        convnet_out: torch.Tensor,
        z0: torch.Tensor,
        rates: torch.Tensor,
        account_cur_rates: torch.Tensor,
        total_margin: torch.Tensor,
        open_trades_sizes: torch.Tensor,
        open_trades_rates: torch.Tensor,
    ):
        # convnet_out: (batch_size, seq_len, n_features)
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
        convnet_out = convnet_out.unsqueeze(2)

        # TODO: rename to latest?
        first_open_trades_sizes_view = open_trades_sizes[..., 0]
        last_open_trades_sizes_view = open_trades_sizes[..., -1]

        account_cur_open_trades_sizes = open_trades_sizes / account_cur_rates.unsqueeze(4)
        open_trades_margins = account_cur_open_trades_sizes / self.leverage

        # TODO: set the right dims. they should be max_trades and n_cur
        used_margins = open_trades_margins.abs().sum(4)
        total_used_margin = used_margins.sum(3)
        total_unused_margin = total_margin - total_used_margin

        # NOTE: the sign of the first trade determines the type of the position (long, short)
        close_rates = torch.where(
            first_open_trades_sizes_view > 0,
            rates[..., 0],
            torch.where(first_open_trades_sizes_view < 0, rates[..., 1], rates.new_ones([])),
        )

        open_trades_pl = account_cur_open_trades_sizes * (1 - close_rates.unsqueeze(4) / open_trades_rates)

        account_value = total_margin + open_trades_pl.sum([3, 4])
        closeout_mask = account_value < 0.5 * total_used_margin

        rel_margins = torch.cat(
            [open_trades_margins.flatten(3, 4), total_unused_margin.unsqueeze(3)], dim=3
        ) / total_margin.unsqueeze(3)
        rel_open_rates = open_trades_rates / rates[:, -1, None].mean(4, keepdim=True)

        input = torch.cat(
            [convnet_out.expand(-1, -1, self.n_samples, -1), rel_margins, rel_open_rates.flatten(3, 4)], dim=3
        )

        z_loc, z_scale = self.trans(input, z0)

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
            fractions > 0, rates[..., 1], torch.where(fractions < 0, rates[..., 0], rates.new_ones([])),
        )

        for i in range(self.n_cur):
            position_sizes = open_trades_margins.sum(4)
            used_margins = position_sizes.abs() / (self.leverage * account_cur_rates)

            total_unused_margin = total_margin - used_margins.sum(3)

            available_margins = total_unused_margin + used_margins[..., i].where(
                opposite_types_mask[..., i], used_margins.new_zeros([])
            )

            exec_sizes = fractions[..., i] * available_margins * self.leverage * account_cur_rates[..., i]
            exec_sizes = exec_sizes.where(exec_mask[..., i], exec_sizes.new_zeros([])).where(
                ~closeout_mask.unsqueeze(3), -position_sizes
            )

            cur_open_trades_sizes_view = open_trades_sizes[..., i, :]
            cur_open_trades_rates_view = open_trades_rates[..., i, :]

            shifted_cur_open_trades_sizes = cur_open_trades_sizes_view.roll(shifts=1, dims=3)
            shifted_cur_open_trades_rates = cur_open_trades_rates_view.roll(shifts=1, dims=3)

            shifted_cur_open_trades_sizes[..., 0] = exec_sizes
            shifted_cur_open_trades_rates[..., 0] = open_rates[..., i]

            cur_open_trades_sizes_view[...] = shifted_cur_open_trades_sizes.where(
                add_trade_mask[..., i, None], cur_open_trades_sizes_view
            )
            cur_open_trades_rates_view[...] = shifted_cur_open_trades_rates.where(
                add_trade_mask[..., i, None], cur_open_trades_rates_view
            )

            left_exec_sizes = exec_sizes.where(opposite_types_mask[..., i], exec_sizes.new_zeros([]))
            already_covered_mask = torch.zeros_like(left_exec_sizes, dtype=torch.bool)
            # TODO: do we keep the lastest at the beginning of the array?
            for j in reversed(range(self.max_trades)):
                # NOTE: even though we are adding here, this is always a difference in the positions
                # where we are interested, as the sign of the two addends is always opposite
                sizes_diffs = left_exec_sizes + cur_open_trades_sizes_view[..., j]
                # NOTE: the <= is important. Without it some of the computations below would end up being wrong
                exec_size_covered_mask = ~already_covered_mask & (sizes_diffs * left_exec_sizes <= 0)
                already_covered_mask |= exec_size_covered_mask

                closed_trades_sizes = torch.where(
                    exec_size_covered_mask, left_exec_sizes, cur_open_trades_sizes_view[..., j]
                ).abs_()
                closed_trades_pl = (
                    closed_trades_sizes
                    # TODO: we can compute this once before the loop and then use it here
                    / account_cur_rates[..., i]
                    * (1 - close_rates[..., i] / cur_open_trades_rates_view[..., j])
                )
                total_margin = total_margin + closed_trades_pl

                cur_open_trades_sizes_view[..., j] = sizes_diffs.where(
                    exec_size_covered_mask, sizes_diffs.new_zeros([])
                )

                cur_open_trades_rates_view[..., j] = cur_open_trades_rates_view[..., j].where(
                    exec_size_covered_mask, open_trades_rates.new_zeros([])
                )

                left_exec_sizes = sizes_diffs.new_zeros([]).where(exec_size_covered_mask, sizes_diffs)

            cur_open_trades_sizes_view[..., 0] = left_exec_sizes.where(
                ~already_covered_mask, cur_open_trades_sizes_view[..., 0]
            )
            cur_open_trades_rates_view[..., 0] = open_rates[..., i].where(
                ~already_covered_mask, cur_open_trades_rates_view[..., 0]
            )

        logprobs = z_logprobs + exec_logprobs.where(
            ~(closeout_mask.unsqueeze(3) | ignore_add_trade_mask), exec_logprobs.new_zeros([])
        ).sum_(3)

        costs = total_margin.log().neg_()
        baselines = costs.mean(2, keepdim=True)
        loss = torch.mean(logprobs * torch.detach(costs - baselines) + costs, dim=2).sum()
        return loss, z_sample, total_margin, open_trades_sizes, open_trades_rates


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
