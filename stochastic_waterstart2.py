import torch
import torch.jit
import torch.distributions as dist
import torch.nn as nn
from waterstart_model import CNN, GatedTrasition, Emitter, NeuralBaseline
from pyro.distributions import TransformModule
from typing import List


class LossEvaluator(nn.Module):
    def __init__(
        self,
        cnn: CNN,
        trans: GatedTrasition,
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
        # batch_data: (seq_len * batch_size, n_features, n_cur, window_size)
        # z0: (n_samples, seq_len, batch_size, z_dim)

        # rates: (2, n_cur, seq_len, batch_size)
        # this is the midpoint for the rate between
        # the base currency of the pair and the account currency (euro)
        # account_cur_rates: (n_cur, seq_len, batch_size)

        # total_margin: (n_samples, seq_len, batch_size)
        # open_trades_sizes: (n_cur, max_trades, n_samples, seq_len, batch_size)
        # open_trades_rates: (n_cur, max_trades, n_samples, seq_len, batch_size)

        # NOTE: to broadcast with the n_samples dim
        rates = rates.unsqueeze(2)
        account_cur_rates = account_cur_rates.unsqueeze(1)

        assert not torch.any((open_trades_sizes == 0) != (open_trades_rates == 0))
        assert torch.all(torch.all(open_trades_sizes >= 0, dim=1) | torch.all(open_trades_sizes <= 0, dim=1))

        # NOTE: the trades go from the oldest to the newest and are aligned to the right
        # TODO: maybe switch the names of these two?
        first_open_trades_sizes_view = open_trades_sizes[:, -1]
        last_open_trades_sizes_view = open_trades_sizes[:, 0]

        account_cur_open_trades_sizes = open_trades_sizes / account_cur_rates.unsqueeze(1)
        open_trades_margins = account_cur_open_trades_sizes / self.leverage

        total_used_margin = open_trades_margins.abs().sum([0, 1])
        total_unused_margin = torch.maximum(total_margin - total_used_margin, total_margin.new_zeros([]))

        # NOTE: the sign of the first trade determines the type of the position (long, short)
        close_rates = torch.where(
            first_open_trades_sizes_view > 0,
            rates[0],
            torch.where(first_open_trades_sizes_view < 0, rates[1], rates.new_ones([])),
        )

        open_trades_pl = account_cur_open_trades_sizes * (1 - open_trades_rates / close_rates.unsqueeze(1))

        account_value = total_margin + open_trades_pl.sum([0, 1])
        closeout_mask = account_value < 0.5 * total_used_margin

        rel_margins = torch.cat([open_trades_margins.flatten(0, 1), total_unused_margin.unsqueeze(0)]) / total_margin
        rel_open_rates = open_trades_rates / rates.mean(0).unsqueeze_(1)

        prev_step_data = torch.cat([rel_margins, rel_open_rates.flatten(0, 1)])

        out = self.cnn(batch_data, prev_step_data.flatten(1, 3).t_()).view(
            self.n_samples, self.seq_len, self.batch_size, -1
        )
        z_loc, z_scale = self.trans(out, z0)

        z_dist = dist.TransformedDistribution(dist.Normal(z_loc, z_scale), self.iafs)
        z_sample = z_dist.rsample()
        z_logprob = z_dist.log_prob(z_sample)

        exec_logits, raw_fractions = (
            self.emitter(z_sample).movedim(3, 0).view(2, self.n_cur, self.n_samples, self.seq_len, self.batch_size)
        )

        fractions = raw_fractions.tanh()
        exec_dist = dist.Bernoulli(logits=exec_logits)

        exec_samples = exec_dist.sample()
        exec_logprobs = exec_dist.log_prob(exec_samples)
        exec_mask = exec_samples == 1

        fractions = fractions.where(exec_mask, fractions.new_zeros([]))

        # TODO: this data is not useful when there is a closeout, but since it's a marginal case for now
        # we keep it like this
        nn_baseline_input = torch.cat(
            [prev_step_data, z_sample.detach().movedim(3, 0), exec_samples, fractions.detach()]
        )
        assert not nn_baseline_input.requires_grad
        baseline = self.nn_baseline(nn_baseline_input.movedim(0, 3)).movedim(3, 0)

        prod_ = fractions * first_open_trades_sizes_view

        add_trade_mask = ~closeout_mask & (prod_ > 0)
        remove_reduce_trade_mask = (closeout_mask & (first_open_trades_sizes_view != 0)) | (prod_ < 0)
        new_pos_mask = ~closeout_mask & (first_open_trades_sizes_view == 0) & (fractions != 0)

        ignore_add_trade_mask = add_trade_mask & (last_open_trades_sizes_view != 0)
        # add_trade_mask = add_trade_mask & (last_open_trades_sizes_view == 0)
        add_trade_mask &= ~ignore_add_trade_mask

        exec_logprobs = exec_logprobs.where(~(closeout_mask | ignore_add_trade_mask), exec_logprobs.new_zeros([]))
        cum_exec_logprobs = exec_logprobs.cumsum(0)

        open_rates = torch.where(fractions > 0, rates[1], torch.where(fractions < 0, rates[0], rates.new_zeros([])))

        surrogate_loss = torch.zeros_like(total_margin)
        loss = torch.zeros_like(total_margin)

        for i in range(self.n_cur):
            pos_sizes = open_trades_sizes.sum(1)
            used_margins = pos_sizes.abs() / (self.leverage * account_cur_rates)

            total_unused_margin = torch.maximum(total_margin - used_margins.sum(0), total_margin.new_zeros([]))

            available_margins = total_unused_margin + used_margins[i].where(
                remove_reduce_trade_mask[i], used_margins.new_zeros([])
            )

            exec_sizes = fractions[i] * available_margins * self.leverage * account_cur_rates[i]
            exec_sizes = pos_sizes[i].neg().where(closeout_mask, exec_sizes)

            cur_open_trades_sizes_view = open_trades_sizes[i]
            cur_open_trades_rates_view = open_trades_rates[i]

            shifted_cur_open_trades_sizes = cur_open_trades_sizes_view.roll(shifts=-1, dims=3)
            shifted_cur_open_trades_rates = cur_open_trades_rates_view.roll(shifts=-1, dims=3)

            shifted_cur_open_trades_sizes[-1] = exec_sizes.detach()
            shifted_cur_open_trades_rates[-1] = open_rates[i]

            cur_open_trades_sizes_view[...] = shifted_cur_open_trades_sizes.where(
                add_trade_mask[i], cur_open_trades_sizes_view
            )
            cur_open_trades_rates_view[...] = shifted_cur_open_trades_rates.where(
                add_trade_mask[i], cur_open_trades_rates_view
            )
            assert not torch.any((cur_open_trades_sizes_view == 0) != (cur_open_trades_rates_view == 0))

            right_cum_size_diffs = exec_sizes + cur_open_trades_sizes_view.cumsum(0)
            close_trades_mask = remove_reduce_trade_mask[i] & (right_cum_size_diffs * exec_sizes >= 0)

            left_cum_size_diffs = torch.empty_like(right_cum_size_diffs)
            left_cum_size_diffs[0] = exec_sizes
            left_cum_size_diffs[1:] = right_cum_size_diffs[:-1]
            reduce_trade_size_mask = left_cum_size_diffs * right_cum_size_diffs < 0
            assert not torch.any(~remove_reduce_trade_mask[i] & reduce_trade_size_mask)
            assert torch.all(reduce_trade_size_mask.sum(0) <= 1)

            closed_trades_sizes = cur_open_trades_sizes_view.where(
                close_trades_mask, left_cum_size_diffs.where(reduce_trade_size_mask, left_cum_size_diffs.new_zeros([]))
            )

            cur_open_trades_sizes_view[...] = open_trades_sizes.new_zeros([]).where(
                close_trades_mask,
                right_cum_size_diffs.detach().where(reduce_trade_size_mask, cur_open_trades_sizes_view),
            )
            cur_open_trades_rates_view[...] = open_trades_rates.new_zeros([]).where(
                close_trades_mask, cur_open_trades_rates_view
            )
            assert not torch.any((cur_open_trades_sizes_view == 0) != (cur_open_trades_rates_view == 0))

            closed_trades_pl = torch.sum(
                closed_trades_sizes.abs_() / account_cur_rates[i] * (1 - open_trades_rates[i] / close_rates[i]), dim=0
            )
            logprob = z_logprob + cum_exec_logprobs[i]
            cost = torch.log1p_(closed_trades_pl / total_margin)

            surrogate_loss = (
                surrogate_loss
                - (logprob * torch.detach_(cost - baseline[i]) + cost)
                + (cost.detach() - baseline[i]) ** 2
            )

            loss = loss - cost.detach()
            total_margin = total_margin + closed_trades_pl
            assert torch.all(total_margin > 0)

            flip_pos_mask = close_trades_mask[-1] & (right_cum_size_diffs[-1] != 0)
            assert not torch.any(closeout_mask & flip_pos_mask)

            cur_open_trades_sizes_view[-1] = (
                right_cum_size_diffs[-1]
                .detach()
                .where(flip_pos_mask, exec_sizes.detach().where(new_pos_mask[i], cur_open_trades_sizes_view[-1]))
            )
            cur_open_trades_rates_view[-1] = open_rates[i].where(
                flip_pos_mask | new_pos_mask[i], cur_open_trades_rates_view[-1]
            )
            assert not torch.any((cur_open_trades_sizes_view == 0) != (cur_open_trades_rates_view == 0))

        assert not torch.any((open_trades_sizes == 0) != (open_trades_rates == 0))
        assert torch.all(torch.all(open_trades_sizes >= 0, dim=1) | torch.all(open_trades_sizes <= 0, dim=1))
        assert not torch.any(closeout_mask & (open_trades_sizes.abs().sum([0, 1]) != 0))
        total_margin = total_margin.detach()
        z_sample = z_sample.detach()
        assert not open_trades_sizes.requires_grad
        assert not open_trades_rates.requires_grad

        return surrogate_loss, loss, z_sample, total_margin, open_trades_sizes, open_trades_rates


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
