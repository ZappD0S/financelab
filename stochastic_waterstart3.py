import torch
import torch.jit
import torch.nn as nn


class LossEvaluator(nn.Module):
    def __init__(
        self, seq_len: int, batch_size: int, n_samples: int, n_cur: int, leverage: float,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.n_cur = n_cur
        self.leverage = leverage

    def forward(
        self,
        samples: torch.Tensor,
        fractions: torch.Tensor,
        x_logprobs: torch.Tensor,
        z_logprobs: torch.Tensor,
        rates: torch.Tensor,
        account_cur_rates: torch.Tensor,
        pos_states: torch.Tensor,
        pos_types: torch.Tensor,
        total_margin: torch.Tensor,
        open_pos_sizes: torch.Tensor,
        open_rates: torch.Tensor,
    ):
        # samples: (seq_len, 3, n_cur, n_samples, batch_size)
        # fractions: (seq_len, n_cur, n_samples, batch_size)
        # x_logprobs: (seq_len, 3, n_cur, n_samples, batch_size)
        # z_logprobs: (seq_len, n_samples, batch_size)
        # rates: (seq_len, 2, n_cur, batch_size)

        # this is the midpoint for the rate between
        # the base currency of the pair and the account currency (euro)
        # account_cur_rates: (seq_len, n_cur, batch_size)

        # pos_state: (n_cur, n_samples, batch_size)
        # pos_type: (n_cur, n_samples, batch_size)
        # total_margin: (n_samples, batch_size)
        # open_pos_sizes: (n_cur, n_samples, batch_size)
        # open_rates: (n_cur, n_samples, batch_size)

        # to broadcast with the n_samples dim
        rates = rates.unsqueeze(3)
        account_cur_rates = account_cur_rates.unsqueeze(2)

        all_pos_states = pos_states.new_zeros(self.seq_len, self.n_cur, self.n_samples, self.batch_size)
        all_pos_types = pos_types.new_zeros(self.seq_len, self.n_cur, self.n_samples, self.batch_size)
        all_total_margins = total_margin.new_ones(self.seq_len, self.n_samples, self.batch_size)
        all_open_pos_sizes = open_pos_sizes.new_zeros(self.seq_len, self.n_cur, self.n_samples, self.batch_size)
        all_open_rates = open_rates.new_zeros(self.seq_len, self.n_cur, self.n_samples, self.batch_size)

        long_pos_type_mask = pos_types == 0
        short_pos_type_mask = pos_types == 1

        cum_z_logprob = samples.new_zeros(self.n_samples, self.batch_size)

        cum_exec_logprobs = samples.new_zeros(self.n_cur, self.n_samples, self.batch_size)
        open_pos_type_logprobs = samples.new_zeros(self.n_cur, self.n_samples, self.batch_size)

        costs = samples.new_zeros(self.n_cur, self.n_samples, self.batch_size)
        loss = samples.new_zeros(self.n_samples, self.batch_size)

        for i in range(self.seq_len):
            open_pos_mask = pos_states == 1
            closed_pos_mask = pos_states == 0

            close_rates = torch.where(
                open_pos_mask & long_pos_type_mask,
                rates[i, 0],
                torch.where(open_pos_mask & short_pos_type_mask, rates[i, 1], rates.new_ones([])),
            )

            # NOTE: we don't need to use torch.where because open_pos_sizes is already zero
            # where there's no open position
            account_cur_open_pos_sizes = open_pos_sizes / account_cur_rates[i]

            open_pos_pl = self.leverage * account_cur_open_pos_sizes * (1 - open_rates / close_rates)

            account_value = total_margin + open_pos_pl.sum(0)
            margin_used = account_cur_open_pos_sizes.sum(0)
            closeout_mask = account_value < 0.5 * margin_used

            cum_z_logprob = cum_z_logprob + z_logprobs[i]

            open_samples, close_samples, pos_type_samples = samples[i].unbind()
            open_mask = closed_pos_mask & ~closeout_mask & (open_samples == 1)
            close_mask = open_pos_mask & (closeout_mask | (close_samples == 1))

            # NOTE: when using the % operator torch.jit.trace causes trouble.
            # Also, this way the operation becomes inplace
            pos_states = torch.where(open_mask | close_mask, (pos_states + 1).remainder_(2), pos_states)
            all_pos_states[i] = pos_states.detach()

            open_logprobs, close_logprobs, pos_type_logprobs = x_logprobs[i].unbind()

            # NOTE: when the closeout is triggered the close_sample gets shadowed,
            # so we ignore its logprob even if we were going to close
            exec_logprobs = torch.where(
                ~closeout_mask & closed_pos_mask,
                open_logprobs,
                torch.where(~closeout_mask & open_pos_mask, close_logprobs, close_logprobs.new_zeros([])),
            )
            cum_exec_logprobs = cum_exec_logprobs + exec_logprobs

            open_pos_type_logprobs = pos_type_logprobs.where(open_mask, open_pos_type_logprobs)

            pos_types = pos_type_samples.type_as(pos_types).where(open_mask, pos_types)
            all_pos_types[i] = pos_types.detach()
            # NOTE: 0 -> long, 1 -> short
            long_pos_type_mask = pos_types == 0
            short_pos_type_mask = pos_types == 1

            open_rates = torch.where(
                open_mask & long_pos_type_mask,
                rates[i, 1],
                torch.where(open_mask & short_pos_type_mask, rates[i, 0], open_rates),
            )
            all_open_rates[i] = open_rates.detach()

            costs = open_pos_pl.where(close_mask, open_pos_pl.new_zeros([]))

            for j in range(self.n_cur):
                cur_open_mask = open_mask[j]

                account_cur_open_pos_sizes = open_pos_sizes / account_cur_rates[i]
                margin_used = account_cur_open_pos_sizes.sum(0)
                margin_available = total_margin - margin_used
                new_pos_size = fractions[i, j] * margin_available * account_cur_rates[i, j]
                open_pos_sizes[j] = new_pos_size.where(cur_open_mask, open_pos_sizes[j])

                # open_pos_size_logprobs[j] = margin_available_logprobs[j].where(cur_open_mask, open_pos_size_logprobs[j])

                cur_close_mask = close_mask[j]

                cost = costs[j]
                # TODO: what if we used as baseline the pos_pl of the most
                # recently closed pos? in that case we'd have to keep track
                # of a tensor called last_costs
                baseline = costs[j].mean(0, keepdim=True)
                # cum_z_logprob + open_pos_size_logprobs[j] + open_pos_type_logprobs[j] + cum_close_logprobs[j]
                cost_logprob = cum_z_logprob + cum_exec_logprobs[j] + open_pos_type_logprobs[j]

                loss = loss + torch.where(
                    cur_close_mask, cost_logprob * torch.detach(cost - baseline) + cost, loss.new_zeros([])
                )
                # reset sizes of positions that were closed
                open_pos_sizes[j] = open_pos_sizes.new_zeros([]).where(cur_close_mask, open_pos_sizes[j])

                total_margin = total_margin + open_pos_pl[j].where(cur_close_mask, open_pos_pl.new_zeros([]))

            all_open_pos_sizes[i] = open_pos_sizes.detach()
            all_total_margins[i] = total_margin.detach()

            cum_exec_logprobs = cum_exec_logprobs.new_zeros([]).where(close_mask, cum_exec_logprobs)

        return loss, all_pos_states, all_pos_types, all_total_margins, all_open_pos_sizes, all_open_rates


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
