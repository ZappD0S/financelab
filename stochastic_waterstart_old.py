import os
from typing import Optional

import numpy as np
import torch
import torch.distributions
import torch.nn as nn


@torch.jit.script
def compute_loss(exec_probs, long_probs, fractions, prices, n_samples: int, device: torch.device):
    p_buy, p_sell = prices.unbind(-1)
    seq_len, batch_size = exec_probs.size(0), exec_probs.size(1)

    rows = torch.arange(batch_size).unsqueeze(-1)
    cols = torch.arange(n_samples)

    loss_terms = torch.empty(seq_len, batch_size, 2, 2, 2).to(device)

    # long, open
    loss_terms[..., 0, 0] = torch.stack((fractions * p_buy, (1 + fractions) / (fractions * p_buy)), dim=-1)
    # long, close
    loss_terms[..., 0, 1] = torch.stack((torch.zeros_like(fractions), -1 / p_sell), dim=-1)
    # short, open
    loss_terms[..., 1, 0] = torch.stack((fractions * p_sell, (1 - fractions) / (fractions * p_sell)), dim=-1)
    # short, close
    loss_terms[..., 1, 1] = torch.stack((torch.zeros_like(fractions), 1 / p_buy), dim=-1)

    pos_state_index = torch.zeros(batch_size, n_samples, dtype=torch.int64, device=device)
    pos_type = torch.zeros(batch_size, n_samples, dtype=torch.int64, device=device)

    exec_samples = torch.bernoulli(exec_probs.detach().unsqueeze(2).expand(-1, -1, n_samples, -1)).type(torch.int64)
    # BUG: così non funziona perché i samples non corrispondono alle probs
    exec_samples[-1, ..., 0] = 0
    exec_samples[-1, ..., 1] = 1

    ls_samples = torch.bernoulli(1 - long_probs.detach().unsqueeze(2).expand(-1, -1, n_samples)).type(torch.int64)

    sample_losses_builder = torch.zeros(batch_size, n_samples, seq_len // 2, 2, 2, device=device)

    sample_exec_probs = torch.zeros(seq_len, batch_size, n_samples, device=device)
    sample_ls_probs = torch.ones(seq_len, batch_size, n_samples, device=device)

    for i in range(seq_len):
        latest_pos = pos_state_index // 2
        pos_state = pos_state_index % 2

        # bisogna aggiornare i pos_type dove il pos_state == 0 (chiuso) e
        # l'exec_sample di apertura (index == 0) é 1

        closed_state_mask = pos_state == 0
        open_event_mask = exec_samples[i, :, :, 0] == 1
        update_pos_type_mask = closed_state_mask & open_event_mask

        # mask_rows, mask_cols = update_pos_type_mask.nonzero(as_tuple=True)
        mask_rows, mask_cols = update_pos_type_mask.nonzero().unbind(1)

        # 0 -> long, 1 -> short
        pos_type[update_pos_type_mask] = ls_samples[i, mask_rows, mask_cols]
        sample_ls_probs[i, mask_rows, mask_cols] = torch.where(
            pos_type == 0, long_probs[i, :, None], 1 - long_probs[i, :, None]
        )[mask_rows, mask_cols]

        # pos_state == 0 -> la posizione è chiusa, quindi vogliamo guardare il sample di apertura
        # pos_state == 1 -> las posizione è aperta, quindi vogliamo guardare il sample di chiusura
        event_mask = exec_samples[i, rows, cols, pos_state] == 1

        # mask_rows, mask_cols = event_mask.nonzero(as_tuple=True)
        mask_rows, mask_cols = event_mask.nonzero().unbind(1)

        sample_losses_builder[mask_rows, mask_cols, latest_pos[event_mask], :, pos_state[event_mask]] = loss_terms[
            i, mask_rows, :, pos_type[event_mask], pos_state[event_mask]
        ]

        # event_mask:          batch_size, n_samples, *
        # exec_probs: seq_len, batch_size,         *, 2

        sample_exec_probs[i] = torch.where(
            event_mask.unsqueeze(-1), exec_probs[i, :, None], 1 - exec_probs[i, :, None]
        )[rows, cols, pos_state]

        pos_state_index[event_mask] += 1

    sample_logprobs = sample_ls_probs.add(1e-45).log().sum(dim=0) + sample_exec_probs.add(1e-45).log().sum(dim=0)
    sample_losses = sample_losses_builder.sum(dim=-1).add(1e-45).log().sum(dim=(-2, -1))

    # return sample_logprobs * (sample_losses - sample_losses.mean(dim=1, keepdim=True))
    return sample_logprobs * sample_losses


# this method implements the variance reduction tecniques described in Pyro docs and in https://arxiv.org/abs/1506.05254
# by removing the all cost terms that are not downstream of (that do not depend on) a given stochastic node.
# Also a baseline is subtracted from the cost in the surrogate loss.
def compute_loss2(exec_prob_logits, short_prob_logits, fractions, prices, n_samples: int, device: torch.device):
    p_buy, p_sell = prices.unbind(-1)
    seq_len, batch_size = exec_prob_logits.size(0), exec_prob_logits.size(1)

    rows = torch.arange(batch_size).unsqueeze(-1)
    cols = torch.arange(n_samples)

    cost_terms = torch.empty(seq_len, batch_size, 2, 2, 2).to(device)

    # long, open
    cost_terms[..., 0, 0] = torch.stack((leverage * fractions * p_buy, (1 + fractions) / (fractions * p_buy)), dim=-1)
    # long, close
    cost_terms[..., 0, 1] = torch.stack((torch.zeros_like(fractions), -1 / p_sell), dim=-1)
    # short, open
    cost_terms[..., 1, 0] = torch.stack((fractions * p_sell, (1 - fractions) / (fractions * p_sell)), dim=-1)
    # short, close
    cost_terms[..., 1, 1] = torch.stack((torch.zeros_like(fractions), 1 / p_buy), dim=-1)

    pos_state = torch.zeros(batch_size, n_samples, dtype=torch.int64, device=device)
    # pos_type = torch.empty(batch_size, n_samples, dtype=torch.int64, device=device)
    pos_type = torch.full(-1, (batch_size, n_samples), dtype=torch.int64, device=device)

    exec_prob_distr = torch.distributions.Bernoulli(logits=exec_prob_logits).expand((-1, -1, n_samples, -1))
    short_prob_distr = torch.distributions.Bernoulli(logits=short_prob_logits).expand((-1, -1, n_samples))

    exec_samples = exec_prob_distr.sample().type(torch.int64)
    exec_samples[-1, ..., 0] = 0
    exec_samples[-1, ..., 1] = 1

    exec_logprobs = exec_prob_distr.log_prob(exec_samples)

    short_samples = short_prob_distr.sample().type(torch.int64)
    short_logprobs = short_prob_distr.log_prob(short_samples)

    pos_cost_terms = torch.empty(batch_size, n_samples, 2, 2)
    cum_logprobs = torch.zeros(batch_size, n_samples)
    losses = torch.zeros(batch_size, n_samples)

    for i in range(seq_len):
        # pos_state == 0 -> la posizione è chiusa, quindi vogliamo guardare il sample di apertura
        # pos_state == 1 -> las posizione è aperta, quindi vogliamo guardare il sample di chiusura
        event_mask = exec_samples[i, rows, cols, pos_state] == 1
        open_event_mask = event_mask & pos_state == 0
        close_event_mask = event_mask & pos_state == 1

        mask_rows, mask_cols = open_event_mask.nonzero().unbind(1)
        # 0 -> long, 1 -> short
        pos_type[open_event_mask] = short_samples[i, mask_rows, mask_cols]

        cum_logprobs[open_event_mask] += torch.where(
            pos_type == 1, short_logprobs[i], torch.log_(1 - short_logprobs[i].exp())
        )[open_event_mask]

        cum_logprobs += torch.where(event_mask.unsqueeze(-1), exec_logprobs[i], torch.log_(1 - exec_logprobs[i].exp()))[
            rows, cols, pos_state
        ]

        assert torch.all(pos_type[event_mask] != -1)
        mask_rows, mask_cols = event_mask.nonzero().unbind(1)
        pos_cost_terms[mask_rows, mask_cols, :, pos_state[event_mask]] = cost_terms[
            i, mask_rows, :, pos_type[event_mask], pos_state[event_mask]
        ]

        logprobs = cum_logprobs[close_event_mask]
        costs = pos_cost_terms[close_event_mask].sum(dim=-1).log_().sum(dim=-1)
        baselines = torch.sum(logprobs.detach().exp_() * costs.detach(), dim=1, keepdim=True)

        losses[close_event_mask] += logprobs * (costs.detach() - baselines) + costs

        pos_state[event_mask].add_(1).remainder_(2)

    return losses


def compute_loss3(
    logits: torch.Tensor, hidden_logprobs: torch.Tensor, prices: torch.Tensor, n_samples: int, device: torch.device
):
    # logits shape: (batch_size, seq_len, n_cur, 4)
    # hidden_logprobs shape: (batch_size, seq_len)
    # prices shape (batch_size, seq_len, 2)
    batch_size, seq_len, n_cur = logits.size(0), logits.size(1), logits.size(2)

    all_batch_inds = torch.arange(batch_size)
    all_sample_inds = torch.arange(n_samples)
    all_cur_inds = torch.arange(n_cur)

    exec_logits, short_logits, fractions = logits.split([2, 1, 1], dim=-1)

    exec_logits = exec_logits.unsqueeze(1).expand(-1, n_samples, -1, -1, -1)
    short_logits = short_logits.squeeze(-1).unsqueeze(1).expand(-1, n_samples, -1, -1)
    fractions = fractions.squeeze(-1)
    # hidden_logprobs = hidden_logprobs.unsqueeze(1).expand(-1, n_samples, -1)
    cum_hidden_logprobs = hidden_logprobs.cumsum(1).unsqueeze(1).expand(-1, n_samples, -1)

    exec_distr = torch.distributions.Bernoulli(logits=exec_logits)
    pos_type_distr = torch.distributions.Bernoulli(logits=short_logits)

    exec_samples = exec_distr.sample().type(torch.int64)
    exec_samples[:, :, -1, :, 0] = 0
    exec_samples[:, :-1, :, 1] = 1

    exec_logprobs = exec_distr.log_prob(exec_samples)

    pos_type_samples = pos_type_distr.sample().type(torch.int64)
    pos_type_logprobs = pos_type_distr.log_prob(pos_type_samples)

    pos_states = torch.zeros(batch_size, n_samples, n_cur, dtype=torch.int64, device=device)
    # pos_types = torch.full(-1, (batch_size, n_samples, n_cur), dtype=torch.int64, device=device)
    pos_types = torch.empty(batch_size, n_samples, n_cur, dtype=torch.int64, device=device)

    cum_portfolio_logprobs = torch.zeros(batch_size, n_samples, device=device)
    pos_exec_logprobs = torch.zeros(batch_size, n_samples, n_cur, device=device)

    cum_portfolio_values = torch.ones(batch_size, n_samples, device=device)
    pos_initial_values = torch.empty(batch_size, n_samples, n_cur, device=device)

    # profit/loss terms
    pos_pl_terms = torch.empty(batch_size, n_samples, n_cur, 2, device=device)

    losses = torch.zeros(batch_size, n_samples, device=device)
    # TODO: rename all variable where cur stands for current

    # for every position that we open we have to keep track of:
    # - the portfolio logprobs when the position was opened
    #   - we should take the value after we have added the logprob correspoding to the open event
    # - the logprobs of closing the position up to the event of closing
    # - the hidden state logprobs, starting from the following time point, and up to the event of closing

    # at all times we must:
    # - update the exec_logprobs, wheter a open or close event happened of not
    # - update the exec_hidden_logprobs
    # - updated the portfolio_logprobs adding the hidden state logprobs of the current
    #   time point.
    #

    # when we open a position we must:
    # - update the portfolio logprobs adding the values from the exec_logprobs
    # - compute the initial position value and store it
    # - store the initial position value logprobs, which is taken from the portfolio logprobs
    # - compute the profit/loss terms that depend on the opening price  and store them
    #   in the corresponding tensor (we set instead of adding, so that the old values get overriwritten)
    # - reset the exec_logprobs

    # when we close a position we must:
    # - update the portfolio logprobs adding the values from the exec_logprobs
    # - compute the profit_loss_logprobs by adding exec_hidden_logprobs, exec_logprobs and
    #   initial position value logprobs
    # compute the profit/loss terms that depend of the closing price and store them in the corresponding tensor
    # - compute the profit/loss values by adding the multiplying the profit/loss terms together and then
    #   by the initial position values. these are the costs that need to be added to the loss fuction.
    # - update the portfolio logprobs adding the values from the exec_logprobs
    # - update the portfolio values by adding back the initial position values and the profit/loss values

    for i in range(seq_len):
        # pos_state == 0 -> the position is closed, so we consider the open event sample
        # pos_state == 1 -> the position is open, so we consider the close event sample
        event_mask = exec_samples[all_batch_inds, all_sample_inds, i, all_cur_inds, pos_states] == 1
        open_event_mask = event_mask & pos_states == 0
        close_event_mask = event_mask & pos_states == 1

        pos_exec_logprobs += exec_logprobs[all_batch_inds, all_sample_inds, i, all_cur_inds, pos_states]

        mask_batch_inds, mask_sample_inds, mask_cur_inds = open_event_mask.nonzero().unbind(1)

        # where the position are opened, take into account the position type probs
        pos_exec_logprobs[open_event_mask] += pos_type_logprobs[mask_batch_inds, mask_sample_inds, i, mask_cur_inds]
        pos_types[open_event_mask] = pos_type_samples[mask_batch_inds, mask_sample_inds, i, mask_cur_inds]

        # compute the profit/loss terms due to opening
        open_event_pos_types = pos_types[open_event_mask]
        open_terms_prices = prices[:, None, i, open_event_pos_types]
        pos_pl_terms[mask_batch_inds, mask_sample_inds, mask_cur_inds, 0] = leverage * open_terms_prices

        coeffs = 1 / leverage + torch.tensor([1, -1], device=device)[open_event_pos_types]
        pos_pl_terms[mask_batch_inds, mask_sample_inds, mask_cur_inds, 1] = coeffs / open_terms_prices

        mask_batch_inds, mask_sample_inds, mask_cur_inds = close_event_mask.nonzero().unbind(1)

        # compute the profit/loss terms due to closing
        close_event_pos_types = pos_types[close_event_mask]
        close_term_prices = prices[:, None, i, close_event_pos_types]
        coeffs = torch.tensor([-1, 1], device=device)[close_event_pos_types]
        pos_pl_terms[mask_batch_inds, mask_sample_inds, mask_cur_inds, 1] += coeffs / close_term_prices

        pos_pl = pos_initial_values[close_event_mask] * pos_pl_terms[close_event_mask].prod(-1)

        for j in range(n_cur):
            cur_open_event_mask = open_event_mask[:, :, j]
            cur_fractions = fractions[:, i, j]

            mask_batch_inds, mask_sample_inds = cur_open_event_mask.nonzero().unbind(1)

            # update portfolio value subtracting amount used to open positions
            new_pos_initial_values = cur_fractions[cur_open_event_mask] * cum_portfolio_values[cur_open_event_mask]
            pos_initial_values[mask_batch_inds, mask_sample_inds, j] = new_pos_initial_values
            cum_portfolio_values[cur_open_event_mask] -= new_pos_initial_values

            # TODO: when we open a position, we have to save the current portfolio logprob

            # update
            cum_portfolio_logprobs

            pass

        # reset position probs where position where closed or opened
        pos_exec_logprobs[event_mask] = 0

        # pos_logprobs[event_mask] += hidden_logprobs[mask_batch_inds, i] + exec_logprobs[event_mask]

        # 0 -> long, 1 -> short

        # cur_portfolio_values_terms = torch.zeros(batch_size, n_samples, n_cur + 1, device=device)
        # cur_portfolio_values_terms[:, :, 0] = cum_portfolio_values

        # cur_fractions = fractions[:, :, i].clone()
        # cur_fractions[~open_event_mask] = 1
        # new_pos_initial_values = cum_portfolio_values.unsqueeze(-1) * cur_fractions.cumprod(-1)
        # pos_initial_values[open_event_mask] = new_pos_initial_values[open_event_mask]

        # cur_portfolio_values_terms[mask_batch_inds, mask_sample_inds, mask_cur_inds + 1] -= pos_initial_values[
        #     open_event_mask
        # ]


        # cur_logprobs_terms = torch.zeros(batch_size, n_samples, n_cur + 1, device=device)
        # cur_logprobs_terms[:, :, 0] = cum_logprobs
        # cur_logprobs_terms[:, :, 1] = hidden_logprobs[:, None, i]

        # BUG: since we are computing the logprobs from the thistribution, this is
        # not necessary
        cur_logprobs_terms[all_batch_inds, all_sample_inds, all_cur_inds + 1] += torch.where(
            event_mask.unsqueeze(-1), exec_logprobs[:, :, i], torch.log_(1 - exec_logprobs[:, :, i].exp())
        )[all_batch_inds, all_sample_inds, all_cur_inds, pos_states]

        # cur_logprobs[open_event_mask] += torch.where(
        #     pos_types == 1, short_logprobs[:, :, i], torch.log_(1 - short_logprobs[:, :, i].exp())
        # )[open_event_mask]

        cur_logprobs_terms[mask_batch_inds, mask_sample_inds, mask_cur_inds + 1] += torch.stack(
            [torch.log_(1 - pos_type_logprobs[:, :, i].exp()), pos_type_logprobs[:, :, i]], dim=-1
        )[mask_batch_inds, mask_sample_inds, mask_cur_inds, open_event_pos_types]

        # cur_logprobs_terms = hidden_logprobs[:, :, i, None] + cur_logprobs_terms.cumsum(-1)


        cur_portfolio_values_terms[mask_batch_inds, mask_sample_inds, mask_cur_inds + 1] -= pos_pl

        cur_cum_logprobs = cur_logprobs_terms.cumsum(-1)
        cur_cum_portfolio_values = cur_portfolio_values_terms.cumsum(-1)

        # cum_logprobs[mask_batch_inds, mask_sample_inds] + cur_logprobs[close_event_mask]
        # cur_cum_logprobs = cum_logprobs.unsqueeze(-1) + cur_logprobs_terms

        baselines = torch.sum(
            cur_cum_logprobs[mask_batch_inds, :, mask_cur_inds].detach().exp_()
            * cur_cum_portfolio_values[mask_batch_inds, :, mask_cur_inds].detach(),
            index=1,
            keepdim=True,
        )

        costs = cur_cum_portfolio_values[close_event_mask].detach()

        cur_losses = torch.zeros(batch_size, n_samples, n_cur)
        cur_losses[close_event_mask] = cur_cum_logprobs[close_event_mask] * (costs - baselines) + costs

        losses += cur_losses.sum(-1)

        # TODO: update cum_logprobs and cum_portfolio_values
        # TODO: update pos_state

    # pos_cost_terms = torch.empty(batch_size, n_samples, 2, 2)
    # cum_logprobs = torch.ones(batch_size, n_samples)
    # losses = torch.zeros(batch_size, n_samples)


save_path = "drive/My Drive/"
# save_path = "./"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

data = np.load(os.path.join(save_path, "train_data/train_data_10s.npz"))
prices = torch.from_numpy(data["prices"])
input = torch.from_numpy(data["input"])

leverage = 50

batch_size = 8
n_samples = 400
seq_len = 100

rows = torch.arange(batch_size).unsqueeze(-1)
cols = torch.arange(n_samples)

n_batches = (len(prices) - seq_len) // batch_size
batch_inds = torch.randperm(n_batches * batch_size).reshape(-1, batch_size)
batch_prices = prices[torch.arange(seq_len).unsqueeze(-1) + batch_inds[0]].to(device)

exec_probs = torch.rand(seq_len, batch_size, 2).to(device)
long_probs = torch.rand(seq_len, batch_size).to(device)
fractions = torch.rand(seq_len, batch_size).to(device)

losses = compute_loss(exec_probs, long_probs, fractions, batch_prices, n_samples, device)
