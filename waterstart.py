import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Optional, Tuple


class PositionOpener(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super(PositionOpener, self).__init__()
        # TODO: maybe add a trainable initial hidden state. Use nn.Parameter
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lin_hidden_to_prob = nn.Linear(hidden_size, 2)

    def forward(self, input: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        hidden, hx = self.rnn(input, hx)
        return torch.sigmoid(self.lin_hidden_to_prob(hidden)), hidden, hx


class PositionCloser(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super(PositionCloser, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lin_close_h_to_prob = nn.Linear(hidden_size, 1)

    def forward(self, input: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        hidden, hx = self.rnn(input, hx)
        return torch.sigmoid(self.lin_close_h_to_prob(hidden)), hx


def compute_compound_probs(
    input_probs: torch.Tensor, dim: int, initial_probs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    perm = list(range(input_probs.ndim))
    perm.insert(0, perm.pop(dim))

    inv_perm = torch.empty(input_probs.ndim, dtype=int)
    inv_perm[perm] = torch.arange(input_probs.ndim)

    not_done_probs = torch.cumprod(1 - input_probs, dim=dim).permute(perm)
    output_probs = input_probs.permute(perm) * initial_probs
    output_probs[1:] *= not_done_probs[:-1]

    return output_probs.permute(*inv_perm), initial_probs * not_done_probs[-1]


def compute_open_data(batch_inds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = batch_inds.nelement()
    cum_prob = torch.zeros(batch_size, device=device)

    n_chunks = min(open_max_chunks, (input.size(0) - batch_inds.max()) // open_chunk_size)
    assert n_chunks > 0

    open_probs = torch.zeros(batch_size, n_chunks * open_chunk_size, 2, device=device)
    open_hidden = torch.zeros(batch_size, n_chunks * open_chunk_size, pos_opener.rnn.hidden_size, device=device)
    open_mask = torch.zeros(batch_size, n_chunks * open_chunk_size, dtype=torch.bool, device=device)
    hx = None

    last_not_exec_probs = torch.ones(batch_size, device=device)
    base_inds = torch.arange(open_chunk_size)
    batch_mask = torch.ones(batch_size, dtype=bool, device=device)

    chunk = 0
    while True:
        chunk_inds = chunk * open_chunk_size + base_inds
        inds = batch_inds.unsqueeze(-1) + chunk_inds
        chunk_open_input = input[inds[batch_mask], :].to(device=device)

        chunk_open_probs, chunk_open_hidden, hx = pos_opener(chunk_open_input, hx)
        chunk_exec_probs, chunk_ls_probs = chunk_open_probs.unbind(2)

        chunk_exec_probs, last_not_exec_probs = compute_compound_probs(chunk_exec_probs, 1, last_not_exec_probs)

        chunk_cum_probs = cum_prob.unsqueeze(-1) + chunk_exec_probs.cumsum(dim=1)
        # chunk_open_mask = ~chunk_cum_probs.isclose(scalar_one)
        chunk_open_mask = torch.empty_like(chunk_cum_probs, dtype=bool)
        chunk_open_mask[:, 0] = True
        chunk_open_mask[:, 1:] = ~chunk_cum_probs[:, :-1].isclose(scalar_one)

        prob_not_saturated = chunk_open_mask.all(dim=1)

        reached_end = chunk == n_chunks - 1
        all_prob_saturated = not prob_not_saturated.any()

        if reached_end and not all_prob_saturated:
            chunk_exec_probs[prob_not_saturated, -1] = 1 - chunk_cum_probs[prob_not_saturated, -2]

        idx = batch_mask.nonzero(as_tuple=False)
        open_probs[idx, chunk_inds] = torch.stack((chunk_exec_probs, chunk_ls_probs), dim=2)
        open_mask[idx, chunk_inds] = chunk_open_mask
        open_hidden[idx, chunk_inds] = chunk_open_hidden

        if reached_end or all_prob_saturated:
            break

        batch_mask[batch_mask] = prob_not_saturated
        cum_prob = chunk_cum_probs[prob_not_saturated, -1]
        last_not_exec_probs = last_not_exec_probs[prob_not_saturated]
        hx = torch.stack(hx)[:, :, prob_not_saturated].unbind()

        chunk += 1

    open_probs = open_probs[:, : (chunk + 1) * open_chunk_size]
    open_hidden = open_hidden[:, : (chunk + 1) * open_chunk_size]
    open_mask = open_mask[:, : (chunk + 1) * open_chunk_size]

    return open_probs, open_hidden, open_mask


def compute_loss(batch_inds: torch.Tensor):
    assert batch_inds.ndim == 1
    probs, open_hidden, open_mask = compute_open_data(batch_inds)

    n_chunks = (probs.size(1) + loss_chunk_size - 1) // loss_chunk_size

    total_loss = 0.0

    global tot_prob, deltas
    tot_prob = 0.0
    deltas[:] = 0.0

    for chunk in range(n_chunks):
        print(f"\r{chunk}/{n_chunks}", end="", flush=True)
        chunk_loss_inds = torch.arange(loss_chunk_size) + chunk * loss_chunk_size

        chunk_loss_probs = probs[:, chunk_loss_inds]
        chunk_loss_open_hidden = open_hidden[:, chunk_loss_inds]
        chunk_loss_open_mask = open_mask[:, chunk_loss_inds]

        if not chunk_loss_open_mask.any():
            break

        inds = batch_inds.unsqueeze(-1) + chunk_loss_inds
        loss = compute_chunk_loss(inds, chunk_loss_probs, chunk_loss_open_hidden, chunk_loss_open_mask)

        loss.backward(retain_graph=True)
        total_loss += loss.item()
        del loss

    print()
    print("tot_prob:", tot_prob / batch_size)
    print("deltas:", deltas)
    return total_loss


def compute_chunk_loss(
    batch_inds: torch.Tensor, probs: torch.Tensor, open_hidden: torch.Tensor, open_mask: torch.Tensor,
) -> torch.Tensor:
    global tot_prob, deltas
    open_probs, ls_probs = probs.unbind(2)

    open_probs = open_probs[open_mask].unsqueeze(-1)
    ls_probs = ls_probs[open_mask].unsqueeze(-1)

    h0 = hidden_state_adapter(open_hidden[open_mask])
    h0 = h0.expand(pos_closer.rnn.num_layers, *h0.shape).contiguous()
    hx = (h0, torch.zeros_like(h0))

    batch_mask = open_mask.clone()

    cum_prob = torch.zeros(open_mask.sum(), device=device)
    last_not_close_probs = torch.ones(open_mask.sum(), device=device)

    n_chunks = min(close_max_chunks, (logp.size(0) - batch_inds[:, -1].max()) // close_chunk_size)
    assert n_chunks > 0

    loss = 0.0

    open_logp = logp[batch_inds[batch_mask], :].unsqueeze(1)
    open_buy_logp, open_sell_logp = open_logp.to(device=device).unbind(2)

    base_inds = torch.arange(close_chunk_size)

    chunk = 0
    while True:
        chunk_inds = chunk * close_chunk_size + base_inds
        inds = batch_inds.unsqueeze(-1) + chunk_inds
        chunk_close_input = input[inds[batch_mask], :].to(device=device)

        chunk_close_probs, hx = pos_closer(chunk_close_input, hx)
        (chunk_close_probs,) = chunk_close_probs.unbind(2)

        chunk_close_probs, last_not_close_probs = compute_compound_probs(chunk_close_probs, 1, last_not_close_probs)

        chunk_cum_probs = cum_prob.unsqueeze(-1) + chunk_close_probs.cumsum(dim=1)
        # chunk_close_mask = ~chunk_cum_probs.isclose(scalar_one)
        chunk_close_mask = torch.empty_like(chunk_close_probs, dtype=bool)
        chunk_close_mask[:, 0] = True
        chunk_close_mask[:, 1:] = ~chunk_cum_probs[:, :-1].isclose(scalar_one)

        prob_not_saturated = chunk_close_mask.all(dim=1)

        reached_end = chunk == n_chunks - 1
        all_prob_saturated = not prob_not_saturated.any()

        if reached_end and not all_prob_saturated:
            chunk_close_probs[prob_not_saturated, -1] = 1 - chunk_cum_probs[prob_not_saturated, -2]

        chunk_close_buy_logp, chunk_close_sell_logp = logp[inds[batch_mask], :].to(device=device).unbind(2)

        with torch.no_grad():
            tot_prob += (open_probs * chunk_close_probs)[chunk_close_mask].sum().item()

            exec_probs = open_probs * chunk_close_probs
            exec_probs[~chunk_close_mask] = 0
            deltas_terms_ = exec_probs * chunk_inds.to(dtype=torch.get_default_dtype(), device=device)
            deltas_terms = torch.zeros_like(batch_mask, dtype=torch.get_default_dtype())
            deltas_terms[batch_mask] = deltas_terms_.sum(axis=1)
            deltas += deltas_terms.sum(axis=1).detach()

        loss_terms = (
            open_probs
            * chunk_close_probs
            * (
                ls_probs * (chunk_close_sell_logp - open_buy_logp)
                + (1 - ls_probs) * (open_sell_logp - chunk_close_buy_logp)
            )
        )

        loss += loss_terms[chunk_close_mask].sum()

        if reached_end or all_prob_saturated:
            break

        cum_prob = chunk_cum_probs[prob_not_saturated, -1]
        batch_mask[batch_mask] = prob_not_saturated
        hx = torch.stack(hx)[:, :, prob_not_saturated].unbind()
        last_not_close_probs = last_not_close_probs[prob_not_saturated]
        open_probs = open_probs[prob_not_saturated]
        ls_probs = ls_probs[prob_not_saturated]

        open_buy_logp = open_buy_logp[prob_not_saturated]
        open_sell_logp = open_sell_logp[prob_not_saturated]

        chunk += 1

    return -loss


save_path = "drive/My Drive/"
# save_path = "./"

batch_size = 128

open_chunk_size = 2000
open_max_chunks = 25

close_chunk_size = 2000
close_max_chunks = 25

loss_chunk_size = 16

num_layers = 1
open_hidden_size = 30
close_hidden_size = 30

torch.set_default_dtype(torch.float32)

device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# this is needed for allclose check
scalar_one = torch.tensor(1.0, device=device)

tot_prob = 0.0
deltas = torch.zeros(batch_size, device=device)

try:
    modulelist = torch.load(os.path.join(save_path, "nn_weights/waterstart/modulelist.pth"))
    print("modulelist found")
except FileNotFoundError:
    modulelist = nn.ModuleList(
        [
            PositionOpener(2, open_hidden_size, num_layers),
            nn.Linear(open_hidden_size, close_hidden_size),
            PositionCloser(2, close_hidden_size, num_layers),
        ]
    )

modulelist.to(device=device)
pos_opener, hidden_state_adapter, pos_closer = modulelist

data = np.load(os.path.join(save_path, "train_data/train_data.npz"))

logp = torch.from_numpy(data["logp"])
input = torch.from_numpy(data["input"])

max_length = open_max_chunks * open_chunk_size + close_max_chunks * close_chunk_size

n_batches = (input.size(0) - max_length) // batch_size

# inds = torch.empty(n_batches, batch_size, dtype=int)
# inds.flatten()[torch.randperm(inds.nelement())] = torch.arange(inds.nelement())
inds = torch.arange(n_batches * batch_size).reshape(-1, batch_size)
inds = inds.flatten()[torch.randperm(inds.nelement())].view_as(inds)
optimizer = torch.optim.Adam(modulelist.parameters(), lr=0.001)

max_count = 5
count = 0
min_loss = float("inf")

for batch_inds in inds:
    optimizer.zero_grad()
    loss = compute_loss(batch_inds)
    optimizer.step()
    print("loss:", loss)
    count += 1
    print("count:", count)

    new_min_loss = loss < min_loss
    reached_max_count = count == max_count

    if new_min_loss or reached_max_count:
        print("saving..")
        torch.save(modulelist, os.path.join(save_path, "nn_weights/waterstart/modulelist.pth"))

        if new_min_loss:
            min_loss = loss

        count = 0
