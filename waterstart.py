import time
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
    input_probs: torch.Tensor, dim: int, initial_probs: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if initial_probs is None:
        shape = input_probs.shape
        initial_probs = torch.ones(shape[:dim] + shape[dim + 1 :], device=device)

    shape = list(input_probs.shape)
    shape.insert(0, shape.pop(dim))

    not_done_probs = torch.cumprod(1 - input_probs, dim=dim).view(shape)
    output_probs = input_probs.view(shape) * initial_probs
    output_probs[1:] *= not_done_probs[:-1]
    initial_probs *= not_done_probs[-1]

    return output_probs.view_as(input_probs), initial_probs


def compute_open_data(inds: torch.Tensor, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = inds.nelement()
    scalar_one = torch.tensor(1.0, device=device)
    cum_prob = torch.zeros(batch_size, device=device)

    n_chunks = min(open_max_chunks, (input.size(0) - inds.max()) // open_chunk_size)
    assert n_chunks > 0

    open_probs = torch.empty(batch_size, n_chunks * open_chunk_size, 2, device=device)
    open_hidden = torch.empty(batch_size, n_chunks * open_chunk_size, pos_opener.rnn.hidden_size, device=device)
    hx = None

    last_not_exec_probs = None
    base_slice = torch.arange(open_chunk_size)

    chunk = 0
    while True:
        print(chunk)
        chunk_slice = chunk * open_chunk_size + base_slice
        chunk_open_slices = inds.view(-1, 1) + chunk_slice

        # shape: (batch, chunck_size, feature)
        chunk_open_input = input[chunk_open_slices, ...]
        chunk_open_input[:, :, 0] -= chunk_open_input[:, 0, None, 0]
        chunk_open_input = chunk_open_input.to(device=device)

        chunk_open_probs, chunk_open_hidden, hx = pos_opener(chunk_open_input, hx)
        chunk_exec_probs, _ = chunk_open_probs.unbind(2)
        chunk_exec_probs, last_not_exec_probs = compute_compound_probs(chunk_exec_probs, 1, last_not_exec_probs)

        cum_prob += torch.sum(chunk_exec_probs[:, :-1], dim=1)
        prob_saturated = cum_prob.allclose(scalar_one)
        reached_end = chunk == n_chunks - 1

        if reached_end and not prob_saturated:
            chunk_exec_probs[:, -1] = 1 - cum_prob

        open_probs[:, chunk_slice] = chunk_open_probs
        open_hidden[:, chunk_slice] = chunk_open_hidden

        if reached_end or prob_saturated:
            break
        else:
            cum_prob += chunk_exec_probs[:, -1]

        chunk += 1

    open_probs = open_probs[:, : (chunk + 1) * open_chunk_size]
    open_hidden = open_hidden[:, : (chunk + 1) * open_chunk_size]

    return open_probs, open_hidden


def compute_loss(inds: torch.Tensor, input: torch.Tensor, logp: torch.Tensor):
    assert inds.ndim == 1
    probs, open_hidden = compute_open_data(inds, input)
    open_probs, ls_probs = probs.unbind(2)

    batch_shape = open_probs.shape
    h0 = hidden_state_adapter(open_hidden.view(-1, *open_hidden.shape[2:]))
    h0 = h0.expand(pos_closer.rnn.num_layers, *h0.shape).contiguous()
    # h0: (num_layers, batch, hidden_size)
    hx = (h0, torch.zeros_like(h0))

    open_slice = torch.arange(batch_shape[1])
    open_slices = inds.view(-1, 1) + open_slice

    scalar_one = torch.tensor(1.0, device=device)
    cum_probs = torch.zeros(batch_shape, device=device)
    last_not_close_probs = torch.ones(batch_shape, device=device)

    n_chunks = (logp.size(0) - max(s[-1] for s in open_slices)) // close_chunk_size
    assert n_chunks > 0

    loss = 0.0

    open_buy_logp = logp[open_slices, 0].unsqueeze(-1).to(device=device)
    open_sell_logp = logp[open_slices, 1].unsqueeze(-1).to(device=device)

    open_probs = open_probs.unsqueeze(-1)
    ls_probs = ls_probs.unsqueeze(-1)

    base_slice = torch.arange(close_chunk_size)

    chunk = 0
    while True:
        print(chunk)
        chunk_slice = chunk * close_chunk_size + base_slice
        chunk_close_slices = inds.view(-1, 1, 1) + open_slice.view(-1, 1) + chunk_slice

        # (batch, n_chunks * open_chunk_size, close_chunk_size, feature)
        chunk_close_input = input[chunk_close_slices, ...]
        chunk_close_input[:, :, :, 0] -= chunk_close_input[:, :, 0, None, 0]
        chunk_close_input = chunk_close_input.to(device=device)

        chunk_close_probs, hx = pos_closer(chunk_close_input.view(-1, *chunk_close_input.shape[2:]), hx)
        (chunk_close_probs,) = chunk_close_probs.view(*batch_shape, *chunk_close_probs.shape[1:]).unbind(3)

        chunk_close_probs, last_not_close_probs = compute_compound_probs(chunk_close_probs, 2, last_not_close_probs)

        chunk_close_buy_logp = logp[chunk_close_slices, 0].to(device=device)
        chunk_close_sell_logp = logp[chunk_close_slices, 1].to(device=device)

        cum_probs += torch.sum(chunk_close_probs[:, :, :-1], dim=2)

        prob_saturated = cum_probs.allclose(scalar_one)
        reached_end = chunk == n_chunks - 1

        if reached_end and not prob_saturated:
            chunk_close_probs[:, :, -1] = 1 - cum_probs

        loss += torch.sum(
            open_probs
            * chunk_close_probs
            * (
                ls_probs * (chunk_close_sell_logp - open_buy_logp)
                + (1 - ls_probs) * (open_sell_logp - chunk_close_buy_logp)
            )
        )

        if reached_end or prob_saturated:
            break
        else:
            cum_probs += chunk_close_probs[:, :, -1]

        chunk += 1

    return -loss


open_chunk_size = 1000
open_max_chunks = 5

close_chunk_size = 10
close_max_chunks = 500

open_hidden_size = 30
close_hidden_size = 30
num_layers = 1

device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

pos_opener = PositionOpener(2, open_hidden_size, num_layers).to(device=device)
pos_closer = PositionCloser(2, close_hidden_size, num_layers).to(device=device)
hidden_state_adapter = nn.Linear(open_hidden_size, close_hidden_size).to(device=device)

data = np.load("train_data.npz")
# data = np.load("drive/My Drive/train_data/train_data.npz")

logp = torch.from_numpy(data["logp"])
input = torch.from_numpy(data["input"])

inds = torch.arange(8)

loss = compute_loss(inds, input, logp)
