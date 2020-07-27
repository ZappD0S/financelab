import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Optional


def compute_compound_probs(input_probs: torch.Tensor, dim: int, initial_probs: Optional[torch.Tensor] = None):
    if initial_probs is None:
        shape = input_probs.shape
        initial_probs = torch.ones(shape[:dim] + shape[dim + 1 :])

    shape = list(input_probs.shape)
    shape.insert(0, shape.pop(dim))

    not_done_probs = torch.cumprod(1 - input_probs, dim=dim).view(shape)
    output_probs = input_probs.view(shape) * initial_probs
    output_probs[1:] *= not_done_probs[:-1]
    initial_probs *= not_done_probs[-1]

    return output_probs.view_as(input_probs), initial_probs


class PositionOpener(nn.Module):
    def __init__(self, hidden_size: int, chunk_size: int, max_chunks: int):
        super(PositionOpener, self).__init__()
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        # TODO: we need a trainable initial hidden state. Use nn.Parameter
        self.lstm = nn.LSTM(2, hidden_size, batch_first=True)
        self.lin_hidden_to_prob = nn.Linear(hidden_size, 2)

    def forward(self, inds: torch.Tensor, input: torch.Tensor):
        batch_size = inds.nelement()
        cum_prob = torch.zeros(batch_size)

        n_chunks = min(self.max_chunks, (input.size(0) - inds.max()) // self.chunk_size)
        assert n_chunks > 0

        open_probs = torch.empty(batch_size, n_chunks * self.chunk_size)
        ls_probs = torch.empty(batch_size, n_chunks * self.chunk_size)
        open_output = torch.empty(batch_size, n_chunks * self.chunk_size, self.lstm.hidden_size)
        hx = None

        last_not_open_probs = None

        for offset in range(n_chunks):
            chunk_slice = torch.arange(offset * self.chunk_size, (offset + 1) * self.chunk_size)
            chunk_open_slices = [i + chunk_slice for i in inds]

            # shape: (batch, chunck_size, feature)
            chunk_open_input = input[chunk_open_slices, ...]
            chunk_open_input[:, :, 0] -= chunk_open_input[:, 0, None, 0]

            output, hx = self.lstm(chunk_open_input, hx)
            chunk_ls_probs, chunk_open_probs = torch.sigmoid(self.lin_hidden_to_prob(output)).unbind(2)
            chunk_open_probs, last_not_open_probs = compute_compound_probs(chunk_open_probs, 1, last_not_open_probs)

            open_probs[:, chunk_slice] = chunk_open_input
            ls_probs[:, chunk_slice] = chunk_ls_probs
            open_output[:, chunk_slice] = output

            cum_prob += torch.sum(chunk_open_probs, dim=1)
            if cum_prob.allclose(torch.tensor(1.0)):
                break

        open_probs[:, -1] = 1 - cum_prob
        open_slices = [i + torch.arange(n_chunks * self.chunk_size) for i in inds]

        return open_probs, ls_probs, open_output, open_slices


class PositionCloser(nn.Module):
    def __init__(self, close_hidden_size: int, open_hidden_size: int, chunk_size: int):
        super(PositionCloser, self).__init__()
        self.chunk_size = chunk_size
        self.lstm = nn.LSTM(2, close_hidden_size, batch_first=True)
        self.lin_close_h_to_prob = nn.Linear(close_hidden_size, 1)
        self.lin_open_h_to_close_h = nn.Linear(open_hidden_size, close_hidden_size)

    def forward(
        self,
        inds: torch.Tensor,
        input: torch.Tensor,
        p: torch.Tensor,
        ls_probs: torch.Tensor,
        open_probs: torch.Tensor,
        open_slices: torch.Tensor,
        open_output: torch.Tensor,
    ):
        batch_shape = open_probs.shape
        h0 = self.lin_open_h_to_close_h(open_output.view(-1, open_output.shape[2:]))
        h0 = h0.expand(self.lstm.num_layers, *h0.shape).contiguous()
        # h0: (num_layers, batch, hidden_size)
        hx = (h0, torch.zeros_like(h0))

        cum_probs = torch.zeros(batch_shape)
        last_not_close_probs = torch.ones(batch_shape)

        n_chunks = (p.size(0) - max(s[-1] for s in open_slices)) // self.chunk_size
        assert n_chunks > 0

        loss = 0.0

        open_buy_logp = torch.log(p[open_slices, 0]).expand(*batch_shape, self.chunk_size)
        open_sell_logp = torch.log(p[open_slices, 1]).expand(*batch_shape, self.chunk_size)

        open_probs = open_probs.expand(*batch_shape, self.chunk_size)
        ls_probs = ls_probs.expand(*batch_shape, self.chunk_size)

        for offset in range(n_chunks):
            chunk_slice = torch.arange(offset * self.chunk_size, (offset + 1) * self.chunk_size)
            chunk_close_slices = [[i + j + chunk_slice for j in range(batch_shape[1])] for i in inds]

            chunk_close_input = input[chunk_close_slices, ...]
            chunk_close_input[:, :, :, 0] -= chunk_close_input[:, :, 0, None, 0]

            output, hx = self.lstm(chunk_close_input.view(-1, *chunk_close_input.shape[2:]), hx)
            chunk_close_probs = torch.sigmoid(self.lin_close_h_to_prob(output))
            chunk_close_probs = output.view(*batch_shape, *chunk_close_probs.shape[1:])

            chunk_close_probs, last_not_close_probs = compute_compound_probs(chunk_close_probs, 1, last_not_close_probs)

            chunk_close_buy_logp = torch.log(p[chunk_close_slices, 0])
            chunk_close_sell_logp = torch.log(p[chunk_close_slices, 1])

            cum_probs += torch.sum(chunk_close_probs, dim=2)

            prob_saturated = cum_probs.allclose(torch.tensor(1.0))

            if not prob_saturated and offset == n_chunks - 1:
                chunk_close_probs[:, :, -1] = 1 - cum_probs

            loss += torch.sum(
                open_probs
                * chunk_close_probs
                * (
                    ls_probs * (chunk_close_sell_logp - open_buy_logp)
                    + (1 - ls_probs) * (open_sell_logp - chunk_close_sell_logp)
                )
            )

            if prob_saturated:
                break

        return -loss
