import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable


class PositionOpener(nn.Module):
    def __init__(self, hidden_size: int, chunk_size: int, max_chunks: int):
        super(PositionOpener, self).__init__()
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        # TODO: we need a trainable initial hidden state. Use nn.Parameter
        self.lstm = nn.LSTM(2, hidden_size, batch_first=True)
        self.lin_hidden_to_prob = nn.Linear(hidden_size, 2)

    def forward(self, inds: torch.Tensor, input: torch.Tensor):
        # questo magari lo facciamo fuori
        open_probs = torch.Tensor()
        ls_probs = torch.Tensor()
        open_hx = torch.Tensor()
        hx = None

        batch_size = inds.nelement()
        cum_prob = torch.zeros(batch_size)

        n_chunks = min(self.max_chunks, (input.size(0) - inds.max()) // self.chunk_size)
        assert n_chunks > 0

        last_not_open_probs = torch.ones(batch_size)

        for offset in range(n_chunks):
            chunk_open_slices = [range(i + offset * self.chunk_size, i + (offset + 1) * self.chunk_size) for i in inds]

            # shape: (batch, chunck_size, feature)
            chunk_open_input = input[chunk_open_slices, ...]
            chunk_open_input[:, :, 0] -= chunk_open_input[:, 0, None, 0]

            output, hx = self.lstm(chunk_open_input, hx)
            chunk_ls_probs, chunk_open_probs = torch.sigmoid(self.lin_hidden_to_prob(output)).unbind(2)
            not_open_probs = torch.cumprod(1 - chunk_open_probs, dim=1)
            chunk_open_probs *= last_not_open_probs
            chunk_open_probs[:, 1:] *= not_open_probs[:, :-1]
            last_not_open_probs *= not_open_probs[:, -1]

            open_probs = torch.cat((open_probs, chunk_open_probs), dim=1)
            ls_probs = torch.cat((ls_probs, chunk_ls_probs), dim=1)
            open_hx = torch.cat((open_hx, hx), dim=1)

            cum_prob += torch.sum(chunk_open_probs, dim=1)
            if cum_prob.allclose(torch.tensor(1.0)):
                break

        open_slices = [range(i, i + n_chunks * self.chunk_size) for i in inds]

        open_probs[:, -1] = 1 - cum_prob
        return open_probs, ls_probs, open_hx, open_slices


class PositionCloser(nn.Module):
    def __init__(self, hidden_size: int, chunk_size: int):
        super(PositionCloser, self).__init__()
        self.chunk_size = chunk_size
        self.lstm = nn.LSTM(2, hidden_size, batch_first=True)
        self.lin_hidden_to_prob = nn.Linear(hidden_size, 1)

    def forward(
        self,
        inds: torch.Tensor,
        input: torch.Tensor,
        p: torch.Tensor,
        ls_probs: torch.Tensor,
        open_probs: torch.Tensor,
        open_slices: torch.Tensor,
        open_hx: torch.Tensor,
    ):
        batch_shape = open_probs.shape
        all_probs = torch.Tensor()
        # (batch_size, T, hidden_state)
        hx = open_hx.view(-1, *open_hx.shape[2:])
        cum_prob = torch.zeros(batch_shape)
        last_not_done_probs = torch.ones(batch_shape)

        n_chunks = (p.size(0) - max(s[-1] for s in open_slices)) // self.chunk_size
        assert n_chunks > 0

        loss = 0.0

        open_buy_logp = torch.log(p[open_slices, 0]).expand(*batch_shape, self.chunk_size)
        open_sell_logp = torch.log(p[open_slices, 1]).expand(*batch_shape, self.chunk_size)

        open_probs = open_probs.expand(*batch_shape, self.chunk_size)
        ls_probs = ls_probs.expand(*batch_shape, self.chunk_size)

        for offset in range(n_chunks):
            chunk_close_slices = [
                [
                    range(i + j + offset * self.chunk_size, i + j + (offset + 1) * self.chunk_size)
                    for j in range(batch_shape[1])
                ]
                for i in inds
            ]

            chunk_close_input = input[chunk_close_slices, ...]
            chunk_close_input[:, :, :, 0] -= chunk_close_input[:, :, 0, None, 0]

            output, hx = self.lstm(chunk_close_input.view(-1, *chunk_close_input.shape[2:]), hx)
            output = output.view(*batch_shape, *output.shape[1:])

            chunk_close_probs = torch.sigmoid(self.lin_hidden_to_prob(output))
            not_done_probs = torch.cumprod(1 - chunk_close_probs, dim=1)
            chunk_close_probs *= last_not_done_probs
            chunk_close_probs[:, :, 1:] *= not_done_probs[:, :, :-1]
            last_not_done_probs *= not_done_probs[:, :, -1]

            chunk_close_buy_logp = torch.log(p[chunk_close_slices, 0])
            chunk_close_sell_logp = torch.log(p[chunk_close_slices, 1])

            cum_prob += torch.sum(chunk_close_probs, dim=2)

            prob_saturated = cum_prob.allclose(torch.tensor(1.0))

            if not prob_saturated and offset == n_chunks - 1:
                chunk_close_probs[:, :, -1] = 1 - cum_prob

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
