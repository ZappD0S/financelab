import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionOpener(nn.Module):
    def __init__(self, hidden_size: int, chunk_size: int, max_chunks: int):
        super(PositionOpener, self).__init__()
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        # TODO: we need a trainable initial hidden state. Use nn.Parameter
        self.lstm = nn.LSTM(2, hidden_size, batch_first=True)
        self.lin_hidden_to_prob = nn.Linear(hidden_size, 2)

    def forward(self, inds: torch.Tensor, p: torch.Tensor):
        # questo magari lo facciamo fuori
        open_probs = torch.Tensor()
        open_hx = torch.Tensor()
        hx = None

        batch_size = inds.nelement()
        cum_prob = torch.zeros(batch_size)

        # attenzione p.size(0) è una dimensione mentre inds.max() è un indice!
        # TODO: aggiusta n_chunks: adesso deve essere 1 in meno
        n_chunks = min(self.max_chunks, (p.size(0) - inds.max()) // self.chunk_size)

        last_not_done_probs = torch.ones(batch_size)
        cum_slices = [range(i, i + self.chunk_size) for i in inds]

        for offset in range(n_chunks):
            slices = [range(i + offset * self.chunk_size, i + (offset + 1) * self.chunk_size) for i in inds]
            cum_slices = [range(old.start, cur.end) for old, cur in zip(cum_slices, slices)]

            chunk_open_p = p[slices, ...]
            # shape: (batch, chunck_size, feature)

            chunk_open_p -= chunk_open_p[:, 0, None, ...]

            output, hx = self.lstm(chunk_open_p, hx)
            chunk_ls_probs, chunk_open_probs = torch.sigmoid(self.lin_hidden_to_prob(output)).unbind(2)
            not_done_probs = last_not_done_probs * torch.cumprod(1 - chunk_open_probs[:, :-1], dim=1)
            chunk_open_probs[:, 1:] *= not_done_probs
            last_not_done_probs = not_done_probs[:, -1]

            cum_prob += torch.sum(chunk_open_probs, dim=1)

            # all_probs = probs if all_probs is None else torch.cat((all_probs, probs), dim=1)
            open_probs = torch.cat((open_probs, chunk_open_probs), dim=1)
            open_hx = torch.cat((open_hx, hx), dim=1)

            if cum_prob.allclose(torch.tensor(1.0)):
                break

        # last_inds = [s[-1] + 1 for s in slices]
        last_inds = [s[-1] + 1 for s in cum_slices]

        return open_probs, open_hx, 1 - cum_prob


class PositionCloser(nn.Module):
    def __init__(self, hidden_size: int, chunk_size: int):
        super(PositionCloser, self).__init__()
        self.chunk_size = chunk_size
        self.lstm = nn.LSTM(2, hidden_size, batch_first=True)
        self.lin_hidden_to_prob = nn.Linear(hidden_size, 1)

    def forward(self, inds: torch.Tensor, p: torch.Tensor, open_hx: torch.Tensor):
        T = p.size(1)
        # per ora lo prendiamo da qui..
        batch_shape = open_hx.shape[:2]
        all_probs = torch.Tensor()
        # (batch_size, T, hidden_state)
        hx = open_hx.view(-1, *open_hx.shape[2:])
        cum_prob = torch.zeros(batch_shape)
        last_not_done_probs = torch.ones(batch_shape)

        n_chunks = (p.size(0) - inds.max()) // self.chunk_size

        loss = 0.

        for offset in range(n_chunks):
            slices = [
                [range(i + j + offset * self.chunk_size, i + j + (offset + 1) * self.chunk_size) for j in range(T)]
                for i in inds
            ]
            # slices = [
            #     range(i + j + offset * self.chunk_size, i + j + (offset + 1) * self.chunk_size)
            #     for i in inds
            #     for j in range(T)
            # ]
            chunk_close_p = p[slices, ...]
            chunk_close_p -= chunk_close_p[:, :, 0, None, ...]

            output, hx = self.lstm(chunk_close_p.view(-1, *chunk_close_p.shape[2:]), hx)
            output = output.view(*batch_shape, *output.shape[1:])

            chunk_close_probs = torch.sigmoid(self.lin_hidden_to_prob(output))
            not_done_probs = last_not_done_probs * torch.cumprod(1 - chunk_close_probs[:, :, :-1], dim=2)
            chunk_close_probs[:, :, 1:] *= not_done_probs
            last_not_done_probs = not_done_probs[:, :, -1]

            # loss += torch.sum(
            #     chunk_ls_probs
            #     * chunk_open_probs
            #     * (torch.log(chunk_close_p) - torch.log(open_p).expand_as(chunk_close_p))

            #     + (1 - chunk_ls_probs) * chunk_open_probs * (torch.log(chunk_close_p) - torch.log(open_p).expand_as(chunk_close_p))

            # )

            # probs[:, 1:] *= torch.cumprod(1 - probs[:, :-1], dim=1)

            # cum_prob += torch.sum(probs, dim=1)
            # all_probs = torch.cat((all_probs, torch), dim=1)

            # if cum_prob.allclose(torch.tensor(1.0)):
            #     break

        return all_probs, 1 - cum_prob
