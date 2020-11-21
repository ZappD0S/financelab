import os
from itertools import chain, repeat
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# class Model(nn.Module):
#     def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
#         super(Model, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.h0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_size))
#         self.rnns = nn.ModuleList([nn.GRUCell(2, 30)] + [nn.GRUCell(30, 30) for _ in range(num_layers - 1)])
#         self.lin = nn.Linear(hidden_size, 1)

#     def forward(
#         self, input: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, force_probs: bool = False
#     ):
#         if hx is None:
#             hx = (self.h0.expand(-1, len(input), -1).contiguous(), torch.ones(len(input), 1).to(self.h0))

#         h_prev, not_done_prob = hx
#         probs = torch.zeros(input.size(0), input.size(1), 1, device=input.device)

#         seq_len = input.size(1)
#         h = torch.zeros_like(h_prev)
#         i = 0
#         for i in range(seq_len):
#             out = torch.cat((input[:, i], not_done_prob), dim=1)
#             for j, rnn in enumerate(self.rnns):
#                 out = h[j] = rnn(out, h_prev[j])

#             h_prev = h
#             h = torch.zeros_like(h_prev)

#             # exec_prob, ls_prob = self.lin(out).sigmoid().unbind(1)
#             exec_prob = self.lin(out).sigmoid().squeeze()

#             if force_probs and i == seq_len - 1:
#                 exec_prob = torch.ones(len(input)).to(exec_prob)

#             probs[:, i] = not_done_prob * exec_prob
#             # probs[:, i, 0] = not_done_prob * exec_prob
#             # probs[:, i, 1] = ls_prob

#             not_done_prob = not_done_prob * (1 - exec_prob)

#         print("i:", i)
#         return probs, (h, not_done_prob)


class Model(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.h0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_size))
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.lin = nn.Linear(hidden_size, 3)

    def forward(self, input: torch.Tensor, hx: Optional[torch.Tensor] = None, force_probs: bool = False):
        if hx is None:
            hx = self.h0.expand(-1, len(input), -1).contiguous()

        out, hx = self.rnn(input, hx)
        exec_probs, long_probs, fractions = self.lin(out).sigmoid().unbind(dim=2)

        if force_probs:
            exec_probs = exec_probs.clone()
            exec_probs[:, -1] = 1

        not_done_logprobs = torch.log(1 - exec_probs + 1e-45).cumsum(dim=1)
        compound_logprobs = torch.log(exec_probs + 1e-45)
        compound_logprobs[:, 1:] += not_done_logprobs[:, :-1]
        exec_probs = torch.exp(compound_logprobs)

        probs = torch.stack((exec_probs, long_probs, fractions), dim=2)

        return probs, hx


save_path = "drive/My Drive/"
# save_path = "./"

torch.set_default_dtype(torch.float32)
torch.autograd.set_detect_anomaly(True)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

data = np.load(os.path.join(save_path, "train_data/train_data_10s.npz"))
prices = torch.from_numpy(data["prices"])
input = torch.from_numpy(data["input"])

# model = torch.jit.script(Model(input_size, hidden_size, num_layers))
model = Model(input_size=1, hidden_size=30, num_layers=1)
# model = torch.load("model.pth")
model.to(device)

torch.manual_seed(1234)

window_length = 10_000

# batch_size = 1
batch_size = 32
batch_rows = torch.arange(batch_size, device=device).unsqueeze(-1)

# i0 = torch.randint(len(input) - window_length, size=(1,))
# inds = i0.unsqueeze(-1) + torch.arange(window_length)

scalar_zero = torch.tensor(0.0, device=device)

# window_input = input[inds].to(device)
# buy_p, sell_p = prices[inds].to(device).unbind(2)
# input = buy_p.log().unsqueeze(-1)
# input = (input - input.mean()) / input.std()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

n_batches = (len(input) - window_length) // batch_size

inds = torch.randperm(n_batches * batch_size).reshape(-1, batch_size)

# for _ in range(100_000):
for batch_inds in inds:
    buy_p, sell_p = prices[batch_inds.unsqueeze(-1) + torch.arange(window_length)].to(device).unbind(2)
    input = buy_p.log().unsqueeze(-1)
    input = (input - input.mean(dim=1, keepdim=True)) / input.std(dim=1, keepdim=True)

    probs, _ = model(input, force_probs=True)
    (exec_probs, long_probs, fractions) = probs.unbind(2)

    exec_probs, sort_inds = exec_probs.sort(dim=1, descending=True)
    mask = ~exec_probs.isclose(scalar_zero)
    # exec_probs = exec_probs[mask]
    # sort_inds = exec_probs[mask]

    long_probs = long_probs[batch_rows, sort_inds]
    fractions = fractions[batch_rows, sort_inds]
    # probs = probs.squeeze().clone()
    # probs[-1] = 1.0

    # not_done_logprobs = torch.log(1 - exec_probs + 1e-45).cumsum(dim=1)
    # compound_logprobs = torch.log(exec_probs + 1e-45)
    # compound_logprobs[1:] += not_done_logprobs[:-1]
    # compound_probs = torch.exp(compound_logprobs)

    # not_done_probs = torch.cumprod(1 - probs, dim=0)
    # compound_probs = probs.clone()
    # compound_probs[1:] *= not_done_probs[:-1]

    # probs, _ = model(window_input.unsqueeze(0), force_probs=True)
    # exec_probs, long_probs = probs.unbind(2)
    # exec_probs = probs.squeeze()

    print("tot prob:", exec_probs.detach().sum(dim=1).mean().item())
    print("mask fraction:", mask.detach().sum().item() / mask.numel())

    # buy_p, sell_p = prices[inds].to(device).unbind(1)

    long_rates = sell_p / buy_p[:, 0, None]
    short_rates = sell_p[:, 0, None] / buy_p

    long_rates = long_rates[batch_rows, sort_inds]
    short_rates = short_rates[batch_rows, sort_inds]

    # loss = torch.sum(exec_probs * (torch.log(window_sell_p) - torch.log(window_buy_p[0])))
    long_gains = torch.log1p(fractions[mask] * (long_rates[mask] - 1))
    short_gains = torch.log1p(fractions[mask] * (short_rates[mask] - 1))

    short_probs = 1 - long_probs
    loss = torch.sum(exec_probs[mask] * (long_probs[mask] * long_gains + short_probs[mask] * short_gains))

    optimizer.zero_grad()
    loss.neg().backward()

    print("loss:", loss.item())
    grad_norm = sum(p.grad.data.norm() ** 2 for p in model.parameters()) ** 0.5
    print("grad_norm:", grad_norm.item())

    optimizer.step()
    print(loss.item())
