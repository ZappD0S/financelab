import gc
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain, repeat
from typing import Iterable, Union, Optional, Tuple, Iterator
from torch.utils.tensorboard import SummaryWriter


class PositionOpener(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super(PositionOpener, self).__init__()
        # TODO: maybe add a trainable initial hidden state. Use nn.Parameter
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lin_hidden_to_prob = nn.Linear(hidden_size, 3)

    def forward(self, input: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        hidden, hx = self.rnn(input, hx)
        return torch.sigmoid(self.lin_hidden_to_prob(hidden)), hidden, hx


class PositionCloser(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super(PositionCloser, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lin_close_h_to_prob = nn.Linear(hidden_size, 1)

    def forward(self, input: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        hidden, hx = self.rnn(input, hx)
        return torch.sigmoid(self.lin_close_h_to_prob(hidden)), hx


class HiddenStateAdapter(nn.Module):
    def __init__(self, open_hidden_size: int, close_hidden_size, num_layers: int = 1):
        super(HiddenStateAdapter, self).__init__()
        self.num_layers = num_layers
        self.close_hidden_size = close_hidden_size
        self.lin = nn.Linear(open_hidden_size, 2 * num_layers * close_hidden_size)

    def forward(self, input: torch.Tensor):
        out1, out2 = self.lin(input).split(num_layers * close_hidden_size, dim=-1)
        out = torch.sigmoid(out1) * torch.tanh(out2)
        out = out.view(*out.shape[:-1], num_layers, close_hidden_size)

        perm = list(range(out.ndim))
        perm.insert(0, perm.pop(-2))

        return out.permute(perm).contiguous()


# def compute_compound_probs(
#     input_probs: torch.Tensor, dim: int, last_not_done_prob: torch.Tensor
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     perm = list(range(input_probs.ndim))
#     perm.insert(0, perm.pop(dim))

#     inv_perm = torch.empty(input_probs.ndim, dtype=int)
#     inv_perm[perm] = torch.arange(input_probs.ndim)

#     not_done_probs = torch.cumprod(1 - input_probs, dim=dim).permute(perm)
#     output_probs = input_probs.permute(perm) * last_not_done_prob
#     output_probs[1:] *= not_done_probs[:-1]

#     return output_probs.permute(*inv_perm), last_not_done_prob * not_done_probs[-1]


def compute_compound_probs(
    input_probs: torch.Tensor, dim: int, last_not_done_prob: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    perm = list(range(input_probs.ndim))
    perm.insert(0, perm.pop(dim))

    inv_perm = torch.empty(input_probs.ndim, dtype=int)
    inv_perm[perm] = torch.arange(input_probs.ndim)

    last_not_done_logprob = torch.log(last_not_done_prob)
    not_done_logprobs = torch.log(1 - input_probs).cumsum(dim).permute(perm)
    output_logprobs = torch.log(input_probs).permute(perm) + last_not_done_logprob
    output_logprobs[1:] += not_done_logprobs[:-1]

    return torch.exp(output_logprobs).permute(*inv_perm), torch.exp(last_not_done_logprob + not_done_logprobs[-1])


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
        chunk_open_input = input[inds[batch_mask], :].to(device)

        chunk_open_probs, chunk_open_hidden, hx = pos_opener(chunk_open_input, hx)
        chunk_exec_probs, chunk_ls_probs, _ = chunk_open_probs.unbind(2)

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


def compute_loss(batch_inds: torch.Tensor, train: bool = True):
    assert batch_inds.ndim == 1

    total_loss = 0.0

    global tot_prob, deltas
    tot_prob = 0.0

    deltas[:] = 0.0

    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(train)

    probs, open_hidden, open_mask = compute_open_data(batch_inds)
    n_chunks = (probs.size(1) + loss_chunk_size - 1) // loss_chunk_size

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

        if train:
            loss.backward(retain_graph=True)

        total_loss += loss.item()
        del loss

    torch.set_grad_enabled(prev)

    print()
    print("loss:", total_loss)
    print("tot_prob:", tot_prob / batch_size)
    print("deltas:", deltas)
    # return total_loss
    return {"loss": total_loss}


def compute_chunk_loss(
    batch_inds: torch.Tensor, probs: torch.Tensor, open_hidden: torch.Tensor, open_mask: torch.Tensor,
) -> torch.Tensor:
    global tot_prob, deltas
    open_probs, ls_probs = probs.unbind(2)

    open_probs = open_probs[open_mask].unsqueeze(-1)
    ls_probs = ls_probs[open_mask].unsqueeze(-1)

    h0 = hidden_state_adapter(open_hidden[open_mask])
    # h0 = h0.expand(pos_closer.rnn.num_layers, *h0.shape).contiguous()
    hx = (h0, torch.zeros_like(h0))

    batch_mask = open_mask.clone()

    cum_prob = torch.zeros(open_mask.sum(), device=device)
    last_not_close_probs = torch.ones(open_mask.sum(), device=device)

    n_chunks = min(close_max_chunks, (logp.size(0) - batch_inds[:, -1].max()) // close_chunk_size)
    assert n_chunks > 0

    loss = 0.0

    open_logp = logp[batch_inds[batch_mask], :].unsqueeze(1)
    open_buy_logp, open_sell_logp = open_logp.to(device).unbind(2)

    base_inds = torch.arange(close_chunk_size)

    chunk = 0
    while True:
        chunk_inds = chunk * close_chunk_size + base_inds
        inds = batch_inds.unsqueeze(-1) + chunk_inds
        chunk_close_input = input[inds[batch_mask], :].to(device)

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

        chunk_close_buy_logp, chunk_close_sell_logp = logp[inds[batch_mask], :].to(device).unbind(2)

        with torch.no_grad():
            exec_probs = open_probs * chunk_close_probs
            exec_probs[~chunk_close_mask] = 0

            tot_prob += (exec_probs)[chunk_close_mask].sum().item()
            deltas_terms_ = exec_probs * chunk_inds.to(dtype=torch.get_default_dtype(), device=device)
            deltas_terms = torch.zeros_like(batch_mask, dtype=torch.get_default_dtype())
            deltas_terms[batch_mask] = deltas_terms_.sum(dim=1)
            deltas += deltas_terms.sum(dim=1).detach()

        long_gain = chunk_close_sell_logp - open_buy_logp
        short_gain = open_sell_logp - chunk_close_buy_logp

        loss_terms = open_probs * chunk_close_probs * (ls_probs * long_gain + (1 - ls_probs) * short_gain)

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


def shuffle(x: torch.Tensor) -> torch.Tensor:
    return x.flatten()[torch.randperm(x.nelement())].view_as(x)


def train_epoch(epoch: int) -> None:
    val_count = 0
    max_loss = float("-inf")

    for i, batch_inds in enumerate(train_inds):
        optimizer.zero_grad()
        compute_loss(batch_inds)
        # torch.nn.utils.clip_grad_norm_(parameters, 10.0)
        # grad_norm = sum(p.grad.data.norm() ** 2 for p in parameters).item() ** 0.5
        print("pos_opener norm:", sum(p.grad.data.norm() ** 2 for p in pos_opener.parameters()).item() ** 0.5)
        print("pos_closer norm:", sum(p.grad.data.norm() ** 2 for p in pos_closer.parameters()).item() ** 0.5)

        optimizer.step()

        if (i + 1) % validation_interval != 0:
            continue

        val_count += 1
        loss = 0
        # avg_lengths = []
        for j, val_batch_inds in enumerate(val_inds):
            data = compute_loss(val_batch_inds, train=False)
            loss += data["loss"]
            # avg_lengths.append(data["avg_lengths"])

        writer.add_scalar("loss", loss, epoch * len(train_inds) + i)
        # writer.add_scalar("grad_norm", grad_norm, epoch * len(train_inds) + i)
        # writer.add_histogram("avg_lengths", torch.cat(avg_lengths), epoch * len(train_inds) + i)
        writer.flush()

        new_max_loss = loss > max_loss
        reached_max_count = val_count == max_val_count

        if new_max_loss or reached_max_count:
            print("saving..")
            torch.save(modulelist.state_dict(), os.path.join(save_path, "nn_weights/waterstart/modulelist.pth"))

            if new_max_loss:
                max_loss = loss

            val_count = 0


validation_interval = 100  # do validation once every
max_val_count = 5
save_path = "drive/My Drive/"
# save_path = "./"

batch_size = 32
# batch_size = 128

n_val_batches = 20

# open_chunk_size = 100
# open_max_chunks = 10_000 // open_chunk_size

# close_chunk_size = 200
# close_max_chunks = 10_000 // close_chunk_size

# loss_chunk_size = open_chunk_size

open_chunk_size = 500
open_max_chunks = 100

close_chunk_size = 500
close_max_chunks = 100

loss_chunk_size = 4
num_layers = 2
open_hidden_size = 30
close_hidden_size = 30

torch.set_default_dtype(torch.float32)
torch.autograd.set_detect_anomaly(True)

device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# this is needed for isclose check
scalar_one = torch.tensor(1.0, device=device)

deltas = torch.zeros(batch_size, device=device)

pos_opener = PositionOpener(2, open_hidden_size, num_layers)
hidden_state_adapter = HiddenStateAdapter(open_hidden_size, close_hidden_size, num_layers)
# hidden_state_adapter = nn.Linear(open_hidden_size, close_hidden_size)
pos_closer = PositionCloser(2, close_hidden_size, num_layers)

modulelist = nn.ModuleList([pos_opener, hidden_state_adapter, pos_closer])

try:
    state_dict = torch.load(os.path.join(save_path, "nn_weights/waterstart/modulelist.pth"))
    modulelist.load_state_dict(state_dict)
    print("modulelist found")
except FileNotFoundError:
    pass

modulelist.to(device)
writer = SummaryWriter()

data = np.load(os.path.join(save_path, "train_data/train_data_10s.npz"))
prices = torch.from_numpy(data["prices"])
logp = torch.log(prices)
input = torch.from_numpy(data["input"])

max_length = open_max_chunks * open_chunk_size + close_max_chunks * close_chunk_size

n_batches = (len(input) - max_length) // batch_size

inds = torch.arange(n_batches * batch_size).reshape(-1, batch_size)
inds = shuffle(inds)
train_inds, val_inds = inds[:-n_val_batches], inds[-n_val_batches:]
# parameters = list(modulelist.parameters())
optimizer = torch.optim.Adam(modulelist.parameters(), lr=1e-2)

train_epoch(0)
