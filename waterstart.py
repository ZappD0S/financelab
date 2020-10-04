import gc
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class PositionOpener(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super(PositionOpener, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.h0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_size))
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.lin_hidden_to_prob = nn.Linear(hidden_size, 3)

    def forward(self, input: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        hidden, hx = self.rnn(input, hx)
        return torch.sigmoid(self.lin_hidden_to_prob(hidden)), hidden, hx


class PositionCloser(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super(PositionCloser, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
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

    last_not_done_logprob = torch.log(last_not_done_prob + 1e-45)
    not_done_logprobs = torch.log(1 - input_probs + 1e-45).cumsum(dim).permute(perm)
    output_logprobs = torch.log(input_probs + 1e-45).permute(perm) + last_not_done_logprob
    output_logprobs[1:] += not_done_logprobs[:-1]

    return torch.exp(output_logprobs).permute(*inv_perm), torch.exp(last_not_done_logprob + not_done_logprobs[-1])


class OpenDataCalculator(nn.Module):
    def __init__(self, pos_opener: PositionOpener):
        super(OpenDataCalculator, self).__init__()
        self.batch_size = None
        self.pos_opener = pos_opener
        self.register_buffer("batch_mask", torch.BoolTensor())
        self.register_buffer("last_not_exec_prob", torch.Tensor())
        self.register_buffer("h", torch.Tensor())

    def init_state(self, batch_size: int):
        self.batch_size = batch_size
        self.h = self.pos_opener.h0.expand(-1, batch_size, -1).contiguous().to(self.h)
        self.batch_mask = torch.ones(self.batch_size, dtype=bool).to(self.batch_mask)
        self.last_not_exec_prob = torch.ones(self.batch_size).to(self.last_not_exec_prob)

    def reset_state(self):
        self.batch_size = None
        self.h = torch.Tensor().to(self.h)
        self.batch_mask = torch.BoolTensor().to(self.batch_mask)
        self.last_not_exec_prob = torch.Tensor().to(self.last_not_exec_prob)

    def forward(
        self, input: torch.Tensor, force_probs: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        if self.batch_size is None or not self.batch_mask.any():
            raise RuntimeError("init_state should be called.")

        if len(input) != self.batch_size:
            raise ValueError(f"input shape mismatch. First dim size should be {self.batc_size}")

        output = torch.zeros(*input.shape[:2], 3).to(input)
        hidden = torch.zeros(*input.shape[:2], self.pos_opener.hidden_size).to(input)
        (output[self.batch_mask], hidden[self.batch_mask], self.h[:, self.batch_mask],) = pos_opener(
            input[self.batch_mask], self.h[:, self.batch_mask]
        )
        exec_probs, ls_probs, fractions = output.unbind(2)

        if force_probs:
            exec_probs = exec_probs.clone()
            exec_probs[self.batch_mask, -1] = 1.0

        compound_exec_probs = torch.zeros_like(exec_probs)
        compound_exec_probs[self.batch_mask], self.last_not_exec_prob[self.batch_mask] = compute_compound_probs(
            exec_probs[self.batch_mask], 1, self.last_not_exec_prob[self.batch_mask]
        )

        output = torch.stack((compound_exec_probs, ls_probs, fractions), dim=2)
        self.batch_mask = self.batch_mask & ~self.last_not_exec_prob.isclose(scalar_zero)

        assert not (force_probs and self.batch_mask.any())

        return output, hidden


class CloseDataCalculator(nn.Module):
    def __init__(self, pos_closer: PositionCloser, hidden_state_adapter: HiddenStateAdapter):
        super(CloseDataCalculator, self).__init__()
        self.batch_shape = None
        self.pos_closer = pos_closer
        self.hidden_state_adapter = hidden_state_adapter
        self.register_buffer("batch_mask", torch.BoolTensor())
        self.register_buffer("last_not_exec_prob", torch.Tensor())
        self.register_buffer("h", torch.Tensor())

    def init_state(self, init_hidden: torch.Tensor, init_mask: torch.BoolTensor):
        assert init_mask.ndim == 2
        self.batch_shape = init_mask.shape
        # self.h = self.hidden_state_adapter(init_hidden).to(self.h)
        self.h = (
            self.hidden_state_adapter(init_hidden)
            .expand(self.pos_closer.num_layers, -1, -1, -1)
            .contiguous()
            .to(self.h)
        )
        self.batch_mask = init_mask.to(self.batch_mask)
        self.last_not_exec_prob = torch.ones(self.batch_shape).to(self.last_not_exec_prob)

    def reset_state(self):
        self.batch_shape = None
        self.h = torch.Tensor().to(self.h)
        self.batch_mask = torch.BoolTensor().to(self.batch_mask)
        self.last_not_exec_prob = torch.Tensor().to(self.last_not_exec_prob)

    def forward(self, input: torch.Tensor, force_probs: bool = False) -> torch.Tensor:
        if self.batch_shape is None or not self.batch_mask.any():
            raise RuntimeError("init_state should be called.")

        if input.shape[:2] != self.batch_shape:
            raise ValueError(f"input shape mismatch. First two dimensions shold have sizes {self.batch_shape}")

        output = torch.zeros(*input.shape[:3], 1).to(input)
        output[self.batch_mask], self.h[:, self.batch_mask] = pos_closer(
            input[self.batch_mask], self.h[:, self.batch_mask]
        )

        (probs,) = output.unbind(3)

        if force_probs:
            rows, cols = self.batch_mask.nonzero(as_tuple=True)
            probs = probs.clone()
            probs[rows, cols, -1] = 1.0

        compound_probs = torch.zeros_like(probs)
        compound_probs[self.batch_mask], self.last_not_exec_prob[self.batch_mask] = compute_compound_probs(
            probs[self.batch_mask], 1, self.last_not_exec_prob[self.batch_mask]
        )

        self.batch_mask = self.batch_mask & ~self.last_not_exec_prob.isclose(scalar_zero)

        assert not (force_probs and self.batch_mask.any())

        return compound_probs


def compute_loss(batch_inds: torch.Tensor, train: bool = True):
    open_calc.train(train)
    close_calc.train(train)

    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(train)

    inds = batch_inds.unsqueeze(-1) + torch.arange(max_length)
    batch_p = prices[inds].to(device)
    batch_input = input[inds].to(device)
    batch_input[:, :, 0] = batch_input[:, :, 0] - batch_input[:, 0, None, 0]

    batch_rows = torch.arange(batch_size, device=device).unsqueeze(-1)

    data = {"loss": 0.0}
    tot_probs = torch.zeros(batch_size)
    data["avg_lengths"] = torch.zeros(batch_size)

    open_output_chunks = []
    open_hidden_chunks = []

    open_calc.init_state(batch_size)
    open_chunk = 0
    while open_calc.batch_mask.any():
        print("open_chunk:", open_chunk)
        open_chunk_inds = open_chunk * open_chunk_size + torch.arange(open_chunk_size, device=device)
        open_chunk_input = batch_input[batch_rows, open_chunk_inds]

        open_output_chunk, open_hidden_chunk = open_calc(
            open_chunk_input, force_probs=open_chunk == open_max_chunks - 1
        )
        open_output_chunks.append(open_output_chunk)
        open_hidden_chunks.append(open_hidden_chunk)
        open_chunk += 1

    open_output = torch.cat(open_output_chunks, dim=1)
    open_hidden = torch.cat(open_hidden_chunks, dim=1)

    del open_output_chunks, open_hidden_chunks, open_output_chunk, open_hidden_chunk, open_chunk_input
    open_calc.reset_state()

    open_probs, ls_probs, fractions = open_output.unbind(2)

    open_probs, open_sort_inds = open_probs.sort(dim=1, descending=True)

    # print(torch.all(open_sort_inds == torch.arange(open_chunk_size, device=device)))

    ls_probs = ls_probs[batch_rows, open_sort_inds]
    fractions = fractions[batch_rows, open_sort_inds]

    open_hidden = open_hidden[batch_rows, open_sort_inds]
    open_mask = ~open_probs.isclose(scalar_zero)

    open_chunk = 0
    open_chunk_slice = slice(open_chunk * open_chunk_size, (open_chunk + 1) * open_chunk_size)
    while open_mask[:, open_chunk_slice].any():
        open_chunk_inds = open_sort_inds[:, open_chunk_slice]

        open_buy_p, open_sell_p = batch_p[batch_rows, open_chunk_inds].unbind(2)

        close_probs_chunks = []
        # in teoria questa non serve, è sufficiente open_mask, ma in alcuni casi risulatavano essere diverse,
        # quindi per ora lo teniamo così
        mask = open_probs[:, open_chunk_slice] > 0
        assert mask.any()
        close_calc.init_state(open_hidden[:, open_chunk_slice], mask)

        close_chunk = 0
        while close_calc.batch_mask.any():
            print("close_chunk:", close_chunk)
            close_chunk_inds = close_chunk * close_chunk_size + torch.arange(close_chunk_size, device=device) + 1
            close_chunk_input = batch_input[batch_rows.unsqueeze(-1), open_chunk_inds.unsqueeze(-1) + close_chunk_inds]

            close_probs_chunk = close_calc(close_chunk_input, force_probs=close_chunk == close_max_chunks - 1)
            close_probs_chunks.append(close_probs_chunk)
            close_chunk += 1

        close_probs = torch.cat(close_probs_chunks, dim=2)
        close_calc.reset_state()
        del close_probs_chunks, close_probs_chunk, close_chunk_input

        close_probs, close_sort_inds = close_probs.sort(dim=2, descending=True)
        close_mask = ~close_probs.isclose(scalar_zero)

        close_chunk = 0
        close_chunk_slice = slice(close_chunk * close_chunk_size, (close_chunk + 1) * close_chunk_size)
        while close_mask[:, :, close_chunk_slice].any():
            close_chunk_inds = close_sort_inds[:, :, close_chunk_slice] + 1

            # close_buy_p, close_sell_p = batch_p[:, open_chunk_inds, close_chunk_inds + 1].unbind(3)
            close_buy_p, close_sell_p = batch_p[
                batch_rows.unsqueeze(-1), open_chunk_inds.unsqueeze(-1) + close_chunk_inds
            ].unbind(3)

            long_gain = torch.log1p(
                leverage * fractions[:, open_chunk_slice, None] * (close_sell_p / open_buy_p.unsqueeze(-1) - 1)
            )
            short_gain = torch.log1p(
                leverage * fractions[:, open_chunk_slice, None] * (open_sell_p.unsqueeze(-1) / close_buy_p - 1)
            )

            # long_gain = torch.log(close_sell_p) - torch.log(open_buy_p).unsqueeze(-1)
            # short_gain = torch.log(open_sell_p).unsqueeze(-1) - torch.log(close_buy_p)

            exec_probs = open_probs[:, open_chunk_slice, None] * close_probs[:, :, close_chunk_slice]
            loss = torch.sum(
                exec_probs
                * (
                    ls_probs[:, open_chunk_slice, None] * long_gain
                    + (1 - ls_probs[:, open_chunk_slice, None]) * short_gain
                )
            )

            print(f"{open_chunk * close_max_chunks + close_chunk}: {torch.cuda.memory_allocated()}")

            if train:
                loss.neg().backward(retain_graph=True)
            # else:

            tot_probs += exec_probs.detach().sum(dim=(1, 2)).cpu()
            assert torch.all((tot_probs < 1) | tot_probs.isclose(torch.tensor(1.0)))

            avg_lengths_terms = torch.detach(close_chunk_inds * exec_probs)
            data["avg_lengths"] += avg_lengths_terms.sum(dim=(1, 2)).cpu()
            del avg_lengths_terms

            data["loss"] += loss.item()
            del close_buy_p, close_sell_p, long_gain, short_gain, exec_probs, loss
            gc.collect()
            torch.cuda.empty_cache()

            close_chunk += 1
            close_chunk_slice = slice(close_chunk * close_chunk_size, (close_chunk + 1) * close_chunk_size)

        del open_buy_p, open_sell_p
        open_chunk += 1
        open_chunk_slice = slice(open_chunk * open_chunk_size, (open_chunk + 1) * open_chunk_size)

    torch.set_grad_enabled(prev)
    print("avg_lengths:", data["avg_lengths"])
    print("loss", data["loss"])
    print("tot_prob:", tot_probs.mean().item())
    return data


# def compute_loss(batch_inds: torch.Tensor, train: bool = True):
#     open_calc.train(train)
#     close_calc.train(train)

#     prev = torch.is_grad_enabled()
#     torch.set_grad_enabled(train)

#     inds = batch_inds.unsqueeze(-1) + torch.arange(max_length)
#     batch_p = prices[inds].to(device)
#     batch_input = input[inds].to(device)
#     # batch_input[:, :, 0] = batch_input[:, :, 0] - batch_input[:, 0, None, 0]
#     batch_rows = torch.arange(batch_size, device=device).unsqueeze(-1)

#     data = {"loss": 0}
#     tot_probs = torch.zeros(batch_size)
#     data["avg_lengths"] = torch.zeros(batch_size)

#     open_calc.init_state(batch_size)

#     open_chunk = 0
#     while open_calc.batch_mask.any():
#         open_chunk_inds = open_chunk * open_chunk_size + torch.arange(open_chunk_size)

#         open_chunk_input = batch_input[batch_rows, open_chunk_inds]
#         open_output, open_hidden = open_calc(open_chunk_input, force_probs=open_chunk == open_max_chunks - 1)
#         open_probs, ls_probs, fractions = open_output.unbind(2)

#         open_buy_p, open_sell_p = batch_p[batch_rows, open_chunk_inds].unbind(2)

#         close_calc.init_state(open_hidden, open_probs > 0)
#         del open_hidden, open_output, open_chunk_input

#         close_chunk = 0
#         while close_calc.batch_mask.any():
#             close_chunk_inds = close_chunk * close_chunk_size + torch.arange(close_chunk_size) + 1

#             close_chunk_input = batch_input[batch_rows.unsqueeze(-1), open_chunk_inds.unsqueeze(-1) + close_chunk_inds]
#             close_probs = close_calc(close_chunk_input, force_probs=close_chunk == close_max_chunks - 1)

#             close_buy_p, close_sell_p = batch_p[
#                 batch_rows.unsqueeze(-1), open_chunk_inds.unsqueeze(-1) + close_chunk_inds
#             ].unbind(3)

#             long_gain = torch.log1p(leverage * fractions.unsqueeze(-1) * (close_sell_p / open_buy_p.unsqueeze(-1) - 1))
#             # long_gain = torch.log1p(leverage * (close_sell_p / open_buy_p.unsqueeze(-1) - 1))
#             # long_gain = torch.log(close_sell_p) - torch.log(open_buy_p).unsqueeze(-1)

#             short_gain = torch.log1p(leverage * fractions.unsqueeze(-1) * (open_sell_p.unsqueeze(-1) / close_buy_p - 1))
#             # short_gain = torch.log1p(leverage * (open_sell_p.unsqueeze(-1) / close_buy_p - 1))
#             # short_gain = torch.log(open_sell_p).unsqueeze(-1) - torch.log(close_buy_p)

#             exec_probs = open_probs.unsqueeze(-1) * close_probs
#             loss = torch.sum(
#                 exec_probs * (ls_probs.unsqueeze(-1) * long_gain + (1 - ls_probs.unsqueeze(-1)) * short_gain)
#             )

#             if train:
#                 loss.neg().backward(retain_graph=True)

#             tot_probs += exec_probs.detach().sum(dim=(1, 2)).cpu()
#             assert torch.all((tot_probs < 1) | tot_probs.isclose(torch.tensor(1.0)))
#             print(f"{open_chunk * close_max_chunks + close_chunk}: {torch.cuda.memory_allocated()}")

#             avg_lengths_terms = torch.detach(close_chunk_inds * exec_probs.cpu())
#             data["avg_lengths"] += avg_lengths_terms.sum(dim=(1, 2))
#             del avg_lengths_terms

#             data["loss"] += loss.item()
#             del loss, exec_probs, long_gain, short_gain, close_buy_p, close_sell_p, close_probs, close_chunk_input
#             gc.collect()
#             torch.cuda.empty_cache()

#             close_chunk += 1

#         close_calc.reset_state()
#         del open_probs, ls_probs, fractions, open_buy_p, open_sell_p
#         open_chunk += 1

#     open_calc.reset_state()
#     torch.set_grad_enabled(prev)
#     print("avg_lengths:", data["avg_lengths"])
#     print("loss", data["loss"])
#     print("tot_prob:", tot_probs.mean().item())
#     return data


def shuffle(x: torch.Tensor) -> torch.Tensor:
    return x.flatten()[torch.randperm(x.nelement())].view_as(x)


# def rolling_window(x: torch.Tensor, dim: int, size: int, step: int) -> torch.Tensor:
#     shape = x.shape
#     shape = shape[:dim] + (shape[dim] - size + 1, size) + shape[dim + 1 :]

#     strides = x.stride()
#     strides = strides[:dim] + (strides[dim],) + strides[dim:]

#     return x.as_strided(shape, strides)


def train_epoch(epoch: int) -> None:
    val_count = 0
    max_loss = float("-inf")

    for i, batch_inds in enumerate(train_inds):
        optimizer.zero_grad()
        compute_loss(batch_inds)
        torch.nn.utils.clip_grad_norm_(modulelist.parameters(), 10.0)
        # grad_norm = sum(p.grad.data.norm() ** 2 for p in parameters).item() ** 0.5
        print("pos_opener norm:", sum(p.grad.data.norm() ** 2 for p in pos_opener.parameters()).item() ** 0.5)
        print("pos_closer norm:", sum(p.grad.data.norm() ** 2 for p in pos_closer.parameters()).item() ** 0.5)

        optimizer.step()

        if (i + 1) % validation_interval != 0:
            continue

        val_count += 1
        loss = 0
        avg_lengths = []
        for j, val_batch_inds in enumerate(val_inds):
            data = compute_loss(val_batch_inds, train=False)
            loss += data["loss"]
            avg_lengths.append(data["avg_lengths"])

        writer.add_scalar("loss", loss, epoch * len(train_inds) + i)
        # writer.add_scalar("grad_norm", grad_norm, epoch * len(train_inds) + i)
        writer.add_histogram("avg_lengths", torch.cat(avg_lengths), epoch * len(train_inds) + i)
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
# leverage = 50
leverage = 1
save_path = "drive/My Drive/"
# save_path = "./"

batch_size = 32
n_val_batches = 20

open_chunk_size = 100
open_max_chunks = 10_000 // open_chunk_size

close_chunk_size = 100
close_max_chunks = 10_000 // close_chunk_size

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
# scalar_one = torch.tensor(1.0, device=device)
scalar_zero = torch.tensor(0.0, device=device)

pos_opener = PositionOpener(2, open_hidden_size, num_layers)
# hidden_state_adapter = HiddenStateAdapter(open_hidden_size, close_hidden_size, num_layers)
hidden_state_adapter = nn.Linear(open_hidden_size, close_hidden_size)
pos_closer = PositionCloser(2, close_hidden_size, num_layers)

modulelist = nn.ModuleList([pos_opener, hidden_state_adapter, pos_closer])

try:
    state_dict = torch.load(os.path.join(save_path, "nn_weights/waterstart/modulelist.pth"))
    modulelist.load_state_dict(state_dict)
    print("modulelist found")
except FileNotFoundError:
    pass

open_calc = OpenDataCalculator(pos_opener).to(device)
close_calc = CloseDataCalculator(pos_closer, hidden_state_adapter).to(device)
writer = SummaryWriter()

data = np.load(os.path.join(save_path, "train_data/train_data_10s.npz"))
prices = torch.from_numpy(data["prices"])
input = torch.from_numpy(data["input"])

max_length = open_max_chunks * open_chunk_size + close_max_chunks * close_chunk_size

n_batches = (len(input) - max_length) // batch_size

inds = torch.arange(n_batches * batch_size).reshape(-1, batch_size)
inds = shuffle(inds)
train_inds, val_inds = inds[:-n_val_batches], inds[-n_val_batches:]
# parameters = list(modulelist.parameters())
optimizer = torch.optim.Adam(modulelist.parameters(), lr=1e-1)

train_epoch(0)
