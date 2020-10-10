import os

import numpy as np
import torch
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

    exec_samples = torch.bernoulli(exec_probs.unsqueeze(2).expand(-1, -1, n_samples, -1)).to(torch.int64)
    exec_samples[-1, ..., 0] = 0
    exec_samples[-1, ..., 1] = 1

    ls_samples = torch.bernoulli(1 - long_probs.unsqueeze(2).expand(-1, -1, n_samples)).to(torch.int64)

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


save_path = "drive/My Drive/"
# save_path = "./"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

data = np.load(os.path.join(save_path, "train_data/train_data_10s.npz"))
prices = torch.from_numpy(data["prices"])
input = torch.from_numpy(data["input"])

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
