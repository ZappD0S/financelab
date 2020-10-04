import os

import numpy as np
import torch
import torch.nn as nn


save_path = "drive/My Drive/"
# save_path = "./"

data = np.load(os.path.join(save_path, "train_data/train_data_10s.npz"))
prices = torch.from_numpy(data["prices"])
input = torch.from_numpy(data["input"])

batch_size = 32
n_samples = 100
seq_len = 100

rows = torch.arange(n_samples)
cols = torch.arange(batch_size)

p_buy, p_sell = prices[:seq_len].unbind(1)

exec_probs = torch.rand(batch_size, seq_len, 2)
long_probs = torch.rand(batch_size, seq_len)
fractions = torch.rand(batch_size, seq_len)


# dim -3: long, short
# dim -2: open, close
loss_terms = torch.stack(
    (
        torch.stack(
            (
                torch.stack((p_buy / fractions, (1 - fractions) / fractions * p_buy), dim=-1),
                torch.stack(torch.zeros_like(fractions), p_sell.expand_as(fractions), dim=-1),
            ),
            dim=-2,
        ),
        torch.stack(
            (
                torch.stack((p_sell * fractions, (1 - fractions) / (fractions * p_sell)), dim=-1),
                torch.stack(torch.zeros_like(fractions), 1 / p_sell.expand_as(fractions), dim=-1),
            ),
            dim=-2,
        ),
    ),
    dim=-3,
)


pos_state_index = torch.zeros(batch_size, n_samples, dtype=int)
pos_type = torch.zeros(batch_size, n_samples, dtype=int)

exec_samples = torch.bernoulli(exec_probs.expand(n_samples, -1, -1, -1))
ls_samples = torch.bernoulli(1 - long_probs.expand(n_samples, -1, -1))

sample_losses_builder = torch.zeros(batch_size, n_samples, seq_len // 2, 2, dtype=int)

sample_exec_probs = torch.zeros(batch_size, n_samples, seq_len)
sample_ls_probs = torch.ones(batch_size, n_samples, seq_len)
# not_done_logprobs = torch.zeros(batch_size, n_samples)

i = 0

latest_pos = pos_state_index // 2
pos_state = pos_state_index % 2

# bisogna aggiornare i pos_type dove il pos_state == 0 (chiuso) e
# l'exec_sample di apertura (index == 0) é 1

closed_state_mask = pos_state == 0
open_event_mask = exec_samples[:, :, i, 0] == 1
update_pos_type_mask = closed_state_mask & open_event_mask

mask_rows, mask_cols = update_pos_type_mask.nonzero(as_tuple=True)
# 0 -> long, 1 -> short
pos_type[update_pos_type_mask] = ls_samples[mask_rows, mask_cols, i]
sample_ls_probs[mask_rows, mask_cols, i] = torch.where(pos_type == 0, long_probs, 1 - long_probs)[
    mask_rows, mask_cols, i
]

# pos_state == 0 -> la posizione è chiusa, quindi vogliamo guardare il sample di apertura
# pos_state == 1 -> las posizione è aperta, quindi vogliamo guardare il sample di chiusura
event_mask = exec_samples[rows, cols, i, pos_state] == 1

mask_rows, mask_cols = event_mask.nonzero(as_tuple=True)
sample_losses_builder[mask_rows, mask_cols, latest_pos[event_mask], pos_state[event_mask]] = loss_terms[
    mask_rows, mask_cols, i, pos_type[event_mask], pos_state[event_mask]
]

sample_exec_probs[:, :, i] = torch.where(event_mask, exec_probs, 1 - exec_probs)[rows, cols, i, pos_state]

pos_state_index[event_mask] += 1
i += 1

sample_logprobs = sample_ls_probs.add(1e-45).log().sum(2) + sample_exec_probs.add(1e-45).log().sum(2)

# queste vanno moltiplicate per le logprob
sample_losses = sample_losses_builder.sum(dim=3).log().sum(dim=2)
