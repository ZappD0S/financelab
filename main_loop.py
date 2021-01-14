import torch
import tables
import numpy as np
from SSMEvaluator import SSMEvaluator
from waterstart_model import CNN, GatedTrasition, Emitter
from stochastic_waterstart5 import LossEvaluator
from pyro.distributions.transforms.affine_autoregressive import affine_autoregressive


def create_tables_file(fname, n_timesteps, n_cur, n_samples, z_dim, filters=None):
    # TODO: we'll have to change the open mode to r+ so that we don't loose the progress
    file = tables.open_file(fname, "w", filters=filters)
    file.create_carray(file.root, "hidden_state", tables.Float32Atom(), (n_timesteps, n_samples, z_dim))
    file.create_carray(file.root, "pos_states", tables.Int64Atom(), (n_timesteps, n_cur, n_samples))
    file.create_carray(file.root, "pos_types", tables.Int64Atom(), (n_timesteps, n_cur, n_samples))
    file.create_carray(file.root, "total_margin", tables.Float32Atom(), (n_timesteps, n_samples))
    file.create_carray(file.root, "open_pos_sizes", tables.Float32Atom(), (n_timesteps, n_cur, n_samples))
    file.create_carray(file.root, "open_rates", tables.Float32Atom(), (n_timesteps, n_cur, n_samples))

    return file


def sample_geometric(size, start, end, bias, min_gap=1):
    assert len(size) == 2
    max_length = end - start - 1
    assert ((max_length - 1) // min_gap + 1) >= size[1]

    min_counts = size[1] - torch.arange(size[1]).unsqueeze(0)

    dist = torch.distributions.Geometric(probs=bias)
    samples = dist.sample(size)
    min_gap_mask = torch.zeros_like(samples, dtype=torch.bool)

    count = 0
    while True:
        samples = samples.sort(dim=1).values

        mask = samples >= end - start
        lengths = max_length - samples.where(~mask, samples.new_zeros([]))

        min_gap_mask[:, 1:] = (samples[:, 1:] - samples[:, :-1]) < min_gap
        mask |= min_gap_mask

        not_enough_space_mask = ((lengths - 1) // min_gap + 1) < min_counts
        mask |= not_enough_space_mask

        row_inds = mask.any(dim=1).nonzero(as_tuple=True)[0]
        to_resample = row_inds.nelement()

        if to_resample > 0:
            col_inds = mask[row_inds, :].type(torch.int8).argmax(dim=1)
        else:
            break

        samples[row_inds, col_inds] = dist.sample([to_resample])
        count += 1

    return (end - 1 - samples).type(torch.long)


data = np.load("train_data/test.npz")
all_rates = torch.from_numpy(data["arr"]).type(torch.float32)
all_input = torch.from_numpy(data["arr2"]).type(torch.float32).transpose(1, 2)

# n_timesteps, n_cur, in_features = all_input.shape
n_timesteps, in_features, n_cur = all_input.shape
seq_len = 109
win_len = 50
n_samples = 100
z_dim = 128
out_features = 256
batch_size = 2
n_iterations = 80_000

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

cnn = CNN(win_len, in_features, out_features, n_cur)
# trans = torch.jit.script(GatedTrasition(out_features, z_dim, 200))
# emitter = torch.jit.script(Emitter(z_dim, n_cur, 200))
trans = GatedTrasition(out_features, z_dim, 200)
emitter = Emitter(z_dim, n_cur, 200)

iafs = [affine_autoregressive(z_dim, [200]) for _ in range(2)]


ssm_eval = SSMEvaluator(trans, emitter, iafs, seq_len, batch_size, n_samples, n_cur).to(device)

loss_eval = LossEvaluator(seq_len, batch_size, n_samples, n_cur, leverage=1)

# dummy_input = (
#     torch.randn(seq_len, batch_size, out_features, device=device),
#     torch.randn(batch_size, n_samples, z_dim, device=device),
# )
# ssm_eval = torch.jit.trace(ssm_eval, dummy_input, check_trace=False)

filters = tables.Filters(complevel=9, complib="blosc")
h5file = create_tables_file("tmp.h5", n_timesteps, n_cur, n_samples, z_dim, filters=filters)

all_start_inds = sample_geometric((n_iterations, batch_size), win_len - 1, n_timesteps + 1 - seq_len, 5e-5)
# all_batch_inds = all_batch_inds.view(n_iterations, batch_size, 1) + torch.arange(win_length)
# all_start_inds = all_start_inds.view(n_iterations, batch_size)

for batch_start_inds in all_start_inds:
    batch_inds = batch_start_inds.unsqueeze(-1) + torch.arange(seq_len)
    # batch_window_inds = batch_inds.unsqueeze(-1) - torch.arange(win_len, 0, -1)
    batch_window_inds = batch_inds.view(-1, 1) - torch.arange(win_len - 1, -1, -1)

    batch_data = all_input[batch_window_inds, ...].movedim(1, -1).to(device)
    # out = cnn(batch_data).squeeze(-1).squeeze(-1)
    out = cnn(batch_data).view(batch_size, seq_len, out_features).transpose(0, 1)
    # now we need to load the last hidden state and call ssmeval
    z0 = torch.from_numpy(h5file.root.hidden_state[batch_start_inds.numpy(), ...]).to(device)

    x_samples, z_samples, x_logprobs, z_logprobs, fractions = ssm_eval(out, z0)

    # TODO: save z_samples

    break


h5file.close()
