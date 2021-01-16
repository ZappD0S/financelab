import torch
import tables
import numpy as np
from SSMEvaluator import SSMEvaluator
from waterstart_model import CNN, GatedTrasition, Emitter
from stochastic_waterstart5 import LossEvaluator
from pyro.distributions.transforms.affine_autoregressive import affine_autoregressive


def create_tables_file(fname, n_timesteps, n_cur, n_samples, z_dim, filters=None):
    file = tables.open_file(fname, "w", filters=filters)
    file.create_carray(file.root, "hidden_state", tables.Float32Atom(), (n_timesteps, n_samples, z_dim))
    file.create_carray(file.root, "pos_states", tables.Int8Atom(), (n_timesteps, n_cur, n_samples))
    file.create_carray(file.root, "pos_types", tables.Int8Atom(), (n_timesteps, n_cur, n_samples))
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

        # mask = samples >= end - start
        mask = samples > max_length
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


np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
torch.autograd.set_detect_anomaly(True)

data = np.load("train_data/train_data.npz")
all_rates = torch.from_numpy(data["arr"]).type(torch.float32).pin_memory()
all_account_cur_rates = torch.from_numpy(data["arr2"]).type(torch.float32).pin_memory()
all_input = torch.from_numpy(data["arr3"]).type(torch.float32).transpose(1, 2).pin_memory()

# n_timesteps, n_cur, in_features = all_input.shape
n_timesteps, in_features, n_cur = all_input.shape
seq_len = 109
win_len = 50
n_samples = 100
z_dim = 128
out_features = 256
batch_size = 3
n_iterations = 80_000

# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")

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

z_samples = torch.zeros(seq_len, batch_size, n_samples, z_dim)
all_pos_states = torch.zeros(seq_len, batch_size, n_cur, n_samples)
all_pos_types = torch.zeros(seq_len, batch_size, n_cur, n_samples)
all_total_margins = torch.zeros(seq_len, batch_size, n_samples)
all_open_pos_sizes = torch.zeros(seq_len, batch_size, n_cur, n_samples)
all_open_rates = torch.zeros(seq_len, batch_size, n_cur, n_samples)

# all_start_inds = sample_geometric((n_iterations, batch_size), win_len - 1, n_timesteps + 1 - seq_len, 5e-5)
all_start_inds_it = iter(
    sample_geometric((n_iterations, batch_size), win_len - 1, n_timesteps + 1 - seq_len, 5e-5, min_gap=seq_len)
)

batch_start_inds = next(all_start_inds_it)
batch_inds = torch.arange(seq_len).unsqueeze(-1) + batch_start_inds
batch_window_inds = batch_inds.view(-1, 1) - torch.arange(win_len - 1, -1, -1)

batch_data = all_input[batch_window_inds, ...].movedim(1, -1).to(device, non_blocking=True)
rates = all_rates[batch_inds, ...].permute([0, 3, 2, 1]).to(device, non_blocking=True)
account_cur_rates = all_account_cur_rates[batch_inds, ...].permute([0, 2, 1]).to(device, non_blocking=True)

done = False

while not done:
    try:
        next_batch_start_inds = next(all_start_inds_it)
    except StopIteration:
        # TODO: in this case can we make this an empty array?
        next_batch_start_inds = batch_start_inds
        done = True

    breakpoint()

    next_batch_inds = torch.arange(seq_len).unsqueeze(-1) + next_batch_start_inds
    next_batch_window_inds = next_batch_inds.view(-1, 1) - torch.arange(win_len - 1, -1, -1)

    batch_data[:, :3] /= batch_data[:, 1, None, :, -1, None]
    batch_data[:, 3:6] /= batch_data[:, 4, None, :, -1, None]
    out = cnn(batch_data).view(seq_len, batch_size, out_features)
    batch_data = all_input[next_batch_window_inds, ...].movedim(1, -1).to(device, non_blocking=True)

    h5file.root.hidden_state[batch_inds.flatten().numpy(), ...] = z_samples.flatten(0, 1).numpy()
    z0 = (
        torch.from_numpy(h5file.root.hidden_state[batch_start_inds.numpy(), ...])
        .pin_memory()
        .to(device, non_blocking=True)
    )

    x_samples, z_samples, x_logprobs, z_logprobs, fractions = ssm_eval(out, z0)

    z_samples = z_samples.to("cpu", non_blocking=True)

    x_samples = x_samples.permute([0, 4, 3, 2, 1])
    x_logprobs = x_logprobs.permute([0, 4, 3, 2, 1])
    z_logprobs = z_logprobs.permute([0, 2, 1])
    fractions = fractions.permute([0, 3, 2, 1])

    h5file.root.pos_states[batch_inds.flatten().numpy(), ...] = all_pos_states.flatten(0, 1).numpy()
    h5file.root.pos_types[batch_inds.flatten().numpy(), ...] = all_pos_types.flatten(0, 1).numpy()
    h5file.root.total_margin[batch_inds.flatten().numpy(), ...] = all_total_margins.flatten(0, 1).numpy()
    h5file.root.open_pos_sizes[batch_inds.flatten().numpy(), ...] = all_open_pos_sizes.flatten(0, 1).numpy()
    h5file.root.open_rates[batch_inds.flatten().numpy(), ...] = all_open_rates.flatten(0, 1).numpy()

    pos_states = (
        torch.from_numpy(h5file.root.pos_states[batch_start_inds.numpy(), ...])
        .movedim(0, -1)
        .pin_memory()
        .to(device, non_blocking=True)
    )
    pos_types = (
        torch.from_numpy(h5file.root.pos_types[batch_start_inds.numpy(), ...])
        .movedim(0, -1)
        .pin_memory()
        .to(device, non_blocking=True)
    )
    total_margin = (
        torch.from_numpy(h5file.root.total_margin[batch_start_inds.numpy(), ...])
        .movedim(0, -1)
        .pin_memory()
        .to(device, non_blocking=True)
    )
    open_pos_sizes = (
        torch.from_numpy(h5file.root.open_pos_sizes[batch_start_inds.numpy(), ...])
        .movedim(0, -1)
        .pin_memory()
        .to(device, non_blocking=True)
    )
    open_rates = (
        torch.from_numpy(h5file.root.open_rates[batch_start_inds.numpy(), ...])
        .movedim(0, -1)
        .pin_memory()
        .to(device, non_blocking=True)
    )

    loss, all_pos_states, all_pos_types, all_total_margins, all_open_pos_sizes, all_open_rates = loss_eval(
        x_samples,
        fractions,
        x_logprobs,
        z_logprobs,
        rates,
        account_cur_rates,
        pos_states,
        pos_types,
        total_margin,
        open_pos_sizes,
        open_rates,
    )

    rates = all_rates[next_batch_inds, ...].permute([0, 3, 2, 1]).to(device, non_blocking=True)
    account_cur_rates = all_account_cur_rates[next_batch_inds, ...].permute([0, 2, 1]).to(device, non_blocking=True)

    # TODO: should we do this computation here or within LossEvaluator?
    loss.mean(dim=0).sum(dim=0).neg().backward()

    # TODO: optimizer step

    all_pos_states = all_pos_states.movedim(-1, 0).to("cpu", non_blocking=True)
    all_pos_types = all_pos_types.movedim(-1, 0).to("cpu", non_blocking=True)
    all_total_margins = all_total_margins.movedim(-1, 0).to("cpu", non_blocking=True)
    all_open_pos_sizes = all_open_pos_sizes.movedim(-1, 0).to("cpu", non_blocking=True)
    all_open_rates = all_open_rates.movedim(-1, 0).to("cpu", non_blocking=True)

    batch_start_inds = next_batch_start_inds
    batch_inds = next_batch_inds
    batch_window_inds = next_batch_window_inds

    # break


h5file.close()
