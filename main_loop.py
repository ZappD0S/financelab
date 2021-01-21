import numpy as np
import tables
import torch
import torch.optim
from pyro.distributions.transforms.affine_autoregressive import affine_autoregressive

from sample_geometric import sample_geometric
from SSMEvaluator import SSMEvaluator
from stochastic_waterstart5 import LossEvaluator
from waterstart_model import CNN, Emitter, GatedTrasition


def create_tables_file(fname, n_timesteps, n_cur, n_samples, z_dim, filters=None):
    file = tables.open_file(fname, "w", filters=filters)
    file.create_carray(file.root, "hidden_state", tables.Float32Atom(), (n_timesteps, n_samples, z_dim))
    file.create_carray(file.root, "pos_states", tables.Int8Atom(), (n_timesteps, n_cur, n_samples))
    file.create_carray(file.root, "pos_types", tables.Int8Atom(), (n_timesteps, n_cur, n_samples))
    file.create_carray(file.root, "total_margin", tables.Float32Atom(), (n_timesteps, n_samples))
    file.create_carray(file.root, "open_pos_sizes", tables.Float32Atom(), (n_timesteps, n_cur, n_samples))
    file.create_carray(file.root, "open_rates", tables.Float32Atom(), (n_timesteps, n_cur, n_samples))

    return file


np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
# TODO: remove this
torch.autograd.set_detect_anomaly(True)

data = np.load("train_data/train_data.npz")
all_rates = torch.from_numpy(data["arr"]).type(torch.float32)
all_account_cur_rates = torch.from_numpy(data["arr2"]).type(torch.float32)
all_input = torch.from_numpy(data["arr3"]).type(torch.float32).transpose(1, 2)

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
trans = GatedTrasition(out_features, z_dim, 200)
emitter = Emitter(z_dim, n_cur, 200)
iafs = [affine_autoregressive(z_dim, [200]) for _ in range(2)]

# TODO: jit trace
ssm_eval = SSMEvaluator(trans, emitter, iafs, seq_len, batch_size, n_samples, n_cur).to(device)
loss_eval = LossEvaluator(seq_len, batch_size, n_samples, n_cur, leverage=1).to(device)

optimizer = torch.optim.Adam(ssm_eval.parameters(), lr=1e-3)

# dummy_input = (
#     torch.randn(seq_len, batch_size, out_features, device=device),
#     torch.randn(batch_size, n_samples, z_dim, device=device),
# )
# ssm_eval = torch.jit.trace(ssm_eval, dummy_input, check_trace=False)

filters = tables.Filters(complevel=9, complib="blosc")
h5file = create_tables_file("tmp.h5", n_timesteps, n_cur, n_samples, z_dim, filters=filters)
# h5file = tables.open_file("tmp.h5", "r+", filters=filters)

z_samples = torch.zeros(seq_len, batch_size, n_samples, z_dim, device=device)
all_pos_states = torch.zeros(seq_len, batch_size, n_cur, n_samples, device=device)
all_pos_types = torch.zeros(seq_len, batch_size, n_cur, n_samples, device=device)
all_total_margins = torch.zeros(seq_len, batch_size, n_samples, device=device)
all_open_pos_sizes = torch.zeros(seq_len, batch_size, n_cur, n_samples, device=device)
all_open_rates = torch.zeros(seq_len, batch_size, n_cur, n_samples, device=device)

all_start_inds = torch.from_numpy(
    sample_geometric(n_iterations, batch_size, win_len - 1, n_timesteps + 1 - seq_len, 5e-5, min_gap=seq_len)
)
all_start_inds_it = iter(all_start_inds)

batch_start_inds = next(all_start_inds_it)
batch_inds = torch.arange(seq_len).unsqueeze(-1) + batch_start_inds
batch_window_inds = batch_inds.view(-1, 1) - torch.arange(win_len - 1, -1, -1)

prev_batch_inds = None

batch_data = all_input[batch_window_inds, ...].movedim(1, -1).pin_memory().to(device, non_blocking=True)
rates = all_rates[batch_inds, ...].permute([0, 3, 2, 1]).pin_memory().to(device, non_blocking=True)
account_cur_rates = all_account_cur_rates[batch_inds, ...].permute([0, 2, 1]).pin_memory().to(device, non_blocking=True)

z0 = (
    torch.from_numpy(h5file.root.hidden_state[batch_start_inds.numpy(), ...]).pin_memory().to(device, non_blocking=True)
)

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

first = True
done = False
count = 0
while not done:
    print(count)
    count += 1

    try:
        next_batch_start_inds = next(all_start_inds_it)
        next_batch_inds = torch.arange(seq_len).unsqueeze(-1) + next_batch_start_inds
        next_batch_window_inds = next_batch_inds.view(-1, 1) - torch.arange(win_len - 1, -1, -1)
    except StopIteration:
        done = True
        next_batch_start_inds = next_batch_inds = next_batch_window_inds = None

    batch_data[:, :3] /= batch_data[:, 1, None, :, -1, None]
    batch_data[:, 3:6] /= batch_data[:, 4, None, :, -1, None]
    out = cnn(batch_data).view(seq_len, batch_size, out_features)

    if not done:
        batch_data = all_input[next_batch_window_inds, ...].movedim(1, -1).pin_memory().to(device, non_blocking=True)

    if not first:
        h5file.root.hidden_state[prev_batch_inds.flatten().numpy(), ...] = z_samples.flatten(0, 1).numpy()

    x_samples, z_samples, x_logprobs, z_logprobs, fractions = ssm_eval(out, z0)

    if not done:
        z_samples = z_samples.to("cpu", non_blocking=True)

        z0 = (
            torch.from_numpy(h5file.root.hidden_state[next_batch_start_inds.numpy(), ...])
            .pin_memory()
            .to(device, non_blocking=True)
        )

    x_samples = x_samples.permute([0, 4, 3, 2, 1])
    x_logprobs = x_logprobs.permute([0, 4, 3, 2, 1])
    z_logprobs = z_logprobs.permute([0, 2, 1])
    fractions = fractions.permute([0, 3, 2, 1])

    if not first:
        h5file.root.pos_states[prev_batch_inds.flatten().numpy(), ...] = all_pos_states.flatten(0, 1).numpy()
        h5file.root.pos_types[prev_batch_inds.flatten().numpy(), ...] = all_pos_types.flatten(0, 1).numpy()
        h5file.root.total_margin[prev_batch_inds.flatten().numpy(), ...] = all_total_margins.flatten(0, 1).numpy()
        h5file.root.open_pos_sizes[prev_batch_inds.flatten().numpy(), ...] = all_open_pos_sizes.flatten(0, 1).numpy()
        h5file.root.open_rates[prev_batch_inds.flatten().numpy(), ...] = all_open_rates.flatten(0, 1).numpy()

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

    if not done:
        all_pos_states = all_pos_states.movedim(-1, 1).to("cpu", non_blocking=True)
        all_pos_types = all_pos_types.movedim(-1, 1).to("cpu", non_blocking=True)
        all_total_margins = all_total_margins.movedim(-1, 1).to("cpu", non_blocking=True)
        all_open_pos_sizes = all_open_pos_sizes.movedim(-1, 1).to("cpu", non_blocking=True)
        all_open_rates = all_open_rates.movedim(-1, 1).to("cpu", non_blocking=True)

        pos_states = (
            torch.from_numpy(h5file.root.pos_states[next_batch_start_inds.numpy(), ...])
            .movedim(0, -1)
            .pin_memory()
            .to(device, non_blocking=True)
        )
        pos_types = (
            torch.from_numpy(h5file.root.pos_types[next_batch_start_inds.numpy(), ...])
            .movedim(0, -1)
            .pin_memory()
            .to(device, non_blocking=True)
        )
        total_margin = (
            torch.from_numpy(h5file.root.total_margin[next_batch_start_inds.numpy(), ...])
            .movedim(0, -1)
            .pin_memory()
            .to(device, non_blocking=True)
        )
        open_pos_sizes = (
            torch.from_numpy(h5file.root.open_pos_sizes[next_batch_start_inds.numpy(), ...])
            .movedim(0, -1)
            .pin_memory()
            .to(device, non_blocking=True)
        )
        open_rates = (
            torch.from_numpy(h5file.root.open_rates[next_batch_start_inds.numpy(), ...])
            .movedim(0, -1)
            .pin_memory()
            .to(device, non_blocking=True)
        )

        rates = all_rates[next_batch_inds, ...].permute([0, 3, 2, 1]).pin_memory().to(device, non_blocking=True)
        account_cur_rates = (
            all_account_cur_rates[next_batch_inds, ...].permute([0, 2, 1]).pin_memory().to(device, non_blocking=True)
        )

    # TODO: should we do this computation here or within LossEvaluator?
    loss.mean(dim=0).sum(dim=0).neg().backward()
    optimizer.step()

    if first:
        first = False

    prev_batch_inds, batch_inds = batch_inds, next_batch_inds

h5file.close()
