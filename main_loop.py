import numpy as np
import tables
import torch
import torch.jit
import torch.nn as nn
import torch.optim
from pyro.distributions.transforms.affine_autoregressive import affine_autoregressive
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from itertools import chain

from sample_geometric import sample_geometric
from stochastic_waterstart2 import LossEvaluator
from waterstart_model import CNN, Emitter, GatedTransition, NeuralBaseline


def create_tables_file(fname, n_timesteps, n_cur, n_samples, z_dim, max_trades, filters=None):
    file = tables.open_file(fname, "w", filters=filters)
    file.create_carray(file.root, "hidden_state", tables.Float32Atom(), (n_timesteps, n_samples, z_dim))
    file.create_carray(file.root, "total_margin", tables.Float32Atom(dflt=1.0), (n_timesteps, n_samples))
    file.create_carray(
        file.root, "open_trades_sizes", tables.Float32Atom(), (n_timesteps, n_cur, max_trades, n_samples)
    )
    file.create_carray(
        file.root, "open_trades_rates", tables.Float32Atom(), (n_timesteps, n_cur, max_trades, n_samples)
    )

    return file


np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
torch.autograd.set_detect_anomaly(True)

eps = torch.finfo(torch.get_default_dtype()).eps
data = np.load("/content/drive/MyDrive/train_data/train_data.npz")
# all_rates: (n_timesteps, 2, n_cur)
all_rates = torch.from_numpy(data["arr"]).type(torch.float32).transpose_(1, 2)
all_rates[all_rates == 1] += eps

# all_account_cur_rates: (n_timesteps, n_cur)
all_account_cur_rates = torch.from_numpy(data["arr2"]).type(torch.float32)
all_account_cur_rates[all_account_cur_rates == 1] += eps

# all_input: (n_timesteps, in_features, n_cur)
all_input = torch.from_numpy(data["arr3"]).type(torch.float32).transpose_(1, 2)

n_timesteps, in_features, n_cur = all_input.shape
seq_len = 109
win_len = 50
n_samples = 10
max_trades = 10
z_dim = 128
# TODO: is this a good value?
out_features = 256
batch_size = 1
n_iterations = 80_000
leverage = 20

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

cnn = CNN(seq_len, n_samples, batch_size, win_len, in_features, out_features, n_cur, max_trades).to(device)

trans = GatedTransition(out_features, z_dim, 200)
emitter = Emitter(z_dim, n_cur, 200)
iafs = [affine_autoregressive(z_dim, [200]) for _ in range(2)]
nn_baseline = NeuralBaseline(out_features, z_dim, 200, n_cur)

loss_eval = LossEvaluator(
    cnn, trans, iafs, emitter, nn_baseline, batch_size, seq_len, n_samples, n_cur, max_trades, z_dim, leverage
).to(device)

sign_mask = torch.randint(2, size=(n_cur, 1, n_samples, seq_len, batch_size), dtype=torch.bool, device=device)
signs = torch.where(sign_mask, torch.ones([], device=device), -torch.ones([], device=device))
dummy_open_trades_sizes = signs * torch.rand(n_cur, max_trades, n_samples, seq_len, batch_size, device=device) / 1000
dummy_open_trades_rates = (
    torch.randn(n_cur, max_trades, n_samples, seq_len, batch_size, device=device).div_(100).add_(1)
)
dummy_open_trades_rates[dummy_open_trades_rates == 1] += eps

rand_inds = torch.randint(max_trades + 1, size=(n_cur, 1, n_samples, seq_len, batch_size))
no_trades_mask = torch.arange(max_trades).view(-1, 1, 1, 1) < rand_inds
dummy_open_trades_sizes[no_trades_mask] = 0
dummy_open_trades_rates[no_trades_mask] = 0

dummy_input = (
    torch.randn(seq_len * batch_size, in_features, n_cur, win_len, device=device),
    torch.randn(n_samples, seq_len, batch_size, z_dim, device=device),
    torch.randn(n_cur, seq_len, batch_size, device=device).div_(100).log1p_().cumsum(1).exp_()
    + torch.tensor([0.0, 1e-4], device=device).view(-1, 1, 1, 1),
    torch.randn(n_cur, seq_len, batch_size, device=device).div_(100).log1p_().cumsum(1).exp_(),
    torch.ones(n_samples, seq_len, batch_size, device=device),
    dummy_open_trades_sizes,
    dummy_open_trades_rates,
)

loss_eval = torch.jit.trace(loss_eval, dummy_input, check_trace=False)
del dummy_input, dummy_open_trades_sizes, dummy_open_trades_rates

parameters_groups = [
    {
        "params": list(
            chain(cnn.parameters(), trans.parameters(), *(iaf.parameters() for iaf in iafs), emitter.parameters())
        )
    },
    {"params": list(nn_baseline.parameters()), "lr": 1e-3},
]

# optimizer = torch.optim.Adam(parameters, lr=3e-4, weight_decay=0.1)
optimizer = torch.optim.Adam(parameters_groups, lr=3e-4, weight_decay=0.1)
parameters = [param for group in parameters_groups for param in group["params"]]
writer = SummaryWriter()
t = tqdm(total=n_iterations)

filters = tables.Filters(complevel=9, complib="blosc")
# TODO: rename the file
h5file = create_tables_file("tmp.h5", n_timesteps, n_cur, n_samples, z_dim, max_trades, filters=filters)
# h5file = tables.open_file("tmp.h5", "r+", filters=filters)

z_samples = torch.zeros(seq_len, batch_size, n_samples, z_dim)
total_margin = torch.ones(seq_len, batch_size, n_samples)
open_trades_sizes = torch.zeros(seq_len, batch_size, n_cur, max_trades, n_samples)
open_trades_rates = torch.zeros(seq_len, batch_size, n_cur, max_trades, n_samples)

all_start_inds = torch.from_numpy(
    # NOTE: in the end parameter we don't have the '+ 1' as the save_batch_inds are shifted by one
    sample_geometric(n_iterations, batch_size, win_len - 1, n_timesteps - seq_len, 5e-5, min_gap=seq_len)
)
all_start_inds_it = iter(all_start_inds)

batch_start_inds = next(all_start_inds_it)
# NOTE: it doesn't really matter what value we use, as long as it's valid
# save_batch_inds = batch_inds
save_batch_start_inds = batch_start_inds

batch_inds = torch.arange(seq_len).unsqueeze_(1) + batch_start_inds
z0 = (
    torch.from_numpy(h5file.root.hidden_state[batch_inds.flatten().numpy(), ...])
    .unflatten(0, (seq_len, batch_size))
    .permute(2, 0, 1, 3)
    .pin_memory()
    .to(device, non_blocking=True)
)

prev_total_margin = (
    torch.from_numpy(h5file.root.total_margin[batch_inds.flatten().numpy(), ...])
    .unflatten(0, (seq_len, batch_size))
    .permute(2, 0, 1)
    .pin_memory()
    .to(device, non_blocking=True)
)
prev_open_trades_sizes = (
    torch.from_numpy(h5file.root.open_trades_sizes[batch_inds.flatten().numpy(), ...])
    .unflatten(0, (seq_len, batch_size))
    .permute(2, 3, 4, 0, 1)
    .pin_memory()
    .to(device, non_blocking=True)
)
prev_open_trades_rates = (
    torch.from_numpy(h5file.root.open_trades_rates[batch_inds.flatten().numpy(), ...])
    .unflatten(0, (seq_len, batch_size))
    .permute(2, 3, 4, 0, 1)
    .pin_memory()
    .to(device, non_blocking=True)
)
rates = all_rates[batch_inds, ...].permute(2, 3, 0, 1).pin_memory().to(device, non_blocking=True)
account_cur_rates = all_account_cur_rates[batch_inds, ...].permute(2, 0, 1).pin_memory().to(device, non_blocking=True)

batch_window_inds = batch_inds.view(-1, 1) - torch.arange(win_len - 1, -1, -1)
batch_data = all_input[batch_window_inds, ...].movedim(1, -1).pin_memory().to(device, non_blocking=True)

done = False
n_iter = 0
while not done:
    save_batch_inds = torch.arange(1, seq_len + 1).unsqueeze_(1) + save_batch_start_inds
    h5file.root.hidden_state[save_batch_inds.flatten().numpy(), ...] = z_samples.flatten(0, 1).numpy()
    h5file.root.total_margin[save_batch_inds.flatten().numpy(), ...] = total_margin.flatten(0, 1).numpy()
    h5file.root.open_trades_sizes[save_batch_inds.flatten().numpy(), ...] = open_trades_sizes.flatten(0, 1).numpy()
    h5file.root.open_trades_rates[save_batch_inds.flatten().numpy(), ...] = open_trades_rates.flatten(0, 1).numpy()

    step_log_rates = total_margin[1:].log() - total_margin[:-1].log()
    positive_log_rates = torch.sum(step_log_rates > 0).item()
    negative_log_rates = torch.sum(step_log_rates < 0).item()
    tot_log_rates = step_log_rates.numel()
    writer.add_scalar("single step/positive fraction", positive_log_rates / tot_log_rates, n_iter)
    writer.add_scalar("single step/negative fraction", negative_log_rates / tot_log_rates, n_iter)

    avg_step_gain = step_log_rates.mean(2).exp_().mean().item()
    writer.add_scalar("single step/average gain", avg_step_gain, n_iter)

    whole_seq_log_rates = total_margin[-1].log() - total_margin[0].log()
    positive_log_rates = torch.sum(whole_seq_log_rates > 0).item()
    negative_log_rates = torch.sum(whole_seq_log_rates < 0).item()
    tot_log_rates = whole_seq_log_rates.numel()
    writer.add_scalar("whole sequence/positive fraction", positive_log_rates / tot_log_rates, n_iter)
    writer.add_scalar("whole sequence/negative fraction", negative_log_rates / tot_log_rates, n_iter)

    avg_whole_seq_gain = whole_seq_log_rates.mean(1).exp_().mean().item()
    writer.add_scalar("whole sequence/average gain", avg_whole_seq_gain, n_iter)

    batch_data[:, :3] /= batch_data[:, 2, None, :, -1, None]
    batch_data[:, 3:6] /= batch_data[:, 5, None, :, -1, None]

    with torch.jit.optimized_execution(False):
        surrogate_loss, loss, z_samples, total_margin, open_trades_sizes, open_trades_rates = loss_eval(
            batch_data, z0, rates, account_cur_rates, prev_total_margin, prev_open_trades_sizes, prev_open_trades_rates
        )

    try:
        load_batch_start_inds = next(all_start_inds_it)
    except StopIteration:
        # NOTE: also here it doesn't really matter what value we use
        load_batch_start_inds = batch_start_inds
        done = True

    load_batch_inds = torch.arange(seq_len).unsqueeze_(1) + load_batch_start_inds
    z0 = (
        torch.from_numpy(h5file.root.hidden_state[load_batch_inds.flatten().numpy(), ...])
        .unflatten(0, (seq_len, batch_size))
        .permute(2, 0, 1, 3)
        .pin_memory()
        .to(device, non_blocking=True)
    )
    prev_total_margin = (
        torch.from_numpy(h5file.root.total_margin[load_batch_inds.flatten().numpy(), ...])
        .unflatten(0, (seq_len, batch_size))
        .permute(2, 0, 1)
        .pin_memory()
        .to(device, non_blocking=True)
    )
    prev_open_trades_sizes = (
        torch.from_numpy(h5file.root.open_trades_sizes[load_batch_inds.flatten().numpy(), ...])
        .unflatten(0, (seq_len, batch_size))
        .permute(2, 3, 4, 0, 1)
        .pin_memory()
        .to(device, non_blocking=True)
    )
    prev_open_trades_rates = (
        torch.from_numpy(h5file.root.open_trades_rates[load_batch_inds.flatten().numpy(), ...])
        .unflatten(0, (seq_len, batch_size))
        .permute(2, 3, 4, 0, 1)
        .pin_memory()
        .to(device, non_blocking=True)
    )
    rates = all_rates[load_batch_inds, ...].permute(2, 3, 0, 1).pin_memory().to(device, non_blocking=True)
    account_cur_rates = (
        all_account_cur_rates[load_batch_inds, ...].permute(2, 0, 1).pin_memory().to(device, non_blocking=True)
    )

    next_batch_window_inds = load_batch_inds.view(-1, 1) - torch.arange(win_len - 1, -1, -1)
    batch_data = all_input[next_batch_window_inds, ...].movedim(1, -1).pin_memory().to(device, non_blocking=True)

    z_samples = z_samples.permute(1, 2, 0, 3).to("cpu", non_blocking=True)
    total_margin = total_margin.permute(1, 2, 0).to("cpu", non_blocking=True)
    open_trades_sizes = open_trades_sizes.permute(3, 4, 0, 1, 2).to("cpu", non_blocking=True)
    open_trades_rates = open_trades_rates.permute(3, 4, 0, 1, 2).to("cpu", non_blocking=True)

    surrogate_loss.mean().backward()
    grad_norm = nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
    writer.add_scalar("gradient norm", grad_norm, n_iter)
    optimizer.step()
    optimizer.zero_grad()

    loss_loc = loss.mean().item()
    loss_scale = loss.std(0).mean().item()
    writer.add_scalars(
        "loss with stdev bounds",
        {"lower bound": loss_loc - loss_scale, "value": loss_loc, "upper bound": loss_loc + loss_scale},
        n_iter,
    )

    t.update()
    save_batch_start_inds, batch_start_inds = batch_start_inds, load_batch_start_inds
    n_iter += 1

t.close()
h5file.close()
