import numpy as np
import tables
import torch
import torch.jit
import torch.nn as nn
import torch.optim
from pyro.distributions.transforms.affine_autoregressive import affine_autoregressive
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sample_geometric import sample_geometric
from stochastic_waterstart2 import LossEvaluator
from waterstart_model import CNN, Emitter, GatedTrasition


def create_tables_file(fname, n_timesteps, n_cur, n_samples, z_dim, max_trades, filters=None):
    file = tables.open_file(fname, "w", filters=filters)
    file.create_carray(file.root, "hidden_state", tables.Float32Atom(), (n_timesteps, n_samples, z_dim))
    file.create_carray(file.root, "total_margin", tables.Float32Atom(dflt=1.0), (n_timesteps, n_samples))
    file.create_carray(
        file.root, "open_trades_sizes", tables.Float32Atom(), (n_timesteps, n_samples, n_cur, max_trades)
    )
    file.create_carray(
        file.root, "open_trades_rates", tables.Float32Atom(), (n_timesteps, n_samples, n_cur, max_trades)
    )

    return file


np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
# torch.autograd.set_detect_anomaly(True)

# data = np.load("/content/drive/MyDrive/train_data/train_data.npz")
data = np.load("train_data/train_data.npz")
all_rates = torch.from_numpy(data["arr"]).type(torch.float32)
all_account_cur_rates = torch.from_numpy(data["arr2"]).type(torch.float32)
all_input = torch.from_numpy(data["arr3"]).type(torch.float32).transpose(1, 2)

n_timesteps, in_features, n_cur = all_input.shape
seq_len = 109
win_len = 50
n_samples = 500
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

cnn = CNN(win_len, in_features, out_features, n_cur, n_samples, max_trades).to(device)

trans = GatedTrasition(out_features, z_dim, 200)
emitter = Emitter(z_dim, n_cur, 200)
iafs = [affine_autoregressive(z_dim, [200]) for _ in range(2)]

loss_eval = LossEvaluator(
    cnn, trans, emitter, iafs, batch_size, seq_len, n_samples, n_cur, max_trades, z_dim, leverage
).to(device)

rand_inds = torch.randint(max_trades, size=(batch_size, seq_len, n_samples, n_cur, 1))
no_trades_mask = torch.arange(max_trades) >= rand_inds

dummy_open_trades_sizes = torch.rand(batch_size, seq_len, n_samples, n_cur, max_trades, device=device) * 100
dummy_open_trades_rates = torch.rand(batch_size, seq_len, n_samples, n_cur, max_trades, device=device)

dummy_open_trades_sizes[no_trades_mask] = 0
dummy_open_trades_rates[no_trades_mask] = 0

dummy_input = (
    torch.randn(batch_size * seq_len, in_features, n_cur, win_len, device=device),
    torch.randn(batch_size, seq_len, n_samples, z_dim, win_len, device=device),
    torch.randn(batch_size, seq_len, n_cur, 1, device=device).div_(100).log1p_().cumsum(1).exp_()
    + torch.tensor([0.0, 1e-4], device=device),
    torch.randn(batch_size, seq_len, n_cur, device=device).div_(100).log1p_().cumsum(1).exp_(),
    torch.ones(batch_size, seq_len, n_samples, device=device),
    dummy_open_trades_sizes,
    dummy_open_trades_rates,
)

loss_eval = torch.jit.trace(loss_eval, dummy_input)
del dummy_input, dummy_open_trades_sizes, dummy_open_trades_rates

parameters = list(loss_eval.parameters())
optimizer = torch.optim.Adam(parameters, lr=3e-4, weight_decay=0.01)
writer = SummaryWriter()
t = tqdm(total=n_iterations)

filters = tables.Filters(complevel=9, complib="blosc")
# TODO: rename the file
h5file = create_tables_file("tmp.h5", n_timesteps, n_cur, n_samples, z_dim, filters=filters)
# h5file = tables.open_file("tmp.h5", "r+", filters=filters)

z_samples = torch.zeros(batch_size, seq_len, n_samples, z_dim, device=device)
prev_total_margin = torch.ones(batch_size, seq_len, n_samples, device=device)
prev_open_trades_sizes = torch.zeros(batch_size, seq_len, n_samples, n_cur, max_trades, device=device)
prev_open_trades_rates = torch.zeros(batch_size, seq_len, n_samples, n_cur, max_trades, device=device)

all_start_inds = torch.from_numpy(
    sample_geometric(n_iterations, batch_size, win_len - 1, n_timesteps - seq_len, 5e-5, min_gap=seq_len)
)
all_start_inds_it = iter(all_start_inds)

batch_start_inds = next(all_start_inds_it)
batch_inds = batch_start_inds.unsqueeze(1) + torch.arange(seq_len)
batch_window_inds = batch_inds.view(-1, 1) - torch.arange(win_len - 1, -1, -1)

prev_batch_inds = None

batch_data = all_input[batch_window_inds, ...].movedim(1, -1).pin_memory().to(device, non_blocking=True)
rates = all_rates[batch_inds, ...].pin_memory().to(device, non_blocking=True)
account_cur_rates = all_account_cur_rates[batch_inds, ...].pin_memory().to(device, non_blocking=True)

z0 = torch.from_numpy(h5file.root.hidden_state[batch_inds.numpy(), ...]).pin_memory().to(device, non_blocking=True)

total_margin = (
    torch.from_numpy(h5file.root.total_margin[batch_inds.numpy(), ...]).pin_memory().to(device, non_blocking=True)
)
open_trades_sizes = (
    torch.from_numpy(h5file.root.open_trades_sizes[batch_inds.numpy(), ...]).pin_memory().to(device, non_blocking=True)
)
open_trades_rates = (
    torch.from_numpy(h5file.root.open_trades_rates[batch_inds.numpy(), ...]).pin_memory().to(device, non_blocking=True)
)

first = True
done = False
count = 0
while not done:
    if not first:
        h5file.root.hidden_state[prev_batch_inds.flatten().numpy(), ...] = z_samples.flatten(0, 1).numpy()
        h5file.root.total_margin[prev_batch_inds.flatten().numpy(), ...] = total_margin.flatten(0, 1).numpy()
        h5file.root.open_trades_sizes[prev_batch_inds.flatten().numpy(), ...] = open_trades_sizes.flatten(0, 1).numpy()
        h5file.root.open_trades_rates[prev_batch_inds.flatten().numpy(), ...] = open_trades_rates.flatten(0, 1).numpy()
    else:
        first = False

    step_log_rates = total_margin[:, 1:].log() - total_margin[:, :-1].log()
    positive_log_rates = torch.sum(step_log_rates > 0).item()
    negative_log_rates = torch.sum(step_log_rates < 0).item()
    pos_neg_ratio = positive_log_rates / (negative_log_rates if negative_log_rates > 0 else 1)
    writer.add_scalar("single step gains positive/negative ratio", pos_neg_ratio)

    avg_step_gain = step_log_rates.mean().item()
    writer.add_scalar("average single step gain", avg_step_gain)

    whole_seq_log_rates = total_margin[:, -1].log() - total_margin[:, 0].log()
    positive_log_rates = torch.sum(whole_seq_log_rates > 0).item()
    negative_log_rates = torch.sum(whole_seq_log_rates < 0).item()
    pos_neg_ratio = positive_log_rates / (negative_log_rates if negative_log_rates > 0 else 1)
    writer.add_scalar("average whole sequence gains positive/negative ratio", pos_neg_ratio)

    avg_whole_seq_gain = whole_seq_log_rates.mean().item()
    writer.add_scalar("average whole sequence gain", avg_whole_seq_gain)

    # assert torch.all(batch_data[:, 2, None, :, -1, None] != 0)
    # assert torch.all(batch_data[:, 5, None, :, -1, None] != 0)
    batch_data[:, :3] /= batch_data[:, 2, None, :, -1, None]
    batch_data[:, 3:6] /= batch_data[:, 5, None, :, -1, None]

    with torch.jit.optimized_execution(False):
        losses, z_samples, total_margin, open_trades_sizes, open_trades_rates = loss_eval(
            batch_data, z0, rates, account_cur_rates, prev_total_margin, prev_open_trades_sizes, prev_open_trades_rates
        )

    try:
        next_batch_start_inds = next(all_start_inds_it)
    except StopIteration:
        next_batch_start_inds = None
        done = True

    if not done:
        next_batch_inds = next_batch_start_inds.unsqueeze(1) + torch.arange(seq_len)
        z0 = (
            torch.from_numpy(h5file.root.hidden_state[next_batch_inds.numpy(), ...])
            .pin_memory()
            .to(device, non_blocking=True)
        )
        prev_total_margin = (
            torch.from_numpy(h5file.root.total_margin[next_batch_inds.numpy(), ...])
            .pin_memory()
            .to(device, non_blocking=True)
        )
        prev_open_trades_sizes = (
            torch.from_numpy(h5file.root.open_trades_sizes[next_batch_inds.numpy(), ...])
            .pin_memory()
            .to(device, non_blocking=True)
        )
        prev_open_trades_rates = (
            torch.from_numpy(h5file.root.open_trades_rates[next_batch_inds.numpy(), ...])
            .pin_memory()
            .to(device, non_blocking=True)
        )
        rates = all_rates[next_batch_inds, ...].pin_memory().to(device, non_blocking=True)
        account_cur_rates = all_account_cur_rates[next_batch_inds, ...].pin_memory().to(device, non_blocking=True)

        next_batch_window_inds = next_batch_inds.view(-1, 1) - torch.arange(win_len - 1, -1, -1)
        batch_data = all_input[next_batch_window_inds, ...].movedim(1, -1).pin_memory().to(device, non_blocking=True)

        z_samples = z_samples.to("cpu", non_blocking=True)
        total_margin = total_margin.to("cpu", non_blocking=True)
        open_trades_sizes = open_trades_sizes.to("cpu", non_blocking=True)
        open_trades_rates = open_trades_rates.to("cpu", non_blocking=True)

        prev_batch_inds, batch_inds = batch_inds, next_batch_inds

    loss = losses.mean(2).sum()
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
    writer.add_scalar("gradient norm", grad_norm)
    optimizer.step()
    optimizer.zero_grad()

    losses = losses.detach()
    loss_loc = losses.mean().item()
    loss_scale = losses.std(2).mean().item()
    writer.add_scalars(
        "loss with stdev bounds",
        {"lower bound": loss_loc - loss_scale, "value": loss_loc, "upper bound": loss_loc + loss_scale},
    )
    t.update()
    count += 1

t.close()
h5file.close()
