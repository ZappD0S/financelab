from functools import partial
from itertools import chain

import numpy as np
import tables
import torch
import torch.jit
import torch.nn as nn
import torch.optim
from numpy.lib.npyio import NpzFile
from pyro.distributions.transforms.affine_autoregressive import affine_autoregressive
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sample_geometric import sample_geometric
from sliding_window_view import sliding_window_view
from stochastic_waterstart3 import LossEvaluator
from waterstart_model import CNN, Emitter, GatedTransition, NeuralBaseline


def create_tables_file(fname, n_timesteps, n_cur, n_samples, z_dim, filters=None):
    file = tables.open_file(fname, "w", filters=filters)
    file.create_carray(file.root, "hidden_state", tables.Float32Atom(), (n_timesteps, n_samples, z_dim))
    file.create_carray(file.root, "total_margin", tables.Float32Atom(dflt=1.0), (n_timesteps, n_samples))
    file.create_carray(file.root, "pos_timesteps", tables.Int32Atom(dflt=-1), (n_timesteps, n_samples, n_cur))

    file.create_carray(file.root, "pos_sizes", tables.Float32Atom(), (n_timesteps, n_samples, n_cur))
    file.create_carray(file.root, "pos_rates", tables.Float32Atom(), (n_timesteps, n_samples, n_cur))

    return file


def load_next_state(
    load_batch_inds: np.ndarray,
    all_rates: np.ndarray,
    all_account_cur_rates: np.ndarray,
    all_market_data: np.ndarray,
    group: tables.Group,
):
    def _prepare_output():
        return (
            pos_timesteps,
            torch.from_numpy(open_pos_mask)
            .permute(2, 1, 0)
            .unflatten(2, (seq_len, batch_size))
            .pin_memory()
            .to(device, non_blocking=True),
            z0.permute(0, 2, 1, 3).unflatten(2, (seq_len, batch_size)),
            prev_total_margin.permute(0, 2, 1).unflatten(2, (seq_len, batch_size)),
            prev_pos_sizes.permute(0, 3, 2, 1).unflatten(3, (seq_len, batch_size)),
            prev_pos_rates.permute(0, 3, 2, 1).unflatten(3, (seq_len, batch_size)),
            rates.permute(0, 4, 3, 2, 1).unflatten(4, (seq_len, batch_size)),
            account_cur_rates.permute(0, 3, 2, 1).unflatten(3, (seq_len, batch_size)),
            market_data.permute(0, 2, 1, 4, 3, 5).unflatten(2, (seq_len, batch_size)),
        )

    # TODO: maybe create these tensors already in their final shape and use transpose on
    # the array that we load from disk before moving it to device
    z0 = torch.zeros(n_cur + 1, seq_len * batch_size, n_samples, z_dim, device=device)
    prev_total_margin = torch.ones(n_cur + 1, seq_len * batch_size, n_samples, device=device)
    prev_pos_sizes = torch.zeros(n_cur + 1, seq_len * batch_size, n_samples, n_cur, device=device)
    prev_pos_rates = torch.zeros(n_cur + 1, seq_len * batch_size, n_samples, n_cur, device=device)
    rates = torch.zeros(n_cur + 1, seq_len * batch_size, n_samples, n_cur, 2, device=device)
    account_cur_rates = torch.ones(n_cur + 1, seq_len * batch_size, n_samples, n_cur, device=device)
    market_data = torch.zeros(n_cur + 1, seq_len * batch_size, n_samples, n_cur, in_features, win_len, device=device)

    z0[-1] = (
        torch.from_numpy(group.hidden_state[load_batch_inds.ravel(), ...]).pin_memory().to(device, non_blocking=True)
    )
    prev_total_margin[-1] = (
        torch.from_numpy(group.total_margin[load_batch_inds.ravel(), ...]).pin_memory().to(device, non_blocking=True)
    )
    prev_pos_sizes[-1] = (
        torch.from_numpy(group.pos_sizes[load_batch_inds.ravel(), ...]).pin_memory().to(device, non_blocking=True)
    )
    prev_pos_rates[-1] = (
        torch.from_numpy(group.pos_rates[load_batch_inds.ravel(), ...]).pin_memory().to(device, non_blocking=True)
    )
    rates[-1] = (
        torch.from_numpy(all_rates[load_batch_inds.ravel(), ...])
        .unsqueeze_(1)
        .pin_memory()
        .to(device, non_blocking=True)
    )
    account_cur_rates[-1] = (
        torch.from_numpy(all_account_cur_rates[load_batch_inds.ravel(), ...])
        .unsqueeze_(1)
        .pin_memory()
        .to(device, non_blocking=True)
    )
    market_data[-1] = (
        torch.from_numpy(all_market_data[load_batch_inds.ravel() - win_len + 1, ...])
        .unsqueeze_(1)
        .pin_memory()
        .to(device, non_blocking=True)
    )

    pos_timesteps = group.pos_timesteps[load_batch_inds.ravel(), ...]
    open_pos_mask = pos_timesteps >= 0

    if not open_pos_mask.any():
        return _prepare_output()

    timestep_inds, sample_inds, cur_inds = open_pos_mask.nonzero()
    open_pos_timesteps = pos_timesteps[timestep_inds, sample_inds, cur_inds]

    unique_timestep_inds, inverse_timestep_inds = np.unique(open_pos_timesteps, return_inverse=True)

    unique_z0 = (
        torch.from_numpy(group.hidden_state[unique_timestep_inds, ...]).pin_memory().to(device, non_blocking=True)
    )
    z0[cur_inds, timestep_inds, sample_inds] = unique_z0[inverse_timestep_inds, sample_inds]

    unique_prev_total_margin = (
        torch.from_numpy(group.total_margin[unique_timestep_inds, ...]).pin_memory().to(device, non_blocking=True)
    )
    prev_total_margin[cur_inds, timestep_inds, sample_inds] = unique_prev_total_margin[
        inverse_timestep_inds, sample_inds
    ]

    unique_prev_pos_sizes = (
        torch.from_numpy(group.pos_sizes[unique_timestep_inds, ...]).pin_memory().to(device, non_blocking=True)
    )
    prev_pos_sizes[cur_inds, timestep_inds, sample_inds] = unique_prev_pos_sizes[inverse_timestep_inds, sample_inds]

    unique_prev_pos_rates = (
        torch.from_numpy(group.pos_rates[unique_timestep_inds, ...]).pin_memory().to(device, non_blocking=True)
    )
    prev_pos_rates[cur_inds, timestep_inds, sample_inds] = unique_prev_pos_rates[inverse_timestep_inds, sample_inds]

    unique_rates = (
        torch.from_numpy(all_rates[unique_timestep_inds, ...]).unsqueeze_(1).pin_memory().to(device, non_blocking=True)
    )
    rates[cur_inds, timestep_inds, sample_inds] = unique_rates[inverse_timestep_inds]

    unique_account_cur_rates = (
        torch.from_numpy(all_account_cur_rates[unique_timestep_inds, ...])
        .unsqueeze_(1)
        .pin_memory()
        .to(device, non_blocking=True)
    )
    account_cur_rates[cur_inds, timestep_inds, sample_inds] = unique_account_cur_rates[inverse_timestep_inds]

    unique_market_data = (
        torch.from_numpy(all_market_data[unique_timestep_inds - win_len + 1, ...])
        .unsqueeze_(2)
        .pin_memory()
        .to(device, non_blocking=True)
    )
    market_data[cur_inds, timestep_inds, sample_inds] = unique_market_data[inverse_timestep_inds]

    return _prepare_output()


def save_prev_state(
    save_batch_inds: np.ndarray,
    pos_timesteps: np.ndarray,
    open_mask: torch.Tensor,
    close_mask: torch.Tensor,
    z_samples: torch.Tensor,
    total_margin: torch.Tensor,
    pos_sizes: torch.Tensor,
    pos_rates: torch.Tensor,
    group: tables.Group,
):
    open_mask = open_mask.flatten(2, 3).numpy().transpose(2, 1, 0)
    close_mask = close_mask.flatten(2, 3).numpy().transpose(2, 1, 0)

    save_batch_pos_timesteps = np.where(
        open_mask, np.reshape(save_batch_inds - 1, [-1, 1, 1]), np.where(close_mask, -1, pos_timesteps),
    )

    group.pos_timesteps[save_batch_inds.ravel(), ...] = save_batch_pos_timesteps
    group.hidden_state[save_batch_inds.ravel(), ...] = z_samples.flatten(1, 2).numpy().transpose(1, 0, 2)
    group.total_margin[save_batch_inds.ravel(), ...] = total_margin.flatten(1, 2).numpy().transpose(1, 0)
    group.pos_sizes[save_batch_inds.ravel(), ...] = pos_sizes.flatten(2, 3).numpy().transpose(2, 1, 0)
    group.pos_rates[save_batch_inds.ravel(), ...] = pos_rates.flatten(2, 3).numpy().transpose(2, 1, 0)


def write_margin_stats(
    total_margin: torch.Tensor, loss: torch.Tensor, grad_norm: torch.Tensor, n_iter: int, writer: SummaryWriter
):
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

    loss_loc = loss.mean().item()
    loss_scale = loss.std(0).mean().item()
    writer.add_scalars(
        "loss with stdev bounds",
        {"lower bound": loss_loc - loss_scale, "value": loss_loc, "upper bound": loss_loc + loss_scale},
        n_iter,
    )

    writer.add_scalar("gradient norm", grad_norm.item(), n_iter)


def move_state_to_cpu(
    z_samples: torch.Tensor,
    total_margin: torch.Tensor,
    pos_sizes: torch.Tensor,
    pos_rates: torch.Tensor,
    open_mask: torch.Tensor,
    close_mask: torch.Tensor,
):
    z_samples = z_samples.to("cpu", non_blocking=True)
    total_margin = total_margin.to("cpu", non_blocking=True)
    pos_sizes = pos_sizes.to("cpu", non_blocking=True)
    pos_rates = pos_rates.to("cpu", non_blocking=True)
    open_mask = open_mask.to("cpu", non_blocking=True)
    close_mask = close_mask.to("cpu", non_blocking=True)

    return z_samples, total_margin, pos_sizes, pos_rates, open_mask, close_mask


def trace_loss_eval(loss_eval: LossEvaluator) -> LossEvaluator:
    sign_mask = torch.randint(
        2, size=(n_cur + 1, n_cur, n_samples, seq_len, batch_size), dtype=torch.bool, device=device
    )
    signs = torch.where(sign_mask, torch.ones([], device=device), torch.ones([], device=device).neg_())
    dummy_pos_sizes = signs * torch.rand(n_cur + 1, n_cur, n_samples, seq_len, batch_size, device=device) / 1000
    dummy_pos_rates = torch.randn(n_cur + 1, n_cur, n_samples, seq_len, batch_size, device=device).div_(100).add_(1)

    no_trades_mask = torch.randint(2, size=(n_cur + 1, n_cur, n_samples, seq_len, batch_size), dtype=torch.bool)
    dummy_pos_sizes[no_trades_mask] = 0
    dummy_pos_rates[no_trades_mask] = 0

    same_pos_mask = torch.randint(2, size=(n_cur, n_samples, seq_len, batch_size), dtype=torch.bool)
    cur_inds, *batch_inds = same_pos_mask.nonzero(as_tuple=True)
    dummy_pos_rates[(-1, cur_inds, *batch_inds)] = dummy_pos_rates[(cur_inds, cur_inds, *batch_inds)]

    dummy_input = (
        torch.randn(n_cur + 1, n_samples, seq_len, batch_size, in_features, n_cur, win_len, device=device),
        torch.randn(n_cur + 1, n_samples, seq_len, batch_size, z_dim, device=device),
        torch.randn(n_cur + 1, 1, n_cur, seq_len, batch_size, device=device).div_(100).log1p_().cumsum(3).exp_()
        + torch.tensor([0.0, 1e-4], device=device).view(-1, 1, 1, 1),
        torch.randn(n_cur + 1, n_cur, seq_len, batch_size, device=device).div_(100).log1p_().cumsum(2).exp_(),
        torch.ones(n_cur + 1, n_samples, seq_len, batch_size, device=device),
        dummy_pos_sizes,
        dummy_pos_rates,
        same_pos_mask,
    )

    return torch.jit.trace(loss_eval, dummy_input, check_trace=False)


def load_rates_and_market_data(file: NpzFile, win_len: int):
    # all_rates: (n_timesteps, n_cur, 2)
    all_rates: np.ndarray = file["arr"].astype(np.float32)

    # all_account_cur_rates: (n_timesteps, n_cur)
    all_account_cur_rates: np.ndarray = file["arr2"].astype(np.float32)

    # all_market_data: (n_timesteps, n_cur, in_features)
    all_market_data: np.ndarray = file["arr3"].astype(np.float32)

    # all_market_data: (n_timesteps, n_cur, in_features, win_len)
    all_market_data = sliding_window_view(all_market_data, window_shape=win_len, axis=0)

    return all_rates, all_account_cur_rates, all_market_data


np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
torch.autograd.set_detect_anomaly(True)

seq_len = 109
win_len = 50
n_samples = 10
# max_trades = 10
z_dim = 128
# TODO: is this a good value?
out_features = 256
batch_size = 1
n_iterations = 80_000
leverage = 20

file = np.load("/content/drive/MyDrive/train_data/train_data.npz")
# data: NpzFile = np.load("train_data/train_data.npz")

all_rates, all_account_cur_rates, all_market_data = load_rates_and_market_data(file, win_len)
n_timesteps, n_cur, in_features, _ = all_market_data.shape

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

cnn = CNN(seq_len, n_samples, batch_size, win_len, in_features, out_features, n_cur).to(device)

trans = GatedTransition(out_features, z_dim, 200)
emitter = Emitter(z_dim, n_cur, 200)
iafs = [affine_autoregressive(z_dim, [200]) for _ in range(2)]
nn_baseline = NeuralBaseline(out_features, z_dim, 200, n_cur)

loss_eval = LossEvaluator(
    cnn, trans, iafs, emitter, nn_baseline, batch_size, seq_len, n_samples, n_cur, z_dim, leverage
).to(device)

loss_eval = trace_loss_eval(loss_eval)

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
h5file = create_tables_file("tmp.h5", n_timesteps, n_cur, n_samples, z_dim, filters=filters)
# h5file = tables.open_file("tmp.h5", "r+", filters=filters)

# NOTE: in the end parameter we don't have the '+ 1' as the save_batch_inds are shifted by one
all_start_inds = sample_geometric(n_iterations, batch_size, win_len - 1, n_timesteps - seq_len, 5e-5, min_gap=seq_len)
all_start_inds_it = iter(all_start_inds)

batch_start_inds = next(all_start_inds_it)
# NOTE: it doesn't really matter what value we use, as long as it's valid
# save_batch_inds = batch_inds
save_batch_start_inds = batch_start_inds

load_next_state = partial(
    load_next_state,
    all_rates=all_rates,
    all_account_cur_rates=all_account_cur_rates,
    all_market_data=all_market_data,
    group=h5file.root,
)
save_prev_state = partial(save_prev_state, group=h5file.root)

batch_inds = np.arange(seq_len).reshape(-1, 1) + batch_start_inds

(
    pos_timesteps,
    open_pos_mask,
    z0,
    prev_total_margin,
    prev_pos_sizes,
    prev_pos_rates,
    rates,
    account_cur_rates,
    market_data,
) = load_next_state(batch_inds)

z_samples = torch.zeros(n_samples, seq_len, batch_size, z_dim)
total_margin = torch.ones(n_samples, seq_len, batch_size)
pos_sizes = torch.zeros(n_cur, n_samples, seq_len, batch_size)
pos_rates = torch.zeros(n_cur, n_samples, seq_len, batch_size)
open_mask = torch.zeros(n_cur, n_samples, seq_len, batch_size)
close_mask = torch.zeros(n_cur, n_samples, seq_len, batch_size)

done = False
n_iter = 0
while not done:
    save_batch_inds = np.arange(1, seq_len + 1).reshape(-1, 1) + save_batch_start_inds

    save_prev_state(
        save_batch_inds, pos_timesteps, open_mask, close_mask, z_samples, total_margin, pos_sizes, pos_rates
    )

    last_close_rates = market_data[..., 2, None, :, -1, None]
    last_close_account_cur_rates = market_data[..., 5, None, :, -1, None]

    market_data[..., :3, :, :] /= last_close_rates.where(last_close_rates != 0, market_data.new_ones([]))
    market_data[..., 3:6, :, :] /= last_close_account_cur_rates.where(
        last_close_account_cur_rates != 0, market_data.new_ones([])
    )

    with torch.jit.optimized_execution(False):
        total_margin, pos_sizes, pos_rates, open_mask, close_mask, surrogate_loss, loss = loss_eval(
            market_data, z0, rates, account_cur_rates, prev_total_margin, prev_pos_sizes, prev_pos_rates, open_pos_mask
        )

    try:
        load_batch_start_inds = next(all_start_inds_it)
    except StopIteration:
        # NOTE: also here it doesn't really matter what value we use
        load_batch_start_inds = batch_start_inds
        done = True

    load_batch_inds = np.arange(seq_len).reshape(-1, 1) + load_batch_start_inds

    (
        pos_timesteps,
        open_pos_mask,
        z0,
        prev_total_margin,
        prev_pos_sizes,
        prev_pos_rates,
        rates,
        account_cur_rates,
        market_data,
    ) = load_next_state(load_batch_inds)

    z_samples, total_margin, pos_sizes, pos_rates, open_mask, close_mask = move_state_to_cpu(
        z_samples, total_margin, pos_sizes, pos_rates, open_mask, close_mask
    )

    surrogate_loss.mean(0).sum().backward()
    # TODO: should we compute the mean over all dims?
    # surrogate_loss.mean().backward()
    grad_norm = nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
    optimizer.step()
    optimizer.zero_grad()

    write_margin_stats(total_margin, loss, grad_norm, n_iter, writer)

    t.update()
    save_batch_start_inds, batch_start_inds = batch_start_inds, load_batch_start_inds
    n_iter += 1

t.close()
h5file.close()
