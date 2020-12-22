import numpy as np
import tables
import torch
import torch.distributions

batch_size = 32
n_timesteps = 35041


def sample(size, end, bias):
    dist = torch.distributions.Geometric(probs=bias)
    samples = dist.sample(size)

    count = 0
    while True:
        mask = samples >= end
        to_resample = mask.sum()
        if to_resample == 0:
            break

        samples[mask] = dist.sample((to_resample,))
        count += 1

    return (end - 1 - samples).type(torch.long)


def create_tables_file(fname, n_timesteps, n_cur, n_samples, filters=None):
    file = tables.open_file(fname, "w", filters=filters)
    file.create_carray(file.root, "pos_states", tables.Int64Atom(), (n_timesteps, n_cur, n_samples))
    file.create_carray(file.root, "pos_types", tables.Int64Atom(), (n_timesteps, n_cur, n_samples))
    file.create_carray(file.root, "total_margin", tables.Float32Atom(), (n_timesteps, n_samples))
    file.create_carray(file.root, "open_pos_sizes", tables.Float32Atom(), (n_timesteps, n_cur, n_samples))
    file.create_carray(file.root, "open_rates", tables.Float32Atom(), (n_timesteps, n_cur, n_samples))

    return file


batch_inds = sample((batch_size,), n_timesteps, 5e-5)

filters = tables.Filters(complevel=9, complib="blosc")
h5f = create_tables_file("tmp.h5", n_timesteps, 10, 500, filters=filters)

np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
batch = torch.from_numpy(h5f.root.pos_types[batch_inds.numpy(), ...])
h5f.close()
