# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'


# %%
import numpy as np
from numba import njit

@njit(fastmath=True)
def sample_geometric(n_batches, batch_size, start, end, bias, min_gap=1):
    max_length = end - start - 1

    if (max_length - 1) // min_gap + 1 < 2 * batch_size:
        raise Exception

    out = np.zeros(n_batches * batch_size, dtype=np.int64)
    inds = np.zeros(2 * batch_size, dtype=np.int64)
    right_inds = np.zeros(batch_size, dtype=np.int64)
    sorted_values = np.empty(2 * batch_size, dtype=np.int64)
    right_sorted_values = np.empty(batch_size, dtype=np.int64)

    tmp_inds = np.zeros(2 * batch_size, dtype=np.int64)
    tmp_right_inds = np.zeros(batch_size, dtype=np.int64)
    tmp_sorted_values = np.empty(2 * batch_size, dtype=np.int64)
    tmp_right_sorted_values = np.empty(batch_size, dtype=np.int64)

    sample_size = 1
    good_samples = 0
    tot_samples = 0

    for i in range(batch_size):
        # print(i)
        found = False
        while not found:
            samples = np.random.geometric(bias, size=sample_size) - 1

            for sample in samples:
                tot_samples += 1

                if sample > max_length:
                    continue

                # assert np.all(np.diff(right_sorted_values[:i]) >= 0)
                ind = np.searchsorted(right_sorted_values[:i], sample)

                if ind > 0 and sample - right_sorted_values[ind - 1] < min_gap:
                    continue

                if ind < i and right_sorted_values[ind] - sample < min_gap:
                    continue

                out[i] = sample
                right_inds[i] = ind

                tmp_right_inds[: i + 1] = right_inds[: i + 1]
                inds_view = tmp_right_inds[:i]
                inds_view[inds_view >= ind] += 1

                inds_view = tmp_right_inds[: i + 1]
                tmp_right_sorted_values[inds_view] = out[: i + 1]
                sorted_values_view = tmp_right_sorted_values[: i + 1]
                # assert np.all(np.diff(sorted_values_view) >= 0)

                places_left = np.sum((np.diff(sorted_values_view) - 1 - (min_gap - 1)) // min_gap)
                places_left += (sorted_values_view[0] - (min_gap - 1)) // min_gap
                places_left += (max_length - sorted_values_view[-1] - (min_gap - 1)) // min_gap

                if places_left < 2 * batch_size - i - 1:
                    continue

                right_inds[: i + 1] = tmp_right_inds[: i + 1]
                right_sorted_values[: i + 1] = tmp_right_sorted_values[: i + 1]
                found = True
                good_samples += 1
                break

            sample_size = int(tot_samples / good_samples) if good_samples > 0 else 2 * sample_size

    for i in range(batch_size, n_batches * batch_size):
        # print(i)
        batch_pos = i % batch_size
        last_batch_start = i - batch_pos - batch_size
        batch_pair_view = out[last_batch_start : last_batch_start + 2 * batch_size]

        if batch_pos == 0:
            inds[:batch_size] = right_inds
            sorted_values[:batch_size] = right_sorted_values

        found = False
        while not found:
            samples = np.random.geometric(bias, size=sample_size) - 1

            for sample in samples:
                tot_samples += 1

                if sample > max_length:
                    continue

                # assert np.all(np.diff(sorted_values[: batch_size + batch_pos]) >= 0)
                ind = np.searchsorted(sorted_values[: batch_size + batch_pos], sample)

                if ind > 0 and sample - sorted_values[ind - 1] < min_gap:
                    continue

                if ind < batch_size + batch_pos and sorted_values[ind] - sample < min_gap:
                    continue

                batch_pair_view[batch_size + batch_pos] = sample
                inds[batch_size + batch_pos] = ind

                tmp_inds[: batch_size + batch_pos + 1] = inds[: batch_size + batch_pos + 1]
                inds_view = tmp_inds[: batch_size + batch_pos]
                inds_view[inds_view >= ind] += 1

                inds_view = tmp_inds[: batch_size + batch_pos + 1]
                tmp_sorted_values[inds_view] = batch_pair_view[: batch_size + batch_pos + 1]
                sorted_values_view = tmp_sorted_values[: batch_size + batch_pos + 1]

                places_left = np.sum((np.diff(sorted_values_view) - 1 - (min_gap - 1)) // min_gap)
                places_left += (sorted_values_view[0] - (min_gap - 1)) // min_gap
                places_left += (max_length - sorted_values_view[-1] - (min_gap - 1)) // min_gap

                if places_left < batch_size - batch_pos - 1:
                    continue

                # assert np.all(np.diff(right_sorted_values[:batch_pos]) >= 0)
                right_ind = np.searchsorted(right_sorted_values[:batch_pos], sample)

                right_inds[batch_pos] = right_ind
                tmp_right_inds[: batch_pos + 1] = right_inds[: batch_pos + 1]
                inds_view = tmp_right_inds[:batch_pos]
                inds_view[inds_view >= right_ind] += 1

                inds_view = tmp_right_inds[: batch_pos + 1]
                tmp_right_sorted_values[inds_view] = batch_pair_view[batch_size : batch_size + batch_pos + 1]
                sorted_values_view = tmp_right_sorted_values[: batch_pos + 1]

                places_left = np.sum((np.diff(sorted_values_view) - 1 - (min_gap - 1)) // min_gap)
                places_left += (sorted_values_view[0] - (min_gap - 1)) // min_gap
                places_left += (max_length - sorted_values_view[-1] - (min_gap - 1)) // min_gap

                if places_left < 2 * batch_size - batch_pos - 1:
                    continue

                inds[: batch_size + batch_pos + 1] = tmp_inds[: batch_size + batch_pos + 1]
                right_inds[: batch_pos + 1] = tmp_right_inds[: batch_pos + 1]
                sorted_values[: batch_size + batch_pos + 1] = tmp_sorted_values[: batch_size + batch_pos + 1]
                right_sorted_values[: batch_pos + 1] = tmp_right_sorted_values[: batch_pos + 1]

                found = True
                good_samples += 1
                break

            sample_size = int(tot_samples / good_samples) if good_samples > 0 else 2 * sample_size

    return end - 1 - out


# res = sample_geometric2(100, 10, 10, 100, 1 / 50, 3).reshape(100, 10)
# res = sample_geometric(100, 10, 10, 100, 1 / 50, 4).reshape(100, 10)

res = sample_geometric(80_000, 10, 30 - 1, 25_000 + 1 - 109, 5e-5, min_gap=109)
