import numpy as np
import pandas as pd
from numba import njit, prange, cuda, float64, float32


@njit(parallel=True)
def compute_stuff(p, lower, upper):
    res = np.zeros((lower.size, upper.size, 2))

    for t0 in prange(p.size):
        for l in prange(lower.size):
            for u in prange(upper.size):
                for t in range(t0 + 1, p.size):
                    if p[t] - p[t0] < lower[l]:
                        res[l, u, 0] += 1
                        break
                    if p[t] - p[t0] > upper[u]:
                        res[l, u, 1] += 1
                        break
    return res


@cuda.jit
def compute_stuff_cuda(p, lower, upper, out):
    shared_out = cuda.shared.array((l_tpb, u_tpb, 2), dtype=float32)

    shared_p = cuda.shared.array(frame_size, dtype=float32)
    shared_lower = cuda.shared.array(l_tpb, dtype=float32)
    shared_upper = cuda.shared.array(u_tpb, dtype=float32)
    i0, lu = cuda.grid(2)

    # (lower.size, upper.size)
    # ((i*J + j)*K + k)*L + l
    l = lu // upper.size
    u = lu % upper.size

    if l >= lower.size or u >= upper.size:
        return

    frames_per_grid = p.size // frame_size

    if i0 >= frames_per_grid * frame_size:
        return

    ti0 = cuda.threadIdx.x
    tlu = cuda.threadIdx.y

    tl = tlu // u_tpb
    tu = tlu % u_tpb

    shared_lower[tl] = lower[tl]
    shared_upper[tu] = upper[tu]

    left_frames = (frames_per_grid * frame_size - i0 + frame_size - 1) // frame_size

    found = False
    for frame in range(left_frames):
        for i in range(multiplier):
            shared_p[ti0 * multiplier + i] = p[i0 + frame * frame_size + ti0 * (multiplier - 1) + i]

        cuda.syncthreads()

        for ti in range(ti0, frame_size):
            if shared_p[ti] - shared_p[ti0] < shared_lower[tl]:
                shared_out[tl, tu, 0] += 1
                found = True
                break
            if shared_p[ti] - shared_p[ti0] > shared_upper[tu]:
                shared_out[tl, tu, 1] += 1
                found = True
                break

        if cuda.syncthreads_and(found):
            break

    cuda.atomic.add(out, (l, u, 0), shared_out[tl, tu, 0])
    cuda.atomic.add(out, (l, u, 1), shared_out[tl, tu, 1])
    # questo funziona?
    # cuda.atomic.add(out, (l, u), shared_out[tl, tu])


df = pd.read_parquet("tick_data/eurusd_2019.parquet.gzip")
# df = pd.read_parquet("drive/My Drive/train_data/eurusd_2019.parquet.gzip")

p = cuda.to_device(df["buy"].values.astype("float32"))

lower = cuda.to_device(-np.linspace(1e-5, 1e-1, 200, dtype="float32"))
upper = cuda.to_device(np.linspace(1e-5, 1e-1, 200, dtype="float32"))


i_tpb = 32
l_tpb = 4
u_tpb = 4

multiplier = 150

frame_size = i_tpb * multiplier

frames_per_grid = p.size // frame_size
i_bpg = (frames_per_grid * frame_size + i_tpb - 1) // i_tpb

l_bpg = (lower.size + l_tpb - 1) // l_tpb
u_bpg = (upper.size + u_tpb - 1) // u_tpb

tpb = (i_tpb, l_tpb * u_tpb)
bpg = (i_bpg, l_bpg * u_bpg)

d_out = cuda.to_device(np.zeros((l_tpb * l_bpg, u_tpb * u_bpg, 2), dtype="float32"))

compute_stuff_cuda[bpg, tpb](p, lower, upper, d_out)
cuda.synchronize()

out = d_out.copy_to_host()
