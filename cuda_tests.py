import numpy as np
import pandas as pd
from numba import njit, prange, cuda, float64


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


tpb = (32, 4 * 4)

multiplier = 1000


@cuda.jit
def compute_stuff_cuda(p, lower, upper, out):
    shared_out = cuda.shared.array((4, 4, 2), dtype=float64)

    shared_p = cuda.shared.array(tpb * multiplier, dype=float64)
    shared_lower = cuda.shared.array(4, dype=float64)
    shared_upper = cuda.shared.array(4, dype=float64)
    i0, lu = cuda.grid(2)

    # (lower.size, upper.size)
    # ((i*J + j)*K + k)*L + l
    l = lu // upper.size
    u = lu % upper.size

    if l >= lower.size or u >= upper.size:
        return

    max_i0 = p.size - p.size % shared_p.size

    if i0 >= max_i0:
        return

    ti0 = cuda.threadIdx.x
    tlu = cuda.threadIdx.y

    tl = tlu // 4
    tu = tlu % 4

    shared_lower[tl] = lower[tl]
    shared_upper[tu] = upper[tu]
    shared_out[tl, tu] = 0

    left_frames = (max_i0 - i0) // shared_p.size + 1

    found = False
    for frame in range(left_frames):
        for i in range(multiplier):
            shared_p[frame * shared_p.size + ti0 * multiplier + i] = p[
                i0 + frame * shared_p.size + ti0 * (multiplier - 1) + i
            ]

        cuda.syncthreads()

        for ti in range(ti0, shared_p.size):
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

    # questo probabilmente non serve
    cuda.syncthreads()
    cuda.atomic.add(out, (l, u, 0), shared_out[tl, tu, 0])
    cuda.atomic.add(out, (l, u, 1), shared_out[tl, tu, 1])
    # questo funziona?
    # cuda.atomic.add(out, (l, u), shared_out[tl, tu])


df = pd.read_parquet("tick_data/eurusd_2019.parquet.gzip")
p = df["buy"].values

lower = -np.linspace(1e-5, 1e-1, 200)
upper = np.linspace(1e-5, 1e-1, 200)

thresholds = np.stack((lower, upper), axis=-1)
# res = compute_stuff(p, thresholds)
