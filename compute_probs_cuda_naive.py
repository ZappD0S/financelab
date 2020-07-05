import numpy as np
import pandas as pd
from numba import cuda, float32, int32


@cuda.jit
def compute_counts_cuda_naive(p, lower, upper, counts):
    shared_counts = cuda.shared.array((l_tpb, u_tpb, 2), dtype=int32)
    i0, lu = cuda.grid(2)

    l = lu // upper.size
    u = lu % upper.size

    if l >= lower.size or u >= upper.size or i0 >= p.size:
        return

    ti0 = cuda.threadIdx.x
    tlu = cuda.threadIdx.y

    tl = tlu // u_tpb
    tu = tlu % u_tpb

    shared_counts[tl, tu, 0] = 0
    shared_counts[tl, tu, 1] = 0

    cuda.syncthreads()

    buy_p = p[i0] + spread
    for i in range(i0, p.size):
        sell_p = p[i]
        if sell_p / buy_p < lower[l]:
            cuda.atomic.add(shared_counts, (tl, tu, 0), 1)
            break
        if sell_p / buy_p > upper[u]:
            cuda.atomic.add(shared_counts, (tl, tu, 1), 1)
            break

    cuda.syncthreads()

    if ti0 == 0:
        cuda.atomic.add(counts, (l, u, 0), shared_counts[tl, tu, 0])
        cuda.atomic.add(counts, (l, u, 1), shared_counts[tl, tu, 1])


df = pd.read_parquet("tick_data/eurusd_2019.parquet.gzip")
# df = pd.read_parquet("drive/My Drive/train_data/eurusd_2019.parquet.gzip")

p = cuda.to_device(df["sell"].values.astype("float32")[: int(1e5)])
spread = 1.5e-4

lower = 1 - np.linspace(1e-5, 2e-4, 50, dtype="float32")
upper = 1 + np.linspace(1e-5, 2e-4, 50, dtype="float32")

i_tpb = 64
l_tpb = 4
u_tpb = 4

i_bpg = (p.size + i_tpb - 1) // i_tpb

l_bpg = (lower.size + l_tpb - 1) // l_tpb
u_bpg = (upper.size + u_tpb - 1) // u_tpb

tpb = (i_tpb, l_tpb * u_tpb)
bpg = (i_bpg, l_bpg * u_bpg)

lower_dev = cuda.to_device(lower)
upper_dev = cuda.to_device(upper)
counts_dev = cuda.to_device(np.zeros((lower.size, upper.size, 2), dtype="int32"))

compute_counts_cuda_naive[bpg, tpb](p, lower_dev, upper_dev, counts_dev)
cuda.synchronize()

counts = counts_dev.copy_to_host()
probs = counts.astype("float64") / counts.sum(axis=-1, keepdims=True)

upper_lower = np.dstack(np.meshgrid(lower, upper, indexing="ij"))
gains = (upper_lower[..., 0] ** probs[..., 0]) * (upper_lower[..., 1] ** probs[..., 1])
