import torch
import torch.jit


def f(x):

    mask1 = torch.randint_like(x, 2, dtype=torch.bool)
    mask2 = torch.randint_like(x, 2, dtype=torch.bool)
    mask3 = mask1 | mask2
    y = torch.rand_like(x)
    for i in range(500):
        z = torch.where(mask3 & mask1, x, torch.where(mask3 & mask2, y, x.new_ones([])))
        # z = torch.where(mask1, x, torch.where(mask2, y, x.new_tensor(float("nan")))).where(mask3, x.new_ones([]))

        x = y
        y = z

    return z


x = torch.rand(100, 100, 100).cuda()

f = torch.jit.trace(f, (x,), check_trace=False)
