import torch
import matplotlib.pyplot as plt

N = 1000
batch_size = 512
n_batches = 1000

counts = torch.zeros(N)
real_probs = torch.rand(N) / 200
real_probs[-1] = 0

for _ in range(n_batches):
    samples = torch.bernoulli(real_probs.expand(batch_size, -1))
    inds = torch.argmax(samples * torch.arange(N, 0, -1), dim=1)
    for i in inds:
        counts[i] += 1

sampled_compound_probs = counts / counts.sum()


# not_done_probs = torch.cumprod(1 - real_probs, dim=0)
not_done_logprobs = torch.log(1 - real_probs + 1e-45).cumsum(dim=0)
# expected_compound_logprobs = not_done_logprobs.clone()
# expected_compound_logprobs[:-1] += torch.log(real_probs[1:])

expected_compound_logprobs = torch.log(real_probs + 1e-45)
expected_compound_logprobs[1:] += not_done_logprobs[:-1]
expected_compound_logprobs[-1] = not_done_logprobs[-1]

expected_compound_probs = torch.exp(expected_compound_logprobs)

plt.plot(sampled_compound_probs)
plt.plot(expected_compound_probs)
plt.show()
