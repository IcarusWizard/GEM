import torch
from torch.functional import F

class Normal:
    """
        Implementation of Normal distribution
    """
    def __init__(self, mean, std, with_batch=True):
        super().__init__()
        self.mean = mean
        self.std = std
        self.with_batch = with_batch
        self.prior = torch.distributions.Normal(mean, std)

    def log_prob(self, x):
        return self.prior.log_prob(x)

    def mode(self):
        return self.mean

    def sample(self, num=None):
        x = self.prior.rsample([num]) if num is not None else self.prior.rsample()
        return x

    def sample_with_logprob(self, num=None):
        x = self.sample(num)
        return x, self.log_prob(x)
    
    def entropy(self):
        entropy = self.prior.entropy()
        return torch.sum(entropy, dim=tuple(range(1, len(entropy.shape)))) if self.with_batch else torch.sum(entropy)