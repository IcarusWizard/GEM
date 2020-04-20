import torch
from torch.functional import F

EPS = 1e-8

class Bernoulli:
    """
        Implementation of Bernoulli distribution
    """
    def __init__(self, p, with_batch=True):
        super().__init__()
        self.p = p
        self.with_batch = with_batch
        self.prior = torch.distributions.Bernoulli(p)

    def log_prob(self, x):
        logprob = self.prior.log_prob(x)
        return torch.sum(logprob, dim=tuple(range(1, len(logprob.shape))))

    def mode(self):
        return (self.p > 0.5).float()

    def sample(self, num=None):
        x = self.prior.sample([num]) if num is not None else self.prior.sample()
        return x

    def sample_with_logprob(self, num=None):
        x = self.sample(num)
        return x, self.log_prob(x)
    
    def entropy(self):
        entropy = self.prior.entropy()
        return torch.sum(entropy, dim=tuple(range(1, len(entropy.shape)))) if self.with_batch else torch.sum(entropy)