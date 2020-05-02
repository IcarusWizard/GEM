import torch
from torch.functional import F
import numpy as np

EPS = 1e-8

class Onehot:
    """
        Implementation of Bernoulli distribution
    """
    def __init__(self, logits, with_batch=True):
        super().__init__()
        self.logits = logits
        self.p = torch.softmax(logits, dim=1)
        self.with_batch = with_batch
        self.prior = torch.distributions.Categorical(logits=logits)

    def log_prob(self, x):
        index = torch.argmax(x, dim=1)
        logprob = self.prior.log_prob(index)
        return torch.sum(logprob, dim=tuple(range(1, len(logprob.shape))))

    def mode(self):
        index = torch.argmax(self.logits, dim=1)
        sample = torch.zeros_like(self.logits)
        sample[np.arange(index.shape[0]), index] = 1
        return sample + self.p - self.p.detach()

    def sample(self, num=None):
        # TODO: implement multi-sample mode
        assert num is None, "currently only support None mode"
        index = self.prior.sample([num]) if num is not None else self.prior.sample()
        sample = torch.zeros_like(self.logits)
        sample[np.arange(index.shape[0]), index] = 1
        return sample + self.p - self.p.detach()

    def sample_with_logprob(self, num=None):
        x = self.sample(num)
        return x, self.log_prob(x)
    
    def entropy(self):
        entropy = self.prior.entropy()
        return torch.sum(entropy, dim=tuple(range(1, len(entropy.shape)))) if self.with_batch else torch.sum(entropy)