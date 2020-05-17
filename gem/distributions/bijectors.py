import torch
from torch.functional import F
import re

from gem.modules import MLP

EPS = 1e-8

class BijectoredDistribution:
    """
        Cover a prior distribution with a bijector
    """
    def __init__(self, prior, bijector):
        super().__init__()
        self.prior = prior
        self.bijector = bijector
        self.with_batch = self.prior.with_batch

    def log_prob(self, x=None, z=None):
        assert x is not None or z is not None, "at least one of x, z should be provided!"
        if x is not None:
            z, logdet = self.bijector(x, reverse=True)
        else:
            x, logdet = self.bijector(z, reverse=False)
            logdet = -logdet

        return logdet + self.prior.log_prob(z)

    def mode(self):
        mode_z = self.prior.mode()
        mode_x, _ = self.bijector(mode_z, reverse=False)
        return mode_x

    def sample(self, num=None):
        z = self.prior.sample(num)
        x, _ = self.bijector(z, reverse=False)
        return x

    def sample_with_logprob(self, num=None):
        z = self.prior.sample(num)
        x, logdet = self.bijector(z, reverse=False)
        logprob = self.prior.log_prob(z) - logdet
        return x, logprob
    
    def entropy(self, samples=100):
        """
            estimate the entropy with samples
        """
        _, logprob = self.sample_with_logprob(samples)
        logprob = torch.mean(logprob, dim=0)
        return - torch.sum(logprob, dim=tuple(range(1, len(logprob.shape)))) if self.with_batch else torch.sum(logprob)

def atanh(x): 
    return 0.5 * torch.log((1 + x) / (1 - x + EPS))

class Bijector(torch.nn.Module):
    """
        Base class for bijectors
    """
    def __init__(self):
        super().__init__()

    def forward(self, inputs, reverse=False):
        return self._backward(inputs) if reverse else self._forward(inputs)

    def _forward(self, z):
        """
            map from the prior space to the real space Z -> X
        """
        raise NotImplementedError

    def _backward(self, x):
        """
            map from the real space to the prior space X -> Z
        """
        raise NotImplementedError  

class TanhBijector(Bijector):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, reverse=False):
        return self._backward(inputs) if reverse else self._forward(inputs)

    def _forward(self, z):
        x = torch.tanh(z)
        logdet = torch.log(1 - x ** 2 + EPS)
        return x, logdet

    def _backward(self, x):
        x = torch.clamp(x, -0.99999997, 0.99999997)
        z = atanh(x)
        logdet = torch.log(1 - x ** 2 + EPS)
        return z, -logdet
        

class AdditiveCoupling1D(Bijector):
    def __init__(self, input_dim, features, hidden_layers, zero_init=True):
        super().__init__()
        self.coupling = MLP(input_dim // 2, input_dim // 2, features, hidden_layers)

        # Initialize this coupling as an identical mapping
        if zero_init:
            with torch.no_grad():
                state_dict = self.coupling.state_dict()
                num = max(map(lambda s: int(re.findall(r'(\d+)', s)[0]), state_dict))
                state_dict['net.{}.weight'.format(num)].fill_(0)
                state_dict['net.{}.bias'.format(num)].fill_(0)

    def _forward(self, z):
        z1, z2 = torch.chunk(z, 2, dim=1)
        t = self.coupling(z2)
        logs = torch.tanh(logs)
        return torch.cat([z2, z1 - t], dim=1), 0
        
    def _backward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        t = self.coupling(x1)
        return torch.cat([x2 + t, x1], dim=1), 0
    
class AffineCoupling1D(Bijector):
    def __init__(self, input_dim, features, hidden_layers, zero_init=True):
        super().__init__()
        self.coupling = MLP(input_dim // 2, input_dim, features, hidden_layers)

        # Initialize this coupling as an identical mapping
        if zero_init:
            with torch.no_grad():
                state_dict = self.coupling.state_dict()
                num = max(map(lambda s: int(re.findall(r'(\d+)', s)[0]), state_dict))
                state_dict['net.{}.weight'.format(num)].fill_(0)
                state_dict['net.{}.bias'.format(num)].fill_(0)

    def _forward(self, z):
        z1, z2 = torch.chunk(z, 2, dim=1)
        logs, t = torch.chunk(self.coupling(z2), 2, dim=1)
        logs = torch.tanh(logs)
        return torch.cat([z2, (z1 - t) * torch.exp(-logs)], dim=1), torch.cat([torch.zeros_like(logs), -logs], dim=1)
        
    def _backward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        logs, t = torch.chunk(self.coupling(x1), 2, dim=1)
        logs = torch.tanh(logs)
        return torch.cat([x2 * torch.exp(logs) + t, x1], dim=1), torch.cat([logs, torch.zeros_like(logs)], dim=1)
    
class RealNVPBijector1D(Bijector):
    def __init__(self, dim, num_transformation=8, features=128, hidden_layers=3):
        super().__init__()
        self.dim = dim

        self.couplings = torch.nn.ModuleList([AffineCoupling1D(dim, features, hidden_layers) for _ in range(num_transformation)])

    def _forward(self, z):
        x = z
        logdet = 0

        for coupling in reversed(self.couplings):
            x, _logdet = coupling(x, reverse=False)
            logdet += _logdet

        return x, logdet

    def _backward(self, x):
        z = x
        logdet = 0

        for coupling in self.couplings:
            z, _logdet = coupling(z, reverse=True)
            logdet += _logdet

        return z, logdet