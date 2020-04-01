import torch
import re

from degmo.modules import MLP

class AffineCoupling1D(torch.nn.Module):
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
        
    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        logs, t = torch.chunk(self.coupling(x1), 2, dim=1)
        logs = torch.tanh(logs)
        return torch.cat([x2 * torch.exp(logs) + t, x1], dim=1), torch.sum(logs, dim=1)
    
    def backward(self, z):
        z1, z2 = torch.chunk(z, 2, dim=1)
        logs, t = torch.chunk(self.coupling(z2), 2, dim=1)
        logs = torch.tanh(logs)
        return torch.cat([z2, (z1 - t) * torch.exp(-logs)], dim=1)

class FlowDistribution1D(torch.nn.Module):
    def __init__(self, dim, num_transfrom, features=128, hidden_layers=3):
        """
            This a general distribution based on Flow
        """
        super().__init__()
        self.dim = dim

        self.couplings = torch.nn.ModuleList([AffineCoupling1D(dim, features, hidden_layers) for _ in range(num_transfrom)])

        self.prior = torch.distributions.Normal(0, 1)

    def forward(self, x):
        # forward is a inference path
        z, logdet = x, 0

        for coupling in self.couplings:
            z, _logdet = coupling(z)
            logdet += _logdet

        return z, logdet

    def log_prob(self, x):
        z, logdet = self.forward(x)
        return logdet + torch.sum(self.prior.log_prob(z), dim=1)

    def sample(self, num):
        device = next(self.parameters()).device

        with torch.no_grad():
            z = self.prior.sample((num, self.dim)).to(device)
            
            x = z

            for coupling in reversed(self.couplings):
                x = coupling.backward(x)

            return x

    def rsample(self, num):
        device = next(self.parameters()).device

        z = self.prior.sample((num, self.dim)).to(device)
        
        x = z

        for coupling in reversed(self.couplings):
            x = coupling.backward(x)

        return x        