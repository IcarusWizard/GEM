import torch
from torch.functional import F
import numpy as np

from .base import MLP, Flatten, Unflatten, ResNet, ResBlock
from functools import partial

from gem.distributions import Normal, Bernoulli, Onehot, TanhBijector, BijectoredDistribution

class MLPDecoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim, features, hidden_layers, activation='elu',
                 dist_type='gauss', min_std=0.1):
        super().__init__()
        self.dist_type = dist_type
        self.min_std = min_std
        _output_dim = 2 * output_dim if self.dist_type == 'gauss' else output_dim
        self.decoder = MLP(input_dim, _output_dim, features, hidden_layers, activation=activation)
    
    def forward(self, x):
        out = self.decoder(x)
        if self.dist_type == 'fix_std':
            return Normal(out, 1)
        elif self.dist_type == 'gauss':
            mu, std = torch.chunk(out, 2, dim=1)
            std = F.softplus(std) + self.min_std
            return Normal(mu, std)
        elif self.dist_type == 'bernoulli':
            p = torch.sigmoid(out)
            return Bernoulli(p)
        else:
            raise ValueError(f"distribution type {self.dist_type} is not suppoted!") 

class ConvDecoder(torch.nn.Module):
    def __init__(self, output_c, h, w, latent_dim, conv_features, down_sampling, res_layers, mlp_features, mlp_layers, batchnorm,
                 dist_type='gauss'):
        super().__init__()
        self.dist_type = dist_type

        res_layers = list(reversed(res_layers))
        
        feature_shape = (conv_features, h // (2 ** down_sampling), w // (2 ** down_sampling))

        decoder_list = [
            MLP(latent_dim, np.prod(feature_shape), mlp_features, mlp_layers, 'leakyrelu'),
            Unflatten(feature_shape),
        ]

        conv_features = conv_features
        for i in range(down_sampling):
            for j in range(res_layers[i]):
                decoder_list.append(ResBlock(conv_features, batchnorm))
            decoder_list.append(torch.nn.ConvTranspose2d(conv_features, conv_features // 2, 4, 2, padding=1))
            decoder_list.append(torch.nn.ReLU(inplace=True))
            conv_features //= 2

        _output_c = 2 * output_c if self.dist_type == 'gauss' else output_c
        decoder_list.append(torch.nn.Conv2d(conv_features, _output_c, 3, stride=1, padding=1))
        
        self.decoder = torch.nn.Sequential(*decoder_list)

    def forward(self, x):
        out = self.decoder(x)
        if self.dist_type == 'fix_std':
            return Normal(out, 1)
        elif self.dist_type == 'gauss':
            mu, logs = torch.chunk(out, 2, dim=1)
            logs = torch.tanh(logs)
            return Normal(mu, torch.exp(logs))
        elif self.dist_type == 'bernoulli':
            p = torch.sigmoid(out)
            return Bernoulli(p)
        else:
            raise ValueError(f"distribution type {self.dist_type} is not suppoted!") 

class FullConvDecoder(torch.nn.Module):
    def __init__(self, input_c, h, w, output_c, conv_features, down_sampling, res_layers, batchnorm):
        super().__init__()
        res_layers = list(reversed(res_layers))

        decoder_list = [
            torch.nn.Conv2d(input_c, conv_features, 3, stride=1, padding=1)
        ]

        conv_features = conv_features
        for i in range(down_sampling):
            for j in range(res_layers[i]):
                decoder_list.append(ResBlock(conv_features, batchnorm))
            decoder_list.append(torch.nn.ConvTranspose2d(conv_features, conv_features // 2, 4, 2, padding=1))
            decoder_list.append(torch.nn.ReLU(inplace=True))
            conv_features //= 2

        _output_c = 2 * output_c if self.dist_type == 'gauss' else output_c
        decoder_list.append(torch.nn.Conv2d(conv_features, _output_c, 3, stride=1, padding=1))
        
        self.decoder = torch.nn.Sequential(*decoder_list)

    def forward(self, x):
        out = self.decoder(x)
        if self.dist_type == 'fix_std':
            return Normal(out, 1)
        elif self.dist_type == 'gauss':
            mu, logs = torch.chunk(out, 2, dim=1)
            logs = torch.tanh(logs)
            return Normal(mu, torch.exp(logs))
        elif self.dist_type == 'bernoulli':
            p = torch.sigmoid(out)
            return Bernoulli(p)
        else:
            raise ValueError(f"distribution type {self.dist_type} is not suppoted!") 


class ActionDecoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim, features, hidden_layers, activation='elu', mode='continuous', init_std=1, min_std=0.01):
        super().__init__()
        self.mode = mode

        if self.mode == 'continuous':
            self.min_std = min_std
            self.init_bais = np.log(np.exp(init_std) - 1)

            self.decoder = MLP(input_dim, 2 * output_dim, features, hidden_layers, activation=activation)
            self.bijector = TanhBijector()
        elif self.mode == 'discrete':
            self.decoder = MLP(input_dim, output_dim, features, hidden_layers, activation=activation)
    
    def forward(self, x):
        if self.mode == 'continuous':
            mu, std = torch.chunk(self.decoder(x), 2, dim=1)
            mu = 5 * torch.tanh(mu / 5) # rescale 
            std = F.softplus(std + self.init_bais) + self.min_std
            dist = Normal(mu, std)
            dist = BijectoredDistribution(dist, self.bijector)
        elif self.mode == 'discrete':
            logits = self.decoder(x)
            dist = Onehot(logits)
        return dist