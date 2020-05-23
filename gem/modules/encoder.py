import torch
from torch.functional import F
import numpy as np

from .base import MLP, Flatten, Unflatten, ResNet, ResBlock, ACTIVATION_MAP
from functools import partial

from gem.distributions import Normal, RealNVPBijector1D, BijectoredDistribution

class MLPEncoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim, features, hidden_layers, activation='relu',
                 dist_type='gauss', min_std=0.01, flow_config={}):
        super().__init__()
        self.dist_type = dist_type
        self.min_std = min_std
        _output_dim = output_dim if self.dist_type == 'fix_std' else 2 * output_dim
        self.encoder = MLP(input_dim, _output_dim, features, hidden_layers, activation=activation)

        if dist_type == 'flow':
            self.flow = RealNVPBijector1D(output_dim, **flow_config)

    def forward(self, x):
        out = self.encoder(x)
        if self.dist_type == 'fix_std':
            return Normal(out, 1)
        elif self.dist_type == 'gauss':
            out_mu, out_std = torch.chunk(out, 2, dim=1)
            out_std = F.softplus(out_std) + self.min_std
            return Normal(out_mu, out_std)
        elif self.dist_type == 'flow':
            out_mu, out_std = torch.chunk(out, 2, dim=1)
            out_std = F.softplus(out_std) + self.min_std
            output_dist = Normal(out_mu, out_std)   
            return BijectoredDistribution(output_dist, self.flow)    
        else:
            raise ValueError(f"distribution type {self.dist_type} is not suppoted!")     

class ConvEncoder(torch.nn.Module):
    def __init__(self, c, h, w, output_dim, conv_features, down_sampling, res_layers, mlp_features, mlp_layers, batchnorm,
                 activation='relu', dist_type='gauss', min_std=0.01, flow_config={}):
        super().__init__()
        self.dist_type = dist_type
        self.min_std = min_std
        activation_fn = ACTIVATION_MAP[activation]

        feature_shape = (conv_features, h // (2 ** down_sampling), w // (2 ** down_sampling))
        
        conv_features = conv_features // (2 ** down_sampling)
        encoder_list = [
            torch.nn.Conv2d(c, conv_features, 3, 1, padding=1),
            activation_fn()
        ]

        for i in range(down_sampling):
            encoder_list.append(torch.nn.Conv2d(conv_features, conv_features * 2, 3, 2, padding=1))
            encoder_list.append(activation_fn())
            conv_features *= 2
            for j in range(res_layers[i]):
                encoder_list.append(ResBlock(conv_features, batchnorm))

        encoder_list.append(Flatten())
        _output_dim = output_dim if self.dist_type == 'fix_std' else 2 * output_dim
        encoder_list.append(MLP(np.prod(feature_shape), _output_dim, mlp_features, mlp_layers, activation))

        self.encoder = torch.nn.Sequential(*encoder_list)

        if dist_type == 'flow':
            self.flow = RealNVPBijector1D(output_dim, **flow_config)

    def forward(self, x):
        out = self.encoder(x)
        if self.dist_type == 'fix_std':
            return Normal(out, 1)
        elif self.dist_type == 'gauss':
            out_mu, out_std = torch.chunk(out, 2, dim=1)
            out_std = F.softplus(out_std) + self.min_std
            return Normal(out_mu, out_std)
        elif self.dist_type == 'flow':
            out_mu, out_std = torch.chunk(out, 2, dim=1)
            out_std = F.softplus(out_std) + self.min_std
            output_dist = Normal(out_mu, out_std)   
            return BijectoredDistribution(output_dist, self.flow)    
        else:
            raise ValueError(f"distribution type {self.dist_type} is not suppoted!")   

class FullConvEncoder(torch.nn.Module):
    def __init__(self, input_c, h, w, output_c, conv_features, down_sampling, res_layers, batchnorm,
                 activation='relu', dist_type='gauss', min_std=0.01, flow_config={}):
        super().__init__()
        activation = ACTIVATION_MAP[activation]
        
        conv_features = conv_features // (2 ** down_sampling)
        encoder_list = [
            torch.nn.Conv2d(input_c, conv_features, 3, 1, padding=1),
            torch.nn.ReLU(inplace=True)
        ]

        for i in range(down_sampling):
            encoder_list.append(torch.nn.Conv2d(conv_features, conv_features * 2, 3, 2, padding=1))
            encoder_list.append(torch.nn.ReLU(inplace=True))
            conv_features *= 2
            for j in range(res_layers[i]):
                encoder_list.append(ResBlock(conv_features, batchnorm))

        _output_c = output_c if self.dist_type == 'fix_std' else 2 * output_c
        encoder_list.append(torch.nn.Conv2d(conv_features, _output_c, 3, stride=1, padding=1))

        self.encoder = torch.nn.Sequential(*encoder_list)

    def forward(self, x):
        out = self.encoder(x)
        if self.dist_type == 'fix_std':
            return Normal(out, 1)
        elif self.dist_type == 'gauss':
            out_mu, out_std = torch.chunk(out, 2, dim=1)
            out_std = F.softplus(out_std) + self.min_std
            return Normal(out_mu, out_std)  
        else:
            raise ValueError(f"distribution type {self.dist_type} is not suppoted!")  