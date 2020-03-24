import torch
from torch.functional import F
import numpy as np

from degmo.flow.distribution import FlowDistribution1D
from .modules import MLPEncoder, MLPDecoder, ConvEncoder, ConvDecoder
from .utils import get_kl, LOG2PI
from .trainer import VAETrainer

class FVAE(torch.nn.Module):
    def __init__(self, c=3, h=32, w=32, latent_dim=2, network_type='conv', config={},  
                 flow_hidden_layers=3, flow_features=64, flow_num_transformation=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = c * h * w
        output_c = 2 * c

        if network_type == 'mlp':
            self.encoder = MLPEncoder(c, h, w, latent_dim, **config)
            self.decoder = MLPDecoder(output_c, h, w, latent_dim, **config)
        elif network_type == 'conv':
            self.encoder = ConvEncoder(c, h, w, latent_dim, **config)
            self.decoder = ConvDecoder(output_c, h, w, latent_dim, **config)
        else:
            raise ValueError('unsupport network type: {}'.format(network_type))
        
        self.prior = FlowDistribution1D(latent_dim, flow_num_transformation, flow_features, flow_hidden_layers)

        self.init_prior = torch.distributions.Normal(0, 1)

    def forward(self, x):
        mu, logs = torch.chunk(self.encoder(x), 2, dim=1)
        logs = torch.clamp_max(logs, 10) # limit the max logs, prevent inf in kl

        # reparameterize trick
        epsilon = torch.randn_like(logs)
        z = mu + epsilon * torch.exp(logs)

        # Use Mento Carlo Estimation to compute kl divergence
        # kl = log q_{\phi}(z|x) - log p_{\theta}(z)
        post_prob = torch.sum(- epsilon ** 2 / 2 - LOG2PI - logs, dim=1) 
        prior_prob = self.prior.log_prob(z)
        init_prior_prob = torch.sum(self.init_prior.log_prob(z), dim=1)

        kl = post_prob - prior_prob

        _x = self.decoder(z)

        # output is a gauss with diagonal variance
        _mu, _logs = torch.chunk(_x, 2, dim=1)
        _logs = torch.tanh(_logs)
        reconstruction_loss = (x - _mu) ** 2 / 2 * torch.exp(-2 * _logs) + LOG2PI + _logs


        kl = torch.mean(kl)
        reconstruction_loss = torch.mean(torch.sum(reconstruction_loss, dim=(1, 2, 3)))
        extra_info = torch.mean(prior_prob - init_prior_prob) # D_KL(p_{\theta} || p_{\theta_{init}})

        return kl + reconstruction_loss, {
            "KL divergence" : kl.item(),
            "reconstruction loss" : reconstruction_loss.item(),
            "extra information" : extra_info.item(),
        }
    
    def encode(self, x):
        mu, logs = torch.chunk(self.encoder(x), 2, dim=1)
        logs = torch.clamp_max(logs, 10)
        return mu

    def decode(self, z, deterministic=True):
        _x = self.decoder(z)

        _mu, _logs = torch.chunk(_x, 2, dim=1)
        _logs = torch.tanh(_logs)
        x = _mu
        if not deterministic:
            x = x + torch.exp(_logs) * torch.randn_like(x)

        return x

    def sample(self, number=1000, deterministic=True):

        z = self.prior.sample(num=number)

        return self.decode(z, deterministic=deterministic)

    def get_trainer(self):
        return VAETrainer