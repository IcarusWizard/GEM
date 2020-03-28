import torch
from torch.functional import F
import numpy as np

from degmo.vae.modules import MLPEncoder, MLPDecoder, ConvEncoder, ConvDecoder
from degmo.vae.utils import get_kl, LOG2PI
from .trainer import VAETrainer

class VAE(torch.nn.Module):
    r"""
        Variational Auto-Encoder: http://arxiv.org/abs/1312.6114
        
        Inputs:

            c : int, channel of the input image
            h : int, height of the input image
            w : int, width of the input image
            latent_dim : int, dimension of the latent variable
            network_type : str, type of the encoder and decoder, choose from conv and mlp, default: conv
            config : dict, parameters for constructe encoder and decoder
            output_type : str, type of the distribution p(x|z), choose from fix_std(std=1), gauss and bernoulli, default: gauss
            use_mce : bool, whether to compute KL by Mento Carlo Estimation, default: False
    """
    def __init__(self, c=3, h=32, w=32, latent_dim=2, network_type='conv', config={}, 
                 output_type='gauss', use_mce=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_type = output_type
        self.use_mce = use_mce
        self.input_dim = c * h * w
        output_c = 2 * c if self.output_type == 'gauss' else c

        if network_type == 'mlp':
            self.encoder = MLPEncoder(c, h, w, latent_dim, **config)
            self.decoder = MLPDecoder(output_c, h, w, latent_dim, **config)
        elif network_type == 'conv':
            self.encoder = ConvEncoder(c, h, w, latent_dim, **config)
            self.decoder = ConvDecoder(output_c, h, w, latent_dim, **config)
        else:
            raise ValueError('unsupport network type: {}'.format(network_type))
        
        self.prior = torch.distributions.Normal(0, 1)

    def forward(self, x):
        mu, logs = torch.chunk(self.encoder(x), 2, dim=1)
        logs = torch.clamp_max(logs, 10) # limit the max logs, prevent inf in kl

        # reparameterize trick
        epsilon = torch.randn_like(logs)
        z = mu + epsilon * torch.exp(logs)

        # compute kl divergence
        if self.use_mce: # Use Mento Carlo Estimation
            # kl = log q_{\phi}(z|x) - log p_{\theta}(z)
            kl = torch.sum(- epsilon ** 2 / 2 - LOG2PI - logs - self.prior.log_prob(z), dim=1)
        else:
            kl = get_kl(mu, logs)

        _x = self.decoder(z)

        # compute reconstruction loss
        if self.output_type == 'fix_std':
            # output is a gauss with a fixed 1 variance,
            # reconstruction loss is mse plus constant
            reconstruction_loss = (x - _x) ** 2 / 2 + LOG2PI

        elif self.output_type == 'gauss':
            # output is a gauss with diagonal variance
            _mu, _logs = torch.chunk(_x, 2, dim=1)
            _logs = torch.tanh(_logs)
            reconstruction_loss = (x - _mu) ** 2 / 2 * torch.exp(-2 * _logs) + LOG2PI + _logs

        elif self.output_type == 'bernoulli':
            # output is the logit of a bernouli distribution,
            # reconstruction loss is cross-entropy
            p = torch.sigmoid(_x)
            reconstruction_loss = - x * torch.log(p + 1e-8) - (1 - x) * torch.log(1 - p + 1e-8)

        kl = torch.mean(kl)
        reconstruction_loss = torch.mean(torch.sum(reconstruction_loss, dim=(1, 2, 3)))

        return kl + reconstruction_loss, {
            "KL divergence" : kl.item(),
            "reconstruction loss" : reconstruction_loss.item(),
        }
    
    def encode(self, x):
        mu, logs = torch.chunk(self.encoder(x), 2, dim=1)
        logs = torch.clamp_max(logs, 10)
        return mu

    def decode(self, z, deterministic=True):
        _x = self.decoder(z)

        if self.output_type == 'fix_std':
            x = _x
            if not deterministic:
                x = x + torch.randn_like(x)

        elif self.output_type == 'gauss':
            _mu, _logs = torch.chunk(_x, 2, dim=1)
            _logs = torch.tanh(_logs)
            x = _mu
            if not deterministic:
                x = x + torch.exp(_logs) * torch.randn_like(x)

        elif self.output_type == 'bernoulli':
            p = torch.sigmoid(_x)
            if not deterministic:
                x = (torch.rand_like(p) < p).float()
            else:
                x = (p > 0.5).float()

        return x

    def sample(self, number=1000, deterministic=True):
        device = next(self.parameters()).device

        z = torch.randn(number, self.latent_dim, device=device)

        return self.decode(z, deterministic=deterministic)

    def get_trainer(self):
        return VAETrainer