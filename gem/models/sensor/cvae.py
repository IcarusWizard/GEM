import torch
from torch.functional import F
import numpy as np

from degmo.vae.modules import MLPEncoder, MLPDecoder, ConvEncoder, ConvDecoder
from .trainer import VAETrainer

from gem.distributions.utils import get_kl
from gem.distributions import Normal, Bernoulli 

class CVAE(torch.nn.Module):
    r"""
        VAE with consistent loss
        
        Inputs:

            c : int, channel of the input image
            h : int, height of the input image
            w : int, width of the input image
            latent_dim : int, dimension of the latent variable
            network_type : str, type of the encoder and decoder, choose from conv and mlp, default: conv
            config : dict, parameters for constructe encoder and decoder
            output_type : str, type of the distribution p(x|z), choose from fix_std(std=1) and gauss, default: gauss
            use_mce : bool, whether to compute KL by Mento Carlo Estimation, default: False
    """
    def __init__(self, c=3, h=32, w=32, latent_dim=2, free_nats=0, network_type='conv', config={},
                 output_type='gauss', use_mce=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.free_nats = free_nats
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
        
        self.register_buffer('prior_mean', torch.zeros(self.latent_dim))
        self.register_buffer('prior_std', torch.ones(self.latent_dim))

    def forward(self, x):
        # encode
        posterior = self.encode(x, output_dist=True)

        # reparameterize trick
        z = posterior.sample()

        # compute kl divergence
        prior = Normal(self.prior_mean, self.prior_std, with_batch=False)
        kl = get_kl(posterior, prior)
        # stop gradient when kl lower than free nats
        kl = torch.sum(kl, dim=1)
        kl = torch.mean(kl)
        if kl < self.free_nats:
            kl = kl.detach()

        # decode
        output_dist = self.decode(z, output_dist=True)

        # compute reconstruction_loss
        reconstruction_loss = - output_dist.log_prob(x)
        reconstruction_loss = torch.sum(reconstruction_loss, dim=(1, 2, 3))
        reconstruction_loss = torch.mean(reconstruction_loss)

        # compute consistent loss
        next_posterior = self.encode(output_dist.mode(), output_dist=True)
        consistent_loss = torch.mean(torch.sum(get_kl(next_posterior, posterior), dim=1))
        
        loss = kl + reconstruction_loss + consistent_loss
        
        return loss, {
            "NELBO" : loss.item(),
            "KL divergence" : kl.item(),
            "reconstruction loss" : reconstruction_loss.item(),
            'consistent loss' : consistent_loss.item()
        }
    
    def encode(self, x, output_dist=False):
        mu, std = torch.chunk(self.encoder(x), 2, dim=1)
        std = F.softplus(std) + 1e-4
        dist = Normal(mu, std)
        return dist if output_dist else dist.mode()

    def decode(self, z, output_dist=False):
        _x = self.decoder(z)

        if self.output_type == 'fix_std':
            # output is a gauss with a fixed 1 variance,
            # reconstruction loss is mse plus constant
            dist = Normal(_x, 1)

        elif self.output_type == 'gauss':
            # output is a gauss with diagonal variance
            _mu, _logs = torch.chunk(_x, 2, dim=1)
            _logs = torch.tanh(_logs)
            dist = Normal(_mu, torch.exp(_logs))

        elif self.output_type == 'bernoulli':
            # output is the logit of a bernouli distribution,
            # reconstruction loss is cross-entropy
            p = torch.sigmoid(_x)
            dist = Bernoulli(p)

        return dist if output_dist else dist.mode()

    def sample(self, number=1000):
        prior = Normal(self.prior_mean, self.prior_std, with_batch=False)
        z = prior.sample(number)

        return self.decode(z)

    def get_trainer(self):
        return VAETrainer