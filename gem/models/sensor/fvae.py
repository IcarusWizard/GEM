import torch
from torch.functional import F
import numpy as np

from .trainer import VAETrainer

from gem.modules.encoder import ConvEncoder
from gem.modules.decoder import ConvDecoder
from gem.distributions import Normal, BijectoredDistribution, RealNVPBijector1D
from gem.distributions.utils import get_kl

class FVAE(torch.nn.Module):
    r"""
        VAE with Flow as prior
        
        Inputs:

            c : int, channel of the input image
            h : int, height of the input image
            w : int, width of the input image
            latent_dim : int, dimension of the latent variable
            free_nats : the amount of information is free to the model
            network_type : str, type of the encoder and decoder, choose from conv and fullconv, default: conv
            config : dict, parameters for constructe encoder and decoder
            output_type : str, type of the distribution p(x|z), choose from fix_std(std=1), gauss and bernoulli, default: gauss
            flow_config : dict, parameters for constructe flow
    """
    def __init__(self, c=3, h=32, w=32, latent_dim=2, free_nats=0, network_type='conv', config={}, output_type='gauss', flow_config={}):
        super().__init__()
        self.latent_dim = latent_dim
        self.free_nats = free_nats
        self.output_type = output_type
        self.input_dim = c * h * w

        if network_type == 'conv':
            self.encoder = ConvEncoder(c, h, w, latent_dim, **config)
            self.decoder = ConvDecoder(c, h, w, latent_dim, dist_type=output_type, **config)
        else:
            raise ValueError('unsupport network type: {}'.format(network_type))

        self.register_buffer('prior_mean', torch.zeros(self.latent_dim))
        self.register_buffer('prior_std', torch.ones(self.latent_dim))

        self.flow = RealNVPBijector1D(latent_dim, **flow_config)

    def forward(self, x):
        # encode
        posterior = self.encode(x, output_dist=True)

        # reparameterize trick
        z = posterior.sample()

        # compute kl divergence
        init_prior = Normal(self.prior_mean, self.prior_std, with_batch=False)
        prior = BijectoredDistribution(init_prior, self.flow)
        kl = get_kl(posterior, prior)
        # stop gradient when kl lower than free nats
        kl = torch.sum(kl, dim=1)
        kl = torch.mean(kl)
        if kl < self.free_nats:
            kl = kl.detach()

        # conpute extra info
        extra_info = prior.log_prob(z) - init_prior.log_prob(z)
        extra_info = torch.mean(torch.sum(extra_info, dim=1))

        # decode
        output_dist = self.decode(z, output_dist=True)

        # compute reconstruction_loss
        reconstruction_loss = - output_dist.log_prob(x)
        reconstruction_loss = torch.sum(reconstruction_loss, dim=(1, 2, 3))
        reconstruction_loss = torch.mean(reconstruction_loss)

        loss = kl + reconstruction_loss

        return loss, {
            "NELBO" : loss.item(),
            "KL divergence" : kl.item(),
            "reconstruction loss" : reconstruction_loss.item(),
            "extra information" : extra_info.item(),
        }

    def encode(self, x, output_dist=False):
        dist = self.encoder(x)
        return dist if output_dist else dist.mode()

    def decode(self, z, output_dist=False):
        dist = self.decoder(z)
        return dist if output_dist else dist.mode()

    def sample(self, number=1000):
        z = self.sample_prior(number)
        return self.decode(z)

    def sample_prior(self, number=1000):
        prior = self.get_prior()
        return prior.sample(number)    

    def get_prior(self):
        init_prior = Normal(self.prior_mean, self.prior_std, with_batch=False)
        return BijectoredDistribution(init_prior, self.flow)

    def get_trainer(self):
        return VAETrainer