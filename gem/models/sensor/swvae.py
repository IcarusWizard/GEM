import torch
from torch.functional import F
import numpy as np

from .trainer import VAETrainer

from gem.modules.encoder import ConvEncoder
from gem.modules.decoder import ConvDecoder
from gem.distributions.utils import get_kl
from gem.distributions import Normal

class SWVAE(torch.nn.Module):
    r"""
        Variational Auto-Encoder: http://arxiv.org/abs/1312.6114
        
        Inputs:

            c : int, channel of the input image
            h : int, height of the input image
            w : int, width of the input image
            latent_dim : int, dimension of the latent variable
            free_nats : the amount of information is free to the model
            network_type : str, type of the encoder and decoder, choose from conv and mlp, default: conv
            config : dict, parameters for constructe encoder and decoder
            output_type : str, type of the distribution p(x|z), choose from fix_std(std=1), gauss and bernoulli, default: gauss
    """
    def __init__(self, c=3, h=32, w=32, latent_dim=2, free_nats=0, kl_scale=1.0, network_type='conv', config={}, output_type='gauss'):
        super().__init__()
        self.latent_dim = latent_dim
        self.free_nats = free_nats
        self.kl_scale = kl_scale
        self.output_type = output_type
        self.input_dim = c * h * w

        if network_type == 'conv':
            self.encoder = ConvEncoder(c, h, w, latent_dim, **config)
            self.decoder = ConvDecoder(c, h, w, latent_dim, dist_type=output_type, **config)
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
        prior = self.get_prior()
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
        old_reconstruction_loss = torch.mean(torch.sum(reconstruction_loss.detach(), dim=(1, 2, 3)))
        rshape = reconstruction_loss.shape
        weight = reconstruction_loss.detach().view(reconstruction_loss.shape[0], -1)
        weight = torch.softmax(weight, dim=-1) * np.prod(rshape[1:])
        weight = weight.view(*rshape)
        reconstruction_loss = reconstruction_loss * weight
        reconstruction_loss = torch.mean(torch.sum(reconstruction_loss, dim=(1, 2, 3)))
        
        loss = kl * self.kl_scale + reconstruction_loss

        return loss, {
            "NELBO" : (kl + old_reconstruction_loss).item(),
            "KL divergence" : kl.item(),
            "weighted reconstruction loss" : reconstruction_loss.item(),
            "reconstruction loss" : old_reconstruction_loss.item(),
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
        return Normal(self.prior_mean, self.prior_std, with_batch=False)

    def get_trainer(self):
        return VAETrainer