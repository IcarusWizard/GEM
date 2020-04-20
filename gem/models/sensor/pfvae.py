import torch
from torch.functional import F
import numpy as np

from degmo.vae.modules import MLPEncoder, MLPDecoder, ConvEncoder, ConvDecoder
from .trainer import VAETrainer

from gem.distributions import Normal, Bernoulli, BijectoredDistribution, RealNVPBijector1D
from gem.distributions.utils import get_kl

class PFVAE(torch.nn.Module):
    r"""
        VAE with Flow as prior
        
        Inputs:

            c : int, channel of the input image
            h : int, height of the input image
            w : int, width of the input image
            latent_dim : int, dimension of the latent variable
            free_nats : the amount of information is free to the model
            network_type : str, type of the encoder and decoder, choose from conv and mlp, default: conv
            config : dict, parameters for constructe encoder and decoder
            output_type : str, type of the distribution p(x|z), choose from fix_std(std=1), gauss and bernoulli, default: gauss
            flow_hidden_layers : int, num of hidden layers in each transforamtion
            flow_features : int, num of features in each transformation
            flow_num_transformation : int, num of transformation in prior
    """
    def __init__(self, c=3, h=32, w=32, latent_dim=2, free_nats=0, network_type='conv', config={}, output_type='gauss',  
                 flow_hidden_layers=3, flow_features=64, flow_num_transformation=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.free_nats = free_nats
        self.output_type = output_type
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
        
        self.register_buffer('prior_mean', torch.zeros(self.latent_dim))
        self.register_buffer('prior_std', torch.ones(self.latent_dim))

        self.flow = RealNVPBijector1D(latent_dim, flow_num_transformation, flow_features, flow_hidden_layers)

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
        
        loss = kl + reconstruction_loss

        return loss, {
            "NELBO" : loss.item(),
            "KL divergence" : kl.item(),
            "reconstruction loss" : reconstruction_loss.item(),
        }

    def encode(self, x, output_dist=False):
        mu, std = torch.chunk(self.encoder(x), 2, dim=1)
        std = F.softplus(std) + 1e-4
        dist = Normal(mu, std)
        dist = BijectoredDistribution(dist, self.flow)
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