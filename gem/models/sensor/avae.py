import torch
from torch.functional import F
import numpy as np

from .trainer import AVAETrainer

from gem.modules.encoder import ConvEncoder
from gem.modules.decoder import ConvDecoder
from gem.distributions.utils import get_kl
from gem.distributions import Normal

class AVAE(torch.nn.Module):
    r"""
        VAE with adversarial training
        
        Inputs:

            c : int, channel of the input image
            h : int, height of the input image
            w : int, width of the input image
            latent_dim : int, dimension of the latent variable
            free_nats : the amount of information is free to the model
            network_type : str, type of the encoder and decoder, choose from conv and mlp, default: conv
            config : dict, parameters for constructe encoder and decoder
            output_type : str, type of the distribution p(x|z), choose from fix_std(std=1) and gauss, default: gauss
    """
    def __init__(self, c=3, h=32, w=32, latent_dim=2, free_nats=0, network_type='conv', config={}, output_type='guass'):
        super().__init__()
        self.latent_dim = latent_dim
        self.free_nats = free_nats
        self.output_type = output_type
        self.input_dim = c * h * w

        if network_type == 'conv':
            self.encoder = ConvEncoder(c, h, w, latent_dim, **config)
            self.decoder = ConvDecoder(c, h, w, latent_dim, dist_type=output_type, **config)
            self.discriminator = torch.nn.Sequential(
                torch.nn.Conv2d(c, 32, 3, 1, padding=1),
                torch.nn.LeakyReLU(0.1, True),
                torch.nn.Conv2d(32, 64, 3, 1, padding=1),
                torch.nn.LeakyReLU(0.1, True),
                torch.nn.Conv2d(64, 128, 3, 1, padding=1),
                torch.nn.LeakyReLU(0.1, True),
                torch.nn.Conv2d(128, 1, 1, 1),
                torch.nn.Softplus()
            )
            for k, v in self.discriminator.state_dict().items():
                if '6' in k:
                    v.data.fill_(0)
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

        reconstruction_loss = - output_dist.log_prob(x)

        mask = self.discriminator(x)
        mask = mask / torch.sum(mask, dim=(2, 3), keepdim=True)

        reconstruction_loss = torch.sum(reconstruction_loss * mask * np.prod(mask.shape[2:]), dim=(1, 2, 3))
        reconstruction_loss = torch.mean(reconstruction_loss)

        kl = torch.mean(kl)

        loss = kl + reconstruction_loss

        return loss, mask, {
            "NELBO" : loss.item(),
            "KL divergence" : kl.item(),
            "reconstruction loss" : reconstruction_loss.item(),
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
        return AVAETrainer