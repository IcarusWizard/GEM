import torch
from torch.functional import F
import numpy as np

from degmo.vae.modules import MLPEncoder, MLPDecoder, ConvEncoder, ConvDecoder
from degmo.gan.modules import ResDiscriminator
from degmo.vae.utils import get_kl, LOG2PI
from .trainer import AVAETrainer

from .utils import get_kl_2normal

class AVAE(torch.nn.Module):
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
    def __init__(self, c=3, h=32, w=32, latent_dim=2, network_type='conv', config={}, 
                 output_type='fix_std', use_mce=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_type = output_type
        self.use_mce = use_mce
        self.input_dim = c * h * w

        assert output_type == 'fix_std', "AVAE only supply fix_std now"
        assert network_type == 'conv', "AVAE only supply conv now"
        output_c = c

        if network_type == 'conv':
            self.encoder = ConvEncoder(c, h, w, latent_dim, **config)
            self.decoder = ConvDecoder(output_c, h, w, latent_dim, **config)
            self.discriminator = ResDiscriminator(c, h, w, features=config['conv_features'], hidden_layers=2)
            self.criterion = torch.nn.BCEWithLogitsLoss()
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

        kl = torch.mean(kl)

        fake_logit = self.discriminator(_x)
        fake_label = torch.ones_like(fake_logit)
        reconstruction_loss = self.criterion(fake_logit, fake_label) * np.prod(x.shape[1:])

        _mu, _logs = torch.chunk(self.encoder(_x), 2, dim=1)
        _logs = torch.clamp_max(_logs, 10) # limit the max logs, prevent inf in kl
        consistent_loss = torch.mean(torch.sum(get_kl_2normal(_mu, _logs, mu, logs), dim=1))
        
        return kl + reconstruction_loss + consistent_loss, {
            "KL divergence" : kl.item(),
            "reconstruction loss" : reconstruction_loss.item(),
            'consistent loss' : consistent_loss.item()
        }

    def get_discriminator_loss(self, x):
        mu, logs = torch.chunk(self.encoder(x), 2, dim=1)
        logs = torch.clamp_max(logs, 10) # limit the max logs, prevent inf in kl

        # reparameterize trick
        epsilon = torch.randn_like(logs)
        z = mu + epsilon * torch.exp(logs)

        _x = self.decoder(z)

        real_logit = self.discriminator(x)
        fake_logit = self.discriminator(_x)

        fake_label = torch.zeros_like(fake_logit)
        real_label = torch.ones_like(real_logit)

        return (self.criterion(real_logit, real_label) + self.criterion(fake_logit, fake_label)) * np.prod(x.shape[1:])
    
    def encode(self, x):
        mu, logs = torch.chunk(self.encoder(x), 2, dim=1)
        logs = torch.clamp_max(logs, 10)
        return mu

    def decode(self, z, deterministic=True):
        _x = self.decoder(z)

        x = _x
        if not deterministic:
            x = x + torch.randn_like(x)

        return x

    def sample(self, number=1000, deterministic=True):
        device = next(self.parameters()).device

        z = torch.randn(number, self.latent_dim, device=device)

        return self.decode(z, deterministic=deterministic)

    def get_trainer(self):
        return AVAETrainer