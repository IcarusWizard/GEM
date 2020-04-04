import torch
from torch.functional import F
import numpy as np

from degmo.vae.modules import MLPEncoder, MLPDecoder, ConvEncoder, ConvDecoder
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
                 output_type='guass', use_mce=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_type = output_type
        self.use_mce = use_mce
        self.input_dim = c * h * w

        assert network_type == 'conv', "AVAE only supply conv now"
        output_c = c * 2

        if network_type == 'conv':
            self.encoder = ConvEncoder(c, h, w, latent_dim, **config)
            self.decoder = ConvDecoder(output_c, h, w, latent_dim, **config)
            self.discriminator = torch.nn.Sequential(
                torch.nn.Conv2d(c, 32, 3, 1, padding=1),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(32, 64, 3, 1, padding=1),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(64, 128, 3, 1, padding=1),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(128, 1, 1, 1)
            )
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

        _mu, _logs = torch.chunk(self.decoder(z), 2, dim=1)
        _logs = torch.tanh(_logs)

        reconstruction_loss = (x - _mu) ** 2 / 2 * torch.exp(-2 * _logs) + LOG2PI + _logs

        mask = self.discriminator(x)
        mask_shape = mask.shape
        mask = mask.view(mask.shape[0], -1)
        mask = F.softmax(mask, dim=1)
        mask = mask.view(*mask_shape)

        reconstruction_loss = torch.sum(reconstruction_loss * mask * np.prod(mask.shape[2:]), dim=(1, 2, 3))
        reconstruction_loss = torch.mean(reconstruction_loss)

        kl = torch.mean(kl)

        loss = kl + reconstruction_loss

        return loss, mask, {
            "NELBO" : loss.item(),
            "KL divergence" : kl.item(),
            "reconstruction loss" : reconstruction_loss.item(),
        }
    
    def encode(self, x):
        mu, logs = torch.chunk(self.encoder(x), 2, dim=1)
        logs = torch.clamp_max(logs, 10)
        return mu

    def decode(self, z, deterministic=True):
        _mu, _logs = torch.chunk(self.decoder(z), 2, dim=1)
        _logs = torch.tanh(_logs)

        x = _mu
        if not deterministic:
            x = x + torch.randn_like(x) * torch.exp(_logs)

        return x

    def sample(self, number=1000, deterministic=True):
        device = next(self.parameters()).device

        z = torch.randn(number, self.latent_dim, device=device)

        return self.decode(z, deterministic=deterministic)

    def get_trainer(self):
        return AVAETrainer