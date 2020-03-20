import torch

def get_kl_2normal(mu1, logs1, mu2, logs2):
    return logs2 - logs1 + (torch.exp(2 * logs1) + (mu1 - mu2) ** 2) * torch.exp(2 * -logs2) * 0.5 - 0.5