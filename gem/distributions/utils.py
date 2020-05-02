import torch
import numpy as np
from .normal import Normal

def get_kl(q, p):
    if isinstance(q, Normal) and isinstance(p, Normal):
        return torch.distributions.kl_divergence(q.prior, p.prior)
    else: 
        # otherwise perform Mento Carlo Estimation
        # kl = log q_{\phi}(z|x) - log p_{\theta}(z)
        sample, q_logprob = q.sample_with_logprob()
        p_logprob = p.log_prob(sample)
        return q_logprob - p_logprob

def stack_normal(dists):
    assert np.all([dist.with_batch for dist in dists]) or np.all([not dist.with_batch for dist in dists]), \
        "all distribution should be in the same form"
    means = torch.stack([dist.mean for dist in dists])
    try:
        stds = torch.stack([dist.std for dist in dists])
    except:
        stds = 1
    return Normal(means, stds, with_batch=dists[0].with_batch)
