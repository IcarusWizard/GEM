import torch
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