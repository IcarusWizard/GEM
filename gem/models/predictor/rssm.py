import torch
from torch.functional import F
from torch.distributions import Normal, kl_divergence

from degmo.modules import MLP
from .trainer import PredictorTrainer

class RSSM(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, stoch_dim, hidden_dim, action_mimic=True, predict_reward=True, 
                 decoder_config={"hidden_layers" : 2, "hidden_features" : 512, "activation" : torch.nn.ELU}):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.stoch_dim = stoch_dim
        self.hidden_dim = hidden_dim
        self.action_minic = action_dim
        self.predict_reward = predict_reward
        self.decoder_config = decoder_config

        self.rnn_cell = torch.nn.GRUCell(stoch_dim + action_dim, hidden_dim)

        self.post = MLP(hidden_dim + obs_dim, 2 * stoch_dim, **decoder_config)
        self.prior = MLP(hidden_dim, 2 * stoch_dim, **decoder_config)

        self.obs_pre = MLP(hidden_dim + stoch_dim, obs_dim, **decoder_config)
        
        if self.action_minic:
            self.action_pre = MLP(hidden_dim + stoch_dim, action_dim, **decoder_config)

        if self.predict_reward:
            self.reward_pre = MLP(hidden_dim + stoch_dim, 1, **decoder_config)

    def reset(self, obs0):
        batch_size = obs0.shape[0]
        dtype = obs0.dtype
        device = obs0.device

        return torch.zeros(batch_size, self.hidden_dim, dtype=dtype, device=device), \
            torch.zeros(batch_size, self.stoch_dim, dtype=dtype, device=device)


    def forward(self, obs, action, reward=None):
        """
            Inputs are all tensor[T, B, *]
        """
        
        T, B = obs.shape[:2]

        h, s = self.reset(obs[0])
        whole_state = torch.cat([h, s], dim=1)

        pre_obs = []
        pre_obs_loss = 0

        pre_action = []
        pre_action_loss = 0

        pre_reward = []
        pre_reward_loss = 0

        kl_loss = 0

        if self.action_minic:
            pre_action.append(self.action_pre(whole_state.detach()))
            action_dis = Normal(pre_action[-1], 1)
            pre_action_loss -= torch.sum(action_dis.log_prob(action[0]))

        for i in range(T):
            _obs = obs[i]
            _action = action[i]

            h, s, posterior_dis, prior_dis = self.obs_step(h, s, _action, _obs)

            kl_loss += torch.sum(kl_divergence(posterior_dis, prior_dis))

            whole_state = torch.cat([h, s], dim=1)
            
            pre_obs.append(self.obs_pre(whole_state))
            obs_dis = Normal(pre_obs[-1], 1)
            pre_obs_loss -= torch.sum(obs_dis.log_prob(_obs))

            if self.action_minic and i < T - 1:
                pre_action.append(self.action_pre(whole_state.detach()))
                action_dis = Normal(pre_action[-1], 1)
                pre_action_loss -= torch.sum(action_dis.log_prob(action[i+1]))

            if self.predict_reward:
                assert reward is not None
                _reward = reward[i]
                pre_reward.append(self.reward_pre(whole_state))
                reward_dis = Normal(pre_reward[-1], 1)
                pre_reward_loss -= torch.sum(reward_dis.log_prob(_reward))

        pre_obs_loss /= T * B
        pre_action_loss /= T * B
        pre_reward_loss /= T * B
        kl_loss /= T * B

        loss = pre_obs_loss + pre_action_loss + pre_reward_loss + kl_loss

        prediction = {
            "obs" : torch.stack(pre_obs),
        }

        info = {
            "loss" : loss.item(),
            "obs_loss" : pre_obs_loss.item(),
            "kl_loss" : kl_loss.item(),
        }      

        if self.action_minic:
            prediction['action'] = torch.stack(pre_action)
            info['action_loss'] = pre_action_loss.item()
        
        if self.predict_reward:
            prediction['reward'] = torch.stack(pre_reward)
            info['reward_loss'] = pre_reward_loss.item()

        return loss, prediction, info

    def generate(self, obs0, horizon, action=None):
        assert action is not None or self.action_minic

        pre_obs = []
        pre_action = []
        pre_reward = []    

        h, s = self.reset(obs0)
        whole_state = torch.cat([h, s], dim=1)

        for i in range(horizon):
            _action = self.action_pre(whole_state.detach()) if action is None else action[i]

            if i == 0:
                h, s, _, _ = self.obs_step(h, s, _action, obs0)
            else:
                h, s, _ = self.img_step(h, s, _action)

            whole_state = torch.cat([h, s], dim=1)

            pre_obs.append(self.obs_pre(whole_state))

            if self.action_minic:
                pre_action.append(_action)

            if self.predict_reward:
                pre_reward.append(self.reward_pre(whole_state))

        prediction = {
            "obs" : torch.stack(pre_obs),
        }

        if self.action_minic:
            prediction['action'] = torch.stack(pre_action)
        
        if self.predict_reward:
            prediction['reward'] = torch.stack(pre_reward)

        return prediction

    def obs_step(self, prev_h, prev_s, prev_a, obs):
        next_h, _, prior_dis = self.img_step(prev_h, prev_s, prev_a)

        posterior_mu, posterior_std = torch.chunk(self.post(torch.cat([next_h, obs], dim=1)), 2, dim=1)
        posterior_std = F.softplus(posterior_std) + 0.01

        posterior_dis = Normal(posterior_mu, posterior_std)

        return next_h, posterior_dis.rsample(), posterior_dis, prior_dis

    def img_step(self, prev_h, prev_s, prev_a):
        next_h = self.rnn_cell(torch.cat([prev_s, prev_a], dim=1), prev_h) 

        prior_mu, prior_std = torch.chunk(self.prior(next_h), 2, dim=1)
        prior_std = F.softplus(prior_std) + 0.01

        prior_dis = Normal(prior_mu, prior_std)

        return next_h, prior_dis.rsample(), prior_dis

    def get_trainer(self):
        return PredictorTrainer