import torch
from torch.functional import F

from gem.distributions.utils import get_kl

from gem.modules.decoder import MLPDecoder, ActionDecoder
from .trainer import PredictorTrainer

class RSSM(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, stoch_dim, hidden_dim, action_mimic=False, predict_reward=True, 
                 decoder_config={"hidden_layers" : 2, "features" : 512, "activation" : 'elu'}):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.stoch_dim = stoch_dim
        self.hidden_dim = hidden_dim
        self.action_minic = action_mimic
        self.predict_reward = predict_reward
        self.decoder_config = decoder_config

        self.state_dim = hidden_dim + stoch_dim

        self.rnn_cell = torch.nn.GRUCell(stoch_dim + action_dim, hidden_dim)

        self.post = MLPDecoder(hidden_dim + obs_dim, stoch_dim, **decoder_config)
        self.prior = MLPDecoder(hidden_dim, stoch_dim, **decoder_config)

        self.obs_pre = MLPDecoder(self.state_dim, obs_dim, dist_type='fix_std', **decoder_config)
        
        if self.action_minic:
            self.action_pre = ActionDecoder(self.state_dim, action_dim, **decoder_config)

        if self.predict_reward:
            self.reward_pre = MLPDecoder(self.state_dim, 1, dist_type='fix_std', **decoder_config)

    def _reset(self, obs0):
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

        h, s = self._reset(obs[0])
        whole_state = torch.cat([h, s], dim=1)

        pre_obs = []
        pre_obs_loss = 0

        pre_action = []
        pre_action_loss = 0

        pre_reward = []
        pre_reward_loss = 0

        kl_loss = 0

        if self.action_minic:
            action_dist = self.action_pre(whole_state.detach())
            pre_action.append(action_dist.mode())
            pre_action_loss -= torch.sum(action_dist.log_prob(action[0]))

        for i in range(T):
            _obs = obs[i]
            _action = action[i]

            h, s, posterior_dist, prior_dist = self.obs_step(h, s, _action, _obs)

            kl_loss += torch.sum(get_kl(posterior_dist, prior_dist))

            whole_state = torch.cat([h, s], dim=1)
            
            obs_dist = self.obs_pre(whole_state)
            pre_obs_loss -= torch.sum(obs_dist.log_prob(_obs))

            pre_obs.append(obs_dist.mode())

            if self.action_minic and i < T - 1:
                action_dist = self.action_pre(whole_state.detach())
                pre_action.append(action_dist.mode())
                pre_action_loss -= torch.sum(action_dist.log_prob(action[i+1]))

            if self.predict_reward:
                assert reward is not None
                _reward = reward[i]
                reward_dist = self.reward_pre(whole_state)
                pre_reward.append(reward_dist.mode())
                pre_reward_loss -= torch.sum(reward_dist.log_prob(_reward))

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

        h, s = self._reset(obs0)
        whole_state = torch.cat([h, s], dim=1)

        for i in range(horizon):
            if action is None:
                action_dist = self.action_pre(whole_state.detach())
                _action = action_dist.mode()
            else:
                _action = action[i]

            if i == 0:
                h, s, _, _ = self.obs_step(h, s, _action, obs0)
            else:
                h, s, _ = self.img_step(h, s, _action)

            whole_state = torch.cat([h, s], dim=1)

            obs_dist = self.obs_pre(whole_state)
            pre_obs.append(obs_dist.mode())

            if self.action_minic:
                pre_action.append(_action)

            if self.predict_reward:
                pre_reward.append(self.reward_pre(whole_state).mode())

        prediction = {
            "obs" : torch.stack(pre_obs),
        }

        if self.action_minic:
            prediction['action'] = torch.stack(pre_action)
        
        if self.predict_reward:
            prediction['reward'] = torch.stack(pre_reward)

        return prediction

    def obs_step(self, prev_h, prev_s, prev_a, obs):
        next_h, _, prior_dist = self.img_step(prev_h, prev_s, prev_a)

        posterior_dist = self.post(torch.cat([next_h, obs], dim=1))

        return next_h, posterior_dist.sample(), posterior_dist, prior_dist

    def img_step(self, prev_h, prev_s, prev_a):
        next_h = self.rnn_cell(torch.cat([prev_s, prev_a], dim=1), prev_h) 

        prior_dist = self.prior(next_h)

        return next_h, prior_dist.sample(), prior_dist

    def get_trainer(self):
        return PredictorTrainer

    # API for environmental rollout
    def reset(self, obs):
        h, s = self._reset(obs)
        h, s, _, _ = self.obs_step(h, s, torch.zeros(obs.shape[0], self.action_dim, dtype=obs.dtype, device=obs.device), obs)
        return torch.cat([h, s], dim=1)

    def step(self, pre_state, action):
        h, s = torch.split(pre_state, [self.hidden_dim, self.stoch_dim], dim=1)
        h, s, _ = self.img_step(h, s, action)
        next_state = torch.cat([h, s], dim=1)
        reward = self.reward_pre(next_state).mode()
        return next_state, reward