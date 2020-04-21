import torch

from gem.distributions import Normal
from gem.modules.decoder import MLPDecoder, ActionDecoder
from .trainer import PredictorTrainer

class RAR(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, action_mimic=False, predict_reward=True, 
                 decoder_config={"hidden_layers" : 2, "features" : 512, "activation" : 'elu'}):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_minic = action_mimic
        self.predict_reward = predict_reward
        self.decoder_config = decoder_config

        self.state_dim = hidden_dim

        self.rnn_cell = torch.nn.GRUCell(obs_dim + action_dim, hidden_dim)

        self.obs_pre = MLPDecoder(hidden_dim, obs_dim, dist_type='fix_std', **decoder_config)
        
        if self.action_minic:
            self.action_pre = ActionDecoder(hidden_dim, action_dim, **decoder_config)

        if self.predict_reward:
            self.reward_pre = MLPDecoder(hidden_dim, 1, dist_type='fix_std', **decoder_config)

    def forward(self, obs, action, reward=None):
        """
            Inputs are all tensor[T, B, *]
        """
        
        T, B = obs.shape[:2]

        pre_obs = []
        pre_obs_loss = 0

        pre_action = []
        pre_action_loss = 0

        pre_reward = []
        pre_reward_loss = 0

        # compute h1 by assuming obs -1 is the same as obs 0, and take no action 
        h = self.rnn_cell(torch.cat([obs[0], torch.zeros_like(action[0])], dim=1)) 

        for i in range(T):
            _obs = obs[i]
            _action = action[i]

            obs_dist = self.obs_pre(h)
            pre_obs_loss -= torch.sum(obs_dist.log_prob(_obs))

            _obs = obs_dist.mode()

            pre_obs.append(_obs)

            if self.action_minic:
                action_dist = self.action_pre(h.detach())
                pre_action.append(action_dist.mode())
                pre_action_loss -= torch.sum(action_dist.log_prob(_action))

            if self.predict_reward:
                assert reward is not None
                _reward = reward[i]
                reward_dist = self.reward_pre(h)
                pre_reward.append(reward_dist.mode())
                pre_reward_loss -= torch.sum(reward_dist.log_prob(_reward))

            h = self.rnn_cell(torch.cat([_obs, _action], dim=1), h) # compute next state

        pre_obs_loss /= T * B
        pre_action_loss /= T * B
        pre_reward_loss /= T * B

        loss = pre_obs_loss + pre_action_loss + pre_reward_loss

        prediction = {
            "obs" : torch.stack(pre_obs),
        }

        info = {
            "loss" : loss.item(),
            "obs_loss" : pre_obs_loss.item(),
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

        # compute h1 by assuming obs -1 is the same as obs 0, and take no action 
        h = self.rnn_cell(torch.cat([obs0, torch.zeros(obs0.shape[0], self.action_dim, dtype=obs0.dtype, device=obs0.device)], dim=1)) 

        for i in range(horizon):
            _obs = self.obs_pre(h).mode()
            _action = self.action_pre(h.detach()).mode() if action is None else action[i]
            
            pre_obs.append(_obs)

            if self.action_minic:
                pre_action.append(_action)

            if self.predict_reward:
                pre_reward.append(self.reward_pre(h).mode())

            h = self.rnn_cell(torch.cat([_obs, _action], dim=1), h) # compute next state    

        prediction = {
            "obs" : torch.stack(pre_obs),
        }

        if self.action_minic:
            prediction['action'] = torch.stack(pre_action)
        
        if self.predict_reward:
            prediction['reward'] = torch.stack(pre_reward)

        return prediction

    def get_trainer(self):
        return PredictorTrainer

    # API for environmental rollout
    def reset(self, obs):
        state = self.rnn_cell(torch.cat([obs, torch.zeros(obs.shape[0], self.action_dim, dtype=obs.dtype, device=obs.device)], dim=1))
        return state 

    def step(self, pre_state, action):
        obs = self.obs_pre(pre_state).mode()
        next_state = self.rnn_cell(torch.cat([obs, action], dim=1), pre_state)
        reward = self.reward_pre(next_state).mode()
        return next_state, reward