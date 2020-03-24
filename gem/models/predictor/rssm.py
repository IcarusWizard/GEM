import torch
from torch.distributions import Normal

from degmo.modules import MLP
from .trainer import PredictorTrainer

class GRUBaseline(torch.nn.Module):
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

        self.obs_pre = MLP(hidden_dim, obs_dim, **decoder_config)
        
        if self.action_minic:
            self.action_pre = MLP(hidden_dim, action_dim, **decoder_config)

        if self.predict_reward:
            self.reward_pre = MLP(hidden_dim, 1, **decoder_config)

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
            
            pre_obs.append(self.obs_pre(h))
            obs_dis = Normal(pre_obs[-1], 1)
            pre_obs_loss -= torch.sum(obs_dis.log_prob(_obs))

            if self.action_minic:
                pre_action.append(self.action_pre(h.detach()))
                action_dis = Normal(pre_action[-1], 1)
                pre_action_loss -= torch.sum(action_dis.log_prob(_action))

            if self.predict_reward:
                assert reward is not None
                _reward = reward[i]
                pre_reward.append(self.reward_pre(h))
                reward_dis = Normal(pre_reward[-1], 1)
                pre_reward_loss -= torch.sum(reward_dis.log_prob(_action))

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
            _obs = self.obs_pre(h)
            _action = self.action_pre(h.detach()) if action is None else action[i]
            
            pre_obs.append(_obs)

            if self.action_minic:
                pre_action.append(_action)

            if self.predict_reward:
                pre_reward.append(self.reward_pre(h))

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