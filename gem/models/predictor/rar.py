import torch

from gem.distributions import Normal
from gem.modules.decoder import MLPDecoder, ActionDecoder
from .trainer import PredictorTrainer

class RAR(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, action_mimic=False, actor_mode='continuous', predict_reward=True, 
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
            self.action_pre = ActionDecoder(hidden_dim, action_dim, mode=actor_mode, **decoder_config)

        if self.predict_reward:
            self.reward_pre = MLPDecoder(hidden_dim, 1, dist_type='fix_std', **decoder_config)

    def forward(self, obs, action, reward=None):
        """
            Inputs are all tensor[T, B, *]
        """
        
        T, B = obs.shape[:2]

        states = []

        # compute state1 by assuming obs -1 is the same as obs 0, and take no action 
        state = self.rnn_cell(torch.cat([obs[0], torch.zeros_like(action[0])], dim=1)) 

        for i in range(T):
            states.append(state)

            _obs = obs[i]
            _action = action[i]

            obs_dist = self.obs_pre(state)

            _obs = obs_dist.mode()

            state = self.rnn_cell(torch.cat([_obs, _action], dim=1), state) # compute next state

        info = {}
        prediction = {}
        loss = 0

        states = torch.cat(states, dim=0).contiguous()
        obs_dist = self.obs_pre(states)
        obs_loss = - torch.sum(obs_dist.log_prob(obs.view(T * B, *obs.shape[2:]))) / (T * B)
        info['obs_loss'] = obs_loss.item()
        prediction['obs'] = obs_dist.mode().view(T, B, *obs.shape[2:])
        loss += obs_loss

        if self.action_minic:
            action_dist = self.action_pre(states[:-B])
            action_loss = - torch.sum(action_dist.log_prob(action[1:].view((T - 1) * B, *action.shape[2:]))) / ((T - 1) * B)
            info['action_loss'] = action_loss.item()
            prediction['action'] = action_dist.mode().view(T - 1, B, *action.shape[2:])
            loss += action_loss
        
        if self.predict_reward:
            reward_dist = self.reward_pre(states)
            reward_loss = - torch.sum(reward_dist.log_prob(reward.view(T * B, 1))) / (T * B)
            info['reward_loss'] = reward_loss.item()
            prediction['reward'] = reward_dist.mode().view(T, B, 1)
            loss += reward_loss

        info['loss'] = loss.item()

        return loss, prediction, info

    def generate(self, obs0, horizon, action=None):
        assert action is not None or self.action_minic

        pre_obs = []
        pre_action = []
        pre_reward = []    

        # compute state1 by assuming obs -1 is the same as obs 0, and take no action 
        state = self.rnn_cell(torch.cat([obs0, torch.zeros(obs0.shape[0], self.action_dim, dtype=obs0.dtype, device=obs0.device)], dim=1)) 

        for i in range(horizon):
            _obs = self.obs_pre(state).mode()
            _action = self.action_pre(state.detach()).mode() if action is None else action[i]
            
            pre_obs.append(_obs)

            if self.action_minic:
                pre_action.append(_action)

            if self.predict_reward:
                pre_reward.append(self.reward_pre(state).mode())

            state = self.rnn_cell(torch.cat([_obs, _action], dim=1), state) # compute next state    

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
        self.last_obs = obs
        state = self.rnn_cell(torch.cat([obs, torch.zeros(obs.shape[0], self.action_dim, dtype=obs.dtype, device=obs.device)], dim=1))
        return state 

    def step(self, pre_state, action, obs=None):
        if self.last_obs is None:
            last_obs = self.obs_pre(pre_state).mode()
        else:
            last_obs = self.last_obs
        next_state = self.rnn_cell(torch.cat([last_obs, action], dim=1), pre_state)
        self.last_obs = obs
        reward = self.reward_pre(next_state).mode()
        return next_state, reward