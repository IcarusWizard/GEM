import torch

from gem.distributions import Normal
from gem.modules.decoder import MLPDecoder, ActionDecoder
from .trainer import PredictorTrainer

class RAR(torch.nn.Module):
    def __init__(self, emb_dim, action_dim, hidden_dim, warm_up=10, action_mimic=False, actor_mode='continuous', predict_reward=True, 
                 decoder_config={"hidden_layers" : 2, "features" : 512, "activation" : 'elu'}):
        super().__init__()

        self.emb_dim = emb_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.warm_up = warm_up
        self.action_minic = action_mimic
        self.predict_reward = predict_reward
        self.decoder_config = decoder_config

        self.state_dim = hidden_dim

        self.rnn_cell = torch.nn.GRUCell(emb_dim + action_dim, hidden_dim)

        self.emb_pre = MLPDecoder(hidden_dim, emb_dim, dist_type='fix_std', **decoder_config)
        
        if self.action_minic:
            self.action_pre = ActionDecoder(hidden_dim, action_dim, mode=actor_mode, **decoder_config)

        if self.predict_reward:
            self.reward_pre = MLPDecoder(hidden_dim, 1, dist_type='fix_std', **decoder_config)

        self.last_emb = None

    def forward(self, emb, action, reward=None, use_emb_loss=True):
        """
            Inputs are all tensor[T, B, *]
        """
        
        T, B = emb.shape[:2]

        states = []

        # compute state1 by assuming emb -1 is the same as emb 0, and take no action 
        state = self.rnn_cell(torch.cat([emb[0], torch.zeros_like(action[0])], dim=1)) 

        for i in range(T):
            states.append(state)

            _action = action[i]

            emb_dist = self.emb_pre(state)

            if i < self.warm_up:
                _emb = emb[i]
            else:
                _emb = emb_dist.mode()

            state = self.rnn_cell(torch.cat([_emb, _action], dim=1), state) # compute next state

        info = {}
        prediction = {"state" : torch.stack(states)}
        loss = 0

        states = torch.cat(states, dim=0).contiguous()
        emb_dist = self.emb_pre(states)
        prediction['emb'] = emb_dist.mode().view(T, B, *emb.shape[2:])
        # if use_emb_loss:
        emb_loss = - torch.sum(emb_dist.log_prob(emb.view(T * B, *emb.shape[2:]))) / (T * B)
        info['emb_loss'] = emb_loss.item()
        loss += emb_loss

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

    def generate(self, emb0, horizon, action=None):
        assert action is not None or self.action_minic

        pre_emb = []
        pre_action = []
        pre_reward = []    

        # compute state1 by assuming emb -1 is the same as emb 0, and take no action 
        state = self.rnn_cell(torch.cat([emb0, torch.zeros(emb0.shape[0], self.action_dim, dtype=emb0.dtype, device=emb0.device)], dim=1)) 

        for i in range(horizon):
            _emb = self.emb_pre(state).mode()
            _action = self.action_pre(state.detach()).mode() if action is None else action[i]
            
            pre_emb.append(_emb)

            if self.action_minic:
                pre_action.append(_action)

            if self.predict_reward:
                pre_reward.append(self.reward_pre(state).mode())

            state = self.rnn_cell(torch.cat([_emb, _action], dim=1), state) # compute next state    

        prediction = {
            "emb" : torch.stack(pre_emb),
        }

        if self.action_minic:
            prediction['action'] = torch.stack(pre_action)
        
        if self.predict_reward:
            prediction['reward'] = torch.stack(pre_reward)

        return prediction

    def get_trainer(self):
        return PredictorTrainer

    # API for environmental rollout
    def reset(self, emb):
        self.last_emb = emb
        state = self.rnn_cell(torch.cat([emb, torch.zeros(emb.shape[0], self.action_dim, dtype=emb.dtype, device=emb.device)], dim=1))
        return state 

    def step(self, pre_state, action, emb=None):
        if self.last_emb is None or not self.last_emb.shape[0] == action.shape[0]:
            last_emb = self.emb_pre(pre_state).mode()
        else:
            last_emb = self.last_emb
        next_state = self.rnn_cell(torch.cat([last_emb, action], dim=1), pre_state)
        self.last_emb = emb
        reward = self.reward_pre(next_state).mode()
        return next_state, reward