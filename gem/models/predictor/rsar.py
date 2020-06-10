import torch

from gem.distributions import Normal
from gem.distributions.utils import get_kl, stack_normal
from gem.modules.decoder import MLPDecoder, ActionDecoder
from .trainer import PredictorTrainer

class RSAR(torch.nn.Module):
    def __init__(self, emb_dim, action_dim, stoch_dim, hidden_dim, free_nats=3.0, kl_scale=1.0,
                 action_mimic=False, actor_mode='continuous', predict_reward=True, 
                 decoder_config={"hidden_layers" : 2, "features" : 512, "activation" : 'elu'}):
        super().__init__()

        self.emb_dim = emb_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.stoch_dim = stoch_dim
        self.free_nats = free_nats
        self.kl_scale = kl_scale
        self.action_minic = action_mimic
        self.predict_reward = predict_reward
        self.decoder_config = decoder_config

        self.state_dim = hidden_dim + stoch_dim

        self.rnn_cell = torch.nn.GRUCell(stoch_dim + action_dim, hidden_dim)

        self.stoch_trans = MLPDecoder(hidden_dim + emb_dim, stoch_dim, **decoder_config)

        self.emb_pre = MLPDecoder(self.state_dim, emb_dim, dist_type='fix_std', **decoder_config)
        
        if self.action_minic:
            self.action_pre = ActionDecoder(self.state_dim, action_dim, mode=actor_mode, **decoder_config)

        if self.predict_reward:
            self.reward_pre = MLPDecoder(self.state_dim, 1, dist_type='fix_std', **decoder_config)

        self.last_emb = None

    def _reset(self, emb0):
        batch_size = emb0.shape[0]
        dtype = emb0.dtype
        device = emb0.device

        return torch.zeros(batch_size, self.hidden_dim, dtype=dtype, device=device), \
            torch.zeros(batch_size, self.stoch_dim, dtype=dtype, device=device)

    def forward(self, emb, action, reward=None, use_emb_loss=True):
        """
            Inputs are all tensor[T, B, *]
        """
        
        T, B = emb.shape[:2]

        h, s = self._reset(emb[0])

        states = []
        posterior_dists = []
        prior_dists = []

        for i in range(T):
            h, s, posterior_dist, prior_dist = self.whole_step(h, s, action[i], emb[max(i-1, 0)])

            state = torch.cat([h, s], dim=1)
            states.append(state)
            posterior_dists.append(posterior_dist)
            prior_dists.append(prior_dist)

        info = {}
        prediction = {"state" : torch.stack(states)}
        loss = 0

        kl_loss = get_kl(stack_normal(posterior_dists), stack_normal(prior_dists))
        prediction['kl'] = kl_loss
        kl_loss = torch.sum(kl_loss) / (T * B)
        info['kl_loss'] = kl_loss.item()
        if kl_loss < self.free_nats:
            kl_loss = kl_loss.detach()
        loss += kl_loss * self.kl_scale

        states = torch.cat(states, dim=0).contiguous()
        emb_dist = self.emb_pre(states)
        prediction['emb'] = emb_dist.mode().view(T, B, *emb.shape[2:])
        if use_emb_loss:
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

        T, B = emb0.shape[:2]

        h, s = self._reset(emb0)
        state = torch.cat([h, s], dim=1)

        pre_emb = []
        pre_action = []
        pre_reward = []    

        for i in range(horizon):
            _action = self.action_pre(state.detach()).mode() if action is None else action[i]
            h, s, _ = self.obs_step(h, s, _action, emb0) if i < 2 else self.img_step(h, s, _action)
            state = torch.cat([h, s], dim=1)
            
            _emb = self.emb_pre(state).mode()
            pre_emb.append(_emb)

            if self.action_minic:
                pre_action.append(_action)

            if self.predict_reward:
                pre_reward.append(self.reward_pre(state).mode())

        prediction = {
            "emb" : torch.stack(pre_emb),
        }

        if self.action_minic:
            prediction['action'] = torch.stack(pre_action)
        
        if self.predict_reward:
            prediction['reward'] = torch.stack(pre_reward)

        return prediction

    def whole_step(self, prev_h, prev_s, prev_a, pre_emb):
        next_h, _, prior_dist = self.img_step(prev_h, prev_s, prev_a)

        posterior_dist = self.stoch_trans(torch.cat([next_h, pre_emb], dim=1))

        return next_h, posterior_dist.sample(), posterior_dist, prior_dist

    def obs_step(self, prev_h, prev_s, prev_a, pre_emb):
        next_h = self.rnn_cell(torch.cat([prev_s, prev_a], dim=1), prev_h) 

        posterior_dist = self.stoch_trans(torch.cat([next_h, pre_emb], dim=1))

        return next_h, posterior_dist.sample(), posterior_dist

    def img_step(self, prev_h, prev_s, prev_a):
        next_h = self.rnn_cell(torch.cat([prev_s, prev_a], dim=1), prev_h) 

        pre_emb = self.emb_pre(torch.cat([prev_h, prev_s], dim=1)).mode()

        prior_dist = self.stoch_trans(torch.cat([next_h, pre_emb], dim=1))

        return next_h, prior_dist.sample(), prior_dist

    def get_trainer(self):
        return PredictorTrainer

    # API for environmental rollout
    def reset(self, emb):
        self.last_emb = emb
        h, s = self._reset(emb)
        h, s, _ = self.obs_step(h, s, torch.zeros(emb.shape[0], self.action_dim, dtype=emb.dtype, device=emb.device), emb)
        return torch.cat([h, s], dim=1)

    def step(self, pre_state, action, emb=None):
        h, s = torch.split(pre_state, [self.hidden_dim, self.stoch_dim], dim=1)
        if self.last_emb is None or not self.last_emb.shape[0] == action.shape[0]:
            h, s, _ = self.img_step(h, s, action)
        else:
            h, s, _ = self.obs_step(h, s, action, self.last_emb)
        self.last_emb = emb
        next_state = torch.cat([h, s], dim=1)
        reward = self.reward_pre(next_state).mode()
        return next_state, reward