import torch
import numpy as np
from itertools import chain

from gem.modules.decoder import ActionDecoder, MLPDecoder
from gem.modules import MLP

from .trainer import VGCtTrainer

class VGC(torch.nn.Module):
    def __init__(self, state_dim, action_dim, features, hidden_layers, actor_mode='continuous'):
        super().__init__()

        self.actor = ActionDecoder(state_dim, action_dim, features, hidden_layers, mode=actor_mode)
        self.critic = MLPDecoder(state_dim, 1, features, hidden_layers, dist_type='fix_std', activation='elu')

    def forward(self, state):
        return self.actor(state), self.critic(state)

    def get_action_dist(self, state):
        return self.actor(state)

    def get_critic_dist(self, state):
        return self.critic(state)

    def get_actor_parameters(self):
        return self.actor.parameters()

    def get_critic_parameters(self):
        return self.critic.parameters()

    def get_trainer(self):
        return VGCtTrainer

class VGCS(torch.nn.Module):
    def __init__(self, state_dim, action_dim, features, hidden_layers, actor_mode='continuous'):
        super().__init__()

        self.feature_net = MLP(state_dim, features, features, hidden_layers-1, activation='elu')
        self.actor_head = ActionDecoder(features, action_dim, features, 0, mode=actor_mode)
        self.critic_head = MLPDecoder(features, 1, features, 0, dist_type='fix_std')

    def forward(self, state):
        features = self.feature_net(state)
        return self.actor_head(features), self.critic_head(features)

    def get_action_dist(self, state):
        return self.actor_head(self.feature_net(state))

    def get_critic_dist(self, state):
        return self.critic_head(self.feature_net(state))

    def get_actor_parameters(self):
        return chain(self.feature_net.parameters(), self.actor_head.parameters())

    def get_critic_parameters(self):
        return chain(self.feature_net.parameters(), self.critic_head.parameters())

    def get_trainer(self):
        return VGCtTrainer