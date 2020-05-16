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
        self.critic_target = MLPDecoder(state_dim, 1, features, hidden_layers, dist_type='fix_std', activation='elu')

        self.update_target()

    def forward(self, state):
        return self.actor(state), self.critic(state)

    def get_action_dist(self, state):
        return self.actor(state)

    def get_critic_dist(self, state):
        return self.critic(state)

    def get_critic_target(self, state):
        return self.critic_target(state).mode()

    def get_actor_parameters(self):
        return self.actor.parameters()

    def get_critic_parameters(self):
        return self.critic.parameters()

    @torch.no_grad()
    def update_target(self):
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.requires_grad_(False)

    def get_trainer(self):
        return VGCtTrainer