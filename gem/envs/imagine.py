import numpy as np
import gym

class Imagine:

    def __init__(self, sensor, predictor):
        super().__init__()
        self.sensor = sensor
        self.predictor = predictor
        self.state_dim = predictor.state_dim
        self.action_dim = predictor.action_dim

    @property
    def observation_space(self):
        return gym.spaces.Box(-np.inf, np.inf, [self.state_dim], dtype=np.float32)

    @property
    def action_space(self):
        return gym.spaces.Box(-1, 1, [self.action_dim], dtype=np.float32)

    def step(self, action):
        self.state, reward = self.predictor.step(self.state, action)

        info = None
        done = False

        return self.state, reward, done, info

    def reset(self, obs):
        emb = self.sensor.encode(obs)
        self.state = self.predictor.reset(emb)
        return self.state

    def render(self, state):
        emb = self.predictor.obs_pre(state).mode()
        return self.sensor.decode(emb)

    def to(self, device):
        self.sensor = self.sensor.to(device)
        self.predictor = self.predictor.to(device)